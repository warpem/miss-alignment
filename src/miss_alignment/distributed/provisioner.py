"""Worker provisioners: spawn local child processes or submit cluster jobs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ClusterConfig

# A worker dir with no heartbeat file yet is assumed alive for this long
# after the dir was created (worker is still starting up).
_STARTUP_GRACE_S = 60.0
# A worker dir whose newest heartbeat is older than this is considered dead.
_STALL_TIMEOUT_S = 120.0


def _live_worker_dirs(running_dir: Path) -> int:
    """Count running/<wid>/ subdirs whose heartbeat is fresh enough to be alive."""
    if not running_dir.exists():
        return 0
    count = 0
    now = time.time()
    for wdir in running_dir.iterdir():
        if not wdir.is_dir():
            continue
        ticks = list(wdir.glob("hb-*"))
        if ticks:
            newest_mtime = None
            for t in ticks:
                try:
                    mtime = t.stat().st_mtime
                    if newest_mtime is None or mtime > newest_mtime:
                        newest_mtime = mtime
                except FileNotFoundError:
                    pass  # heartbeat thread rotated this tick; ignore
            if newest_mtime is not None and now - newest_mtime < _STALL_TIMEOUT_S:
                count += 1
        else:
            # No heartbeat yet — consider alive if dir was created recently.
            try:
                if now - wdir.stat().st_mtime < _STARTUP_GRACE_S:
                    count += 1
            except FileNotFoundError:
                pass  # dir vanished between iterdir and stat; skip
    return count


class WorkerProvisioner(ABC):
    @abstractmethod
    def ensure_workers(self, n_workers: int) -> None:
        """Ensure workers are running.

        Called once at startup and on each scheduler tick to respawn dead workers.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Terminate all managed workers."""

    @abstractmethod
    def live_worker_count(self) -> int:
        """Return the number of workers currently considered alive."""

    def worker_counts_by_type(self) -> dict[str, int]:
        """Return a dict of label → live count for display purposes."""
        return {"workers": self.live_worker_count()}


class LocalProvisioner(WorkerProvisioner):
    """Spawns one miss-alignment worker subprocess per GPU device."""

    def __init__(self, queue_dir: Path, devices: list[int]) -> None:
        self._queue_dir = queue_dir
        self._devices = devices
        self._procs: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def live_worker_count(self) -> int:
        with self._lock:
            return sum(1 for p in self._procs.values() if p.poll() is None)

    def worker_counts_by_type(self) -> dict[str, int]:
        return {"local": self.live_worker_count()}

    def ensure_workers(self, n_workers: int) -> None:
        with self._lock:
            for device in self._devices:
                proc = self._procs.get(device)
                if proc is not None and proc.poll() is None:
                    continue  # still running
                new_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "miss_alignment",
                        "worker",
                        "--queue-dir",
                        str(self._queue_dir),
                        "--device",
                        str(device),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=sys.stderr,
                )
                self._procs[device] = new_proc

    def shutdown(self) -> None:
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self._procs.values():
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()


# ---------------------------------------------------------------------------
# Per-scheduler status parsers.
# Each entry maps a comma-separated "id,STATUS" token to alive/dead.
# The status_list command is expected to emit one "id,STATUS" pair per line
# (e.g. SLURM: squeue -u $USER -h -o "%i,%T").
# ---------------------------------------------------------------------------

# Statuses where the job is actively executing on a node.
_RUNNING_STATUSES: dict[str, set[str]] = {
    "slurm": {"RUNNING", "R", "COMPLETING", "CG", "RESIZING"},
    "lsf":   {"RUN"},
    "pbs":   {"R", "E"},
    "sge":   {"r", "t", "Rr"},
}

# Statuses where the job is alive but not yet on a node.
_PENDING_STATUSES: dict[str, set[str]] = {
    "slurm": {"PENDING", "PD", "SUSPENDED", "S"},
    "lsf":   {"PEND", "SSUSP", "USUSP", "PSUSP"},
    "pbs":   {"Q", "H", "W", "T", "S"},
    "sge":   {"qw", "Rq", "hqw", "hRwq"},
}


# Jobs absent from squeue for longer than this are considered gone.
# Must be long enough to cover scheduler registration delay (usually <30s)
# plus a full squeue poll cycle.
_JOB_GRACE_S = 120.0


def _parse_status_output(
    output: str,
    our_job_ids: set[str],
    scheduler: str,
    custom_alive_statuses: list[str],
    custom_status_regex: str,
) -> dict[str, str]:
    """Parse status_list output; return dict[job_id → 'running'|'pending'].

    Only jobs from our_job_ids that appear in the output are returned.
    Jobs absent from the output are not included — the caller decides
    whether to prune them based on how long they have been absent.
    """
    result: dict[str, str] = {}

    if scheduler == "auto":
        schedulers_to_try = ["slurm", "lsf", "pbs", "sge"]
    elif scheduler == "custom":
        schedulers_to_try = []
    else:
        schedulers_to_try = [scheduler]

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        if "," in line:
            job_id, _, status_token = line.partition(",")
            job_id = job_id.strip()
            status_token = status_token.strip()
        else:
            job_id = line
            status_token = ""

        if job_id not in our_job_ids:
            continue

        if scheduler == "custom":
            if status_token in custom_alive_statuses or not status_token:
                result[job_id] = "pending"
            # Otherwise not recognised as alive — leave absent from result.
            continue

        if not status_token:
            result[job_id] = "pending"
            continue

        for sched in schedulers_to_try:
            if status_token in _RUNNING_STATUSES[sched]:
                result[job_id] = "running"
                break
            if status_token in _PENDING_STATUSES[sched]:
                result[job_id] = "pending"
                break
        # Jobs with unrecognised status are simply absent from the result.

    return result


class ClusterProvisioner(WorkerProvisioner):
    """Submits cluster jobs; tracks liveness by querying the batch scheduler.

    The status_list command (configured in cluster_config.json) is called each
    scheduler tick to get the set of alive jobs. Supports SLURM, LSF, PBS, SGE,
    and custom schedulers. Scheduler type is auto-detected from status output
    unless 'scheduler' is set explicitly in the config.

    Jobs absent from squeue output are kept for _JOB_GRACE_S seconds before
    being pruned — this covers the window between submission and scheduler
    registration, which can be tens of seconds on busy clusters.

    Recommended status_list format (one job per line, id,STATUS):
      SLURM: squeue -u $USER -h -o "%i,%T"
    """

    def __init__(self, queue_dir: Path, config: ClusterConfig) -> None:
        self._queue_dir = queue_dir
        self._config = config
        # job_id → submission timestamp
        self._job_submit_time: dict[str, float] = {}
        # Monotonically increasing index for unique script names so indices
        # never repeat even after preemption-triggered resubmissions.
        self._next_index: int = 0
        self._scripts_dir = queue_dir / "cluster"
        self._scripts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _job_ids(self) -> list[str]:
        return list(self._job_submit_time.keys())

    def _query_job_states(self) -> dict[str, str]:
        """Query the scheduler; return dict[job_id → 'running'|'pending'].

        Jobs absent from squeue are pruned only after _JOB_GRACE_S seconds,
        allowing for scheduler registration delay. On error, returns all
        tracked jobs as 'pending'.
        """
        if not self._job_submit_time:
            return {}
        our_ids = set(self._job_submit_time.keys())
        status_cmd = self._config.status_list.replace(
            "{{user}}", os.environ.get("USER", os.environ.get("USERNAME", ""))
        )
        try:
            result = subprocess.run(
                status_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=30,
            )
            states = _parse_status_output(
                output=result.stdout,
                our_job_ids=our_ids,
                scheduler=self._config.scheduler,
                custom_alive_statuses=self._config.custom_alive_statuses,
                custom_status_regex=self._config.custom_status_regex,
            )
            now = time.time()
            # Prune jobs absent from squeue only after the grace period.
            for jid in list(self._job_submit_time):
                if jid not in states:
                    age = now - self._job_submit_time[jid]
                    if age > _JOB_GRACE_S:
                        del self._job_submit_time[jid]
            return states
        except Exception:
            return {jid: "pending" for jid in self._job_submit_time}

    def live_worker_count(self) -> int:
        return len(self._job_submit_time)

    def worker_counts_by_type(self) -> dict[str, int]:
        states = self._query_job_states()
        running = sum(1 for s in states.values() if s == "running")
        # pending = squeue-visible pending + not-yet-visible (within grace period)
        visible = set(states.keys())
        pending = sum(1 for s in states.values() if s == "pending")
        pending += sum(1 for jid in self._job_submit_time if jid not in visible)
        counts = {}
        if running:
            counts["cluster-running"] = running
        if pending:
            counts["cluster-pending"] = pending
        return counts

    def ensure_workers(self, n_workers: int) -> None:
        """Submit new jobs until tracked job count reaches n_workers.

        Prunes grace-expired absent jobs first, then submits the deficit.
        """
        self._query_job_states()
        deficit = n_workers - len(self._job_submit_time)
        for _ in range(deficit):
            self._submit_one()

    def _render_script(self, index: int) -> Path:
        template_text = self._config.script_path.read_text()
        # $(hostname) and $$ are expanded by the compute node's shell at runtime.
        command = (
            f"miss-alignment worker"
            f" --queue-dir {self._queue_dir}"
            f" --device 0"
            f' --worker-id "$(hostname)-$$-{index}"'
        )
        logs_dir = self._queue_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        rendered = template_text.replace("{{command}}", command)
        rendered = rendered.replace("{{tasks_dir}}", str(self._queue_dir))
        rendered = rendered.replace("{{logs_dir}}", str(logs_dir))
        for key, value in os.environ.items():
            if key.startswith("MISS_CLUSTER_VAR_"):
                var_name = key[len("MISS_CLUSTER_VAR_") :].lower()
                rendered = rendered.replace(f"{{{{{var_name}}}}}", value)

        script_path = self._scripts_dir / f"worker-{index}.sh"
        script_path.write_text(rendered)
        return script_path

    def _submit_one(self) -> None:
        index = self._next_index
        self._next_index += 1
        script_path = self._render_script(index)
        submit_cmd = self._config.submit.replace(
            "{{script_path}}", str(script_path)
        )
        result = subprocess.run(
            submit_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        match = re.search(self._config.submit_job_id_regex, result.stdout)
        if match:
            self._job_submit_time[match.group(1)] = time.time()

    def shutdown(self) -> None:
        def _cancel(job_id: str) -> None:
            cancel_cmd = self._config.cancel.replace("{{job_id}}", job_id)
            subprocess.run(cancel_cmd, shell=True, stderr=subprocess.DEVNULL)

        job_ids = list(self._job_submit_time.keys())
        with ThreadPoolExecutor(max_workers=min(32, len(job_ids) or 1)) as pool:
            list(as_completed([pool.submit(_cancel, jid) for jid in job_ids]))
        self._job_submit_time.clear()


class CompositeProvisioner(WorkerProvisioner):
    """Delegates to multiple provisioners simultaneously.

    Used to run local GPU workers alongside cluster workers so local GPUs
    are not left idle during cluster-distributed alignment phases.
    """

    def __init__(self, provisioners: list[WorkerProvisioner]) -> None:
        self._provisioners = provisioners

    def live_worker_count(self) -> int:
        return sum(p.live_worker_count() for p in self._provisioners)

    def worker_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self._provisioners:
            for label, n in p.worker_counts_by_type().items():
                counts[label] = counts.get(label, 0) + n
        return counts

    def ensure_workers(self, n_workers: int) -> None:
        for p in self._provisioners:
            p.ensure_workers(n_workers)

    def shutdown(self) -> None:
        for p in self._provisioners:
            p.shutdown()
