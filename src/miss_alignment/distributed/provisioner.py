"""Worker provisioners: spawn local child processes or submit cluster jobs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
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

    def live_worker_count(self) -> int:
        return sum(1 for p in self._procs.values() if p.poll() is None)

    def worker_counts_by_type(self) -> dict[str, int]:
        return {"local": self.live_worker_count()}

    def ensure_workers(self, n_workers: int) -> None:
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


class ClusterProvisioner(WorkerProvisioner):
    """Submits cluster jobs; resubmits when alive workers fall below target."""

    def __init__(self, queue_dir: Path, config: ClusterConfig) -> None:
        self._queue_dir = queue_dir
        self._config = config
        self._job_ids: list[str] = []
        # Monotonically increasing index for unique script names. Separate
        # from _job_ids so we never reuse a script index even after replenishment.
        self._next_index: int = 0
        self._scripts_dir = queue_dir / "cluster"
        self._scripts_dir.mkdir(parents=True, exist_ok=True)

    def live_worker_count(self) -> int:
        return _live_worker_dirs(self._queue_dir / "running")

    def worker_counts_by_type(self) -> dict[str, int]:
        return {"cluster": self.live_worker_count()}

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
            self._job_ids.append(match.group(1))

    def ensure_workers(self, n_workers: int) -> None:
        """Submit new jobs until live worker count reaches n_workers."""
        alive = self.live_worker_count()
        deficit = n_workers - alive
        for _ in range(deficit):
            self._submit_one()

    def shutdown(self) -> None:
        for job_id in self._job_ids:
            cancel_cmd = self._config.cancel.replace("{{job_id}}", job_id)
            subprocess.run(cancel_cmd, shell=True, stderr=subprocess.DEVNULL)
        self._job_ids.clear()


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
