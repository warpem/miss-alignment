"""Worker provisioners: spawn local child processes or submit cluster jobs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from .config import ClusterConfig


class WorkerProvisioner(ABC):
    @abstractmethod
    def ensure_workers(self, n_workers: int) -> None:
        """Ensure workers are running.

        Called once at startup and on each scheduler tick to respawn dead workers.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Terminate all managed workers."""


class LocalProvisioner(WorkerProvisioner):
    """Spawns one miss-alignment worker subprocess per GPU device."""

    def __init__(self, queue_dir: Path, devices: list[int]) -> None:
        self._queue_dir = queue_dir
        self._devices = devices
        self._procs: dict[int, subprocess.Popen] = {}

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
    """Submits exactly n_workers cluster jobs, each running until the queue drains."""

    def __init__(self, queue_dir: Path, config: ClusterConfig) -> None:
        self._queue_dir = queue_dir
        self._config = config
        self._job_ids: list[str] = []
        self._scripts_dir = queue_dir / "cluster"
        self._scripts_dir.mkdir(parents=True, exist_ok=True)

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

    def ensure_workers(self, n_workers: int) -> None:
        already = len(self._job_ids)
        for i in range(already, n_workers):
            script_path = self._render_script(i)
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

    def ensure_workers(self, n_workers: int) -> None:
        for p in self._provisioners:
            p.ensure_workers(n_workers)

    def shutdown(self) -> None:
        for p in self._provisioners:
            p.shutdown()
