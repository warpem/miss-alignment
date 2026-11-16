"""Head-node coordinator: writes tasks, starts provisioner, blocks until done."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from uuid import uuid4
from pathlib import Path

import tqdm

from .config import load_cluster_config
from .provisioner import (
    ClusterProvisioner,
    CompositeProvisioner,
    LocalProvisioner,
    WorkerProvisioner,
)
from .queue import (
    QueueLayout,
    TaskSpec,
    clear_queue,
    compute_fingerprint,
    write_pending,
)

_POLL_INTERVAL_S = 0.5
_SCHEDULER_INTERVAL_S = 10.0
_MANAGER_HB_INTERVAL_S = 5.0
_WORKER_STALL_TIMEOUT_S = 120.0


def _format_task_id(index: int, tilt_series_path: Path) -> str:
    return f"{index:07d}-{tilt_series_path.stem}"


def _write_manager_hb(layout: QueueLayout, seq: int) -> None:
    new_hb = layout.manager_hb / f"hb-{seq}"
    new_hb.write_text("")
    if seq > 0:
        (layout.manager_hb / f"hb-{seq - 1}").unlink(missing_ok=True)


def _sweep_stalled_workers(layout: QueueLayout) -> None:
    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        ticks = list(worker_dir.glob("hb-*"))
        latest_mtime = None
        for tick in ticks:
            try:
                mtime = tick.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            except FileNotFoundError:
                pass  # heartbeat thread replaced this tick; ignore
        if latest_mtime is not None:
            age = time.time() - latest_mtime
        else:
            age = time.time() - worker_dir.stat().st_mtime

        if age <= _WORKER_STALL_TIMEOUT_S:
            continue

        for task_file in worker_dir.glob("*.json"):
            # Skip tasks that already completed while the worker was being swept.
            if (layout.done / task_file.name).exists():
                task_file.unlink(missing_ok=True)
                continue
            if (layout.failed / task_file.name).exists():
                task_file.unlink(missing_ok=True)
                continue
            dest = layout.pending / task_file.name
            try:
                os.rename(task_file, dest)
            except FileNotFoundError:
                pass
        for hb in worker_dir.glob("hb-*"):
            hb.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass


def _format_worker_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{label}={n}" for label, n in sorted(counts.items()))


def _heartbeat_thread(
    layout: QueueLayout, stop_event: threading.Event, error_box: list
) -> None:
    # Scheduler queries/submission can block; keep the manager heartbeat independent.
    seq = 1
    try:
        while not stop_event.wait(_MANAGER_HB_INTERVAL_S):
            _write_manager_hb(layout, seq)
            seq += 1
    except Exception as exc:
        error_box.append(exc)
        stop_event.set()


def _scheduler_thread(
    layout: QueueLayout,
    provisioner: WorkerProvisioner,
    n_workers: int,
    stop_event: threading.Event,
    error_box: list,
    worker_counts: dict,
) -> None:
    # This thread alone provisions workers, including at startup. In particular,
    # never race a cluster submission against another ensure_workers() call.
    first = True
    try:
        while not stop_event.is_set():
            _sweep_stalled_workers(layout)
            n_pending = len(list(layout.pending.glob("*.json")))
            if n_pending and not stop_event.is_set():
                provisioner.ensure_workers(
                    n_workers if first else min(n_workers, n_pending)
                )
            first = False
            worker_counts.clear()
            worker_counts.update(provisioner.worker_counts_by_type())
            stop_event.wait(_SCHEDULER_INTERVAL_S)
    except Exception as exc:
        error_box.append(exc)
        stop_event.set()


class DistributedPool:
    """Own workers across sequential batches, closing them together at run end.

    Persistent sessions use a unique queue directory: a delayed worker from an
    earlier run can never claim a new run's tasks. Each batch also has unique
    task IDs, so late result publication cannot cross an iteration barrier.
    Initialization is lazy, so an unused pool does not submit any workers.
    """

    def __init__(
        self,
        devices: list[int],
        n_cluster_workers: int | None,
        queue_root: Path,
        *,
        persistent: bool = False,
    ) -> None:
        self.devices = devices
        self.n_cluster_workers = n_cluster_workers
        self.persistent = persistent
        self._queue_parent = Path(queue_root).resolve()
        self.layout = QueueLayout(
            self._queue_parent / uuid4().hex if persistent else self._queue_parent
        )
        self._stop = threading.Event()
        self._errors: list[Exception] = []
        self._worker_counts: dict[str, int] = {}
        self._provisioner: WorkerProvisioner | None = None
        self._threads: list[threading.Thread] = []
        self._batch = 0
        self._owns_queue = False
        self._closed = False

    def __enter__(self) -> DistributedPool:
        if self._closed:
            raise RuntimeError("Worker pool is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _initialize(self) -> None:
        self.layout.ensure_directories()
        self._owns_queue = True
        clear_queue(self.layout)
        self.layout.stop_file.unlink(missing_ok=True)
        if self.n_cluster_workers is not None:
            if self.n_cluster_workers < 1:
                raise ValueError("n_cluster_workers must be positive")
            cluster = ClusterProvisioner(
                queue_dir=self.layout.root,
                config=load_cluster_config(),
                persistent=self.persistent,
            )
            if self.devices:
                local = LocalProvisioner(
                    queue_dir=self.layout.root,
                    devices=self.devices,
                    persistent=self.persistent,
                )
                self._provisioner = CompositeProvisioner([cluster, local])
            else:
                self._provisioner = cluster
            self._n_workers = self.n_cluster_workers
        else:
            if not self.devices:
                raise RuntimeError(
                    "No local GPUs are available; use --n-cluster-workers "
                    "to run alignment on cluster workers."
                )
            self._provisioner = LocalProvisioner(
                queue_dir=self.layout.root,
                devices=self.devices,
                persistent=self.persistent,
            )
            self._n_workers = len(self.devices)
        _write_manager_hb(self.layout, seq=0)

    def _start(self) -> None:
        # Publishing a large first batch on shared storage can take a while.
        # Workers must see a fresh heartbeat as soon as they start.
        _write_manager_hb(self.layout, seq=0)
        self._threads = [
            threading.Thread(
                target=_heartbeat_thread,
                args=(self.layout, self._stop, self._errors),
                daemon=True,
            ),
            threading.Thread(
                target=_scheduler_thread,
                args=(
                    self.layout,
                    self._provisioner,
                    self._n_workers,
                    self._stop,
                    self._errors,
                    self._worker_counts,
                ),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def run(
        self,
        tilt_series_list: list[Path],
        model_checkpoint: Path,
        output_directory: Path,
        setting: str | tuple,
        patch_size: int,
        patch_overlap: float,
        batch_size: int,
        apply_ctf: bool,
        downsample: int,
        task_type: str = "alignment",
        desired_pixel_size: float | None = None,
        lowpass_cutoff: float | None = None,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Worker pool is closed")
        if self._errors:
            raise RuntimeError("Worker pool scheduler crashed") from self._errors[0]
        if not tilt_series_list:
            return {}
        if not self._owns_queue:
            self._initialize()
        self._batch += 1
        checkpoint = str(Path(model_checkpoint).resolve())
        serial_setting = setting if isinstance(setting, str) else list(setting)
        fingerprint = (
            compute_fingerprint(
                checkpoint,
                serial_setting,
                patch_size,
                patch_overlap,
                batch_size,
                apply_ctf,
                downsample,
            )
            if task_type == "alignment"
            else ""
        )
        pending_ids = set()
        batch_prefix = f"{self._batch:06d}-"
        for i, ts_path in enumerate(tilt_series_list):
            task_id = batch_prefix + _format_task_id(i, ts_path)
            pending_ids.add(task_id)
            write_pending(
                self.layout,
                TaskSpec(
                    task_id=task_id,
                    model_checkpoint_path=checkpoint
                    if task_type == "alignment"
                    else "",
                    tilt_series_path=str(Path(ts_path).resolve()),
                    output_directory=str(Path(output_directory).resolve()),
                    setting=serial_setting,
                    patch_size=patch_size,
                    patch_overlap=patch_overlap,
                    batch_size=batch_size,
                    apply_ctf=apply_ctf,
                    downsample=downsample,
                    init_fingerprint=fingerprint,
                    task_type=task_type,
                    desired_pixel_size=desired_pixel_size,
                    lowpass_cutoff=lowpass_cutoff,
                ),
            )
        if not self._threads:
            self._start()

        losses: dict[str, float] = {}
        failed_series: list[str] = []
        desc = {
            "alignment": "Tilt series alignment",
            "prepare_stacks": "Preparing stacks",
            "cross_correlation": "Cross-correlation alignment",
        }.get(task_type, task_type)
        with tqdm.tqdm(total=len(pending_ids), desc=desc, file=sys.stdout) as pbar:
            while pending_ids:
                if self._closed:
                    raise RuntimeError("Worker pool is closed")
                if self._errors:
                    raise RuntimeError(
                        "Worker pool scheduler crashed"
                    ) from self._errors[0]
                pbar.set_postfix_str(_format_worker_counts(self._worker_counts.copy()))
                # Scan terminal entries rather than stat'ing every unfinished
                # task on shared storage. Ignore earlier batches without rereading
                # their JSON; retain them until shutdown for orphan recovery.
                for directory, failed in (
                    (self.layout.done, False),
                    (self.layout.failed, True),
                ):
                    for result_file in directory.glob(f"{batch_prefix}*.json"):
                        task_id = result_file.stem
                        if task_id not in pending_ids:
                            continue
                        data = json.loads(result_file.read_text())
                        name = Path(data["tilt_series_path"]).stem
                        if failed:
                            failed_series.append(name)
                            print(
                                f"[manager] FAILED {name}: {data.get('error', '')}",
                                file=sys.stderr,
                            )
                        else:
                            losses[name] = data.get("final_loss", float("nan"))
                        pending_ids.remove(task_id)
                        pbar.update(1)
                if pending_ids:
                    time.sleep(_POLL_INTERVAL_S)
        if failed_series:
            raise RuntimeError(
                f"Alignment failed for {len(failed_series)} tilt series: "
                + ", ".join(failed_series)
            )
        return losses

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._owns_queue:
            try:
                self.layout.stop_file.touch()
            except OSError:
                pass  # still stop/cancel workers if the queue became unwritable
        if self._provisioner is not None:
            self._provisioner.request_stop()
        # Join in-flight submissions before shutdown so returned job IDs are
        # included in cancellation.
        for thread in self._threads:
            thread.join()
        try:
            if self._provisioner is not None:
                self._provisioner.shutdown()
        finally:
            if self._owns_queue:
                shutil.rmtree(self.layout.root, ignore_errors=True)
                if self.persistent:
                    try:
                        self._queue_parent.rmdir()
                    except OSError:
                        pass  # another session or legacy queue still uses it


def run_distributed(
    tilt_series_list: list[Path],
    model_checkpoint: Path,
    output_directory: Path,
    setting: str | tuple,
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    devices: list[int],
    n_cluster_workers: int | None,
    queue_root: Path,
    task_type: str = "alignment",
    desired_pixel_size: float | None = None,
    lowpass_cutoff: float | None = None,
    pool: DistributedPool | None = None,
) -> dict[str, float]:
    """Run one batch, using an existing inference pool when supplied.

    Without a pool, workers and queue belong to this call and close on exit,
    retaining the training/standalone behavior.
    """
    kwargs = dict(
        tilt_series_list=tilt_series_list,
        model_checkpoint=model_checkpoint,
        output_directory=output_directory,
        setting=setting,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        downsample=downsample,
        task_type=task_type,
        desired_pixel_size=desired_pixel_size,
        lowpass_cutoff=lowpass_cutoff,
    )
    if pool is not None:
        return pool.run(**kwargs)
    with DistributedPool(devices, n_cluster_workers, queue_root) as standalone:
        return standalone.run(**kwargs)
