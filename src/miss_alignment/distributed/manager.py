"""Head-node coordinator: writes tasks, starts provisioner, blocks until done."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
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


def _scheduler_thread(
    layout: QueueLayout,
    provisioner: WorkerProvisioner,
    n_workers: int,
    stop_event: threading.Event,
    error_box: list,
    worker_counts: dict,
) -> None:
    hb_seq = 1  # seq 0 written before thread starts
    last_hb = time.time()
    # Start from now so the first sweep is delayed by _SCHEDULER_INTERVAL_S;
    # run_distributed calls ensure_workers() explicitly at startup instead.
    last_sweep = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()

            if now - last_hb >= _MANAGER_HB_INTERVAL_S:
                _write_manager_hb(layout, hb_seq)
                hb_seq += 1
                last_hb = now

            if now - last_sweep >= _SCHEDULER_INTERVAL_S:
                _sweep_stalled_workers(layout)
                n_pending = len(list(layout.pending.glob("*.json")))
                if n_pending > 0:
                    # Cap at n_pending: no point submitting more workers than tasks.
                    # This also prevents resubmission when workers exit cleanly on
                    # an empty queue — only re-fill if the sweep re-pended orphans.
                    provisioner.ensure_workers(min(n_workers, n_pending))
                worker_counts.update(provisioner.worker_counts_by_type())
                last_sweep = now

            stop_event.wait(timeout=1.0)
    except Exception as exc:
        error_box.append(exc)
        stop_event.set()  # wake the poll loop so it notices immediately


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
) -> dict[str, float]:
    """Write tasks, provision workers, block until all tasks are terminal.

    Returns dict[series_name → final_loss]. Raises RuntimeError listing all
    failed series if any task ends in failed/. Deletes queue_root on exit.

    task_type selects the worker dispatch path:
      "alignment"          — evaluate_tilt_series (default)
      "prepare_stacks"     — _prepare_single_tilt_series
      "cross_correlation"  — _run_cross_correlation_single
    """
    layout = QueueLayout(queue_root)
    layout.ensure_directories()
    clear_queue(layout)

    # Fingerprint only meaningful for alignment tasks (amortizes model load).
    if task_type == "alignment":
        fingerprint = compute_fingerprint(
            model_checkpoint_path=str(model_checkpoint),
            setting=setting if isinstance(setting, str) else list(setting),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
        )
    else:
        fingerprint = ""

    task_ids = []
    for i, ts_path in enumerate(tilt_series_list):
        task_id = _format_task_id(i, ts_path)
        task_ids.append(task_id)
        spec = TaskSpec(
            task_id=task_id,
            model_checkpoint_path=(
                str(model_checkpoint) if task_type == "alignment" else ""
            ),
            tilt_series_path=str(ts_path),
            output_directory=str(output_directory),
            setting=setting if isinstance(setting, str) else list(setting),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
            init_fingerprint=fingerprint,
            task_type=task_type,
            desired_pixel_size=desired_pixel_size,
            lowpass_cutoff=lowpass_cutoff,
        )
        write_pending(layout, spec)

    # Write first heartbeat before starting workers so workers never see a
    # missing heartbeat on startup (would cause immediate exit).
    _write_manager_hb(layout, seq=0)

    if n_cluster_workers is not None:
        cluster_config = load_cluster_config()
        cluster = ClusterProvisioner(queue_dir=queue_root, config=cluster_config)
        if devices:
            # Also use local GPUs — the queue is shared so local and cluster
            # workers pull from the same task list with no extra coordination.
            local = LocalProvisioner(queue_dir=queue_root, devices=devices)
            provisioner: WorkerProvisioner = CompositeProvisioner([cluster, local])
        else:
            provisioner = cluster
        n_workers = n_cluster_workers
    else:
        provisioner = LocalProvisioner(queue_dir=queue_root, devices=devices)
        n_workers = len(devices)

    stop_event = threading.Event()
    scheduler_errors: list = []
    worker_counts: dict = provisioner.worker_counts_by_type()
    scheduler = threading.Thread(
        target=_scheduler_thread,
        args=(
            layout, provisioner, n_workers, stop_event, scheduler_errors, worker_counts
        ),
        daemon=True,
    )
    scheduler.start()
    provisioner.ensure_workers(n_workers)

    pending_ids = set(task_ids)
    losses: dict[str, float] = {}
    failed_series: list[str] = []

    _desc = {
        "alignment": "Tilt series alignment",
        "prepare_stacks": "Preparing stacks",
        "cross_correlation": "Cross-correlation alignment",
    }.get(task_type, task_type)
    pbar = tqdm.tqdm(total=len(task_ids), desc=_desc, file=sys.stdout)
    try:
        while pending_ids:
            time.sleep(_POLL_INTERVAL_S)

            if scheduler_errors:
                raise RuntimeError("Scheduler thread crashed") from scheduler_errors[0]

            pbar.set_postfix_str(_format_worker_counts(worker_counts))

            for done_file in layout.done.glob("*.json"):
                data = json.loads(done_file.read_text())
                tid = data["task_id"]
                if tid in pending_ids:
                    ts_name = Path(data["tilt_series_path"]).stem
                    losses[ts_name] = data.get("final_loss", float("nan"))
                    pending_ids.discard(tid)
                    pbar.update(1)

            for fail_file in layout.failed.glob("*.json"):
                data = json.loads(fail_file.read_text())
                tid = data["task_id"]
                if tid in pending_ids:
                    ts_name = Path(data["tilt_series_path"]).stem
                    failed_series.append(ts_name)
                    pending_ids.discard(tid)
                    pbar.update(1)
                    print(
                        f"[manager] FAILED {ts_name}: {data.get('error', '')}",
                        file=sys.stderr,
                    )
    finally:
        pbar.close()
        stop_event.set()
        scheduler.join(timeout=5.0)
        provisioner.shutdown()
        shutil.rmtree(queue_root, ignore_errors=True)

    if failed_series:
        raise RuntimeError(
            f"Alignment failed for {len(failed_series)} tilt series: "
            + ", ".join(failed_series)
        )

    return losses
