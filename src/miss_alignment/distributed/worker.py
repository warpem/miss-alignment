"""Worker subcommand: claims tasks from the queue and runs evaluate_tilt_series.

Each worker processes many series per run. The model checkpoint is loaded once
when the first task is claimed, then reused for all subsequent tasks that share
the same init_fingerprint (all tasks in one alignment phase do).

Usage (launched by provisioner):
    miss-alignment worker --queue-dir <path> --device <int> [--worker-id <str>]
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import typer

from ..alignment.tilt_series import evaluate_tilt_series
from ..models.models import MissAlignment
from ..prepare_stacks import _prepare_single_tilt_series
from ..preprocessing import _run_cross_correlation_single
from .queue import (
    QueueLayout,
    TaskSpec,
    claim_one,
    mark_done,
    mark_failed,
)

_MANAGER_HB_TIMEOUT_S = 120.0
_HB_INTERVAL_S = 5.0


def _write_worker_hb(worker_dir: Path, seq: int) -> None:
    new_hb = worker_dir / f"hb-{seq}"
    new_hb.write_text("")
    if seq > 0:
        (worker_dir / f"hb-{seq - 1}").unlink(missing_ok=True)


def _manager_hb_age_s(layout: QueueLayout) -> float:
    """Seconds since the manager's most recent heartbeat tick, or infinity."""
    ticks = list(layout.manager_hb.glob("hb-*"))
    if not ticks:
        return float("inf")
    latest = max(ticks, key=lambda p: p.stat().st_mtime)
    return time.time() - latest.stat().st_mtime


def _load_model(checkpoint_path: str) -> MissAlignment:
    model = MissAlignment.load_from_checkpoint(checkpoint_path, map_location="cpu")
    # Unwrap torch.compile: incompatible with spawned processes (see tilt_series.py).
    if hasattr(model.net, "_orig_mod"):
        model.net = model.net._orig_mod
    return model


def _device_int(device: str) -> int | None:
    """Convert 'cuda:0' → 0, 'cpu' → None."""
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return None


def _execute_task(
    spec: TaskSpec,
    device: str,
    cached_model: MissAlignment | None,
) -> float:
    """Run the work described by spec and return a scalar result (final loss or 0.0)."""
    if spec.task_type == "alignment":
        # Convert setting back to tuple if serialised as list.
        setting = (
            tuple(spec.setting) if isinstance(spec.setting, list) else spec.setting
        )
        _, loss_values = evaluate_tilt_series(
            model_checkpoint_path=Path(spec.model_checkpoint_path),
            tilt_series_path=Path(spec.tilt_series_path),
            output_directory=Path(spec.output_directory),
            setting=setting,
            patch_size=spec.patch_size,
            patch_overlap=spec.patch_overlap,
            batch_size=spec.batch_size,
            apply_ctf=spec.apply_ctf,
            downsample=spec.downsample,
            device=device,
            model=cached_model,
        )
        return float(loss_values[-1]) if loss_values else float("nan")

    elif spec.task_type == "prepare_stacks":
        _prepare_single_tilt_series(
            xml_path=Path(spec.tilt_series_path),
            desired_pixel_size=spec.desired_pixel_size,
            device=_device_int(device),
        )
        return 0.0

    elif spec.task_type == "cross_correlation":
        pretilt_range = (
            tuple(spec.pretilt_search_range)
            if spec.pretilt_search_range is not None
            else (-30.0, 30.0)
        )
        _run_cross_correlation_single(
            xml_file=Path(spec.tilt_series_path),
            device=_device_int(device),
            lowpass_cutoff=spec.lowpass_cutoff or 0.25,
            pretilt_search_range=pretilt_range,
        )
        return 0.0

    else:
        raise ValueError(f"Unknown task_type: {spec.task_type!r}")


def run_worker_loop(
    layout: QueueLayout,
    worker_id: str,
    device: str,
    manager_hb_timeout_s: float = _MANAGER_HB_TIMEOUT_S,
) -> str:
    """Main worker loop: claim → evaluate → write result. Repeat until queue empty.

    Returns an exit-reason string suitable for writing to logs/<worker_id>.exit.
    """
    worker_dir = layout.worker_dir(worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)

    last_fingerprint: str | None = None
    cached_model: MissAlignment | None = None
    hb_seq = 0
    last_hb_time = 0.0
    tasks_done = 0
    tasks_failed = 0

    while True:
        age = _manager_hb_age_s(layout)
        if age > manager_hb_timeout_s:
            return (
                f"manager heartbeat stale ({age:.0f}s > {manager_hb_timeout_s:.0f}s); "
                f"done={tasks_done} failed={tasks_failed}"
            )

        now = time.time()
        if now - last_hb_time >= _HB_INTERVAL_S:
            _write_worker_hb(worker_dir, hb_seq)
            hb_seq += 1
            last_hb_time = now

        spec = claim_one(layout, worker_id)
        if spec is None:
            return f"queue empty; done={tasks_done} failed={tasks_failed}"

        # For alignment tasks, (re)load the model when the fingerprint changes.
        if spec.task_type == "alignment" and spec.init_fingerprint != last_fingerprint:
            cached_model = _load_model(spec.model_checkpoint_path)
            last_fingerprint = spec.init_fingerprint

        try:
            final_loss = _execute_task(spec, device, cached_model)
            mark_done(layout, worker_id, spec, final_loss=final_loss, device=device)
            tasks_done += 1
        except Exception:
            error = traceback.format_exc()
            mark_failed(layout, worker_id, spec, error=error)
            tasks_failed += 1
            print(
                f"[{worker_id}] Failed {spec.task_id}:\n{error}",
                file=sys.stderr,
            )


def worker_miss_align(
    queue_dir: Path = typer.Option(..., help="Path to the tasks/ queue directory."),
    device: int = typer.Option(0, help="GPU device index to use."),
    worker_id: Optional[str] = typer.Option(
        None, help="Unique worker ID. Defaults to local-<pid>-gpu<device>."
    ),
) -> None:
    """Claim and run inference tasks from the distributed queue."""
    if worker_id is None:
        worker_id = f"local-{os.getpid()}-gpu{device}"

    layout = QueueLayout(queue_dir)
    layout.ensure_directories()

    cuda_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    exit_reason = "unknown (unhandled exception)"
    try:
        exit_reason = run_worker_loop(layout, worker_id, cuda_device)
    except Exception:
        exit_reason = f"unhandled exception:\n{traceback.format_exc()}"
        raise
    finally:
        exit_file = layout.logs / f"{worker_id}.exit"
        try:
            exit_file.write_text(exit_reason)
        except Exception:
            pass  # don't mask the original error
