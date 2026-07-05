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


def run_worker_loop(
    layout: QueueLayout,
    worker_id: str,
    device: str,
    manager_hb_timeout_s: float = _MANAGER_HB_TIMEOUT_S,
) -> None:
    """Main worker loop: claim → evaluate → write result. Repeat until queue empty."""
    worker_dir = layout.worker_dir(worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)

    last_fingerprint: str | None = None
    cached_model: MissAlignment | None = None
    hb_seq = 0
    last_hb_time = 0.0

    while True:
        age = _manager_hb_age_s(layout)
        if age > manager_hb_timeout_s:
            print(
                f"[{worker_id}] Manager heartbeat stale ({age:.0f}s). Exiting.",
                file=sys.stderr,
            )
            return

        now = time.time()
        if now - last_hb_time >= _HB_INTERVAL_S:
            _write_worker_hb(worker_dir, hb_seq)
            hb_seq += 1
            last_hb_time = now

        spec = claim_one(layout, worker_id)
        if spec is None:
            return  # queue empty, exit cleanly

        print(f"[{worker_id}] Claimed {spec.task_id}", file=sys.stderr)

        if spec.init_fingerprint != last_fingerprint:
            cached_model = _load_model(spec.model_checkpoint_path)
            last_fingerprint = spec.init_fingerprint

        # Convert setting back to tuple if it was serialized as a list.
        setting = (
            tuple(spec.setting) if isinstance(spec.setting, list) else spec.setting
        )

        try:
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
            final_loss = float(loss_values[-1]) if loss_values else float("nan")
            mark_done(layout, worker_id, spec, final_loss=final_loss, device=device)
            print(
                f"[{worker_id}] Done {spec.task_id} loss={final_loss:.4f}",
                file=sys.stderr,
            )
        except Exception:
            error = traceback.format_exc()
            mark_failed(layout, worker_id, spec, error=error)
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
    run_worker_loop(layout, worker_id, cuda_device)
