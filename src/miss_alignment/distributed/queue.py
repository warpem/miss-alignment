"""Filesystem queue: task JSON files + atomic-rename claim protocol.

Directory layout under <root>/:
    pending/                  one JSON per queued task
    running/<worker_id>/      claimed task + heartbeat ticks
    done/                     completed task JSONs (result fields appended)
    failed/                   failed task JSONs (error field appended)
    manager/                  manager heartbeat ticks
    cluster/                  rendered cluster submission scripts
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class QueueLayout:
    root: Path

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    @property
    def running(self) -> Path:
        return self.root / "running"

    @property
    def done(self) -> Path:
        return self.root / "done"

    @property
    def failed(self) -> Path:
        return self.root / "failed"

    @property
    def manager_hb(self) -> Path:
        return self.root / "manager"

    @property
    def cluster(self) -> Path:
        return self.root / "cluster"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure_directories(self) -> None:
        for d in (
            self.pending,
            self.running,
            self.done,
            self.failed,
            self.manager_hb,
            self.cluster,
            self.logs,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def worker_dir(self, worker_id: str) -> Path:
        return self.running / worker_id


@dataclass
class TaskSpec:
    task_id: str
    model_checkpoint_path: str
    tilt_series_path: str
    output_directory: str
    setting: str | list
    patch_size: int
    patch_overlap: float
    batch_size: int
    apply_ctf: bool
    downsample: int
    init_fingerprint: str
    # Task type and optional parameters for non-alignment tasks.
    task_type: str = "alignment"
    desired_pixel_size: float | None = None
    lowpass_cutoff: float | None = None


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file + rename."""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_spec(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    return TaskSpec(
        **{k: v for k, v in data.items() if k in TaskSpec.__dataclass_fields__}
    )


def compute_fingerprint(
    model_checkpoint_path: str,
    setting: str | list,
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
) -> str:
    """SHA-256 over the fields that require reloading the model/settings."""
    payload = json.dumps(
        {
            "model_checkpoint_path": model_checkpoint_path,
            "setting": setting,
            "patch_size": patch_size,
            "patch_overlap": patch_overlap,
            "batch_size": batch_size,
            "apply_ctf": apply_ctf,
            "downsample": downsample,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_pending(layout: QueueLayout, spec: TaskSpec) -> None:
    _atomic_write(layout.pending / f"{spec.task_id}.json", asdict(spec))


def claim_one(layout: QueueLayout, worker_id: str) -> TaskSpec | None:
    """Attempt to claim a pending task via atomic rename.

    Returns the claimed TaskSpec, or None if the queue is empty.
    """
    worker_dir = layout.worker_dir(worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(layout.pending.glob("*.json"))
    random.shuffle(candidates)

    for candidate in candidates:
        dest = worker_dir / candidate.name
        try:
            os.rename(candidate, dest)
            return _read_spec(dest)
        except FileNotFoundError:
            # another worker claimed it first
            continue

    return None


def mark_done(
    layout: QueueLayout,
    worker_id: str,
    spec: TaskSpec,
    final_loss: float,
    device: str,
) -> None:
    """Write result to done/, then remove from running/ (publish-before-delete)."""
    data = asdict(spec)
    data["final_loss"] = final_loss
    data["device"] = device
    _atomic_write(layout.done / f"{spec.task_id}.json", data)
    (layout.worker_dir(worker_id) / f"{spec.task_id}.json").unlink(missing_ok=True)


def mark_failed(
    layout: QueueLayout,
    worker_id: str,
    spec: TaskSpec,
    error: str,
) -> None:
    """Write error to failed/, then remove from running/ (publish-before-delete)."""
    data = asdict(spec)
    data["error"] = error
    data["worker_id"] = worker_id
    _atomic_write(layout.failed / f"{spec.task_id}.json", data)
    (layout.worker_dir(worker_id) / f"{spec.task_id}.json").unlink(missing_ok=True)


def clear_queue(layout: QueueLayout) -> None:
    """Delete stale queue state from a prior run; recover running orphans to pending.

    Order matters: wipe pending/done/failed first, THEN recover orphans into
    the now-empty pending/ so they are not immediately re-deleted.
    """
    for directory in (layout.pending, layout.done, layout.failed):
        for f in directory.glob("*.json"):
            f.unlink(missing_ok=True)

    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        for task_file in worker_dir.glob("*.json"):
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
            pass  # not empty yet; scheduler will sweep it
