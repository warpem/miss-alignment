# Distributed Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disk-based task queue that distributes per-tilt-series alignment inference across cluster nodes (SLURM/PBS), while keeping the existing local multi-GPU behaviour unchanged.

**Architecture:** A new `miss_alignment/distributed/` package implements a filesystem queue (atomic rename as the claim mutex), a manager that writes tasks and blocks until done, two provisioners (local subprocess and cluster batch scheduler), and a `miss-alignment worker` subcommand. `run_alignment_parallel` in `alignment/parallel.py` is updated to drive the manager instead of `_parallel.run_device_pool`. No changes to `train.py`, `infer.py`, or `evaluate_tilt_series`.

**Tech Stack:** Python 3.10+, stdlib only (`pathlib`, `os`, `subprocess`, `threading`, `hashlib`, `json`, `re`, `time`, `signal`) — no new dependencies.

## Global Constraints

- Python ≥ 3.10 (project minimum).
- No new runtime dependencies beyond stdlib.
- `ruff check --fix && ruff format` must pass (line length 88, ignore E712).
- `pytest --color=yes` must pass with no warnings-as-errors regressions.
- All new files under `src/miss_alignment/distributed/`.
- Cluster mode activated only when **both** `MISS_CLUSTER_CONFIG` and `MISS_CLUSTER_SCRIPT` env vars are set; otherwise `LocalProvisioner` is used.
- `evaluate_tilt_series` is never modified.
- `train.py` and `infer.py` are never modified.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/miss_alignment/distributed/__init__.py` | Create | Re-export public API |
| `src/miss_alignment/distributed/queue.py` | Create | Directory layout, task JSON, atomic rename claim |
| `src/miss_alignment/distributed/manager.py` | Create | Head-node coordinator, scheduler thread, poll loop |
| `src/miss_alignment/distributed/provisioner.py` | Create | `WorkerProvisioner` ABC, `LocalProvisioner`, `ClusterProvisioner` |
| `src/miss_alignment/distributed/worker.py` | Create | `miss-alignment worker` subcommand logic |
| `src/miss_alignment/distributed/config.py` | Create | Read env vars, return `ClusterConfig \| None` |
| `src/miss_alignment/alignment/parallel.py` | Modify | Replace `run_device_pool` call with manager call |
| `src/miss_alignment/_cli.py` | Modify | Register `worker` subcommand |
| `src/miss_alignment/__init__.py` | Modify | Export `worker_miss_align` |
| `src/miss_alignment/_parallel.py` | Delete (after Task 6) | Superseded by `LocalProvisioner` |
| `tests/distributed/test_queue.py` | Create | Queue layer unit tests |
| `tests/distributed/test_manager.py` | Create | Manager + provisioner integration tests |
| `tests/distributed/test_worker.py` | Create | Worker claim loop unit tests |
| `tests/distributed/test_config.py` | Create | Config env-var parsing tests |
| `tests/test_parallel.py` | Modify | Update import from new path |

---

## Task 1: Queue layer (`distributed/queue.py`)

**Files:**
- Create: `src/miss_alignment/distributed/__init__.py`
- Create: `src/miss_alignment/distributed/queue.py`
- Create: `tests/distributed/__init__.py`
- Create: `tests/distributed/test_queue.py`

**Interfaces:**
- Produces:
  - `QueueLayout(root: Path)` — manages subdirectory creation; attributes `pending`, `running`, `done`, `failed`, `manager_hb`, each a `Path`.
  - `TaskSpec` — `dataclass` with fields: `task_id: str`, `model_checkpoint_path: str`, `tilt_series_path: str`, `output_directory: str`, `setting: str | list`, `patch_size: int`, `patch_overlap: float`, `batch_size: int`, `apply_ctf: bool`, `downsample: int`, `init_fingerprint: str`.
  - `write_pending(layout: QueueLayout, spec: TaskSpec) -> None` — writes `layout.pending/<spec.task_id>.json`.
  - `claim_one(layout: QueueLayout, worker_id: str) -> TaskSpec | None` — shuffled rename claim; returns `None` when queue empty.
  - `mark_done(layout: QueueLayout, worker_id: str, spec: TaskSpec, final_loss: float, device: str) -> None`
  - `mark_failed(layout: QueueLayout, worker_id: str, spec: TaskSpec, error: str) -> None`
  - `compute_fingerprint(model_checkpoint_path: str, setting: str | list, patch_size: int, patch_overlap: float, batch_size: int, apply_ctf: bool, downsample: int) -> str` — SHA-256 hex.
  - `clear_queue(layout: QueueLayout) -> None` — deletes all JSON files in `pending/`, `done/`, `failed/`; moves any `running/<wid>/<id>.json` back to `pending/` (orphan recovery).

- [ ] **Step 1: Create directory skeleton and write failing tests**

```bash
mkdir -p tests/distributed
touch tests/distributed/__init__.py
```

```python
# tests/distributed/test_queue.py
import json
import os
from pathlib import Path
import pytest
from miss_alignment.distributed.queue import (
    QueueLayout,
    TaskSpec,
    clear_queue,
    claim_one,
    compute_fingerprint,
    mark_done,
    mark_failed,
    write_pending,
)


@pytest.fixture()
def layout(tmp_path):
    layout = QueueLayout(tmp_path / "tasks")
    layout.ensure_directories()
    return layout


def _spec(task_id="0000001-ts01"):
    return TaskSpec(
        task_id=task_id,
        model_checkpoint_path="/data/model.ckpt",
        tilt_series_path="/data/ts01.xml",
        output_directory="/data/out",
        setting="anchoring",
        patch_size=96,
        patch_overlap=0.1,
        batch_size=32,
        apply_ctf=False,
        downsample=2,
        init_fingerprint="abc123",
    )


def test_write_pending_creates_json(layout):
    write_pending(layout, _spec())
    assert (layout.pending / "0000001-ts01.json").exists()


def test_claim_one_returns_spec_and_moves_file(layout):
    write_pending(layout, _spec())
    result = claim_one(layout, "worker-0")
    assert result is not None
    assert result.task_id == "0000001-ts01"
    assert not (layout.pending / "0000001-ts01.json").exists()
    assert (layout.running / "worker-0" / "0000001-ts01.json").exists()


def test_claim_one_returns_none_when_empty(layout):
    assert claim_one(layout, "worker-0") is None


def test_claim_one_exclusive(layout, tmp_path):
    """Two concurrent claimers: exactly one wins."""
    write_pending(layout, _spec())
    results = []
    results.append(claim_one(layout, "worker-0"))
    results.append(claim_one(layout, "worker-1"))
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1


def test_mark_done_writes_done_and_removes_running(layout):
    spec = _spec()
    write_pending(layout, spec)
    claim_one(layout, "worker-0")
    mark_done(layout, "worker-0", spec, final_loss=0.042, device="cuda:0")
    done_path = layout.done / "0000001-ts01.json"
    assert done_path.exists()
    data = json.loads(done_path.read_text())
    assert data["final_loss"] == pytest.approx(0.042)
    assert data["device"] == "cuda:0"
    assert not (layout.running / "worker-0" / "0000001-ts01.json").exists()


def test_mark_failed_writes_failed_and_removes_running(layout):
    spec = _spec()
    write_pending(layout, spec)
    claim_one(layout, "worker-0")
    mark_failed(layout, "worker-0", spec, error="CUDA OOM")
    failed_path = layout.failed / "0000001-ts01.json"
    assert failed_path.exists()
    data = json.loads(failed_path.read_text())
    assert data["error"] == "CUDA OOM"
    assert not (layout.running / "worker-0" / "0000001-ts01.json").exists()


def test_clear_queue_recovers_orphans(layout):
    spec = _spec()
    write_pending(layout, spec)
    claim_one(layout, "worker-0")
    # simulate crash: running file remains, we clear and recover
    clear_queue(layout)
    # orphan should be back in pending
    assert (layout.pending / "0000001-ts01.json").exists()


def test_compute_fingerprint_is_deterministic():
    fp1 = compute_fingerprint("/ckpt", "anchoring", 96, 0.1, 32, False, 2)
    fp2 = compute_fingerprint("/ckpt", "anchoring", 96, 0.1, 32, False, 2)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_compute_fingerprint_differs_on_change():
    fp1 = compute_fingerprint("/ckpt", "anchoring", 96, 0.1, 32, False, 2)
    fp2 = compute_fingerprint("/other.ckpt", "anchoring", 96, 0.1, 32, False, 2)
    assert fp1 != fp2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tegunovd/dev/miss-alignment
pytest tests/distributed/test_queue.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'miss_alignment.distributed'`

- [ ] **Step 3: Create `__init__.py` skeleton**

```python
# src/miss_alignment/distributed/__init__.py
"""Disk-based distributed task queue for miss-alignment inference."""
```

- [ ] **Step 4: Implement `queue.py`**

```python
# src/miss_alignment/distributed/queue.py
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

    def ensure_directories(self) -> None:
        for d in (
            self.pending,
            self.running,
            self.done,
            self.failed,
            self.manager_hb,
            self.cluster,
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


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file + rename."""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_spec(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    return TaskSpec(**{k: v for k, v in data.items() if k in TaskSpec.__dataclass_fields__})


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
    running_path = layout.worker_dir(worker_id) / f"{spec.task_id}.json"
    running_path.unlink(missing_ok=True)


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
    running_path = layout.worker_dir(worker_id) / f"{spec.task_id}.json"
    running_path.unlink(missing_ok=True)


def clear_queue(layout: QueueLayout) -> None:
    """Delete stale queue state from a prior run; recover running orphans to pending."""
    # recover orphaned running tasks back to pending
    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        for task_file in worker_dir.glob("*.json"):
            dest = layout.pending / task_file.name
            try:
                os.rename(task_file, dest)
            except FileNotFoundError:
                pass
        try:
            worker_dir.rmdir()
        except OSError:
            pass  # not empty; will be swept later

    for directory in (layout.pending, layout.done, layout.failed):
        for f in directory.glob("*.json"):
            f.unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/distributed/test_queue.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/miss_alignment/distributed/__init__.py src/miss_alignment/distributed/queue.py tests/distributed/__init__.py tests/distributed/test_queue.py
git commit -m "feat: add distributed queue layer with atomic rename claim protocol"
```

---

## Task 2: Cluster configuration (`distributed/config.py`)

**Files:**
- Create: `src/miss_alignment/distributed/config.py`
- Create: `tests/distributed/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ClusterConfig` — `dataclass` with fields: `submit: str`, `submit_job_id_regex: str`, `cancel: str`, `script_path: Path`.
  - `load_cluster_config() -> ClusterConfig | None` — reads `MISS_CLUSTER_CONFIG` and `MISS_CLUSTER_SCRIPT`; returns `None` if either is unset.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_config.py
import os
import json
import pytest
from pathlib import Path
from miss_alignment.distributed.config import ClusterConfig, load_cluster_config


@pytest.fixture()
def cluster_json(tmp_path):
    cfg = {
        "submit": "sbatch {{script_path}}",
        "submit_job_id_regex": r"Submitted batch job (\d+)",
        "cancel": "scancel {{job_id}}",
    }
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def cluster_script(tmp_path):
    p = tmp_path / "worker.sh"
    p.write_text("#!/bin/bash\n{{command}}\n")
    return p


def test_load_cluster_config_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("MISS_CLUSTER_CONFIG", raising=False)
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    assert load_cluster_config() is None


def test_load_cluster_config_returns_none_when_only_one_set(monkeypatch, cluster_json):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(cluster_json))
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    assert load_cluster_config() is None


def test_load_cluster_config_returns_config(monkeypatch, cluster_json, cluster_script):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(cluster_json))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    cfg = load_cluster_config()
    assert isinstance(cfg, ClusterConfig)
    assert "sbatch" in cfg.submit
    assert cfg.script_path == cluster_script
    assert r"(\d+)" in cfg.submit_job_id_regex


def test_load_cluster_config_raises_on_missing_file(monkeypatch, tmp_path, cluster_script):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(tmp_path / "nonexistent.json"))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    with pytest.raises(FileNotFoundError):
        load_cluster_config()


def test_load_cluster_config_raises_on_missing_key(monkeypatch, tmp_path, cluster_script):
    bad = tmp_path / "bad.json"
    bad.write_text('{"submit": "sbatch {{script_path}}"}')
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(bad))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    with pytest.raises(KeyError):
        load_cluster_config()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/distributed/test_config.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `config.py`**

```python
# src/miss_alignment/distributed/config.py
"""Read cluster configuration from environment variables.

Cluster mode is activated when both MISS_CLUSTER_CONFIG and MISS_CLUSTER_SCRIPT
are set. If either is absent, load_cluster_config() returns None and
LocalProvisioner is used instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusterConfig:
    submit: str
    submit_job_id_regex: str
    cancel: str
    script_path: Path


def load_cluster_config() -> ClusterConfig | None:
    """Return ClusterConfig if both env vars are set, else None."""
    config_path_str = os.environ.get("MISS_CLUSTER_CONFIG")
    script_path_str = os.environ.get("MISS_CLUSTER_SCRIPT")

    if not config_path_str or not script_path_str:
        return None

    config_path = Path(config_path_str)
    script_path = Path(script_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"MISS_CLUSTER_CONFIG not found: {config_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"MISS_CLUSTER_SCRIPT not found: {script_path}")

    data = json.loads(config_path.read_text())
    return ClusterConfig(
        submit=data["submit"],
        submit_job_id_regex=data["submit_job_id_regex"],
        cancel=data["cancel"],
        script_path=script_path,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/distributed/test_config.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/config.py tests/distributed/test_config.py
git commit -m "feat: add cluster config reader (MISS_CLUSTER_CONFIG + MISS_CLUSTER_SCRIPT)"
```

---

## Task 3: Worker subcommand (`distributed/worker.py`)

**Files:**
- Create: `src/miss_alignment/distributed/worker.py`
- Create: `tests/distributed/test_worker.py`

**Interfaces:**
- Consumes:
  - `QueueLayout`, `TaskSpec`, `claim_one`, `mark_done`, `mark_failed` from `distributed/queue.py`
- Produces:
  - `worker_miss_align(queue_dir: Path, device: int, worker_id: str | None)` — Typer command entry point; loops until queue empty or manager heartbeat stale.
  - `run_worker_loop(layout: QueueLayout, worker_id: str, device: str, manager_hb_timeout_s: float) -> None` — testable loop body.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_worker.py
"""Unit tests for the worker claim loop.

These tests do NOT call evaluate_tilt_series; they mock it to keep
tests fast and free of CUDA/warpylib dependencies.
"""
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from miss_alignment.distributed.queue import (
    QueueLayout,
    TaskSpec,
    write_pending,
)
from miss_alignment.distributed.worker import run_worker_loop


@pytest.fixture()
def layout(tmp_path):
    layout = QueueLayout(tmp_path / "tasks")
    layout.ensure_directories()
    return layout


def _write_manager_hb(layout, seq=0):
    """Write a fresh manager heartbeat tick."""
    for old in layout.manager_hb.glob("hb-*"):
        old.unlink(missing_ok=True)
    (layout.manager_hb / f"hb-{seq}").write_text("")


def _spec(task_id="0000001-ts01"):
    return TaskSpec(
        task_id=task_id,
        model_checkpoint_path="/data/model.ckpt",
        tilt_series_path="/data/ts01.xml",
        output_directory="/data/out",
        setting="anchoring",
        patch_size=96,
        patch_overlap=0.1,
        batch_size=32,
        apply_ctf=False,
        downsample=2,
        init_fingerprint="abc123",
    )


def test_worker_processes_task_and_writes_done(layout, tmp_path):
    _write_manager_hb(layout)
    write_pending(layout, _spec())

    fake_loss = [0.5, 0.3, 0.1]
    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        return_value=(Path("/data/ts01.xml"), fake_loss),
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)

    done = layout.done / "0000001-ts01.json"
    assert done.exists()
    data = json.loads(done.read_text())
    assert data["final_loss"] == pytest.approx(0.1)


def test_worker_writes_failed_on_exception(layout):
    _write_manager_hb(layout)
    write_pending(layout, _spec())

    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=RuntimeError("CUDA OOM"),
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)

    failed = layout.failed / "0000001-ts01.json"
    assert failed.exists()
    data = json.loads(failed.read_text())
    assert "CUDA OOM" in data["error"]


def test_worker_exits_when_manager_hb_stale(layout):
    # Write a manager heartbeat that is already old
    hb_file = layout.manager_hb / "hb-0"
    hb_file.write_text("")
    # Make it appear 200 seconds old by back-dating mtime
    old_time = time.time() - 200
    import os
    os.utime(hb_file, (old_time, old_time))

    write_pending(layout, _spec())

    called = []
    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=lambda **kw: called.append(True),
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=120.0)

    # Worker should exit without processing the task
    assert called == []


def test_worker_reuses_model_when_fingerprint_matches(layout):
    """Model is loaded once when two tasks share the same init_fingerprint."""
    _write_manager_hb(layout)
    spec1 = _spec("0000001-ts01")
    spec2 = TaskSpec(
        task_id="0000002-ts02",
        model_checkpoint_path="/data/model.ckpt",
        tilt_series_path="/data/ts02.xml",
        output_directory="/data/out",
        setting="anchoring",
        patch_size=96,
        patch_overlap=0.1,
        batch_size=32,
        apply_ctf=False,
        downsample=2,
        init_fingerprint="abc123",  # same fingerprint
    )
    write_pending(layout, spec1)
    write_pending(layout, spec2)

    load_calls = []

    def fake_evaluate(**kwargs):
        return (Path(kwargs["tilt_series_path"]), [0.1])

    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=fake_evaluate,
    ):
        with patch(
            "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        ) as mock_load:
            mock_load.return_value = mock_load  # return self as stub
            run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)
            # Model should only be loaded once despite two tasks
            assert mock_load.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/distributed/test_worker.py -v 2>&1 | head -20
```

Expected: `ImportError` for `miss_alignment.distributed.worker`.

- [ ] **Step 3: Implement `worker.py`**

```python
# src/miss_alignment/distributed/worker.py
"""Worker subcommand: claims tasks from the queue and runs evaluate_tilt_series.

Usage (launched by provisioner):
    miss-alignment worker --queue-dir <path> --device <int> [--worker-id <str>]
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

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

# Seconds without a manager heartbeat tick before the worker exits.
_MANAGER_HB_TIMEOUT_S = 120.0
# Seconds between heartbeat writes.
_HB_INTERVAL_S = 5.0


def _write_worker_hb(worker_dir: Path, seq: int) -> None:
    """Write a new heartbeat tick, removing the previous one."""
    new_hb = worker_dir / f"hb-{seq}"
    new_hb.write_text("")
    if seq > 0:
        old_hb = worker_dir / f"hb-{seq - 1}"
        old_hb.unlink(missing_ok=True)


def _manager_hb_age_s(layout: QueueLayout) -> float:
    """Seconds since the manager's most recent heartbeat tick, or infinity."""
    ticks = list(layout.manager_hb.glob("hb-*"))
    if not ticks:
        return float("inf")
    latest = max(ticks, key=lambda p: p.stat().st_mtime)
    return time.time() - latest.stat().st_mtime


def run_worker_loop(
    layout: QueueLayout,
    worker_id: str,
    device: str,
    manager_hb_timeout_s: float = _MANAGER_HB_TIMEOUT_S,
) -> None:
    """Main worker loop: claim → check heartbeat → evaluate → write result.

    Separated from the Typer command for testability.
    """
    worker_dir = layout.worker_dir(worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)

    last_fingerprint: str | None = None
    loaded_model = None
    hb_seq = 0
    last_hb_time = 0.0

    while True:
        # Check manager heartbeat before every claim attempt.
        age = _manager_hb_age_s(layout)
        if age > manager_hb_timeout_s:
            print(
                f"[{worker_id}] Manager heartbeat stale ({age:.0f}s > "
                f"{manager_hb_timeout_s:.0f}s). Exiting.",
                file=sys.stderr,
            )
            return

        # Write our own heartbeat if due.
        now = time.time()
        if now - last_hb_time >= _HB_INTERVAL_S:
            _write_worker_hb(worker_dir, hb_seq)
            hb_seq += 1
            last_hb_time = now

        spec = claim_one(layout, worker_id)
        if spec is None:
            return  # queue empty, exit cleanly

        print(f"[{worker_id}] Claimed {spec.task_id}", file=sys.stderr)

        # Load model only when fingerprint changes.
        if spec.init_fingerprint != last_fingerprint:
            loaded_model = MissAlignment.load_from_checkpoint(
                spec.model_checkpoint_path, map_location="cpu"
            )
            last_fingerprint = spec.init_fingerprint

        try:
            _, loss_values = evaluate_tilt_series(
                model_checkpoint_path=Path(spec.model_checkpoint_path),
                tilt_series_path=Path(spec.tilt_series_path),
                output_directory=Path(spec.output_directory),
                setting=spec.setting,
                patch_size=spec.patch_size,
                patch_overlap=spec.patch_overlap,
                batch_size=spec.batch_size,
                apply_ctf=spec.apply_ctf,
                downsample=spec.downsample,
                device=device,
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
    worker_id: str | None = typer.Option(
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/distributed/test_worker.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/worker.py tests/distributed/test_worker.py
git commit -m "feat: add worker subcommand with claim loop and model fingerprint reuse"
```

---

## Task 4: Provisioners (`distributed/provisioner.py`)

**Files:**
- Create: `src/miss_alignment/distributed/provisioner.py`
- Create: `tests/distributed/test_provisioner.py`

**Interfaces:**
- Consumes: `ClusterConfig` from `distributed/config.py`.
- Produces:
  - `WorkerProvisioner` — ABC with `ensure_workers(n_tasks: int) -> None` and `shutdown() -> None`.
  - `LocalProvisioner(queue_dir: Path, devices: list[int])` — spawns `miss-alignment worker` child processes.
  - `ClusterProvisioner(queue_dir: Path, config: ClusterConfig, n_tasks: int)` — submits cluster jobs.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_provisioner.py
"""Tests for LocalProvisioner and ClusterProvisioner."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from miss_alignment.distributed.config import ClusterConfig
from miss_alignment.distributed.provisioner import ClusterProvisioner, LocalProvisioner


def test_local_provisioner_spawns_one_process_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1])
        p.ensure_workers(n_tasks=10)

        assert mock_popen.call_count == 2
        # Each call should pass --device 0 and --device 1
        calls_str = [str(c) for c in mock_popen.call_args_list]
        assert any("--device" in s and "0" in s for s in calls_str)
        assert any("--device" in s and "1" in s for s in calls_str)


def test_local_provisioner_shutdown_terminates_processes(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_tasks=5)
        p.shutdown()

        mock_proc.terminate.assert_called()


def test_local_provisioner_does_not_respawn_running_processes(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_tasks=5)
        p.ensure_workers(n_tasks=5)  # second call should not spawn again

        assert mock_popen.call_count == 1


def test_cluster_provisioner_submits_one_job_per_task(tmp_path):
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/bash\n{{command}}\n")
    cfg = ClusterConfig(
        submit="sbatch {{script_path}}",
        submit_job_id_regex=r"Submitted batch job (\d+)",
        cancel="scancel {{job_id}}",
        script_path=script,
    )

    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        result = MagicMock()
        result.stdout = "Submitted batch job 12345\n"
        return result

    with patch("miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_tasks=3)

    assert len(submitted) == 3


def test_cluster_provisioner_cancels_jobs_on_shutdown(tmp_path):
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/bash\n{{command}}\n")
    cfg = ClusterConfig(
        submit="sbatch {{script_path}}",
        submit_job_id_regex=r"Submitted batch job (\d+)",
        cancel="scancel {{job_id}}",
        script_path=script,
    )

    cancel_calls = []

    def fake_run(cmd, **kwargs):
        if "sbatch" in cmd:
            result = MagicMock()
            result.stdout = "Submitted batch job 99999\n"
            return result
        cancel_calls.append(cmd)
        return MagicMock()

    with patch("miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_tasks=2)
        p.shutdown()

    assert len(cancel_calls) == 2
    assert all("scancel" in c for c in cancel_calls)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/distributed/test_provisioner.py -v 2>&1 | head -20
```

Expected: `ImportError` for `miss_alignment.distributed.provisioner`.

- [ ] **Step 3: Implement `provisioner.py`**

```python
# src/miss_alignment/distributed/provisioner.py
"""Worker provisioners: spawn local child processes or submit cluster jobs."""

from __future__ import annotations

import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template

from .config import ClusterConfig


class WorkerProvisioner(ABC):
    @abstractmethod
    def ensure_workers(self, n_tasks: int) -> None:
        """Ensure sufficient workers are running for n_tasks tasks."""

    @abstractmethod
    def shutdown(self) -> None:
        """Terminate all managed workers."""


class LocalProvisioner(WorkerProvisioner):
    """Spawns miss-alignment worker child processes, one per GPU device."""

    def __init__(self, queue_dir: Path, devices: list[int]) -> None:
        self._queue_dir = queue_dir
        self._devices = devices
        self._procs: dict[int, subprocess.Popen] = {}  # device -> process

    def ensure_workers(self, n_tasks: int) -> None:
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
    """Submits one cluster job per task via a configurable submit command."""

    def __init__(self, queue_dir: Path, config: ClusterConfig) -> None:
        self._queue_dir = queue_dir
        self._config = config
        self._job_ids: list[str] = []
        self._scripts_dir = queue_dir / "cluster"
        self._scripts_dir.mkdir(parents=True, exist_ok=True)

    def _render_script(self, index: int) -> Path:
        """Render the .sh template for one worker and write it to tasks/cluster/."""
        template_text = self._config.script_path.read_text()
        # The {{command}} in the template uses shell-evaluated $(hostname) and $$
        # so worker IDs are unique per compute node at runtime.
        command = (
            f"miss-alignment worker"
            f" --queue-dir {self._queue_dir}"
            f" --device 0"
            f' --worker-id "$(hostname)-$$-{index}"'
        )
        # Replace {{command}} and any {{MISS_CLUSTER_VAR_*}} env var placeholders.
        import os

        rendered = template_text.replace("{{command}}", command)
        for key, value in os.environ.items():
            if key.startswith("MISS_CLUSTER_VAR_"):
                var_name = key[len("MISS_CLUSTER_VAR_"):].lower()
                rendered = rendered.replace(f"{{{{{var_name}}}}}", value)

        script_path = self._scripts_dir / f"worker-{index}.sh"
        script_path.write_text(rendered)
        return script_path

    def ensure_workers(self, n_tasks: int) -> None:
        already = len(self._job_ids)
        for i in range(already, n_tasks):
            script_path = self._render_script(i)
            submit_cmd = self._config.submit.replace(
                "{{script_path}}", str(script_path)
            )
            result = subprocess.run(
                submit_cmd,
                shell=True,
                capture_output=False,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/distributed/test_provisioner.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/provisioner.py tests/distributed/test_provisioner.py
git commit -m "feat: add LocalProvisioner and ClusterProvisioner"
```

---

## Task 5: Manager (`distributed/manager.py`)

**Files:**
- Create: `src/miss_alignment/distributed/manager.py`
- Create: `tests/distributed/test_manager.py`

**Interfaces:**
- Consumes:
  - `QueueLayout`, `TaskSpec`, `write_pending`, `clear_queue`, `compute_fingerprint` from `distributed/queue.py`
  - `WorkerProvisioner` from `distributed/provisioner.py`
- Produces:
  - `run_distributed(tilt_series_list: list[Path], model_checkpoint: Path, output_directory: Path, setting: str | tuple, patch_size: int, patch_overlap: float, batch_size: int, apply_ctf: bool, downsample: int, devices: list[int], queue_root: Path, cluster_config: ClusterConfig | None) -> dict[str, float]`

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_manager.py
"""Integration tests for the manager coordinator."""
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.manager import run_distributed
from miss_alignment.distributed.queue import QueueLayout, mark_done, mark_failed


def _fake_provisioner_class(layout):
    """Returns a provisioner that writes done files for each pending task."""

    class FakeProvisioner:
        def __init__(self, *a, **kw):
            pass

        def ensure_workers(self, n_tasks):
            pass

        def shutdown(self):
            pass

    return FakeProvisioner


def _make_xml(tmp_path, name):
    p = tmp_path / f"{name}.xml"
    p.write_text(f"<TiltSeries><Name>{name}</Name></TiltSeries>")
    return p


def test_run_distributed_returns_losses(tmp_path):
    """Manager resolves all tasks completed by a simulated worker thread."""
    xml1 = _make_xml(tmp_path, "ts01")
    xml2 = _make_xml(tmp_path, "ts02")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("")

    queue_root = tmp_path / "tasks"

    # Simulate a worker: poll pending/ and write done/ files
    def fake_worker(layout_root):
        layout = QueueLayout(layout_root)
        deadline = time.time() + 10
        done_count = 0
        while done_count < 2 and time.time() < deadline:
            for f in list(layout.pending.glob("*.json")):
                data = json.loads(f.read_text())
                task_id = data["task_id"]
                running_dir = layout.running / "fake-worker"
                running_dir.mkdir(parents=True, exist_ok=True)
                import os
                try:
                    os.rename(f, running_dir / f.name)
                except FileNotFoundError:
                    continue
                done_data = {**data, "final_loss": 0.01, "device": "cpu"}
                (layout.done / f"{task_id}.json").write_text(
                    json.dumps(done_data)
                )
                (running_dir / f"{task_id}.json").unlink(missing_ok=True)
                done_count += 1
            time.sleep(0.05)

    worker_thread = threading.Thread(target=fake_worker, args=(queue_root,), daemon=True)
    worker_thread.start()

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        _fake_provisioner_class(None),
    ):
        losses = run_distributed(
            tilt_series_list=[xml1, xml2],
            model_checkpoint=ckpt,
            output_directory=tmp_path,
            setting="anchoring",
            patch_size=96,
            patch_overlap=0.1,
            batch_size=32,
            apply_ctf=False,
            downsample=2,
            devices=[0],
            queue_root=queue_root,
            cluster_config=None,
        )

    assert set(losses.keys()) == {"ts01", "ts02"}
    assert all(v == pytest.approx(0.01) for v in losses.values())


def test_run_distributed_raises_if_any_task_fails(tmp_path):
    """Manager raises RuntimeError if any series ends in failed/."""
    xml1 = _make_xml(tmp_path, "ts01")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("")

    queue_root = tmp_path / "tasks"

    def fake_failing_worker(layout_root):
        layout = QueueLayout(layout_root)
        deadline = time.time() + 10
        while time.time() < deadline:
            for f in list(layout.pending.glob("*.json")):
                data = json.loads(f.read_text())
                task_id = data["task_id"]
                running_dir = layout.running / "fake-worker"
                running_dir.mkdir(parents=True, exist_ok=True)
                import os
                try:
                    os.rename(f, running_dir / f.name)
                except FileNotFoundError:
                    continue
                fail_data = {**data, "error": "boom", "worker_id": "fake-worker"}
                (layout.failed / f"{task_id}.json").write_text(
                    json.dumps(fail_data)
                )
                (running_dir / f"{task_id}.json").unlink(missing_ok=True)
                return
            time.sleep(0.05)

    worker_thread = threading.Thread(
        target=fake_failing_worker, args=(queue_root,), daemon=True
    )
    worker_thread.start()

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        _fake_provisioner_class(None),
    ):
        with pytest.raises(RuntimeError, match="ts01"):
            run_distributed(
                tilt_series_list=[xml1],
                model_checkpoint=ckpt,
                output_directory=tmp_path,
                setting="anchoring",
                patch_size=96,
                patch_overlap=0.1,
                batch_size=32,
                apply_ctf=False,
                downsample=2,
                devices=[0],
                queue_root=queue_root,
                cluster_config=None,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/distributed/test_manager.py -v 2>&1 | head -20
```

Expected: `ImportError` for `miss_alignment.distributed.manager`.

- [ ] **Step 3: Implement `manager.py`**

```python
# src/miss_alignment/distributed/manager.py
"""Head-node coordinator: writes tasks, starts provisioner, blocks until done."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import tqdm

from .config import ClusterConfig
from .provisioner import ClusterProvisioner, LocalProvisioner, WorkerProvisioner
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
        old_hb = layout.manager_hb / f"hb-{seq - 1}"
        old_hb.unlink(missing_ok=True)


def _sweep_stalled_workers(layout: QueueLayout) -> None:
    """Move tasks from stalled worker dirs back to pending/."""
    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        ticks = list(worker_dir.glob("hb-*"))
        if ticks:
            latest = max(ticks, key=lambda p: p.stat().st_mtime)
            age = time.time() - latest.stat().st_mtime
        else:
            # No heartbeat yet — use dir mtime as proxy
            age = time.time() - worker_dir.stat().st_mtime

        if age <= _WORKER_STALL_TIMEOUT_S:
            continue

        # Worker is stalled — recover its tasks
        for task_file in worker_dir.glob("*.json"):
            dest = layout.pending / task_file.name
            try:
                import os
                os.rename(task_file, dest)
                print(
                    f"[manager] Recovered stalled task {task_file.name} to pending",
                    file=sys.stderr,
                )
            except FileNotFoundError:
                pass
        # Clean up heartbeat files
        for hb in worker_dir.glob("hb-*"):
            hb.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass


def _scheduler_thread(
    layout: QueueLayout,
    provisioner: WorkerProvisioner,
    n_tasks: int,
    stop_event: threading.Event,
) -> None:
    hb_seq = 0
    last_hb = 0.0
    last_sweep = 0.0

    while not stop_event.is_set():
        now = time.time()

        if now - last_hb >= _MANAGER_HB_INTERVAL_S:
            _write_manager_hb(layout, hb_seq)
            hb_seq += 1
            last_hb = now

        if now - last_sweep >= _SCHEDULER_INTERVAL_S:
            _sweep_stalled_workers(layout)
            provisioner.ensure_workers(n_tasks)
            last_sweep = now

        stop_event.wait(timeout=1.0)


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
    queue_root: Path,
    cluster_config: ClusterConfig | None,
) -> dict[str, float]:
    """Write tasks, provision workers, block until all tasks are terminal.

    Returns a dict mapping tilt-series name to final loss.
    Raises RuntimeError listing all failed series if any task ends in failed/.
    """
    layout = QueueLayout(queue_root)
    layout.ensure_directories()
    clear_queue(layout)

    # Fingerprint is the same for all tasks in one alignment phase.
    fingerprint = compute_fingerprint(
        model_checkpoint_path=str(model_checkpoint),
        setting=setting if isinstance(setting, str) else list(setting),
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        downsample=downsample,
    )

    task_ids = []
    for i, ts_path in enumerate(tilt_series_list):
        task_id = _format_task_id(i, ts_path)
        task_ids.append(task_id)
        spec = TaskSpec(
            task_id=task_id,
            model_checkpoint_path=str(model_checkpoint),
            tilt_series_path=str(ts_path),
            output_directory=str(output_directory),
            setting=setting if isinstance(setting, str) else list(setting),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
            init_fingerprint=fingerprint,
        )
        write_pending(layout, spec)

    n_tasks = len(tilt_series_list)
    if cluster_config is not None:
        provisioner: WorkerProvisioner = ClusterProvisioner(
            queue_dir=queue_root, config=cluster_config
        )
    else:
        provisioner = LocalProvisioner(queue_dir=queue_root, devices=devices)

    stop_event = threading.Event()
    scheduler = threading.Thread(
        target=_scheduler_thread,
        args=(layout, provisioner, n_tasks, stop_event),
        daemon=True,
    )
    scheduler.start()
    provisioner.ensure_workers(n_tasks)

    pending_ids = set(task_ids)
    losses: dict[str, float] = {}
    failed_series: list[str] = []

    pbar = tqdm.tqdm(total=n_tasks, desc="Tilt series alignment", file=sys.stdout)
    try:
        while pending_ids:
            time.sleep(_POLL_INTERVAL_S)

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

    if failed_series:
        raise RuntimeError(
            f"Alignment failed for {len(failed_series)} tilt series: "
            + ", ".join(failed_series)
        )

    return losses
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/distributed/test_manager.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run all distributed tests together**

```bash
pytest tests/distributed/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/miss_alignment/distributed/manager.py tests/distributed/test_manager.py
git commit -m "feat: add distributed manager with scheduler thread and poll loop"
```

---

## Task 6: Wire up `alignment/parallel.py` and CLI; delete `_parallel.py`

**Files:**
- Modify: `src/miss_alignment/alignment/parallel.py`
- Modify: `src/miss_alignment/_cli.py`
- Modify: `src/miss_alignment/__init__.py`
- Modify: `tests/test_parallel.py`
- Delete: `src/miss_alignment/_parallel.py`

**Interfaces:**
- Consumes:
  - `run_distributed` from `distributed/manager.py`
  - `load_cluster_config` from `distributed/config.py`
  - `worker_miss_align` from `distributed/worker.py`
- Produces: `run_alignment_parallel` — same signature as before, same return type `dict[str, float]`.

- [ ] **Step 1: Update `tests/test_parallel.py` to use the new import**

The existing `test_parallel.py` tests `_parallel.run_device_pool` directly. Since `_parallel.py` is being deleted, update the tests to verify the equivalent behaviour through `LocalProvisioner` instead. Replace the file entirely:

```python
# tests/test_parallel.py
"""Tests for the distributed worker provisioner (replaces _parallel.py tests)."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.provisioner import LocalProvisioner


def test_local_provisioner_spawns_worker_per_device(tmp_path):
    """LocalProvisioner starts one worker process per GPU device."""
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_tasks=10)

        assert mock_popen.call_count == 3


def test_local_provisioner_does_not_double_spawn(tmp_path):
    """Calling ensure_workers twice does not spawn extra processes."""
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_tasks=5)
        p.ensure_workers(n_tasks=5)

        assert mock_popen.call_count == 1


@pytest.mark.filterwarnings("ignore")
def test_local_provisioner_shutdown_terminates(tmp_path):
    """shutdown() terminates all spawned processes."""
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_tasks=3)
        p.shutdown()

        mock_proc.terminate.assert_called()
```

- [ ] **Step 2: Run updated tests to verify they pass before touching sources**

```bash
pytest tests/test_parallel.py -v
```

Expected: all 3 tests PASS (they import from `distributed.provisioner` which already exists).

- [ ] **Step 3: Update `alignment/parallel.py`**

Replace the file entirely:

```python
# src/miss_alignment/alignment/parallel.py
from pathlib import Path

from ..distributed.config import load_cluster_config
from ..distributed.manager import run_distributed


def run_alignment_parallel(
    model_checkpoint: Path,
    tilt_series_list: list[Path],
    output_directory: Path,
    setting: str | tuple[int, int] | tuple[int, int, int, int],
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    devices_list: list[int],
) -> dict[str, float]:
    """Distribute per-tilt-series alignment across local GPUs or a cluster.

    With no cluster env vars set, workers are spawned as local child processes
    (one per GPU in devices_list). Set MISS_CLUSTER_CONFIG and MISS_CLUSTER_SCRIPT
    to fan work out to a batch scheduler instead.

    Returns a dict mapping tilt-series stem names to their final loss values.
    """
    cluster_config = load_cluster_config()
    # output_directory is the training directory in train.py and data_directory in
    # infer.py — both are the top-level data dir, so tasks/ lives alongside the XMLs.
    queue_root = output_directory / "tasks"

    return run_distributed(
        tilt_series_list=tilt_series_list,
        model_checkpoint=model_checkpoint,
        output_directory=output_directory,
        setting=setting,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        downsample=downsample,
        devices=devices_list,
        queue_root=queue_root,
        cluster_config=cluster_config,
    )
```

- [ ] **Step 4: Register `worker` subcommand in `_cli.py`**

```python
# src/miss_alignment/_cli.py
from click import Context
import typer
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(cls=OrderCommands, add_completion=False, no_args_is_help=True)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}

from .distributed.worker import worker_miss_align  # noqa: E402

cli.command(name="worker")(worker_miss_align)
```

- [ ] **Step 5: Export `worker_miss_align` from `__init__.py`**

```python
# src/miss_alignment/__init__.py
"""She has a chaotic good alignment for tilt-series."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miss_alignment")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"
__all__ = [
    "__version__",
    "cli",
    "train_miss_align",
    "infer_miss_align",
    "worker_miss_align",
]

from ._cli import cli
from .train import train_miss_align
from .infer import infer_miss_align
from .distributed.worker import worker_miss_align
```

- [ ] **Step 6: Delete `_parallel.py`**

```bash
git rm src/miss_alignment/_parallel.py
```

- [ ] **Step 7: Run the full test suite**

```bash
pytest --color=yes -v
```

Expected: all tests PASS, no warnings-as-errors regressions. If `test_infer.py` or `test_train.py` import `_parallel` directly, fix those imports to remove them (the module no longer exists).

- [ ] **Step 8: Run linter**

```bash
ruff check --fix src/miss_alignment/
ruff format src/miss_alignment/
```

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/miss_alignment/alignment/parallel.py src/miss_alignment/_cli.py src/miss_alignment/__init__.py tests/test_parallel.py
git commit -m "feat: wire distributed queue into run_alignment_parallel; add worker subcommand; delete _parallel.py"
```

---

## Task 7: Update `distributed/__init__.py` and run full suite

**Files:**
- Modify: `src/miss_alignment/distributed/__init__.py`

**Interfaces:**
- Produces: public re-exports for any consumer that imports directly from `miss_alignment.distributed`.

- [ ] **Step 1: Update `__init__.py`**

```python
# src/miss_alignment/distributed/__init__.py
"""Disk-based distributed task queue for miss-alignment inference."""

from .config import ClusterConfig, load_cluster_config
from .manager import run_distributed
from .provisioner import ClusterProvisioner, LocalProvisioner, WorkerProvisioner
from .queue import (
    QueueLayout,
    TaskSpec,
    claim_one,
    clear_queue,
    compute_fingerprint,
    mark_done,
    mark_failed,
    write_pending,
)
from .worker import run_worker_loop, worker_miss_align

__all__ = [
    "ClusterConfig",
    "load_cluster_config",
    "run_distributed",
    "ClusterProvisioner",
    "LocalProvisioner",
    "WorkerProvisioner",
    "QueueLayout",
    "TaskSpec",
    "claim_one",
    "clear_queue",
    "compute_fingerprint",
    "mark_done",
    "mark_failed",
    "write_pending",
    "run_worker_loop",
    "worker_miss_align",
]
```

- [ ] **Step 2: Run full test suite and linter**

```bash
pytest --color=yes --cov --cov-report=term-missing
ruff check src/miss_alignment/
ruff format --check src/miss_alignment/
```

Expected: all tests PASS, coverage report shows `distributed/` coverage, no ruff errors.

- [ ] **Step 3: Verify `miss-alignment worker --help` works**

```bash
miss-alignment worker --help
```

Expected output includes `--queue-dir`, `--device`, `--worker-id` options.

- [ ] **Step 4: Final commit**

```bash
git add src/miss_alignment/distributed/__init__.py
git commit -m "feat: export distributed public API from __init__.py"
```
