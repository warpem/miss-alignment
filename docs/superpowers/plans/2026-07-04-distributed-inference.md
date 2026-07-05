# Distributed Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disk-based task queue that distributes per-tilt-series alignment inference across cluster nodes (SLURM/PBS), while keeping the existing local multi-GPU behaviour unchanged.

**Architecture:** A new `miss_alignment/distributed/` package implements a filesystem queue (atomic rename as the claim mutex), a manager that writes tasks and blocks until done, two provisioners (local subprocess and cluster batch scheduler), and a `miss-alignment worker` subcommand. Each worker processes many series per run; checkpoint loading is amortized by passing the resident model into `evaluate_tilt_series`. Cluster mode is triggered by a new `--n-cluster-workers` CLI arg; without it local multi-GPU mode is unchanged.

**Tech Stack:** Python 3.10+, stdlib only for `distributed/` (`pathlib`, `os`, `subprocess`, `threading`, `hashlib`, `json`, `re`, `time`, `signal`). `tqdm` (already a dependency) used in manager. `typer` (already a dependency) used in worker CLI.

## Global Constraints

- Python ≥ 3.10 (project minimum).
- No new runtime dependencies beyond stdlib + existing deps.
- `ruff check --fix && ruff format` must pass (line length 88, ignore E712).
- `pytest --color=yes` must pass with no warnings-as-errors regressions.
- All new queue infrastructure under `src/miss_alignment/distributed/`.
- Cluster mode activated only by `--n-cluster-workers N`; without it local mode runs unchanged.
- `MISS_CLUSTER_CONFIG` and `MISS_CLUSTER_SCRIPT` are **required** when `--n-cluster-workers` is set; `config.py` raises `RuntimeError` if either is absent (never silently falls back to local mode).
- `evaluate_tilt_series` gains one optional `model` parameter and is otherwise unchanged.
- `train.py` and `infer.py` are modified only to add the `--n-cluster-workers` option and pass it through.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/miss_alignment/__main__.py` | Create | `python -m miss_alignment` entry point for `LocalProvisioner` subprocess launch |
| `src/miss_alignment/distributed/__init__.py` | Create | Public re-exports |
| `src/miss_alignment/distributed/queue.py` | Create | Directory layout, task JSON, atomic rename claim |
| `src/miss_alignment/distributed/manager.py` | Create | Head-node coordinator, scheduler thread, poll loop |
| `src/miss_alignment/distributed/provisioner.py` | Create | `WorkerProvisioner` ABC, `LocalProvisioner`, `ClusterProvisioner` |
| `src/miss_alignment/distributed/worker.py` | Create | `miss-alignment worker` subcommand logic |
| `src/miss_alignment/distributed/config.py` | Create | Read env vars, return `ClusterConfig` or raise |
| `src/miss_alignment/alignment/tilt_series.py` | Modify | Add optional `model` parameter to `evaluate_tilt_series` |
| `src/miss_alignment/alignment/parallel.py` | Modify | Replace `run_device_pool` call with manager; add `n_cluster_workers` param |
| `src/miss_alignment/train.py` | Modify | Add `--n-cluster-workers` option; pass to `run_alignment_parallel` |
| `src/miss_alignment/infer.py` | Modify | Add `--n-cluster-workers` option; pass to `run_alignment_parallel` |
| `src/miss_alignment/_cli.py` | Modify | Register `worker` subcommand |
| `src/miss_alignment/__init__.py` | Modify | Export `worker_miss_align` |
| `src/miss_alignment/_parallel.py` | Delete (Task 7) | Superseded by `LocalProvisioner` |
| `tests/distributed/__init__.py` | Create | Test package |
| `tests/distributed/test_queue.py` | Create | Queue layer unit tests |
| `tests/distributed/test_manager.py` | Create | Manager integration tests |
| `tests/distributed/test_worker.py` | Create | Worker claim loop unit tests |
| `tests/distributed/test_config.py` | Create | Config env-var parsing tests |
| `tests/distributed/test_provisioner.py` | Create | Provisioner unit tests |
| `tests/test_parallel.py` | Modify | Update to test new `LocalProvisioner` path |

---

## Task 1: Queue layer (`distributed/queue.py`)

**Files:**
- Create: `src/miss_alignment/__main__.py`
- Create: `src/miss_alignment/distributed/__init__.py`
- Create: `src/miss_alignment/distributed/queue.py`
- Create: `tests/distributed/__init__.py`
- Create: `tests/distributed/test_queue.py`

**Interfaces:**
- Produces:
  - `QueueLayout(root: Path)` — dataclass; `ensure_directories() -> None`; properties: `pending`, `running`, `done`, `failed`, `manager_hb`, `cluster` each returning `Path`; `worker_dir(worker_id: str) -> Path`.
  - `TaskSpec` — dataclass with fields: `task_id: str`, `model_checkpoint_path: str`, `tilt_series_path: str`, `output_directory: str`, `setting: str | list`, `patch_size: int`, `patch_overlap: float`, `batch_size: int`, `apply_ctf: bool`, `downsample: int`, `init_fingerprint: str`.
  - `compute_fingerprint(model_checkpoint_path, setting, patch_size, patch_overlap, batch_size, apply_ctf, downsample) -> str` — SHA-256 hex.
  - `write_pending(layout: QueueLayout, spec: TaskSpec) -> None`
  - `claim_one(layout: QueueLayout, worker_id: str) -> TaskSpec | None`
  - `mark_done(layout: QueueLayout, worker_id: str, spec: TaskSpec, final_loss: float, device: str) -> None`
  - `mark_failed(layout: QueueLayout, worker_id: str, spec: TaskSpec, error: str) -> None`
  - `clear_queue(layout: QueueLayout) -> None` — deletes `pending/done/failed` contents first, then recovers `running/` orphans into the now-empty `pending/`.

- [ ] **Step 1: Create directory skeleton**

```bash
mkdir -p tests/distributed
touch tests/distributed/__init__.py
```

- [ ] **Step 2: Write failing tests**

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


def test_claim_one_exclusive(layout):
    """Two sequential claimers: exactly one wins."""
    write_pending(layout, _spec())
    r0 = claim_one(layout, "worker-0")
    r1 = claim_one(layout, "worker-1")
    claimed = [r for r in (r0, r1) if r is not None]
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
    # simulate crash: running file remains; clear should put it back in pending
    clear_queue(layout)
    assert (layout.pending / "0000001-ts01.json").exists()


def test_clear_queue_wipes_done_and_failed(layout):
    spec = _spec()
    write_pending(layout, spec)
    claim_one(layout, "worker-0")
    mark_done(layout, "worker-0", spec, final_loss=0.1, device="cpu")
    # write a second spec directly to failed
    spec2 = _spec("0000002-ts02")
    write_pending(layout, spec2)
    claim_one(layout, "worker-0")
    mark_failed(layout, "worker-0", spec2, error="boom")

    clear_queue(layout)
    assert list(layout.done.glob("*.json")) == []
    assert list(layout.failed.glob("*.json")) == []


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

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/tegunovd/dev/miss-alignment
pytest tests/distributed/test_queue.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'miss_alignment.distributed'`

- [ ] **Step 4: Create `__main__.py` and `distributed/__init__.py` skeleton**

```python
# src/miss_alignment/__main__.py
from miss_alignment import cli

cli()
```

```python
# src/miss_alignment/distributed/__init__.py
"""Disk-based distributed task queue for miss-alignment inference."""
```

- [ ] **Step 5: Implement `queue.py`**

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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/distributed/test_queue.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/miss_alignment/__main__.py \
        src/miss_alignment/distributed/__init__.py \
        src/miss_alignment/distributed/queue.py \
        tests/distributed/__init__.py \
        tests/distributed/test_queue.py
git commit -m "feat: add distributed queue layer with atomic rename claim protocol"
```

---

## Task 2: Cluster configuration (`distributed/config.py`)

**Files:**
- Create: `src/miss_alignment/distributed/config.py`
- Create: `tests/distributed/test_config.py`

**Interfaces:**
- Produces:
  - `ClusterConfig` — dataclass with fields: `submit: str`, `submit_job_id_regex: str`, `cancel: str`, `script_path: Path`.
  - `load_cluster_config() -> ClusterConfig` — reads `MISS_CLUSTER_CONFIG` and `MISS_CLUSTER_SCRIPT`; raises `RuntimeError` if either is unset, `FileNotFoundError` if a path doesn't exist, `KeyError` if a required JSON key is missing.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_config.py
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


def test_load_cluster_config_raises_when_config_unset(monkeypatch):
    monkeypatch.delenv("MISS_CLUSTER_CONFIG", raising=False)
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="MISS_CLUSTER_CONFIG"):
        load_cluster_config()


def test_load_cluster_config_raises_when_script_unset(monkeypatch, cluster_json):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(cluster_json))
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="MISS_CLUSTER_SCRIPT"):
        load_cluster_config()


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

Expected: `ImportError` for `miss_alignment.distributed.config`.

- [ ] **Step 3: Implement `config.py`**

```python
# src/miss_alignment/distributed/config.py
"""Read cluster configuration from environment variables.

load_cluster_config() is called only when --n-cluster-workers is set.
It raises RuntimeError immediately if either required env var is absent,
so the user gets a clear error rather than a silent fallback to local mode.
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


def load_cluster_config() -> ClusterConfig:
    """Return ClusterConfig. Raises RuntimeError if env vars are missing."""
    config_path_str = os.environ.get("MISS_CLUSTER_CONFIG")
    if not config_path_str:
        raise RuntimeError(
            "MISS_CLUSTER_CONFIG environment variable is required when "
            "--n-cluster-workers is set. Point it to a JSON file with "
            "'submit', 'submit_job_id_regex', and 'cancel' keys."
        )

    script_path_str = os.environ.get("MISS_CLUSTER_SCRIPT")
    if not script_path_str:
        raise RuntimeError(
            "MISS_CLUSTER_SCRIPT environment variable is required when "
            "--n-cluster-workers is set. Point it to a shell script template "
            "containing a {{command}} placeholder."
        )

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
git commit -m "feat: add cluster config reader (raises if env vars missing)"
```

---

## Task 3: Worker subcommand (`distributed/worker.py`)

**Files:**
- Create: `src/miss_alignment/distributed/worker.py`
- Create: `tests/distributed/test_worker.py`

**Interfaces:**
- Consumes:
  - `QueueLayout`, `TaskSpec`, `claim_one`, `mark_done`, `mark_failed` from `distributed/queue.py`
  - `MissAlignment` from `..models.models`
  - `evaluate_tilt_series` from `..alignment.tilt_series` (after Task 4 adds the `model` param)
- Produces:
  - `run_worker_loop(layout, worker_id, device, manager_hb_timeout_s) -> None` — testable loop body.
  - `worker_miss_align(queue_dir, device, worker_id)` — Typer command.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_worker.py
"""Unit tests for the worker claim loop.

evaluate_tilt_series is mocked throughout; these tests validate the claim,
model-reuse, heartbeat-exit, and result-write logic without CUDA.
"""
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.queue import QueueLayout, TaskSpec, write_pending
from miss_alignment.distributed.worker import run_worker_loop


@pytest.fixture()
def layout(tmp_path):
    layout = QueueLayout(tmp_path / "tasks")
    layout.ensure_directories()
    return layout


def _write_manager_hb(layout, seq=0):
    for old in layout.manager_hb.glob("hb-*"):
        old.unlink(missing_ok=True)
    (layout.manager_hb / f"hb-{seq}").write_text("")


def _spec(task_id="0000001-ts01", fingerprint="abc123"):
    return TaskSpec(
        task_id=task_id,
        model_checkpoint_path="/data/model.ckpt",
        tilt_series_path=f"/data/{task_id}.xml",
        output_directory="/data/out",
        setting="anchoring",
        patch_size=96,
        patch_overlap=0.1,
        batch_size=32,
        apply_ctf=False,
        downsample=2,
        init_fingerprint=fingerprint,
    )


def test_worker_processes_task_and_writes_done(layout):
    _write_manager_hb(layout)
    write_pending(layout, _spec())

    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        return_value=(Path("/data/ts01.xml"), [0.5, 0.3, 0.1]),
    ), patch(
        "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        return_value=MagicMock(),
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
    ), patch(
        "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        return_value=MagicMock(),
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)

    failed = layout.failed / "0000001-ts01.json"
    assert failed.exists()
    data = json.loads(failed.read_text())
    assert "CUDA OOM" in data["error"]


def test_worker_exits_when_manager_hb_stale(layout):
    # Write a heartbeat file that is 200 seconds old
    hb_file = layout.manager_hb / "hb-0"
    hb_file.write_text("")
    old_time = time.time() - 200
    os.utime(hb_file, (old_time, old_time))

    write_pending(layout, _spec())

    called = []
    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=lambda **kw: called.append(True),
    ), patch(
        "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        return_value=MagicMock(),
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=120.0)

    assert called == []


def test_worker_reuses_model_when_fingerprint_matches(layout):
    """Model is loaded once when two tasks share the same init_fingerprint."""
    _write_manager_hb(layout)
    write_pending(layout, _spec("0000001-ts01", fingerprint="same"))
    write_pending(layout, _spec("0000002-ts02", fingerprint="same"))

    load_calls = []

    def fake_evaluate(**kwargs):
        return (Path(kwargs["tilt_series_path"]), [0.1])

    def fake_load(path, map_location=None):
        load_calls.append(path)
        return MagicMock()

    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=fake_evaluate,
    ), patch(
        "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        side_effect=fake_load,
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)

    assert len(load_calls) == 1  # loaded once despite two tasks


def test_worker_reloads_model_when_fingerprint_changes(layout):
    """Model is reloaded when fingerprint differs between tasks."""
    _write_manager_hb(layout)
    write_pending(layout, _spec("0000001-ts01", fingerprint="fp-a"))
    write_pending(layout, _spec("0000002-ts02", fingerprint="fp-b"))

    load_calls = []

    def fake_evaluate(**kwargs):
        return (Path(kwargs["tilt_series_path"]), [0.1])

    def fake_load(path, map_location=None):
        load_calls.append(path)
        return MagicMock()

    with patch(
        "miss_alignment.distributed.worker.evaluate_tilt_series",
        side_effect=fake_evaluate,
    ), patch(
        "miss_alignment.distributed.worker.MissAlignment.load_from_checkpoint",
        side_effect=fake_load,
    ):
        run_worker_loop(layout, "worker-0", "cpu", manager_hb_timeout_s=30.0)

    assert len(load_calls) == 2
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

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/worker.py tests/distributed/test_worker.py
git commit -m "feat: add worker subcommand with model fingerprint reuse across tasks"
```

---

## Task 4: Add `model` parameter to `evaluate_tilt_series`

**Files:**
- Modify: `src/miss_alignment/alignment/tilt_series.py`

**Interfaces:**
- Produces: `evaluate_tilt_series(..., model: MissAlignment | None = None)` — when `model` is provided, uses it directly instead of loading from disk. All existing callers unaffected.

- [ ] **Step 1: Read the current model-loading block**

Open `src/miss_alignment/alignment/tilt_series.py` at lines 118–186. The relevant section is:

```python
# line 131 ends the signature
) -> tuple[Path, list[float]]:
    ...
    # line 172
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path,
        map_location="cpu",
    )
    # line 184
    if hasattr(model.net, "_orig_mod"):
        model.net = model.net._orig_mod
```

- [ ] **Step 2: Add the `model` parameter to the signature**

In `src/miss_alignment/alignment/tilt_series.py`, add `model` as the last parameter before the closing `)`:

```python
# Before:
    n_control_points: int = 7,
) -> tuple[Path, list[float]]:

# After:
    n_control_points: int = 7,
    model: "MissAlignment | None" = None,
) -> tuple[Path, list[float]]:
```

Use a string annotation to avoid a circular import (the type is already imported at the top of the file — verify with `grep "MissAlignment" src/miss_alignment/alignment/tilt_series.py` before committing; if already imported, use the bare type).

- [ ] **Step 3: Replace the model-loading block**

```python
# Before (lines ~172-185):
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path,
        map_location="cpu",
    )
    if hasattr(model.net, "_orig_mod"):
        model.net = model.net._orig_mod

# After:
    if model is None:
        model = MissAlignment.load_from_checkpoint(
            model_checkpoint_path,
            map_location="cpu",
        )
        if hasattr(model.net, "_orig_mod"):
            model.net = model.net._orig_mod
```

- [ ] **Step 4: Run the existing alignment tests to verify nothing broke**

```bash
pytest tests/alignment/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/alignment/tilt_series.py
git commit -m "feat: add optional model param to evaluate_tilt_series for checkpoint reuse"
```

---

## Task 5: Provisioners (`distributed/provisioner.py`)

**Files:**
- Create: `src/miss_alignment/distributed/provisioner.py`
- Create: `tests/distributed/test_provisioner.py`

**Interfaces:**
- Consumes: `ClusterConfig` from `distributed/config.py`.
- Produces:
  - `WorkerProvisioner` — ABC with `ensure_workers(n_workers: int) -> None` and `shutdown() -> None`.
  - `LocalProvisioner(queue_dir: Path, devices: list[int])` — spawns `python -m miss_alignment worker` child processes, one per device. Ignores `n_workers`.
  - `ClusterProvisioner(queue_dir: Path, config: ClusterConfig)` — submits exactly `n_workers` cluster jobs on the first `ensure_workers` call.

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_provisioner.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.config import ClusterConfig
from miss_alignment.distributed.provisioner import ClusterProvisioner, LocalProvisioner


def test_local_provisioner_spawns_one_process_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_workers=10)

        assert mock_popen.call_count == 3
        # verify device args
        all_args = [str(c) for c in mock_popen.call_args_list]
        assert any("'0'" in s or '"0"' in s or "0" in s for s in all_args)


def test_local_provisioner_does_not_respawn_running(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)

        assert mock_popen.call_count == 1


def test_local_provisioner_respawns_dead_worker(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)  # should respawn because poll() != None

        assert mock_popen.call_count == 2


def test_local_provisioner_shutdown_terminates(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.shutdown()

        mock_proc.terminate.assert_called()


def test_cluster_provisioner_submits_n_workers_jobs(tmp_path):
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

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=4)

    assert len(submitted) == 4


def test_cluster_provisioner_cancels_on_shutdown(tmp_path):
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

    with patch(
        "miss_alignment.distributed.provisioner.subprocess.run", side_effect=fake_run
    ):
        p = ClusterProvisioner(queue_dir=tmp_path, config=cfg)
        p.ensure_workers(n_workers=3)
        p.shutdown()

    assert len(cancel_calls) == 3
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
        """Ensure workers are running. Called once at startup and each scheduler tick."""

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
        rendered = template_text.replace("{{command}}", command)
        for key, value in os.environ.items():
            if key.startswith("MISS_CLUSTER_VAR_"):
                var_name = key[len("MISS_CLUSTER_VAR_"):].lower()
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/distributed/test_provisioner.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/provisioner.py tests/distributed/test_provisioner.py
git commit -m "feat: add LocalProvisioner and ClusterProvisioner"
```

---

## Task 6: Manager (`distributed/manager.py`)

**Files:**
- Create: `src/miss_alignment/distributed/manager.py`
- Create: `tests/distributed/test_manager.py`

**Interfaces:**
- Consumes:
  - `QueueLayout`, `TaskSpec`, `write_pending`, `clear_queue`, `compute_fingerprint` from `distributed/queue.py`
  - `WorkerProvisioner` from `distributed/provisioner.py`
- Produces:
  - `run_distributed(tilt_series_list, model_checkpoint, output_directory, setting, patch_size, patch_overlap, batch_size, apply_ctf, downsample, devices, n_cluster_workers, queue_root) -> dict[str, float]`

- [ ] **Step 1: Write failing tests**

```python
# tests/distributed/test_manager.py
"""Integration tests for the manager coordinator.

A fake worker thread simulates cluster workers by polling pending/ and
writing done/ or failed/ files.
"""
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from miss_alignment.distributed.manager import run_distributed
from miss_alignment.distributed.queue import QueueLayout


def _make_xml(tmp_path, name):
    p = tmp_path / f"{name}.xml"
    p.write_text(f"<TiltSeries><Name>{name}</Name></TiltSeries>")
    return p


def _fake_worker_thread(queue_root, n_tasks, fail=False):
    """Simulates a worker: claims pending tasks, writes done or failed."""

    def _run():
        layout = QueueLayout(queue_root)
        done_count = 0
        deadline = time.time() + 15
        while done_count < n_tasks and time.time() < deadline:
            for f in list(layout.pending.glob("*.json")):
                data = json.loads(f.read_text())
                task_id = data["task_id"]
                running_dir = layout.running / "fake-worker"
                running_dir.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(f, running_dir / f.name)
                except FileNotFoundError:
                    continue
                if fail:
                    fail_data = {**data, "error": "boom", "worker_id": "fake-worker"}
                    (layout.failed / f"{task_id}.json").write_text(
                        json.dumps(fail_data)
                    )
                else:
                    done_data = {**data, "final_loss": 0.01, "device": "cpu"}
                    (layout.done / f"{task_id}.json").write_text(
                        json.dumps(done_data)
                    )
                (running_dir / f"{task_id}.json").unlink(missing_ok=True)
                done_count += 1
            time.sleep(0.05)

    return threading.Thread(target=_run, daemon=True)


class _NoOpProvisioner:
    def ensure_workers(self, n_workers):
        pass

    def shutdown(self):
        pass


def test_run_distributed_returns_losses(tmp_path):
    xml1 = _make_xml(tmp_path, "ts01")
    xml2 = _make_xml(tmp_path, "ts02")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("")
    queue_root = tmp_path / "tasks"

    worker = _fake_worker_thread(queue_root, n_tasks=2)
    worker.start()

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        return_value=_NoOpProvisioner(),
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
            n_cluster_workers=None,
            queue_root=queue_root,
        )

    assert set(losses.keys()) == {"ts01", "ts02"}
    assert all(v == pytest.approx(0.01) for v in losses.values())


def test_run_distributed_raises_on_any_failure(tmp_path):
    xml1 = _make_xml(tmp_path, "ts01")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("")
    queue_root = tmp_path / "tasks"

    worker = _fake_worker_thread(queue_root, n_tasks=1, fail=True)
    worker.start()

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        return_value=_NoOpProvisioner(),
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
                n_cluster_workers=None,
                queue_root=queue_root,
            )


def test_run_distributed_cleans_up_tasks_dir(tmp_path):
    xml1 = _make_xml(tmp_path, "ts01")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("")
    queue_root = tmp_path / "tasks"

    worker = _fake_worker_thread(queue_root, n_tasks=1)
    worker.start()

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        return_value=_NoOpProvisioner(),
    ):
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
            n_cluster_workers=None,
            queue_root=queue_root,
        )

    assert not queue_root.exists()
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
import shutil
import sys
import threading
import time
from pathlib import Path

import tqdm

from .config import load_cluster_config
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
        (layout.manager_hb / f"hb-{seq - 1}").unlink(missing_ok=True)


def _sweep_stalled_workers(layout: QueueLayout) -> None:
    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        ticks = list(worker_dir.glob("hb-*"))
        if ticks:
            age = time.time() - max(ticks, key=lambda p: p.stat().st_mtime).stat().st_mtime
        else:
            age = time.time() - worker_dir.stat().st_mtime

        if age <= _WORKER_STALL_TIMEOUT_S:
            continue

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
        for hb in worker_dir.glob("hb-*"):
            hb.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass


def _scheduler_thread(
    layout: QueueLayout,
    provisioner: WorkerProvisioner,
    n_workers: int,
    stop_event: threading.Event,
) -> None:
    hb_seq = 1  # seq 0 written before thread starts
    last_hb = time.time()
    last_sweep = 0.0

    while not stop_event.is_set():
        now = time.time()

        if now - last_hb >= _MANAGER_HB_INTERVAL_S:
            _write_manager_hb(layout, hb_seq)
            hb_seq += 1
            last_hb = now

        if now - last_sweep >= _SCHEDULER_INTERVAL_S:
            _sweep_stalled_workers(layout)
            provisioner.ensure_workers(n_workers)
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
    n_cluster_workers: int | None,
    queue_root: Path,
) -> dict[str, float]:
    """Write tasks, provision workers, block until all tasks are terminal.

    Returns dict[series_name → final_loss]. Raises RuntimeError listing all
    failed series if any task ends in failed/. Deletes queue_root on exit.
    """
    layout = QueueLayout(queue_root)
    layout.ensure_directories()
    clear_queue(layout)

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

    # Write the first manager heartbeat before starting workers so workers
    # never see a missing heartbeat on startup.
    _write_manager_hb(layout, seq=0)

    if n_cluster_workers is not None:
        cluster_config = load_cluster_config()
        provisioner: WorkerProvisioner = ClusterProvisioner(
            queue_dir=queue_root, config=cluster_config
        )
        n_workers = n_cluster_workers
    else:
        provisioner = LocalProvisioner(queue_dir=queue_root, devices=devices)
        n_workers = len(devices)

    stop_event = threading.Event()
    scheduler = threading.Thread(
        target=_scheduler_thread,
        args=(layout, provisioner, n_workers, stop_event),
        daemon=True,
    )
    scheduler.start()
    provisioner.ensure_workers(n_workers)

    pending_ids = set(task_ids)
    losses: dict[str, float] = {}
    failed_series: list[str] = []

    pbar = tqdm.tqdm(total=len(task_ids), desc="Tilt series alignment", file=sys.stdout)
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
        shutil.rmtree(queue_root, ignore_errors=True)

    if failed_series:
        raise RuntimeError(
            f"Alignment failed for {len(failed_series)} tilt series: "
            + ", ".join(failed_series)
        )

    return losses
```

- [ ] **Step 4: Run all distributed tests**

```bash
pytest tests/distributed/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/miss_alignment/distributed/manager.py tests/distributed/test_manager.py
git commit -m "feat: add distributed manager with scheduler thread, poll loop, and cleanup"
```

---

## Task 7: Wire up CLI, parallel, train, infer; delete `_parallel.py`

**Files:**
- Modify: `src/miss_alignment/alignment/parallel.py`
- Modify: `src/miss_alignment/train.py`
- Modify: `src/miss_alignment/infer.py`
- Modify: `src/miss_alignment/_cli.py`
- Modify: `src/miss_alignment/__init__.py`
- Modify: `src/miss_alignment/distributed/__init__.py`
- Modify: `tests/test_parallel.py`
- Delete: `src/miss_alignment/_parallel.py`

**Interfaces:**
- `run_alignment_parallel` gains `n_cluster_workers: int | None = None` parameter.
- `train_miss_align` and `infer_miss_align` each gain `n_cluster_workers: Optional[int] = typer.Option(None, ...)`.

- [ ] **Step 1: Update `tests/test_parallel.py`**

Replace the file entirely to test through the new `LocalProvisioner` path:

```python
# tests/test_parallel.py
"""Tests for the distributed worker provisioner (replaces _parallel.py tests)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miss_alignment.distributed.provisioner import LocalProvisioner


def test_local_provisioner_spawns_worker_per_device(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0, 1, 2])
        p.ensure_workers(n_workers=10)

        assert mock_popen.call_count == 3


def test_local_provisioner_does_not_double_spawn(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.ensure_workers(n_workers=5)

        assert mock_popen.call_count == 1


def test_local_provisioner_shutdown_terminates(tmp_path):
    with patch("miss_alignment.distributed.provisioner.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        p = LocalProvisioner(queue_dir=tmp_path, devices=[0])
        p.ensure_workers(n_workers=5)
        p.shutdown()

        mock_proc.terminate.assert_called()
```

- [ ] **Step 2: Run updated `test_parallel.py` to verify it passes now**

```bash
pytest tests/test_parallel.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 3: Update `alignment/parallel.py`**

```python
# src/miss_alignment/alignment/parallel.py
from pathlib import Path

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
    n_cluster_workers: int | None = None,
) -> dict[str, float]:
    """Distribute per-tilt-series alignment across local GPUs or a cluster.

    Without --n-cluster-workers, one worker subprocess is spawned per GPU in
    devices_list (local mode, unchanged behaviour). Set --n-cluster-workers N
    to submit N cluster jobs instead; requires MISS_CLUSTER_CONFIG and
    MISS_CLUSTER_SCRIPT to be set.

    Returns dict mapping tilt-series stem names to their final loss values.
    """
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
        n_cluster_workers=n_cluster_workers,
        queue_root=queue_root,
    )
```

- [ ] **Step 4: Add `--n-cluster-workers` to `train.py`**

Add the new option to `train_miss_align`'s signature. Find the `preprocess: bool` option (last existing option, around line 308) and add after it:

```python
    n_cluster_workers: Optional[int] = typer.Option(
        None,
        help="Number of cluster jobs to submit for the alignment phase. "
        "When set, activates cluster mode; requires MISS_CLUSTER_CONFIG "
        "and MISS_CLUSTER_SCRIPT environment variables to be set. "
        "When absent, local multi-GPU mode is used.",
    ),
```

Then pass it through to `run_alignment_parallel` in the call at line ~487:

```python
        run_alignment_parallel(
            model_checkpoint=str(training_model_path),
            tilt_series_list=tilt_series_list,
            output_directory=training_directory,
            setting=iteration_settings["alignment"],
            patch_size=alignment_config["patch_size"],
            patch_overlap=alignment_config["patch_overlap"],
            batch_size=alignment_config["batch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            devices_list=devices_alignment,
            n_cluster_workers=n_cluster_workers,
        )
```

- [ ] **Step 5: Add `--n-cluster-workers` to `infer.py`**

Add the same option to `infer_miss_align`'s signature after `preprocess: bool` (around line 41):

```python
    n_cluster_workers: Optional[int] = typer.Option(
        None,
        help="Number of cluster jobs to submit for the alignment phase. "
        "When set, activates cluster mode; requires MISS_CLUSTER_CONFIG "
        "and MISS_CLUSTER_SCRIPT environment variables to be set. "
        "When absent, local multi-GPU mode is used.",
    ),
```

Then pass it through to the `run_alignment_parallel` call at line ~161:

```python
        run_alignment_parallel(
            model_checkpoint=str(model_checkpoint),
            tilt_series_list=tilt_series_list,
            output_directory=data_directory,
            setting=iteration_settings["alignment"],
            patch_size=alignment_config["patch_size"],
            patch_overlap=alignment_config["patch_overlap"],
            batch_size=alignment_config["batch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            devices_list=devices_alignment,
            n_cluster_workers=n_cluster_workers,
        )
```

- [ ] **Step 6: Register the `worker` subcommand in `_cli.py`**

```python
# src/miss_alignment/_cli.py
from click import Context
import typer
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)


cli = typer.Typer(cls=OrderCommands, add_completion=False, no_args_is_help=True)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}

from .distributed.worker import worker_miss_align  # noqa: E402

cli.command(name="worker")(worker_miss_align)
```

- [ ] **Step 7: Export `worker_miss_align` from `__init__.py`**

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

- [ ] **Step 8: Update `distributed/__init__.py`**

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

- [ ] **Step 9: Delete `_parallel.py`**

```bash
git rm src/miss_alignment/_parallel.py
```

- [ ] **Step 10: Run full test suite and linter**

```bash
pytest --color=yes -v
ruff check --fix src/miss_alignment/
ruff format src/miss_alignment/
```

Expected: all tests PASS. If any test imports `miss_alignment._parallel` directly, fix that import (the module no longer exists). No ruff errors.

- [ ] **Step 11: Verify the worker subcommand is registered**

```bash
miss-alignment --help
miss-alignment worker --help
```

Expected: `worker` appears in the command list; `--queue-dir`, `--device`, `--worker-id` appear in its help.

- [ ] **Step 12: Commit**

```bash
git add \
  src/miss_alignment/alignment/parallel.py \
  src/miss_alignment/train.py \
  src/miss_alignment/infer.py \
  src/miss_alignment/_cli.py \
  src/miss_alignment/__init__.py \
  src/miss_alignment/distributed/__init__.py \
  tests/test_parallel.py
git commit -m "feat: wire distributed queue into run_alignment_parallel; add --n-cluster-workers; register worker subcommand; delete _parallel.py"
```
