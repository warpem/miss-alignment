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
