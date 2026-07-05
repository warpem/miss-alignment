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
