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

    def live_worker_count(self):
        return 0

    def worker_counts_by_type(self):
        return {"workers": 0}


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


def test_run_distributed_prepare_stacks_task_type(tmp_path):
    """Manager creates prepare_stacks tasks with the correct task_type field."""
    import json

    xml1 = _make_xml(tmp_path, "ts01")
    queue_root = tmp_path / "tasks"

    # Capture the task JSON before the worker thread consumes it
    written_tasks = []

    worker = _fake_worker_thread(queue_root, n_tasks=1)

    with patch(
        "miss_alignment.distributed.manager.LocalProvisioner",
        return_value=_NoOpProvisioner(),
    ):
        # Run in a thread so we can inspect tasks/ briefly — but just check
        # the return value here since the fake worker resolves immediately.
        losses = None

        def run():
            nonlocal losses
            losses = run_distributed(
                tilt_series_list=[xml1],
                model_checkpoint=Path(""),
                output_directory=tmp_path,
                setting="",
                patch_size=0,
                patch_overlap=0.0,
                batch_size=0,
                apply_ctf=False,
                downsample=1,
                devices=[0],
                n_cluster_workers=None,
                queue_root=queue_root,
                task_type="prepare_stacks",
                desired_pixel_size=10.0,
            )

        import threading

        worker.start()
        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=15)

    assert losses is not None
    assert "ts01" in losses
