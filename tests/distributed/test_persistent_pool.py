"""Exercise real queue/worker loops across inference batches without a GPU."""

import os
import threading
import time
from pathlib import Path

import pytest

from miss_alignment import infer as infer_module
from miss_alignment.distributed import manager, worker
from miss_alignment.distributed.manager import DistributedPool
from miss_alignment.distributed.provisioner import WorkerProvisioner
from miss_alignment.distributed.queue import QueueLayout


class ThreadProvisioner(WorkerProvisioner):
    """Replace process/scheduler launches, retaining the real worker protocol."""

    def __init__(self, queue_dir, persistent=False, **_kwargs):
        self.layout = QueueLayout(queue_dir)
        self.persistent = persistent
        self.threads = []
        self.errors = []
        self.shutdown_calls = 0

    def ensure_workers(self, n_workers):
        if self.errors:
            raise self.errors[0]
        if n_workers and not any(t.is_alive() for t in self.threads):
            name = f"test-worker-{id(self)}-{len(self.threads)}"

            def run():
                try:
                    worker.run_worker_loop(
                        self.layout, name, "cpu", persistent=self.persistent
                    )
                except SystemExit:
                    # Simulate a preempted process: its last heartbeat goes stale.
                    for tick in self.layout.worker_dir(name).glob("hb-*"):
                        os.utime(tick, (0, 0))
                except BaseException as exc:
                    self.errors.append(exc)

            thread = threading.Thread(target=run, name=name, daemon=True)
            self.threads.append(thread)
            thread.start()

    def live_worker_count(self):
        return sum(t.is_alive() for t in self.threads)

    def shutdown(self):
        self.shutdown_calls += 1
        self.layout.stop_file.touch()
        for thread in self.threads:
            thread.join(timeout=2)
            assert not thread.is_alive(), "worker did not stop with its pool"
        assert not self.errors


@pytest.fixture()
def pool_workers(monkeypatch):
    provisioners = []
    loaded = []
    executed = []

    def provisioner(**kwargs):
        result = ThreadProvisioner(**kwargs)
        provisioners.append(result)
        return result

    def load(path):
        loaded.append((threading.current_thread().name, path))
        return path

    def execute(spec, device, cached_model):
        if spec.task_type == "alignment":
            assert cached_model == spec.model_checkpoint_path
            (
                Path(spec.output_directory)
                / f"{Path(spec.tilt_series_path).stem}_alignment_loss.json"
            ).write_text("[0.1]")
        executed.append((threading.current_thread().name, spec))
        return float(spec.downsample)

    monkeypatch.setattr(manager, "LocalProvisioner", provisioner)
    monkeypatch.setattr(manager, "ClusterProvisioner", provisioner)
    monkeypatch.setattr(manager, "load_cluster_config", lambda: object())
    monkeypatch.setattr(manager, "_POLL_INTERVAL_S", 0.005)
    monkeypatch.setattr(manager, "_SCHEDULER_INTERVAL_S", 0.01)
    monkeypatch.setattr(manager, "_MANAGER_HB_INTERVAL_S", 0.01)
    monkeypatch.setattr(worker, "_HB_INTERVAL_S", 0.01)
    monkeypatch.setattr(worker, "_IDLE_POLL_S", 0.005)
    monkeypatch.setattr(worker, "_load_model", load)
    monkeypatch.setattr(worker, "_execute_task", execute)
    return provisioners, loaded, executed


def _batch(pool, tmp_path, iteration=1):
    paths = [tmp_path / "a.xml", tmp_path / "b.xml"]
    for path in paths:
        path.touch()
    return pool.run(
        tilt_series_list=paths,
        model_checkpoint=tmp_path / f"iter{iteration}" / "model.ckpt",
        output_directory=tmp_path,
        setting="global" if iteration == 1 else [3, 3],
        patch_size=96,
        patch_overlap=0.1,
        batch_size=32,
        apply_ctf=False,
        downsample=iteration,
    )


@pytest.mark.parametrize("cluster", [False, True])
def test_same_workers_and_allocations_survive_empty_iteration_gaps(
    tmp_path, pool_workers, cluster
):
    provisioners, loaded, executed = pool_workers
    with DistributedPool(
        [] if cluster else [0],
        1 if cluster else None,
        tmp_path / "tasks",
        persistent=True,
    ) as pool:
        assert _batch(pool, tmp_path, 1) == {"a": 1.0, "b": 1.0}
        first_ids = {spec.task_id for _, spec in executed}
        assert len(provisioners) == 1
        provider = provisioners[0]
        assert provider.persistent
        thread = provider.threads[0]
        heartbeat = {p.name for p in pool.layout.manager_hb.glob("hb-*")}
        # No pending work, including while the manager snapshots an iteration.
        time.sleep(0.05)
        assert thread.is_alive()
        assert provider.shutdown_calls == 0
        assert {p.name for p in pool.layout.manager_hb.glob("hb-*")} != heartbeat
        assert _batch(pool, tmp_path, 2) == {"a": 2.0, "b": 2.0}
        assert provider.threads == [thread]
        second_ids = {spec.task_id for _, spec in executed} - first_ids
        assert len(second_ids) == 2
        assert {name for name, _ in executed} == {thread.name}
        assert [Path(path).parent.name for _, path in loaded] == ["iter1", "iter2"]
    assert provider.shutdown_calls == 1
    assert not thread.is_alive()
    assert not (tmp_path / "tasks").exists()


def test_previous_iteration_results_cannot_complete_next_batch(
    tmp_path, monkeypatch, pool_workers
):
    _provisioners, _loaded, executed = pool_workers
    waiting, release = threading.Event(), threading.Event()
    original = worker._execute_task

    def execute(spec, *args):
        if spec.downsample == 2:
            waiting.set()
            assert release.wait(2)
        return original(spec, *args)

    monkeypatch.setattr(worker, "_execute_task", execute)
    errors, results = [], []
    with DistributedPool([0], None, tmp_path / "tasks", persistent=True) as pool:
        _batch(pool, tmp_path, 1)
        assert len(list(pool.layout.done.glob("*.json"))) == 2

        def second_batch():
            try:
                results.append(_batch(pool, tmp_path, 2))
            except BaseException as exc:
                errors.append(exc)

        second = threading.Thread(target=second_batch, daemon=True)
        second.start()
        try:
            assert waiting.wait(2)
            assert second.is_alive()
            assert not results
        finally:
            release.set()
            second.join(2)
        assert not second.is_alive()
        assert not errors
        assert results == [{"a": 2.0, "b": 2.0}]
        assert len({spec.task_id for _, spec in executed}) == 4


@pytest.mark.parametrize("mode", ["local", "cluster", "mixed"])
def test_full_inference_shares_pool_with_preparation_and_all_eight_models(
    tmp_path, monkeypatch, pool_workers, mode
):
    provisioners, loaded, executed = pool_workers
    data = tmp_path / "data"
    data.mkdir()
    for name in ("a", "b"):
        (data / f"{name}.xml").write_text("XML stub")
    models = tmp_path / "models"
    for i in range(1, 9):
        (models / f"iter{i}").mkdir(parents=True)
        (models / f"iter{i}" / "model.ckpt").touch()
    monkeypatch.setattr(infer_module, "get_pretrained_models", lambda: models)
    monkeypatch.setattr(
        infer_module, "exclude_nonfinite_alignment_tilt_series", lambda _: []
    )
    monkeypatch.setattr(
        infer_module.torch.cuda, "device_count", lambda: 0 if mode == "cluster" else 1
    )
    infer_module.infer_miss_align(
        data_directory=data,
        start_at_iteration=0,
        prepare_stacks=10,
        preprocess=True,
        n_cluster_workers=None if mode == "local" else 1,
    )
    assert len(provisioners) == (2 if mode == "mixed" else 1)
    assert all(p.persistent and len(p.threads) == 1 for p in provisioners)
    assert all(p.shutdown_calls == 1 for p in provisioners)
    assert len(executed) == 20  # two series: prep, coarse alignment, eight models
    assert len({spec.task_id for _, spec in executed}) == 20
    assert {Path(path).parent.name for _, path in loaded} == {
        f"iter{i}" for i in range(1, 9)
    }
    for i in range(1, 9):
        assert (data / f"iter{i}" / "a.xml").is_file()
        assert (data / f"iter{i}" / "model_source.txt").read_text().strip() == str(
            models / f"iter{i}" / "model.ckpt"
        )
    assert not (data / "tasks").exists()


@pytest.mark.parametrize("body_error", [False, True])
def test_pool_closes_once_on_task_failure_or_iteration_snapshot_error(
    tmp_path, monkeypatch, pool_workers, body_error
):
    provisioners, _loaded, _executed = pool_workers
    if not body_error:

        def fail(*_args):
            raise ValueError("bad model output")

        monkeypatch.setattr(worker, "_execute_task", fail)
    with pytest.raises((RuntimeError, OSError)):
        with DistributedPool([0], None, tmp_path / "tasks", persistent=True) as pool:
            _batch(pool, tmp_path)
            raise OSError("iteration snapshot failed")
    assert len(provisioners) == 1
    assert provisioners[0].shutdown_calls == 1
    assert not (tmp_path / "tasks").exists()


def test_delayed_old_workers_cannot_join_a_new_queue(tmp_path, pool_workers):
    # A delayed job from a previous invocation must not join the next invocation.
    first = DistributedPool([0], None, tmp_path / "tasks", persistent=True)
    second = DistributedPool([0], None, tmp_path / "tasks", persistent=True)
    assert first.layout.root != second.layout.root
    first.close()
    assert (
        worker.run_worker_loop(first.layout, "late-worker", "cpu", persistent=True)
        == "queue removed"
    )
    assert not first.layout.root.exists()
    with second:
        assert _batch(second, tmp_path) == {"a": 1.0, "b": 1.0}


def test_pool_startup_failure_cleans_its_owned_queue(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("missing cluster configuration")

    monkeypatch.setattr(manager, "load_cluster_config", fail)
    with pytest.raises(RuntimeError, match="missing cluster configuration"):
        with DistributedPool([], 1, tmp_path / "tasks", persistent=True) as pool:
            _batch(pool, tmp_path)
    assert not (tmp_path / "tasks").exists()


def test_preempted_worker_is_replaced_and_claimed_task_is_recovered(
    tmp_path, monkeypatch, pool_workers
):
    provisioners, _loaded, executed = pool_workers
    original = worker._execute_task
    attempts = []

    def execute(spec, *args):
        attempts.append(spec.task_id)
        if len(attempts) == 1:
            raise SystemExit("simulated worker preemption")
        return original(spec, *args)

    monkeypatch.setattr(worker, "_execute_task", execute)
    with DistributedPool([0], None, tmp_path / "tasks", persistent=True) as pool:
        assert _batch(pool, tmp_path) == {"a": 1.0, "b": 1.0}
        assert len(provisioners) == 1
        assert len(provisioners[0].threads) == 2  # replacement, not a new pool
        assert attempts.count(attempts[0]) == 2
        assert len(executed) == 2


def test_idle_workers_keep_manager_heartbeat_during_slow_scheduler_query(
    tmp_path, monkeypatch, pool_workers
):
    provisioners, _loaded, _executed = pool_workers
    query_entered, release_query = threading.Event(), threading.Event()
    with DistributedPool([0], None, tmp_path / "tasks", persistent=True) as pool:
        _batch(pool, tmp_path)
        provider = provisioners[0]

        def query():
            query_entered.set()
            assert release_query.wait(2)
            return {"workers": provider.live_worker_count()}

        provider.worker_counts_by_type = query
        try:
            assert query_entered.wait(2)
            before = {p.name for p in pool.layout.manager_hb.glob("hb-*")}
            time.sleep(0.05)
            after = {p.name for p in pool.layout.manager_hb.glob("hb-*")}
            assert before != after
            assert provider.threads[0].is_alive()
        finally:
            release_query.set()


def test_close_waits_for_late_submission_before_cancelling(tmp_path, monkeypatch):
    cfg = type("Config", (), {})()
    # Exercise the real ClusterProvisioner with a scheduler response delayed
    # until close has started, so shutdown must retain the returned job ID.
    cfg.script_path = tmp_path / "worker.sh"
    cfg.script_path.write_text("#!/bin/bash\n{{command}}\n")
    cfg.submit = "submit {{script_path}}"
    cfg.submit_job_id_regex = r"job (\d+)"
    cfg.cancel = "cancel {{job_id}}"
    cfg.status_list = "status"
    cfg.scheduler, cfg.custom_alive_statuses, cfg.custom_status_regex = "slurm", [], ""
    monkeypatch.setattr(manager, "load_cluster_config", lambda: cfg)
    entered, release = threading.Event(), threading.Event()
    cancelled, errors = [], []
    from miss_alignment.distributed import provisioner

    def run(command, **kwargs):
        if command.startswith("submit"):
            entered.set()
            assert release.wait(2)
            return type("Result", (), {"stdout": "job 123"})()
        if command.startswith("cancel"):
            cancelled.append(command)
        return type("Result", (), {"stdout": "123,PENDING\n"})()

    monkeypatch.setattr(provisioner.subprocess, "run", run)
    pool = DistributedPool([], 1, tmp_path / "tasks", persistent=True)

    def submit_batch():
        try:
            with pool:
                _batch(pool, tmp_path)
        except BaseException as exc:
            errors.append(exc)

    batch = threading.Thread(target=submit_batch, daemon=True)
    batch.start()
    assert entered.wait(2)
    closing = threading.Thread(target=pool.close, daemon=True)
    closing.start()
    try:
        assert closing.is_alive()
        assert not cancelled
    finally:
        release.set()
    closing.join(2)
    batch.join(2)
    assert not closing.is_alive()
    assert not batch.is_alive()
    assert len(errors) == 1 and "closed" in str(errors[0])
    assert cancelled == ["cancel 123"]
    assert not pool.layout.root.exists()


def test_real_local_worker_process_is_reused_across_batches(tmp_path, monkeypatch):
    import subprocess
    from miss_alignment.distributed import provisioner

    # Execute the actual worker CLI/heartbeat loop in a child interpreter;
    # only model loading and scientific computation are replaced in that child.
    bootstrap = """
import json, os, sys
from pathlib import Path
from miss_alignment import cli
from miss_alignment.distributed import worker
worker._load_model = lambda path: path

def execute(spec, device, cached_model):
    assert cached_model == spec.model_checkpoint_path
    output = Path(spec.output_directory) / (spec.task_id + ".result")
    output.write_text(json.dumps({"pid": os.getpid(), "model": cached_model}))
    return float(spec.downsample)
worker._execute_task = execute
sys.argv = [sys.argv[0], "worker", *sys.argv[1:]]
cli()
"""
    original_popen = subprocess.Popen
    children = []

    def spawn(args, **kwargs):
        child = original_popen([args[0], "-c", bootstrap, *args[4:]], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(provisioner.subprocess, "Popen", spawn)
    monkeypatch.setattr(manager, "_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(manager, "_SCHEDULER_INTERVAL_S", 0.1)
    pool = DistributedPool([0], None, tmp_path / "tasks", persistent=True)
    errors = []

    def run():
        try:
            with pool:
                assert _batch(pool, tmp_path, 1) == {"a": 1.0, "b": 1.0}
                time.sleep(0.6)  # longer than the worker's empty-queue poll
                assert _batch(pool, tmp_path, 2) == {"a": 2.0, "b": 2.0}
        except BaseException as exc:
            errors.append(exc)

    coordinator = threading.Thread(target=run, daemon=True)
    coordinator.start()
    coordinator.join(25)
    if coordinator.is_alive():
        pool.close()
        coordinator.join(2)
        pytest.fail("subprocess worker did not complete two batches")
    assert not errors
    assert len(children) == 1
    import json

    results = [json.loads(path.read_text()) for path in tmp_path.glob("*.result")]
    assert len(results) == 4
    assert {r["pid"] for r in results} == {children[0].pid}
    assert {Path(r["model"]).parent.name for r in results} == {"iter1", "iter2"}
    assert children[0].poll() is not None
    assert not (tmp_path / "tasks").exists()
