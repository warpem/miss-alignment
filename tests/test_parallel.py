"""Tests for the shared one-process-per-device work-queue helper."""

import queue

import pytest

from miss_alignment._parallel import run_device_pool


def _square_runner(device, task_queue, result_queue):
    """Process int jobs by squaring them (device is ignored; no CUDA needed)."""
    while True:
        try:
            n = task_queue.get_nowait()
        except queue.Empty:
            break
        result_queue.put_nowait(n * n)


def _failing_runner(device, task_queue, result_queue):
    """Raise on a specific job to exercise worker-failure detection."""
    while True:
        try:
            n = task_queue.get_nowait()
        except queue.Empty:
            break
        if n == 3:
            raise ValueError("boom")
        result_queue.put_nowait(n * n)


@pytest.mark.filterwarnings("ignore")
def test_run_device_pool_processes_all_jobs():
    """Every job is processed exactly once across multiple worker processes."""
    jobs = [1, 2, 3, 4, 5]
    results = run_device_pool(
        jobs, _square_runner, runner_args=(), devices=[0, 1], desc="test"
    )
    assert sorted(results) == [1, 4, 9, 16, 25]


@pytest.mark.filterwarnings("ignore")
def test_run_device_pool_raises_on_worker_failure():
    """A worker that dies with a non-zero exit code surfaces as RuntimeError."""
    jobs = [1, 2, 3, 4, 5]
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        run_device_pool(jobs, _failing_runner, runner_args=(), devices=[0], desc="test")
