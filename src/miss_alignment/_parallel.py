"""Shared one-process-per-GPU work-queue helper.

Mirrors the scheme used by ``alignment.run_alignment_parallel``: start exactly
one worker process per unique device, feed all jobs through a shared queue, and
let each worker pull jobs until the queue is empty. This binds a device to a
*process* (deterministic, one job per GPU at a time) instead of to a task index,
and uses every available device rather than a fixed process count.
"""

import multiprocessing as mp
import sys
import time
from collections.abc import Callable
from typing import Any

import tqdm


def run_device_pool(
    jobs: list[Any],
    runner: Callable,
    runner_args: tuple,
    devices: list[int] | None,
    desc: str,
) -> list[Any]:
    """Run ``jobs`` across one worker process per unique device.

    Each worker runs ``runner(device, task_queue, result_queue, *runner_args)``,
    pulling jobs off ``task_queue`` until empty and putting one result on
    ``result_queue`` per finished job. One process is started per unique entry in
    ``devices`` (or a single default-device process when ``devices`` is falsy).

    Returns the list of results collected from the workers. Raises ``RuntimeError``
    if any worker exits with a non-zero code (all workers are then terminated).
    """
    ctx = mp.get_context("spawn")
    device_slots = sorted(set(devices)) if devices else [None]

    with ctx.Manager() as manager:
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        for job in jobs:
            task_queue.put_nowait(job)

        procs = [
            ctx.Process(
                target=runner,
                args=(device, task_queue, result_queue, *runner_args),
            )
            for device in device_slots
        ]
        [p.start() for p in procs]

        results: list[Any] = []
        pbar = tqdm.tqdm(
            total=len(jobs),
            desc=desc,
            file=sys.stdout,
        )
        while len(results) < len(jobs):
            while not result_queue.empty():
                results.append(result_queue.get_nowait())
                pbar.update(1)

            for p in procs:
                # a worker that died with a non-zero exit code means a job failed;
                # tear everything down rather than hang waiting for its result
                if not p.is_alive() and p.exitcode != 0:
                    for x in procs:
                        x.terminate()
                    for x in procs:
                        x.join(timeout=5.0)
                    pbar.close()
                    raise RuntimeError(
                        f"A worker process for '{desc}' stopped unexpectedly."
                    )

            time.sleep(0.1)

        pbar.close()
        [p.join() for p in procs]

    return results
