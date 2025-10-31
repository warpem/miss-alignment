import queue
import multiprocessing as mp
import time
import torch
import tqdm
from multiprocessing.managers import BaseProxy
from pathlib import Path
from typing import Optional

from .tilt_series import evaluate_tilt_series


def gpu_runner(
        device: str,
        task_queue: BaseProxy,
        result_queue: BaseProxy,
) -> None:
    """Start a GPU runner, each runner should be initialized to a
    multiprocessing.Process() and manage running jobs on a single GPU. Each runner will
    grab jobs from the task_queue and assign jobs to the result_queue once they finish.
    When the task_queue is empty the gpu_runner will stop.

    Parameters
    ----------
    device: str
        a GPU index to assign to the runner
    task_queue: mp.managers.BaseProxy
        shared queue from multiprocessing with jobs to run
    result_queue: mp.manager.BaseProxy
        shared queue from multiprocessing for finished jobs
    """
    torch.set_num_threads(1)
    while True:
        try:
            task_parameters = task_queue.get_nowait()
            _ = evaluate_tilt_series(
                **task_parameters,
                device=device,
            )
            # place the name of the finished tilt_series
            result_queue.put_nowait(task_parameters['tilt_series_path'].stem)
        except queue.Empty:
            break


def run_alignment_parallel(
        model_checkpoint: Path,
        tilt_series_list: list[Path],
        output_directory: Path,
        setting: str |
                 tuple[int, int, int] |
                 tuple[int, int, int, int],
        patch_size: int,
        batch_size: int,
        apply_ctf: bool,
        devices_list: list[int],
):
    """Run a job in parallel over a single or multiple GPUs. If no volume_splits are
    given the search is parallelized by splitting the angular search. If volume_splits
    are provided the job will first be split by volume, if there are still more GPUs
    available, the subvolume jobs are still further split by angular search.

    Parameters
    ----------
    model_checkpoint: Path
    tilt_series_list: list[Path]
    patches_per_dim: tuple[int, int, int]
    patch_size: int
    tomogram_shape: tuple[int, int, int]
    output_directory: Path
    devices_list: list[int]
    ground_truth_list: list[Path]
    """
    jobs = []
    for i, tilt_series in enumerate(tilt_series_list):
        jobs.append({
            'model_checkpoint_path': model_checkpoint,
            'tilt_series_path': tilt_series,
            'output_directory': output_directory,
            'setting': setting,
            'patch_size': patch_size,
            'batch_size': batch_size,
            'apply_ctf': apply_ctf,
        })

    results = []
    with mp.Manager() as manager:
        task_queue = (
            manager.Queue()
        )  # the list of tasks where processes can get there next task from
        result_queue = (
            manager.Queue()
        )  # this will accumulate results from the processes

        [task_queue.put_nowait(j) for j in jobs]  # put all tasks

        # set the processes and start them!
        procs = [
            mp.Process(
                target=gpu_runner,
                args=(
                    'cuda:' + str(g),
                    task_queue,
                    result_queue,
                ),
            )
            for g in devices_list
        ]
        [p.start() for p in procs]

        pbar = tqdm.tqdm(total=len(jobs), desc='tilt-series alignment')
        while True:
            while not result_queue.empty():
                results.append(result_queue.get_nowait())
                pbar.update(1)

            if len(results) == len(jobs):
                # its done if all the results from the spawn were send back
                break

            for p in procs:
                # if one of the processes is no longer alive and has a failed exit
                # we should error
                if not p.is_alive() and p.exitcode != 0:  # to prevent a deadlock
                    [
                        x.terminate() for x in procs
                    ]  # kill all spawned processes if something broke
                    raise RuntimeError(
                        "One or more of the processes stopped unexpectedly."
                    )

            time.sleep(.1)
        pbar.close()

        [p.join() for p in procs]
