import queue
import torch
from multiprocessing.managers import BaseProxy
from pathlib import Path

from .._parallel import run_device_pool
from .tilt_series import evaluate_tilt_series


def gpu_runner(
    device: int,
    task_queue: BaseProxy,
    result_queue: BaseProxy,
) -> None:
    """Start a GPU runner, each runner should be initialized to a
    multiprocessing.Process() and manage running jobs on a single GPU. Each runner will
    grab jobs from the task_queue and assign jobs to the result_queue once they finish.
    When the task_queue is empty the gpu_runner will stop.

    Parameters
    ----------
    device: int
        a GPU index to assign to the runner
    task_queue: mp.managers.BaseProxy
        shared queue from multiprocessing with jobs to run
    result_queue: mp.manager.BaseProxy
        shared queue from multiprocessing for finished jobs
    """
    torch.set_num_threads(1)
    cuda_device = f"cuda:{device}"
    while True:
        try:
            task_parameters = task_queue.get_nowait()
            tilt_series_path, loss_values = evaluate_tilt_series(
                **task_parameters,
                device=cuda_device,
            )
            # place the name and final loss of the finished tilt_series
            final_loss = float(loss_values[-1]) if loss_values else None
            result_queue.put_nowait(
                {
                    "name": tilt_series_path.stem,
                    "final_loss": final_loss,
                }
            )
        except queue.Empty:
            break


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

    Returns
    -------
    dict[str, float]
        Dictionary mapping tilt-series names to their final loss values.
    """
    jobs = [
        {
            "model_checkpoint_path": model_checkpoint,
            "tilt_series_path": tilt_series,
            "output_directory": output_directory,
            "setting": setting,
            "patch_size": patch_size,
            "patch_overlap": patch_overlap,
            "batch_size": batch_size,
            "apply_ctf": apply_ctf,
            "downsample": downsample,
        }
        for tilt_series in tilt_series_list
    ]

    # one worker process per unique GPU, each pulling jobs from a shared queue
    results = run_device_pool(
        jobs=jobs,
        runner=gpu_runner,
        runner_args=(),
        devices=devices_list,
        desc="Tilt series alignment",
    )

    # Convert results to dictionary of losses
    losses = {result["name"]: result["final_loss"] for result in results}
    return losses
