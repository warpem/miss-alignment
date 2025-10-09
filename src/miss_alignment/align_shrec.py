import typer
from pathlib import Path

from ._cli import OPTION_PROMPT_KWARGS, cli

import torch
from lightning.pytorch import seed_everything

from miss_alignment.models import MissAlignment
from miss_alignment.data import MissAlignmentDataModule
from miss_alignment.alignment import evaluate_tilt_series


@cli.command(name="align_shrec", no_args_is_help=True)
def align_shrec(
    model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data_ground_truth: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    nboxes: int = 1,
    seed: int = 45132,
    patch_size: int = 64,
    device: str = typer.Option(
        "cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu')"
    ),
    iterations: int = typer.Option(
        50, help="Number of iterations for optimization alignment"
    ),
    misalignments_file: Path = typer.Option(
        None, help="Path to a previously generated misalignments JSON file"
    ),
) -> None:
    """Optimize alignment of tilt series using a trained model.

    Parameters
    ----------
    model_checkpoint : Path
        Path to the trained model checkpoint.
    test_data_directory : Path
        Directory containing test data.
    test_data_ground_truth : Path
        Directory containing the ground truth alignments.
    output_directory : Path
        Directory where results will be saved.
    nboxes : int, optional
        Number of boxes to use for optimization, by default 1.
    seed : int, optional
        Random seed for reproducibility, by default 45132.
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', 'cpu'), by default "cuda:0".
    iterations : int, optional
        Number of iterations for optimization alignment, by default 50.
        Controls how many sets of maps are optimized for better statistics.
    misalignments_file : Path, optional
        Path to a previously generated misalignments JSON file, by default None.
        If provided, the misalignments from this file will be used instead of generating new ones.
    """
    seed_everything(seed, workers=True)
    torch.set_printoptions(precision=2, sci_mode=False)

    # get model
    model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
    model.freeze()

    # initialize the dataset
    datamodule = MissAlignmentDataModule(
        test_data_directory,
        dataset_type="SHREC",
    )
    datamodule.setup(stage="validate")
    ground_truth = MissAlignmentDataModule(
        test_data_ground_truth,
        dataset_type="SHREC",
    )
    ground_truth.setup(stage="validate")
    tomograms = datamodule.test_dataset.tomos
    tomograms_ground_truth = ground_truth.test_dataset.tomos

    for test, ground_truth in zip(tomograms, tomograms_ground_truth):
        file_path, tilt_series = test
        file_name = file_path.stem
        _, tilt_series_ground_truth = ground_truth
        evaluate_tilt_series(
            model,
            file_name,
            tilt_series,
            (1, 4, 4),
            128,
            (180, 512, 512),
            output_directory,
            tilt_series_ground_truth=tilt_series_ground_truth,
            device="cpu",
        )

    return None
