import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli

import torch
import einops
import numpy as np
from pytorch_lightning import seed_everything
import napari

from miss_alignment.data import MissAlignmentDataModule
from miss_alignment.models import MissAlignment
# from miss_alignment.align_backend import (
#     center_of_mass,
# )


def optimize_shifts_shrec(
    model,
    tilt_series,
    positions,
):
    """Find shifts to optimize model score.

    Parameters
    ----------
    model: torch.nn.Module
        Model that produces the optimization target.
    tilt_image_dfts: torch.Tensor
        DFTs of tilt images to use for reconstruction.
    tilt_rotation_matrices: torch.Tensor:
        Tilt rotation matrices to use for back projection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list]
        - Detected translational alignment.
        - List of loss values for each optimization iteration.
    """
    patch_size = 64
    tilt_series.sample_translations.requires_grad_()

    alignment_optimizer = torch.optim.LBFGS(
        [
            tilt_series.sample_translations,
        ],
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        volumes = []
        for zyx in positions:
            subtomo = tilt_series.reconstruct_subvolume(zyx, patch_size)
            volumes.append(subtomo)
        volumes = torch.stack(volumes)

        # new_com = center_of_mass(volumes)
        # distance = torch.sum((new_com - gt_com) ** 2)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, "c d h w -> c 1 d h w")

        # Get loss and compute backward pass
        loss = model(volumes)
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        print(loss.item())
        loss_values.append(loss.item())

        return loss

    n_iters = 2  # 5 iterations should give convergence
    for x in range(n_iters):
        print(f"Optimizing... {x + 1}/{n_iters}")
        alignment_optimizer.step(closure)

    # remove gradients
    tilt_series.sample_translations.detach()

    return tilt_series, loss_values


@cli.command(name="align_shrec", no_args_is_help=True)
def align_shrec(
    model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    nboxes: int = 1,
    seed: int = 45132,
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
    model.eval()

    # initialize the dataset
    datamodule = MissAlignmentDataModule(
        test_data_directory,
        dataset_type="SHREC",
    )
    datamodule.setup(stage="validate")
    tomograms = datamodule.test_dataset.tomos
    tomogram_shape = (180, 512, 512)
    d, h, w = tomogram_shape
    tomogram_center = tuple(x // 2 for x in tomogram_shape)
    dc, hc, wc = tomogram_center

    position_grid = []
    ys = np.linspace(32, h - 32, 4) - hc
    xs = np.linspace(32, w - 32, 4) - wc
    for y in ys:
        for x in xs:
            position_grid.append((0, y, x))

    for _, tilt_series in tomograms:
        initial_reconstruction = tilt_series.reconstruct_tomogram(tomogram_shape, 64)

        aligned_tilt_series, loss = optimize_shifts_shrec(
            model, tilt_series, position_grid
        )
        aligned_reconstruction = tilt_series.reconstruct_tomogram(tomogram_shape, 64)

        # print(loss)

        viewer = napari.Viewer()
        viewer.add_image(initial_reconstruction.detach().numpy(), name="initial")
        viewer.add_image(aligned_reconstruction.detach().numpy(), name="aligned")
        napari.run()

        del initial_reconstruction, aligned_reconstruction

    return None
