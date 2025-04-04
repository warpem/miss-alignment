import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli

import torch
import einops
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d, insert_central_slices_rfft_3d
)
from torch_fourier_shift import fourier_shift_dft_2d,  fourier_shift_dft_3d
from torch_grid_utils import fftfreq_grid

from miss_alignment.data import EMDBDataset
from miss_alignment.models import MissAlignment


def prep_tilts(
        volume: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        tilt_image_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = fftfreq_grid(
        image_shape=volume.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device
    )

    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    volume_dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    volume_dft = torch.fft.rfftn(volume_dft, dim=(-3, -2, -1))
    volume_dft = torch.fft.fftshift(volume_dft, dim=(-3, -2,))  # actual fftshift of 3D rfft

    # apply random 3d shift
    random_shift = torch.tensor(
        np.random.normal(
            loc=0, scale=.1 * volume.shape[-1], size=(3,)
        ),
        dtype=torch.float32
    )
    volume_dft = fourier_shift_dft_3d(
        dft=volume_dft,
        image_shape=(volume.shape),
        shifts=random_shift,
        rfft=True,
        fftshifted=True
    )

    # get random rotation offset for slice extraction
    random_rotation = torch.tensor(
        R.random().as_matrix(),
        dtype=torch.float32
    )

    # extract tilt dfts
    tilt_dfts = extract_central_slices_rfft_3d(
        volume_rfft=volume_dft,
        image_shape=volume.shape,
        rotation_matrices=random_rotation @ tilt_rotation_matrices,
        fftfreq_max=0.5,  # ~2x less coords to rotate
    )

    # phase shift to apply translations
    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_dfts,
        image_shape=volume.shape[-2:],
        shifts=tilt_image_shifts,
        rfft=True,
        fftshifted=True
    )

    ground_truth_reconstruction = reconstruct(
        tilt_dfts,
        torch.zeros_like(tilt_image_shifts),
        tilt_rotation_matrices,
        volume.shape[-2:],
        volume.shape[-3:],
    )
    misaligned_reconstruction = reconstruct(
        tilt_dfts_shifted,
        torch.zeros_like(tilt_image_shifts),
        tilt_rotation_matrices,
        volume.shape[-2:],
        volume.shape[-3:],
    )

    return (
        tilt_dfts_shifted,
        ground_truth_reconstruction,
        misaligned_reconstruction
    )


def reconstruct(
        tilt_image_dfts: torch.Tensor,
        predicted_shifts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        image_shape: tuple[int, int],
        volume_shape: tuple[int, int, int],
) -> torch.Tensor:

    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_image_dfts,
        image_shape=image_shape,
        shifts=predicted_shifts,
        rfft=True,
        fftshifted=True
    )

    # Rest of computation...
    volume_dft, weights = insert_central_slices_rfft_3d(
        image_rfft=tilt_dfts_shifted,
        volume_shape=volume_shape,
        rotation_matrices=tilt_rotation_matrices,
        fftfreq_max=0.5
    )

    valid_weights = weights > 1e-3
    volume_dft[valid_weights] /= weights[valid_weights]

    volume_dft = torch.fft.ifftshift(volume_dft, dim=(-3, -2))
    volume = torch.fft.irfftn(volume_dft, dim=(-3, -2, -1))
    volume = torch.fft.ifftshift(volume, dim=(-3, -2, -1))
    volume = torch.real(volume).to(torch.float32)
    return volume


def optimize_shifts(
        model: MissAlignment,
        tilt_image_dfts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find shifts to optimize model score.

    Parameters
    ----------
    model: torch.nn.Module
        Model that produces the optimization target.
    tilt_image_dfts: torch.Tensor
        DFTs of tilt images to use for reconstruction.
    tilt_rotation_matrices: torch.Tensor:
        Tilt rotation matrices to use for back projection.
    tolerance: float

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Reconstruction before optimization.
        - Reconstruction after optimization
        - Detected translational alignment.
    """
    n_tilts = tilt_rotation_matrices.shape[0]
    box_size = tilt_image_dfts.shape[-2]
    image_shape = (box_size, ) * 2
    volume_shape = (box_size, ) * 3
    device = tilt_image_dfts.device

    predicted_shifts = torch.zeros(
        size=(n_tilts, 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    alignment_optimizer = torch.optim.LBFGS(
        [predicted_shifts,],
        line_search_fn="strong_wolfe",
        # max_iter=10
    )

    def closure():
        alignment_optimizer.zero_grad()
        volume = reconstruct(
            tilt_image_dfts=tilt_image_dfts,
            predicted_shifts=predicted_shifts,
            tilt_rotation_matrices=tilt_rotation_matrices,
            image_shape=image_shape,
            volume_shape=volume_shape,
        )
        volume = torch.real(volume).to(torch.float32)
        volume = einops.rearrange(volume, 'd h w -> 1 1 d h w')

        # Get loss and compute backward pass
        loss = model(volume)
        loss.backward()

        print(loss.item())
        # loss_graph += [loss.item()]
        return loss

    for i in range(5):
        print(f"Iteration {i}")
        alignment_optimizer.step(closure)

    predicted_shifts = predicted_shifts.detach()
    volume = reconstruct(
        tilt_image_dfts=tilt_image_dfts,
        predicted_shifts=predicted_shifts,
        tilt_rotation_matrices=tilt_rotation_matrices,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )

    return volume, predicted_shifts


@cli.command(name="optimize_alignment", no_args_is_help=True)
def optimize_alignment(
        model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
) -> None:
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint,
        map_location="cpu"
    )
    model.eval()

    dataset = EMDBDataset(test_data_directory)
    test_data = dataset.prepare_test_boxes()

    for x in test_data:
        tilt_image_dfts, ground_truth, misaligned = prep_tilts(
            volume=x["volume"],
            tilt_rotation_matrices=x["rotations"],
            tilt_image_shifts=x["translations"],
        )
        final, shifts = optimize_shifts(
            model=model,  #.to("cuda:0"),
            tilt_image_dfts=tilt_image_dfts,  # .to("cuda:0"),
            tilt_rotation_matrices=x["rotations"],  #.to("cuda:0"),
        )

        name = x["map_name"]

        print(name)
        print(torch.abs(x["translations"] + shifts.cpu()).sum())

        np.save(
            output_directory / f"{name}_ground_truth.npy",
            ground_truth.cpu().numpy()
        )
        np.save(
            output_directory / f"{name}_misaligned.npy",
            misaligned.cpu().numpy()
        )
        np.save(
            output_directory / f"{name}_final.npy",
            final.cpu().numpy()
        )

    return None