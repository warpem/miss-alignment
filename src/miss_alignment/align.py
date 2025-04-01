import einops
import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli

import torch
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
) -> torch.Tensor:
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
    random_rotation = torch.tensor(R.random().as_matrix(), dtype=torch.float32)

    # extract tilt dfts
    tilt_dfts = extract_central_slices_rfft_3d(
        volume_rfft=volume_dft,
        image_shape=volume.shape,
        rotation_matrices=tilt_rotation_matrices @ random_rotation,
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

    return tilt_dfts_shifted


def optimize_shifts(
        model: MissAlignment,
        tilt_image_dfts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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
        size=(n_tilts, 2), dtype=torch.float32, requires_grad=True, device=device
    )

    alignment_optimizer = torch.optim.SGD(
        params=[predicted_shifts, ],
        lr=0.1,
        momentum=0.9,
    )
    volume = None
    initial_volume = None
    prev_loss = None

    for i in range(10000):
        # Need to detach and reattach with each iteration
        predicted_shifts = predicted_shifts.detach().clone()
        predicted_shifts.requires_grad = True

        # Update the optimizer parameters without recreating the optimizer
        alignment_optimizer.param_groups[0]['params'] = [predicted_shifts]

        alignment_optimizer.zero_grad()

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
        volume = torch.real(
            torch.fft.ifftshift(volume, dim=(-3, -2, -1))
        ).to(torch.float32)
        volume = einops.rearrange(volume, 'd h w -> 1 1 d h w')

        if initial_volume is None:  # store the initial, misaligned
            # reconstruction
            initial_volume = volume.clone().detach()

        # Get loss and compute backward pass
        loss = model(volume)
        loss.backward()

        alignment_optimizer.step()

        if prev_loss is not None and abs(loss.item() - prev_loss) < tolerance:
            break

        prev_loss = loss.item()

        if i % 100 == 0:
            print(loss.item())

    return initial_volume, volume.detach(), predicted_shifts.detach()


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
        tilt_image_dfts = prep_tilts(
            volume=x["volume"],
            tilt_rotation_matrices=x["rotations"],
            tilt_image_shifts=x["translations"],
        )
        initial, final, shifts = optimize_shifts(
            model=model.to("cuda:0"),
            tilt_image_dfts=tilt_image_dfts.to("cuda:0"),
            tilt_rotation_matrices=x["rotations"].to("cuda:0"),
        )

        print(x["translations"])
        print(- shifts.cpu())

        np.save(output_directory / "start.npy", initial.cpu().numpy())
        np.save(output_directory / "end.npy", final.cpu().numpy())

        break

    return None