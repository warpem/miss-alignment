import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli
import tqdm
import random

import matplotlib.pyplot as plt
import torch
import einops
import numpy as np
from pytorch_lightning import seed_everything
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d, insert_central_slices_rfft_3d
)
from torch_fourier_shift import fourier_shift_dft_2d,  fourier_shift_dft_3d
from torch_grid_utils import fftfreq_grid, coordinate_grid

from miss_alignment.data import EMDBDataset
from miss_alignment.models import MissAlignment
from miss_alignment.data.augmentation import (
    generate_aligned_and_misaligned_shifts
)


def prep_tilts(
        volume_dft: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        tilt_image_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = volume_dft.shape[-3]
    shape = (size, ) * 3
    # random shift not needed for evaluation
    random_shift = torch.tensor(
        np.random.normal(
            loc=0, scale=.1 * size, size=(3,)
        ),
        dtype=torch.float32
    )
    volume_dft = fourier_shift_dft_3d(
        dft=volume_dft,
        image_shape=shape,
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
        image_shape=shape,
        rotation_matrices=random_rotation @ tilt_rotation_matrices,
        fftfreq_max=0.5,  # ~2x less coords to rotate
    )

    # phase shift to apply translations
    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_dfts,
        image_shape=shape[-2:],
        shifts=tilt_image_shifts,
        rfft=True,
        fftshifted=True
    )

    ground_truth_reconstruction = reconstruct(
        tilt_dfts,
        torch.zeros_like(tilt_image_shifts),
        tilt_rotation_matrices,
        shape[-2:],
        shape[-3:],
    )
    misaligned_reconstruction = reconstruct(
        tilt_dfts_shifted,
        torch.zeros_like(tilt_image_shifts),
        tilt_rotation_matrices,
        shape[-2:],
        shape[-3:],
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

    grid = fftfreq_grid(
        image_shape=volume.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device,
    )
    volume = volume / torch.sinc(grid) ** 2

    volume = torch.real(volume).to(torch.float32)
    return volume


def center_of_mass(volume: torch.Tensor) -> torch.Tensor:
    """Calculate the center of mass of a 3D tensor.

    Parameters
    ----------
    volume : torch.Tensor
        Input 3D tensor representing mass distribution

    Returns
    -------
    torch.Tensor
        Center of mass coordinates [z, y, x]
    """
    device = volume.device
    volume = volume ** 2
    grid = coordinate_grid(volume.shape[-3:], device=device)
    grid = einops.rearrange(grid, "d h w zyx -> zyx d h w")
    mass = torch.sum(volume, dim=(-3, -2, -1))
    center_of_mass = torch.sum(grid * volume, dim=(-3, -2, -1)) / mass
    return center_of_mass


def optimize_shifts(
        model: MissAlignment,
        tilt_image_dfts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        gt_com: torch.Tensor,
        start_shifts: torch.Tensor | None = None,
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

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Reconstruction before optimization.
        - Reconstruction after optimization
        - Detected translational alignment.
    """
    n_tilts = tilt_rotation_matrices.shape[0]
    box_size = tilt_image_dfts[0].shape[-2]
    image_shape = (box_size, ) * 2
    volume_shape = (box_size, ) * 3
    device = tilt_image_dfts[0].device

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
        volumes = []
        for t in tilt_image_dfts:
            volume = reconstruct(
                tilt_image_dfts=t,
                predicted_shifts=predicted_shifts,
                tilt_rotation_matrices=tilt_rotation_matrices,
                image_shape=image_shape,
                volume_shape=volume_shape,
            )
            volume = torch.real(volume).to(torch.float32)
            volumes.append(volume)

        volumes = torch.stack(volumes)
        volumes = einops.rearrange(volumes, 'b d h w -> b 1 d h w')

        new_com = center_of_mass(volumes)
        distance = torch.sum((new_com - gt_com) ** 2)
        # print(distance)

        # Get loss and compute backward pass
        loss = model(volumes) + distance
        loss = loss.mean()
        loss.backward()

        # print(loss.item())
        # loss_graph += [loss.item()]
        return loss

    for _ in tqdm.tqdm(range(10)):
        alignment_optimizer.step(closure)

    predicted_shifts = predicted_shifts.detach()  # center the shifts
    predicted_shifts = predicted_shifts - predicted_shifts.mean(axis=0)
    # volume = reconstruct(
    #     tilt_image_dfts=tilt_image_dfts,
    #     predicted_shifts=predicted_shifts,
    #     tilt_rotation_matrices=tilt_rotation_matrices,
    #     image_shape=image_shape,
    #     volume_shape=volume_shape,
    # )

    return predicted_shifts


@cli.command(name="optimize_alignment", no_args_is_help=True)
def optimize_alignment(
        model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        nboxes: int = 1,
        seed: int = 45132,
) -> None:
    seed_everything(seed, workers=True)
    torch.set_printoptions(precision=2, sci_mode=False)

    # get model
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint,
        map_location="cpu"
    )
    model.eval()

    # initialize the dataset
    dataset = EMDBDataset(test_data_directory)
    dataset_size = len(dataset)

    # tilt angles used for forward and back projection
    tilt_angles = R.from_euler(
        seq="Y", angles=np.arange(-51, 54, 3), degrees=True
    )
    rotations = torch.tensor(tilt_angles.as_matrix()).float()

    # for storing results over all examples
    x1, x2 = [], []

    for i in range(50):
        # between 0 and misaligned_std
        _, misaligned_translations = (
            generate_aligned_and_misaligned_shifts(
                rotations.shape[0],
                dataset.target_size[0] * .25,  # 1/4 max shift of image size
                outlier_probability=0.
            )
        )
        misalignment = (
                misaligned_translations - misaligned_translations.mean(axis=0)
        )
        tilts, coms = [], []
        for idx in [random.randint(0, dataset_size - 1) for _ in range(
                nboxes)]:
            tilt, gt, _ = prep_tilts(
                dataset.volumes[idx][1],
                rotations,
                misalignment,
            )
            tilts += [tilt]
            coms += [center_of_mass(gt)]
        coms = torch.stack(coms)

        shifts = optimize_shifts(
            model=model.to("cuda:0"),
            tilt_image_dfts=[x.to("cuda:0") for x in tilts],
            tilt_rotation_matrices=rotations.to("cuda:0"),
            gt_com=coms.to("cuda:0"),
        ).to("cpu")

        # print("center of mass distance:", torch.sqrt(torch.sum(
        #     (ground_truth_com - center_of_mass(final ** 2)) ** 2
        # )))
        print("sum of misalignment:",torch.sum(torch.abs(misalignment), dim=0))
        print("sum of alignment:",torch.sum(torch.abs(misalignment + shifts),
                                       dim=0))

        x1.append(torch.sum(torch.abs(misalignment), dim=0).tolist())
        x2.append(torch.sum(torch.abs(misalignment + shifts),
                                       dim=0).tolist())

        diff = misalignment + shifts

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                               figsize=(8, 4))
        ax[0].plot(misalignment[:, 0], label="misaligned")
        ax[1].plot(misalignment[:, 1], label="misaligned")
        ax[0].plot(-shifts[:, 0], label="correction")
        ax[1].plot(-shifts[:, 1], label="correction")
        ax[0].plot(diff[:, 0], label="diff")
        ax[1].plot(diff[:, 1], label="diff")
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f"y shifts (abs. diff. "
                        f"{torch.sum(torch.abs(diff[:, 0])):.2f})")
        ax[1].set_title(f"x shifts (abs. diff. "
                        f"{torch.sum(torch.abs(diff[:, 1])):.2f})")
        ax[0].set_xlabel("tilt image index")
        ax[1].set_xlabel("tilt image index")
        ax[0].set_ylim(-6, 6)
        ax[1].set_ylim(-6, 6)
        plt.savefig(output_directory / f"{i}_graph.png",
                    bbox_inches="tight", dpi=300)

    ids = list(range(len(x1)))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))
    ax[0].set_title("total y shift")
    ax[0].plot(ids, [a[0] for a in x1], "o", alpha=.8, label="misaligned")
    ax[0].plot(ids, [a[0] for a in x2], "o", alpha=.8, label="aligned")
    ax[1].set_title("total x shift")
    ax[1].plot(ids, [a[1] for a in x1], "o", alpha=.8, label="misaligned")
    ax[1].plot(ids, [a[1] for a in x2], "o", alpha=.8, label="aligned")
    ax[1].legend()
    ax[0].set_xlabel("example id")
    ax[1].set_xlabel("example id")
    ax[0].set_ylabel("sum of shifts")
    ax[0].set_ylim(0, 100)
    plt.savefig(output_directory / f"corrections_graph.png",
                bbox_inches="tight", dpi=300)

    return None