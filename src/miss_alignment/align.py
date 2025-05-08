import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli
import tqdm
import random
import json

import matplotlib.pyplot as plt
import torch
import einops
import numpy as np
from pytorch_lightning import seed_everything
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return (
        tilt_dfts,
        tilt_dfts_shifted,
        random_shift,
        random_rotation
    )


def batch_reconstruct(
        tilt_image_dfts: torch.Tensor,
        predicted_shifts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        image_shape: tuple[int, int],
        volume_shape: tuple[int, int, int],
) -> torch.Tensor:

    predicted_shifts = einops.rearrange(
        predicted_shifts, "n yx -> n 1 yx"
    )
    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_image_dfts,
        image_shape=image_shape,
        shifts=predicted_shifts,
        rfft=True,
        fftshifted=True
    )

    # Rest of computation...
    volume_dft, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=tilt_dfts_shifted,
        volume_shape=volume_shape,
        rotation_matrices=tilt_rotation_matrices,
        fftfreq_max=0.5
    )

    valid_weights = weights > 1e-3
    volume_dft[..., valid_weights] /= weights[..., valid_weights]

    volume_dft = torch.fft.ifftshift(volume_dft, dim=(-3, -2))
    volume = torch.fft.irfftn(volume_dft, dim=(-3, -2, -1))
    volume = torch.fft.ifftshift(volume, dim=(-3, -2, -1))

    grid = fftfreq_grid(
        image_shape=volume.shape[-3:],
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
    volume = einops.rearrange(volume, "... d h w -> ... d h w 1")
    mass = torch.sum(volume, dim=(-4, -3, -2))
    center_of_mass = torch.sum(grid * volume, dim=(-4, -3, -2)) / mass
    return center_of_mass


def optimize_shifts(
        model: MissAlignment,
        tilt_image_dfts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        gt_com: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list]:
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
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()
        volumes = batch_reconstruct(
            tilt_image_dfts,
            predicted_shifts,
            tilt_rotation_matrices,
            image_shape,
            volume_shape,
        )

        new_com = center_of_mass(volumes)
        distance = torch.sum((new_com - gt_com) ** 2)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, 'c d h w -> c 1 d h w')

        # Get loss and compute backward pass
        loss = model(volumes) + distance
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        return loss

    for _ in tqdm.tqdm(range(10)):
        alignment_optimizer.step(closure)

    predicted_shifts = predicted_shifts.detach()  # center the shifts
    predicted_shifts = predicted_shifts - predicted_shifts.mean(axis=0)
    volumes = batch_reconstruct(
        tilt_image_dfts=tilt_image_dfts,
        predicted_shifts=predicted_shifts,
        tilt_rotation_matrices=tilt_rotation_matrices,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )

    return volumes, predicted_shifts, loss_values


@cli.command(name="optimize_alignment", no_args_is_help=True)
def optimize_alignment(
        model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
        nboxes: int = 1,
        seed: int = 45132,
        device: str = typer.Option("cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu')"),
        iterations: int = typer.Option(50, help="Number of iterations for optimization alignment"),
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
    """
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

    # Create a data structure to store all the required information
    optimization_report = {
        "iterations": []
    }

    for i in range(iterations):
        # between 0 and misaligned_std
        _, misaligned_translations = (
            generate_aligned_and_misaligned_shifts(
                rotations.shape[0],
                dataset.target_size[0] * .25,  # 1/4 max shift of image size
                # outlier_probability=0.
            )
        )
        misalignment = (
                misaligned_translations - misaligned_translations.mean(axis=0)
        )

        # Store map data for the JSON report
        map_data, map_names, ground_truth_tilts, misaligned_tilts = (
            [], [], [], []
        )

        # Store map names and indices for the current iteration
        map_indices = [
            random.randint(0, dataset_size - 1) for _ in range(nboxes)
        ]

        for idx in map_indices:
            map_name = dataset.volumes[idx][0]  # Get the map name
            map_names.append(map_name)

            ground_truth, misaligned, random_shift, random_rotation = (
                prep_tilts(
                dataset.volumes[idx][1],
                rotations,
                misalignment,
            ))
            misaligned_tilts += [misaligned]
            ground_truth_tilts += [ground_truth]

            # Store map data for the JSON report
            map_data.append({
                "map_name": map_name,
                "random_shift": random_shift.tolist(),
                "random_rotation": random_rotation.tolist()
            })

        misaligned_tilts = torch.stack(misaligned_tilts)
        misaligned_tilts = einops.rearrange(
            misaligned_tilts, "b n h w -> n b h w"
        )
        misaligned_reconstruction = batch_reconstruct(
            misaligned_tilts,
            torch.zeros((len(tilt_angles), 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )
        ground_truth_tilts = torch.stack(ground_truth_tilts)
        ground_truth_tilts = einops.rearrange(
            ground_truth_tilts, "b n h w -> n b h w"
        )
        ground_truth_reconstruction = batch_reconstruct(
            ground_truth_tilts,
            torch.zeros((len(tilt_angles), 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )

        np.save(output_directory / f"{i}_ground_truth.npy",
                ground_truth_reconstruction.numpy())

        coms = center_of_mass(ground_truth_reconstruction)

        # Call optimize_shifts and get both shifts and loss values
        volumes, shifts, loss_values = optimize_shifts(
            model=model.to(device),  # Use the user-specified device
            tilt_image_dfts=misaligned_tilts.to(device),  # Use the user-specified device
            tilt_rotation_matrices=rotations.to(device),  # Use the user-specified device
            gt_com=coms.to(device),  # Use the user-specified device
        )
        shifts = shifts.cpu()
        volumes = volumes.cpu()

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

        # Store all the required information for this iteration
        iteration_data = {
            "iteration": i,
            "map_indices": map_indices,
            "maps": map_data,  # Include map data with random shifts and rotations
            "loss_values": loss_values,
            "misaligned_shifts": misalignment.tolist(),
            "final_shifts": shifts.tolist(),
            "sum_misalignment": torch.sum(torch.abs(misalignment), dim=0).tolist(),
            "sum_alignment": torch.sum(torch.abs(misalignment + shifts), dim=0).tolist()
        }
        optimization_report["iterations"].append(iteration_data)

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
        plt.close()
        np.save(output_directory / f"{i}_volumes.npy", volumes.numpy())

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

    # Save the optimization report as a JSON file
    with open(output_directory / "optimization_report.json", "w") as f:
        json.dump(optimization_report, f, indent=2)

    return None
