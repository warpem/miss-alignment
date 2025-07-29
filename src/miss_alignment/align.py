import typer
from pathlib import Path
from ._cli import OPTION_PROMPT_KWARGS, cli
import random
import json

import matplotlib.pyplot as plt
import torch
import einops
import numpy as np
from pytorch_lightning import seed_everything
from scipy.spatial.transform import Rotation as R
from torch_tiltxcorr import tiltxcorr_no_stretch

from miss_alignment.data import EMDBDataset
from miss_alignment.models import MissAlignment
from miss_alignment.data.shift_generation import (
    generate_shifts,
    project_shifts_3d_to_2d,
)
from miss_alignment.align_backend import (
    prep_tilts,
    batch_reconstruct,
    center_of_mass,
    optimize_shifts,
)


@cli.command(name="optimize_alignment", no_args_is_help=True)
def optimize_alignment(
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
    model.freeze()

    # initialize the dataset
    dataset = EMDBDataset(test_data_directory, train=False)
    dataset_size = len(dataset)

    # tilt angles used for forward and back projection
    tilt_angles_raw = np.arange(-51, 54, 3)
    transition_idx = np.where(tilt_angles_raw >= 0)[0][0]
    tilt_angles = R.from_euler(seq="Y", angles=tilt_angles_raw, degrees=True)
    rotations = torch.tensor(tilt_angles.as_matrix()).float()

    # for storing results over all examples
    x1, x2, x3 = [], [], []

    # Create a data structure to store all the required information
    optimization_report = {"iterations": []}

    # Create a structure to store all misalignments for saving to disk
    all_misalignments = {"misalignments": []}

    # Load misalignments from file if provided
    loaded_misalignments = None
    if misalignments_file is not None and misalignments_file.exists():
        try:
            with open(misalignments_file, "r") as f:
                loaded_misalignments = json.load(f)
                print(f"Loaded misalignments from {misalignments_file}")
        except Exception as e:
            print(f"Error loading misalignments file: {e}")
            loaded_misalignments = None

    for i in range(iterations):
        # Use loaded misalignments if available, otherwise generate new ones
        if loaded_misalignments is not None and i < len(
            loaded_misalignments["misalignments"]
        ):
            # Convert the loaded misalignment from list to tensor
            misalignment = torch.tensor(loaded_misalignments["misalignments"][i])
            print(f"Using loaded misalignment for iteration {i}")
        else:
            # Generate new misalignment
            # between 0 and misaligned_std
            misaligned_translations = generate_shifts(
                rotations.shape[0],
                dataset.target_size[0] * 0.25,  # 1/4 max shift of image size
                # outlier_probability=0.0,
            )
            misalignment = project_shifts_3d_to_2d(misaligned_translations, rotations)
            misalignment = (
                misalignment - misalignment[transition_idx : transition_idx + 1]
            )

            # Store the generated misalignment for saving to disk
            all_misalignments["misalignments"].append(misalignment.tolist())

        # Store map data for the JSON report
        map_data, map_names, ground_truth_tilts, misaligned_tilts = ([], [], [], [])

        # Store map names and indices for the current iteration
        map_indices = [random.randint(0, dataset_size - 1) for _ in range(nboxes)]

        xcorr_shifts = torch.zeros((len(tilt_angles_raw), 2))

        for idx in map_indices:
            map_name = dataset.volumes[idx][0]  # Get the map name
            map_names.append(map_name)

            ground_truth, misaligned, random_shift, random_rotation = prep_tilts(
                dataset.volumes[idx][1],
                rotations,
                misalignment,
            )
            misaligned_tilts += [misaligned]
            ground_truth_tilts += [ground_truth]

            # Store map data for the JSON report
            map_data.append(
                {
                    "map_name": map_name,
                    "random_shift": random_shift.tolist(),
                    "random_rotation": random_rotation.tolist(),
                }
            )

            tilts = torch.fft.ifftshift(misaligned, dim=(-2))
            tilts = torch.fft.irfftn(tilts, dim=(-2, -1))
            tilts = torch.fft.ifftshift(tilts, dim=(-2, -1))
            shifts = tiltxcorr_no_stretch(tilts, tilt_angles_raw, low_pass_cutoff=0.5)
            xcorr_shifts = xcorr_shifts + shifts  # (shifts - shifts.mean(
            # axis=0))

        xcorr_shifts = xcorr_shifts / nboxes

        misaligned_tilts = torch.stack(misaligned_tilts)
        misaligned_tilts = einops.rearrange(misaligned_tilts, "b n h w -> n b h w")
        misaligned_reconstruction = batch_reconstruct(
            misaligned_tilts,
            torch.zeros((len(tilt_angles), 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )
        ground_truth_tilts = torch.stack(ground_truth_tilts)
        ground_truth_tilts = einops.rearrange(ground_truth_tilts, "b n h w -> n b h w")
        ground_truth_reconstruction = batch_reconstruct(
            ground_truth_tilts,
            torch.zeros((len(tilt_angles), 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )

        np.save(
            output_directory / f"{i}_ground_truth.npy",
            ground_truth_reconstruction.numpy(),
        )

        coms = center_of_mass(ground_truth_reconstruction)

        # Call optimize_shifts and get both shifts and loss values
        volumes, shifts, loss_values = optimize_shifts(
            model=model.to(device),  # Use the user-specified device
            tilt_image_dfts=misaligned_tilts.to(
                device
            ),  # Use the user-specified device
            tilt_rotation_matrices=rotations.to(
                device
            ),  # Use the user-specified device
            gt_com=coms.to(device),  # Use the user-specified device
            initial_shifts=xcorr_shifts.to(device),
        )
        shifts = shifts.cpu()
        volumes = volumes.cpu()

        # print("center of mass distance:", torch.sqrt(torch.sum(
        #     (ground_truth_com - center_of_mass(final ** 2)) ** 2
        # )))
        print("sum of misalignment:", torch.sum(torch.abs(misalignment), dim=0))
        print("sum of alignment:", torch.sum(torch.abs(misalignment + shifts), dim=0))
        xcorr_diff = misalignment + xcorr_shifts
        xcorr_diff_sum = torch.sum(torch.abs(xcorr_diff), dim=0)
        print(
            "sum of xcorr alignment:",
            xcorr_diff_sum,
        )

        x1.append(torch.sum(torch.abs(misalignment), dim=0).tolist())
        x2.append(torch.sum(torch.abs(misalignment + shifts), dim=0).tolist())
        x3.append(xcorr_diff_sum.tolist())

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
            "sum_alignment": torch.sum(
                torch.abs(misalignment + shifts), dim=0
            ).tolist(),
        }
        optimization_report["iterations"].append(iteration_data)

        fig, ax = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4)
        )
        ax[0].plot(misalignment[:, 0], label="misalignment")
        ax[1].plot(misalignment[:, 1], label="misalignment")
        # ax[0].plot(-shifts[:, 0], label="correction")
        # ax[1].plot(-shifts[:, 1], label="correction")
        ax[0].plot(xcorr_diff[:, 0], label="xcorr")
        ax[1].plot(xcorr_diff[:, 1], label="xcorr")
        ax[0].plot(diff[:, 0], label="MissAlignment")
        ax[1].plot(diff[:, 1], label="MissAlignment")
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f"y shifts (abs. diff. {torch.sum(torch.abs(diff[:, 0])):.2f})")
        ax[1].set_title(f"x shifts (abs. diff. {torch.sum(torch.abs(diff[:, 1])):.2f})")
        ax[0].set_xlabel("tilt image index")
        ax[1].set_xlabel("tilt image index")
        ax[0].set_ylim(-6, 6)
        ax[1].set_ylim(-6, 6)
        plt.savefig(output_directory / f"{i}_graph.png", bbox_inches="tight", dpi=300)
        plt.close()
        np.save(output_directory / f"{i}_volumes.npy", volumes.numpy())

    ids = list(range(len(x1)))
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    ax[0].set_title("total y shift")
    ax[0].plot(ids, [a[0] for a in x1], "o", alpha=0.8, label="misaligned")
    ax[0].plot(ids, [a[0] for a in x2], "o", alpha=0.8, label="aligned")
    ax[0].plot(ids, [a[0] for a in x3], "o", alpha=0.8, label="xcorr")
    ax[1].set_title("total x shift")
    ax[1].plot(ids, [a[1] for a in x1], "o", alpha=0.8, label="misaligned")
    ax[1].plot(ids, [a[1] for a in x2], "o", alpha=0.8, label="aligned")
    ax[1].plot(ids, [a[1] for a in x3], "o", alpha=0.8, label="xcorr")
    ax[1].legend()
    ax[0].set_xlabel("example id")
    ax[1].set_xlabel("example id")
    ax[0].set_ylabel("sum of shifts")
    ax[0].set_ylim(0, 100)
    plt.savefig(
        output_directory / "corrections_graph.png", bbox_inches="tight", dpi=300
    )

    # Save the optimization report as a JSON file
    with open(output_directory / "optimization_report.json", "w") as f:
        json.dump(optimization_report, f, indent=2)

    # Save the generated misalignments to a JSON file
    if all_misalignments["misalignments"]:
        with open(output_directory / "misalignments.json", "w") as f:
            json.dump(all_misalignments, f, indent=2)
        print(
            f"Saved generated misalignments to {output_directory / 'misalignments.json'}"
        )

    return None
