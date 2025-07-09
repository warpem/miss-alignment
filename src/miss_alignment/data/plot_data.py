import os
import typer
from pathlib import Path
from .._cli import OPTION_PROMPT_KWARGS, cli

import einops
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from miss_alignment.data import MissAlignmentDataModule
from miss_alignment.models import MissAlignment


def plot_volume_slices(
    volume_dict: Dict[str, torch.Tensor],
    save_path: str,
    idx: int,
    scores: tuple[str],
) -> None:
    """
    Plot center slices of volume data in three planes (xy, xz, yz).

    Parameters
    ----------
    volume_dict : Dict[str, torch.Tensor]
        Dictionary with keys 'volume', 'aligned', 'misaligned' containing 4D tensors (C,D,H,W)
    save_path : str
        Directory to save the plot
    idx : int
        Index of the volume in the dataset
    """
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row labels
    row_labels = list(volume_dict.keys())

    # Column labels
    col_labels = ["XY Slice", "XZ Slice", "YZ Slice"]

    # Get the volumes from dictionary and convert to numpy if needed
    volumes = {
        key: vol.cpu().numpy() if isinstance(vol, torch.Tensor) else vol
        for key, vol in volume_dict.items()
        if key in row_labels
    }

    # Calculate center indices for each dimension
    d, h, w = volumes[row_labels[0]].shape
    center_d = d // 2
    center_h = h // 2
    center_w = w // 2

    # List of volumes to plot in order
    volume_keys = row_labels

    # Plot each volume
    for i, key in enumerate(volume_keys):
        vol = volumes[key]

        # XY Slice (center of depth)
        im = axes[i, 0].imshow(np.mean(vol, axis=0), cmap="gray", vmin=-2, vmax=2)
        axes[i, 0].set_title(f"{row_labels[i]} - {col_labels[0]} | " + scores[i])
        axes[i, 0].axis("off")

        # XZ Slice (center of height)
        im = axes[i, 1].imshow(np.mean(vol, axis=1), cmap="gray", vmin=-2, vmax=2)
        axes[i, 1].set_title(f"{row_labels[i]} - {col_labels[1]}")
        axes[i, 1].axis("off")

        # YZ Slice (center of width)
        im = axes[i, 2].imshow(np.mean(vol, axis=2), cmap="gray", vmin=-2, vmax=2)
        axes[i, 2].set_title(f"{row_labels[i]} - {col_labels[2]}")
        axes[i, 2].axis("off")

        # Add colorbar to the right of each row
        fig.colorbar(im, ax=axes[i, :], shrink=0.8)

    # plt.subplots_adjust(top=0.9)

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the figure
    plt.savefig(
        os.path.join(save_path, f"volume_{idx:04d}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_dataset_samples(
    datamodule,
    num_samples: int = 3,
    output_dir: Path = "output_plots",
    model: MissAlignment | None = None,
) -> None:
    """
    Plot samples from the training and validation datasets.

    Parameters
    ----------
    datamodule : MRCDataModule
        The datamodule containing the datasets
    num_samples : int
        Number of samples to plot from each dataset
    output_dir : str
        Directory to save the plots
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("Plotting training samples...")
    train_loader = datamodule.train_dataloader()
    for i, batch in enumerate(train_loader):
        if i >= num_samples:
            break

        # Get one sample from the batch
        x1, x2, x3, t = batch
        t = t.squeeze().tolist()
        sample = {
            "x1 : " + str(t[0]): einops.rearrange(x1, "1 1 d h w -> d h w"),
            "x2 : " + str(t[1]): einops.rearrange(x2, "1 1 d h w -> d h w"),
            "x3 : " + str(t[2]): einops.rearrange(x3, "1 1 d h w -> d h w"),
        }
        if model is not None:
            with torch.no_grad():
                scores = (
                    f"{model(x1).item(): .3f}",
                    f"{model(x2).item(): .3f}",
                    f"{model(x3).item(): .3f}",
                )
        else:
            scores = ("N/A",) * 3

        # Plot and save
        plot_volume_slices(sample, train_dir, i, scores)
        print(f"  Saved training sample {i + 1}/{num_samples}")

    print("Plotting validation samples...")
    val_loader = datamodule.val_dataloader()
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        # Get one sample from the batch
        x1, x2, x3, t = batch
        t = t.squeeze().tolist()
        sample = {
            "x1 : " + str(t[0]): einops.rearrange(x1, "1 1 d h w -> d h w"),
            "x2 : " + str(t[1]): einops.rearrange(x2, "1 1 d h w -> d h w"),
            "x3 : " + str(t[2]): einops.rearrange(x3, "1 1 d h w -> d h w"),
        }
        if model is not None:
            with torch.no_grad():
                scores = (
                    f"{model(x1).item(): .3f}",
                    f"{model(x2).item(): .3f}",
                    f"{model(x3).item(): .3f}",
                )
        else:
            scores = ("N/A",) * 3

        # Plot and save
        plot_volume_slices(sample, val_dir, i, scores)
        print(f"  Saved validation sample {i + 1}/{num_samples}")


@cli.command(name="plot_dataset", no_args_is_help=True)
def plot_dataset(
    dataset_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    dataset_type: str = "SHREC",
    patch_size: int = 64,
    model_checkpoint: Path | None = None,
) -> None:
    """Plot the boxes generated by the EMDBDataset class."""
    # Initialize the data module
    data_module = MissAlignmentDataModule(
        dataset_directory=dataset_directory,
        dataset_type=dataset_type,
        batch_size=1,  # Use batch size of 1 for visualization
        target_size=patch_size,
        train_val_split=(0.8, 0.2),  # No test set for this example
        num_workers=0,  # Use at least 1 for reproducibility
    )

    model = None
    if model_checkpoint is not None:
        # get model
        model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
        model.eval()

    # Set up the data module
    # data_module.prepare_data()
    data_module.setup(stage="fit")

    # Plot samples from both train and validation sets
    plot_dataset_samples(
        data_module,
        num_samples=10,
        output_dir=output_directory,
        model=model,
    )

    print(f"Plots saved to {output_directory}")
    return None
