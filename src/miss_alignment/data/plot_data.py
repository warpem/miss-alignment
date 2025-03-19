import os
import torch
import matplotlib.pyplot as plt
from typing import Dict

# Import your classes
from training_datamodule import EMDBDataModule


def plot_volume_slices(
    volume_dict: Dict[str, torch.Tensor], save_path: str, idx: int, file_path: str
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
    file_path : str
        Original file path of the MRC file
    """
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Set plot title
    filename = os.path.basename(file_path)
    fig.suptitle(f"Volume {idx}: {filename}", fontsize=16)

    # Row labels
    row_labels = ["Original", "Aligned", "Misaligned"]

    # Column labels
    col_labels = ["XY Slice", "XZ Slice", "YZ Slice"]

    # Get the volumes from dictionary and convert to numpy if needed
    volumes = {
        key: vol.cpu().numpy()[0] if isinstance(vol, torch.Tensor) else vol[0]
        for key, vol in volume_dict.items()
        if key in ["volume", "aligned", "misaligned"]
    }

    # Calculate center indices for each dimension
    d, h, w = volumes["volume"].shape
    center_d = d // 2
    center_h = h // 2
    center_w = w // 2

    # List of volumes to plot in order
    volume_keys = ["volume", "aligned", "misaligned"]

    # Plot each volume
    for i, key in enumerate(volume_keys):
        vol = volumes[key]

        # XY Slice (center of depth)
        im = axes[i, 0].imshow(vol[center_d, :, :], cmap="viridis")
        axes[i, 0].set_title(f"{row_labels[i]} - {col_labels[0]}")
        axes[i, 0].axis("off")

        # XZ Slice (center of height)
        im = axes[i, 1].imshow(vol[:, center_h, :], cmap="viridis")
        axes[i, 1].set_title(f"{row_labels[i]} - {col_labels[1]}")
        axes[i, 1].axis("off")

        # YZ Slice (center of width)
        im = axes[i, 2].imshow(vol[:, :, center_w], cmap="viridis")
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
    datamodule, num_samples: int = 3, output_dir: str = "output_plots"
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
        sample = {
            "volume": batch["volume"][0],
            "aligned": batch["aligned"][0],
            "misaligned": batch["misaligned"][0],
        }
        file_path = batch["file_path"][0]

        # Plot and save
        plot_volume_slices(sample, train_dir, i, file_path)
        print(f"  Saved training sample {i + 1}/{num_samples}")

    print("Plotting validation samples...")
    val_loader = datamodule.val_dataloader()
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        # Get one sample from the batch
        sample = {
            "volume": batch["volume"][0],
            "aligned": batch["aligned"][0],
            "misaligned": batch["misaligned"][0],
        }
        file_path = batch["file_path"][0]

        # Plot and save
        plot_volume_slices(sample, val_dir, i, file_path)
        print(f"  Saved validation sample {i + 1}/{num_samples}")


def main():
    # Define data directory (update this to your actual data directory)
    data_dir = "/home/marten/work/datasets/emdb/train"

    # Initialize the data module
    data_module = EMDBDataModule(
        dataset_directory=data_dir,
        batch_size=1,  # Use batch size of 1 for visualization
        target_size=64,
        train_val_split=(0.8, 0.2),  # No test set for this example
        num_workers=1,  # Use at least 1 for reproducibility
    )

    # Set up the data module
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Plot samples from both train and validation sets
    output_dir = "volume_plots"
    plot_dataset_samples(data_module, num_samples=3, output_dir=output_dir)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
