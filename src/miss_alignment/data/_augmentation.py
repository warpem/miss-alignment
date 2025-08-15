import random
import torch


def random_contrast(volume):
    std_change, mean_change = (
        torch.normal(1, 0.1, (1,)), torch.normal(0, 0.1, (1,))
    )
    return volume * std_change + mean_change


def random_cube_mask(
        volume: torch.Tensor, p=.3, size_range=(0.1, 0.3)
) -> torch.Tensor:
    """Apply the same random cube mask to multiple volumes.

    Args:
        volumes: 3D volumes of shape [D, H, W] without channel dimension
        p: Probability of applying the mask
        size_range: Range for the size of the cube as a fraction of the volume dimensions

    Returns:
        List of masked volumes
    """
    if random.random() > 1 - p:
        # Get the dimensions of the first volume (assuming all have same dimensions)
        d, h, w = volume.shape

        # Determine mask size as fraction of volume dimensions
        mask_fraction = random.uniform(size_range[0], size_range[1])
        mask_d = max(1, int(d * mask_fraction))
        mask_h = max(1, int(h * mask_fraction))
        mask_w = max(1, int(w * mask_fraction))

        # Random starting positions for the mask
        start_d = random.randint(0, d - mask_d)
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)

        # Apply the same mask to all volumes
        volume[
            start_d:start_d + mask_d,
            start_h:start_h + mask_h,
            start_w:start_w + mask_w
        ] = random.random() - .5
    return volume


def random_mirror(volume, p=.5):
    if random.random() > 1 - p:
        axis = random.randint(0, 2)
        volume = torch.flip(volume, [axis,])
    return volume


def random_edge_mask(
        volume: torch.Tensor,
        p: float = 0.5,
        edge_width: tuple[int, int] = (1, 8),
        mask_value: float = 0.0
) -> torch.Tensor:
    """Randomly mask all edges of a 3D volume for data augmentation."""
    if random.random() > 1 - p:
        width = random.randint(edge_width[0], edge_width[1])

        # Mask all edges in one line using slicing tricks
        volume[..., :width, :, :] = volume[..., -width:, :, :] = \
            volume[..., :, :width, :] = volume[..., :, -width:, :] = \
            volume[..., :, :, :width] = volume[..., :, :, -width:] = mask_value

    return volume
