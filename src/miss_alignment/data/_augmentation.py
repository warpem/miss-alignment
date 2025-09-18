import random
import torch


def random_contrast(volume):
    """Slightly alter the mean and std of a pre-normalised tensor"""
    std_change, mean_change = (
        torch.normal(1, 0.1, (1,)), torch.normal(0, 0.1, (1,))
    )
    return volume * std_change + mean_change


def random_mirror(volumes: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply same mirroring operation to a list of volumes"""
    flip_dims = [random.choice([True, False]) for _ in range(3)]

    # Apply flips to the last 3 dimensions
    dims_to_flip = []
    for i, should_flip in enumerate(flip_dims):
        if should_flip:
            dims_to_flip.append(-(3 - i))  # -3, -2, -1 for D, H, W

    if dims_to_flip:
        volumes = [torch.flip(v, dims_to_flip) for v in volumes]

    return volumes


def random_cube_mask(
        volume: torch.Tensor, p=.3, size_range=(0.1, 0.3)
) -> torch.Tensor:
    """Mask out a random cube in a volume with probability p"""
    if random.random() > 1 - p:
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

        volume[
            start_d:start_d + mask_d,
            start_h:start_h + mask_h,
            start_w:start_w + mask_w
        ] = random.random() - .5
    return volume


def random_edge_mask(
        volume: torch.Tensor,
        p: float = 0.5,
        edge_width: tuple[int, int] = (1, 8),
        mask_value: float = 0.0
) -> torch.Tensor:
    """Mask out all edges of a volume with probability p"""
    if random.random() > 1 - p:
        width = random.randint(edge_width[0], edge_width[1])

        # Mask all edges in one line using slicing tricks
        volume[..., :width, :, :] = volume[..., -width:, :, :] = \
            volume[..., :, :width, :] = volume[..., :, -width:, :] = \
            volume[..., :, :, :width] = volume[..., :, :, -width:] = mask_value

    return volume
