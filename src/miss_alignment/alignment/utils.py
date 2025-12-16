"""Utility functions for tilt-series alignment."""

import torch
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d


def project_volume_shift_to_image_alignment(
    shift_3d: torch.Tensor,  # (3, )  zyx shift
    projection_matrices: torch.Tensor,  # (n_tilts, 2, 3)
) -> torch.Tensor:  # (n_tilts, 2)  yx shift
    """Project a 3D volume shift to 2D image alignment shifts.

    Takes a single 3D shift vector and projects it to 2D shifts for each tilt image
    using the tilt-series projection geometry.

    Parameters
    ----------
    shift_3d : torch.Tensor
        3D shift vector with shape (3,) in ZYX order
    projection_matrices : torch.Tensor
        Projection matrices for each tilt with shape (n_tilts, 2, 3)

    Returns
    -------
    torch.Tensor
        2D shifts for each tilt with shape (n_tilts, 2) in YX order
    """
    n_tilts = projection_matrices.shape[0]
    shift_3d = shift_3d.repeat(n_tilts, 1)

    shifts_2d = project_shifts_3d_to_2d(shift_3d, projection_matrices)
    return shifts_2d
