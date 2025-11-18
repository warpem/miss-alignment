import torch
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d


def calculate_cross_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the 3D cross correlation between volumes of the same size.

    The position of the maximum relative to the center of the volume gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.

    Parameters
    ----------
    a : torch.Tensor
        First 3D volume with shape (..., D, H, W)
    b : torch.Tensor
        Second 3D volume with shape (..., D, H, W)

    Returns
    -------
    torch.Tensor
        3D cross-correlation volume
    """
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    d, h, w = a.shape[-3:]
    fta = torch.fft.rfftn(a, dim=(-3, -2, -1))
    ftb = torch.fft.rfftn(b, dim=(-3, -2, -1))
    result = fta * torch.conj(ftb)
    result = torch.fft.irfftn(result, dim=(-3, -2, -1), s=(d, h, w))
    result = torch.fft.ifftshift(result, dim=(-3, -2, -1))
    result /= d * h * w  # normalize the result
    return result


def get_shift_from_correlation_image(correlation_image: torch.Tensor) -> torch.Tensor:
    """
    Extract shift from 3D correlation volume.

    The shift should be applied to img2 to align with img1.
    Uses parabolic interpolation for sub-voxel accuracy.

    Parameters
    ----------
    correlation_image : torch.Tensor
        3D correlation volume

    Returns
    -------
    torch.Tensor
        3D shift vector [z, y, x]
    """
    d, h, w = correlation_image.shape
    dtype, device = correlation_image.dtype, correlation_image.device

    image_shape = torch.as_tensor(correlation_image.shape, device=device, dtype=dtype)
    center = torch.divide(image_shape, 2, rounding_mode="floor")

    # Find peak location
    flat_idx = torch.argmax(correlation_image)
    peak_z = flat_idx // (h * w)
    peak_y = (flat_idx % (h * w)) // w
    peak_x = flat_idx % w

    # Check if peak is on border
    if (
        peak_z == 0
        or peak_z == d - 1
        or peak_y == 0
        or peak_y == h - 1
        or peak_x == 0
        or peak_x == w - 1
    ):
        shift = (
            torch.tensor([peak_z, peak_y, peak_x], device=device, dtype=dtype) - center
        )
        return shift

    # Parabolic interpolation in z direction
    f_z0 = correlation_image[peak_z - 1, peak_y, peak_x]
    f_z1 = correlation_image[peak_z, peak_y, peak_x]
    f_z2 = correlation_image[peak_z + 1, peak_y, peak_x]
    subpixel_peak_z = peak_z + 0.5 * (f_z0 - f_z2) / (f_z0 - 2 * f_z1 + f_z2)

    # Parabolic interpolation in y direction
    f_y0 = correlation_image[peak_z, peak_y - 1, peak_x]
    f_y1 = correlation_image[peak_z, peak_y, peak_x]
    f_y2 = correlation_image[peak_z, peak_y + 1, peak_x]
    subpixel_peak_y = peak_y + 0.5 * (f_y0 - f_y2) / (f_y0 - 2 * f_y1 + f_y2)

    # Parabolic interpolation in x direction
    f_x0 = correlation_image[peak_z, peak_y, peak_x - 1]
    f_x1 = correlation_image[peak_z, peak_y, peak_x]
    f_x2 = correlation_image[peak_z, peak_y, peak_x + 1]
    subpixel_peak_x = peak_x + 0.5 * (f_x0 - f_x2) / (f_x0 - 2 * f_x1 + f_x2)

    subpixel_shift = (
        torch.tensor(
            [subpixel_peak_z, subpixel_peak_y, subpixel_peak_x],
            device=device,
            dtype=dtype,
        )
        - center
    )

    return subpixel_shift


def project_volume_shift_to_image_alignment(
    shift_3d: torch.Tensor,  # (3, )  zyx shift
    projection_matrices: torch.Tensor,  # (n_tilts, 2, 3)
) -> torch.Tensor:  # (n_tilts, 2)  yx shift
    n_tilts = projection_matrices.shape[0]
    shift_3d = -1 * shift_3d  # get the forward shift for the imaging model
    shift_3d = shift_3d.repeat(n_tilts, 1)

    shifts_2d = project_shifts_3d_to_2d(shift_3d, projection_matrices)
    return shifts_2d
