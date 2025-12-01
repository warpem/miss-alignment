import torch
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from warpylib import rescale


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


def get_shift_from_correlation_image(
    correlation_image: torch.Tensor,
    patch_size: int = 16,
    upsample_size: int = 512,
) -> torch.Tensor:
    """
    Extract shift from 3D correlation volume.

    The shift should be applied to img2 to align with img1.
    Uses Fourier upsampling for sub-voxel accuracy: extracts a region around the
    integer peak, upsamples it using bandwidth-limited Fourier rescaling, and finds
    the peak position in the upsampled volume.

    Parameters
    ----------
    correlation_image : torch.Tensor
        3D correlation volume
    patch_size : int
        Size of the cubic region to extract around the integer peak (must be even).
        Default is 16.
    upsample_size : int
        Size to upsample the extracted region to (must be even). Default is 512.

    Returns
    -------
    torch.Tensor
        3D shift vector [z, y, x]
    """
    dtype, device = correlation_image.dtype, correlation_image.device
    shape = torch.tensor(correlation_image.shape, device=device, dtype=dtype)
    center = torch.div(shape, 2, rounding_mode="floor")

    # Find integer peak location
    flat_idx = torch.argmax(correlation_image)
    peak_coords = torch.tensor(
        torch.unravel_index(flat_idx, correlation_image.shape),
        device=device,
        dtype=dtype,
    )

    half_patch = patch_size // 2

    # Check if we can extract a full patch around the peak
    if torch.any(peak_coords < half_patch) or torch.any(
        peak_coords >= shape - half_patch
    ):
        return peak_coords - center

    # Extract patch around peak
    pz, py, px = peak_coords.int().tolist()
    patch = correlation_image[
        pz - half_patch : pz + half_patch,
        py - half_patch : py + half_patch,
        px - half_patch : px + half_patch,
    ]

    # Upsample using Fourier rescaling
    upsampled = rescale(patch, size=(upsample_size, upsample_size, upsample_size))

    # Find peak in upsampled volume
    up_flat_idx = torch.argmax(upsampled)
    up_peak_coords = torch.tensor(
        torch.unravel_index(up_flat_idx, upsampled.shape),
        device=device,
        dtype=dtype,
    )

    # Convert upsampled peak position back to original coordinates
    upsample_factor = upsample_size / patch_size
    up_center = upsample_size / 2
    offset = (up_peak_coords - up_center) / upsample_factor
    subpixel_peak = peak_coords + offset

    return subpixel_peak - center


def project_volume_shift_to_image_alignment(
    shift_3d: torch.Tensor,  # (3, )  zyx shift
    projection_matrices: torch.Tensor,  # (n_tilts, 2, 3)
) -> torch.Tensor:  # (n_tilts, 2)  yx shift
    n_tilts = projection_matrices.shape[0]
    shift_3d = shift_3d.repeat(n_tilts, 1)

    shifts_2d = project_shifts_3d_to_2d(shift_3d, projection_matrices)
    return shifts_2d
