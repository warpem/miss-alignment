import torch

from miss_alignment.alignment.tilt_series import calculate_cross_correlation, get_shift_from_correlation_image
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d

def read_aretomo_alignment(filepath):
    with open(filepath) as fstream:
        data = fstream.readlines()
    data = [d.strip().split() for d in data if not d.startswith('#')]
    data = torch.tensor([[float(d[4]), float(d[3])] for d in data])
    return data

def center_tomogram(tilt_series_ground_truth, tilt_series, tomo_shape):
    n_tilts = len(tilt_series.tilt_angles)
    ground_truth_reconstruction = tilt_series_ground_truth.reconstruct_tomogram(
        tomo_shape, 128
    )

    aligned_reconstruction = tilt_series.reconstruct_tomogram(tomo_shape, 128)

    correlation = calculate_cross_correlation(
        ground_truth_reconstruction, aligned_reconstruction
    )
    shift_3d = get_shift_from_correlation_image(correlation)
    shift_3d = -1 * shift_3d  # get the forward shift for the imaging model
    shift_3d = shift_3d.repeat(n_tilts, 1)

    shifts_2d = project_shifts_3d_to_2d(
        shift_3d, tilt_series.projection_matrices[..., :3, :3]
    )
    tilt_series.sample_translations += shifts_2d
    return tilt_series