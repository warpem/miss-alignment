from pathlib import Path
from miss_alignment.data.io import read_tomogram_from_pickle
import torch
import einops


def angle_to_rotation_matrix(angle):
    """Convert angle to 2D rotation matrix.
    
    Parameters
    ----------
    angle : float or torch.Tensor
        Rotation angle in radians
        
    Returns
    -------
    torch.Tensor
        2x2 rotation matrix
    """
    angle_rad = torch.deg2rad(angle)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    return torch.tensor([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])


def write_transform_file(matrix, shifts, filepath, precision=7):
    """
    Write transformation matrices to .xf file with exact spacing.

    Parameters
    ----------
    transforms : torch.Tensor or np.ndarray
        Shape (N, 3, 3) homogeneous transformation matrices
    filepath : str or Path
        Output file path
    precision : int
        Number of decimal places (default 7 to match your format)
    """
    a11, a12, a21, a22 = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
    with open(filepath, 'w') as f:
        for s in shifts:
            tx, ty = s[0], s[1]

            # Write with exact spacing (12 characters per field, right-aligned)
            line = f"{a11:12.7f}{a12:12.7f}{a21:12.7f}{a22:12.7f}{tx:12.3f}{ty:12.3f}\n"
            f.write(line)


out_path = Path('MissAlignments_alignments_run9_250926')
out_path.mkdir(exist_ok=True)
in_path = Path('run9(2)/iter5/')

for in_file in in_path.iterdir():
    if in_file.suffix != '.pickle':
        continue
    out_file = out_path / (in_file.stem + '.xf')
    tilt_series = read_tomogram_from_pickle(in_file)
    
    matrix = angle_to_rotation_matrix(torch.mean(tilt_series.tilt_axis_angle * -1.0))
    # flip to make xy, then invert to correspond with IMOD's back projection model
    shifts = torch.flip(tilt_series.sample_translations, dims=(1,)) * -1.0

    shifts = einops.rearrange(shifts, 'n xy -> n xy 1')
    out_shifts = matrix @ shifts
    out_shifts = einops.rearrange(out_shifts, 'n xy 1 -> n xy')
    write_transform_file(matrix, out_shifts, out_file)

