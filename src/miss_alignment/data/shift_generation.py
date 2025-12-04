import random
import torch
import einops
import warnings
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class ShiftConfig:
    """Configuration for a single shift type."""

    name: str
    probability: float
    generator: Callable[[int, str], torch.Tensor]

    def should_apply(self) -> bool:
        """Check if this shift should be applied based on probability."""
        return self.probability > 0.0 and random.random() <= self.probability


class ShiftGenerator:
    """Composable 3D shift generator."""

    def __init__(
        self, shift_configs: List[ShiftConfig], ensure_at_least_one: bool = True
    ):
        """
        Initialize the shift generator.

        Parameters
        ----------
        shift_configs : List[ShiftConfig]
            List of shift configurations to apply
        ensure_at_least_one : bool
            If True, ensures at least one shift is applied when possible
        """
        self.shift_configs = shift_configs
        self.ensure_at_least_one = ensure_at_least_one

        for config in self.shift_configs:
            if config.probability == 0.0:
                warnings.warn(f"ShiftConfig {config.name} has probability 0.0")

    def __call__(self, num_points: int, device: str = "cpu") -> torch.Tensor:
        """Generate combined shifts for the given number of points."""
        shifts = torch.zeros((num_points, 3), device=device)

        # Determine which shifts to apply
        shifts_to_apply = [
            config for config in self.shift_configs if config.should_apply()
        ]

        # If no shifts selected and we need at least one, pick from available configs
        if not shifts_to_apply and self.ensure_at_least_one:
            available_configs = [
                config for config in self.shift_configs if config.probability > 0.0
            ]
            if available_configs:
                selected_config = random.choice(available_configs)
                shifts_to_apply = [selected_config]

        # Apply selected shifts
        for config in shifts_to_apply:
            shift_contribution = config.generator(num_points, device)
            shifts = shifts + shift_contribution

        return shifts


def select_random_indices(
    sequence,
    max_sequence_length: int = 5,
    edge_only: bool = False,
):
    """
    Randomly select 1 to 4 connected indices from a sequence.

    Parameters
    ----------
    sequence : torch.Tensor
        A 1D tensor from which to select indices

    Returns
    -------
    torch.Tensor
        A 1D tensor containing the selected indices
    """
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 1:
        raise ValueError("Input must be a 1D torch tensor")

    seq_length = len(sequence)
    if seq_length == 0:
        return torch.tensor([], dtype=torch.long)

    # Decide how many indices to select (1-4)
    num_indices = min(random.randint(1, max_sequence_length), seq_length)

    # Select a starting index
    max_start = seq_length - num_indices
    if edge_only:
        if random.random() > 0.5:  # only do outliers on the last 0 to 5 tilts
            start_idx = 0
        else:
            start_idx = max_start
    else:
        start_idx = random.randint(0, max(0, max_start))

    # Create the connected indices
    selected_indices = torch.arange(
        start_idx, start_idx + num_indices, dtype=torch.long
    )

    return selected_indices


def parabola_from_control_points(
    y_controls: torch.Tensor, n_points: int = 100, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generate parabolas through 3 evenly-spaced control points (vectorized).

    Parameters
    ----------
    y_controls : torch.Tensor
        Tensor of shape (..., 3) with y-values at x = [0, 0.5, 1].
        Can be batched along any leading dimensions.
    n_points : int, default=100
        Number of points to generate along each parabola.
    device : str or torch.device, default="cpu"
        Device to run computation on ('cpu', 'cuda', etc.).

    Returns
    -------
    y : torch.Tensor
        Y coordinates of parabolas, shape (..., n_points).

    Examples
    --------
    >>> y_ctrl = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
    >>> y = parabola_from_control_points(y_ctrl, n_points=5)
    >>> y.shape
    torch.Size([2, 5])
    """
    y_controls = y_controls.to(device)

    if y_controls.shape[-1] != 3:
        raise ValueError("Last dimension must be 3 control points")

    # Extract control points: (..., 3) -> 3 tensors of shape (...)
    y0 = y_controls[..., 0]
    y1 = y_controls[..., 1]
    y2 = y_controls[..., 2]

    # Compute parabola coefficients
    c = y0
    a = 2 * (y0 - 2 * y1 + y2)
    b = -3 * y0 + 4 * y1 - y2

    # Generate x coordinates
    x = torch.linspace(0, 1, n_points, device=device)

    # Compute y = ax² + bx + c (broadcasting over batch dims)
    # a, b, c: shape (...), x: shape (n_points,)
    # Reshape for broadcasting: (..., 1) * (n_points,) -> (..., n_points)
    y = a[..., None] * x**2 + b[..., None] * x + c[..., None]

    return y


def generate_shift_trajectory(
    num_points: int,
    breakpoint_p: float = 0.5,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a trajectory with an optional shift in direction.

    This function creates a smooth trajectory in 3D space. With probability
    `breakpoint_p`, it will generate a trajectory with a sudden change in
    direction by stitching together two smooth sub-trajectories.

    Parameters
    ----------
    num_points : int
        Number of points in the final trajectory.
    breakpoint_p : float, optional
        Probability of generating a trajectory with a breakpoint, by default 0.5.
    device : str, optional

    Returns
    -------
    tuple
        A tuple containing:
            - x_positions : torch.Tensor
                X-coordinates of the trajectory points.
            - y_positions : torch.Tensor
                Y-coordinates of the trajectory points.
            - z_positions : torch.Tensor
                Z-coordinates of the trajectory points.
    """
    if num_points < 4:
        raise ValueError(f"num_points must be at least 4, got {num_points}")

    shifts = []
    if random.random() > 1.0 - breakpoint_p:
        t1 = random.randint(2, num_points - 2)
        t2 = num_points - t1
        for _ in range(3):
            controls_1 = torch.rand(3, device=device) * 2 - 1
            controls_2 = torch.cat(
                [controls_1[2:3], torch.rand(2, device=device) * 2 - 1]
            )
            s1 = parabola_from_control_points(controls_1, t1, device=device)
            s2 = parabola_from_control_points(controls_2, t2 + 1, device=device)
            shifts += [torch.cat([s1, s2[1:]])]
    else:
        for _ in range(3):
            controls = torch.rand(3, device=device) * 2 - 1
            shifts += [
                parabola_from_control_points(controls, num_points, device=device)
            ]
    return shifts


class TrajectoryGenerator:
    """Picklable trajectory shift generator."""

    def __init__(self, trajectory_max_shift: float):
        self.trajectory_max_shift = trajectory_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        x_shifts, y_shifts, z_shifts = generate_shift_trajectory(
            num_points, device=device
        )
        x_shifts = x_shifts * self.trajectory_max_shift
        y_shifts = y_shifts * self.trajectory_max_shift
        z_shifts = z_shifts * self.trajectory_max_shift
        shifts = torch.stack([z_shifts, y_shifts, x_shifts], dim=1)
        shifts -= einops.reduce(shifts, "n zyx -> 1 zyx", reduction="mean")
        return shifts


class JitterGenerator:
    """Picklable jitter shift generator."""

    def __init__(self, jitter_max_std: float):
        self.jitter_max_std = jitter_max_std

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        std = random.random() * self.jitter_max_std
        return torch.normal(
            mean=0.0, std=float(std), size=(num_points, 3), device=device
        )


class OutlierGenerator:
    """Picklable outlier shift generator."""

    def __init__(self, outlier_max_shift: float):
        self.outlier_max_shift = outlier_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        shifts = torch.zeros((num_points, 3), device=device)
        ids = select_random_indices(
            torch.arange(num_points),
            max_sequence_length=1,
            edge_only=False,
        )
        outliers = (
            torch.rand(3, device=device) * (2 * self.outlier_max_shift)
            - self.outlier_max_shift
        )
        shifts[ids] = outliers
        return shifts


class FractureGenerator:
    """Picklable fracture shift generator."""

    def __init__(self, fracture_max_shift: float):
        self.fracture_max_shift = fracture_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        shifts = torch.zeros((num_points, 3), device=device)
        fraction_idx = random.randint(1, num_points)
        outliers = (
            torch.rand(3, device=device) * (2 * self.fracture_max_shift)
            - self.fracture_max_shift
        )
        if random.random() > 0.5:  # make it linear
            z = torch.linspace(outliers[0], 0, fraction_idx, device=device)
            y = torch.linspace(outliers[1], 0, fraction_idx, device=device)
            x = torch.linspace(outliers[2], 0, fraction_idx, device=device)
            outliers = torch.stack([z, y, x], dim=1)
        # randomly invert direction
        if random.random() > 0.5:
            shifts[-fraction_idx:] = torch.flip(outliers, dims=(0,))
        else:
            shifts[:fraction_idx] = outliers
        shifts -= einops.reduce(shifts, "n zyx -> 1 zyx", reduction="mean")
        return shifts


def create_default_generator(
    trajectory_probability: float = 0.5,
    trajectory_max_shift: float = 10.0,
    jitter_probability: float = 0.5,
    jitter_max_std: float = 2.0,
    outlier_probability: float = 0.4,
    outlier_max_shift: float = 20.0,
    fracture_probability: float = 0.4,
    fracture_max_shift: float = 30.0,
) -> ShiftGenerator:
    """Create a picklable shift generator."""

    shift_configs = [
        ShiftConfig(
            name="trajectory",
            probability=trajectory_probability,
            generator=TrajectoryGenerator(trajectory_max_shift),
        ),
        ShiftConfig(
            name="jitter",
            probability=jitter_probability,
            generator=JitterGenerator(jitter_max_std),
        ),
        ShiftConfig(
            name="outlier",
            probability=outlier_probability,
            generator=OutlierGenerator(outlier_max_shift),
        ),
        ShiftConfig(
            name="fracture",
            probability=fracture_probability,
            generator=FractureGenerator(fracture_max_shift),
        ),
    ]

    return ShiftGenerator(shift_configs, ensure_at_least_one=True)


def project_shifts_3d_to_2d(
    shifts_3d: torch.Tensor,  # contains (n, zyx) shifts
    projection_matrices: torch.Tensor,  # contains (n, 2, 3)
) -> torch.Tensor:
    """Project 3D shifts to 2D."""
    y, x = projection_matrices.shape[-2:]
    if (y, x) != (2, 3):
        raise ValueError("Shift projection matrices must have shape (2, 3)")
    shifts_3d = einops.rearrange(shifts_3d, "b zyx -> b zyx 1")
    shifts_2d = projection_matrices @ shifts_3d
    shifts_2d = einops.rearrange(shifts_2d, "b yx 1 -> b yx")
    return shifts_2d
