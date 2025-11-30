from pathlib import Path
from typing import Callable

import math
import torch
import einops
from warpylib.cubic_grid import CubicGrid
from warpylib import TiltSeries
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data

from miss_alignment.data.io import TiltSeriesData
from miss_alignment.models import MissAlignment


def generate_position_grid(
    volume_dimensions_physical: tuple[float, float, float] | torch.Tensor,
    pixel_size: float,
    patch_size: int,
    patch_overlap: float,
) -> torch.Tensor:
    # calculate patches per dimensions
    stride = int(patch_size * (1 - patch_overlap))
    patches_per_dim = []
    for s in volume_dimensions_physical:
        dim = s // pixel_size
        patches_per_dim += [max(int(dim // stride), 1)]

    # extract patches per dim and physical size
    px, py, pz = patches_per_dim
    x, y, z = volume_dimensions_physical

    # get the reconstruction positions to optimize over
    zs = ((torch.arange(pz) + 0.5) / pz) * z
    ys = ((torch.arange(py) + 0.5) / py) * y
    xs = ((torch.arange(px) + 0.5) / px) * x

    # merge positions in tensor
    position_grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    # ensure we get a list of xyz positions!!
    position_grid = einops.rearrange(position_grid, "d h w xyz -> (d h w) xyz")
    return position_grid


def optimize_shifts(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    setting: str | tuple[int, int, int] | tuple[int, int, int, int] = "global",
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
):
    """Find shifts to optimize model score.

    Parameters
    ----------
    setting: type of alingment to run
        str - 'global': optimizes a single shift per image
        tuple[int, int, int] - (3, 3, 41): a single 2d field per image
        tuple[int, int, int, int] - (3, 3, 2, 10): a volume warp grid

    Returns
    -------
    tuple[torch.Tensor, list]
        - Detected translational alignment.
        - List of loss values for each optimization iteration.
    """
    # move all modules to device in place
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    # move images to device
    images = images.to(device)

    parameters = None
    if setting == "global":
        # store the initial tilt_series alignment
        initial_tilt_axis_offset_y = tilt_series.tilt_axis_offset_y.clone()
        initial_tilt_axis_offset_x = tilt_series.tilt_axis_offset_x.clone()

        # create the alignment parameters
        shifts_y = torch.zeros_like(
            initial_tilt_axis_offset_x,
            requires_grad=True,
            device=device,
        )
        shifts_x = torch.zeros_like(
            initial_tilt_axis_offset_x,
            requires_grad=True,
            device=device,
        )
        parameters = [shifts_y, shifts_x]
    elif len(setting) == 3:  # TODO add case of starting from existent grid
        # movement grids - these should receive gradients
        tilt_series.grid_movement_x = tilt_series.grid_movement_x.resize(
            new_size=setting
        ).to(device)
        leaf_variable_x = tilt_series.grid_movement_x.values.requires_grad_(True)
        tilt_series.grid_movement_x = CubicGrid(setting, leaf_variable_x)

        tilt_series.grid_movement_y = tilt_series.grid_movement_y.resize(
            new_size=setting
        ).to(device)
        leaf_variable_y = tilt_series.grid_movement_y.values.requires_grad_(True)
        tilt_series.grid_movement_y = CubicGrid(setting, leaf_variable_y)

        parameters = [leaf_variable_x, leaf_variable_y]
    elif len(setting) == 4:  # TODO add case of starting from existent grid
        tilt_series.grid_volume_warp_x = tilt_series.grid_volume_warp_x.resize(
            new_size=setting
        ).to(device)
        leaf_variable_x = tilt_series.grid_volume_warp_x.values.requires_grad_(True)
        tilt_series.grid_volume_warp_x = CubicGrid(setting, leaf_variable_x)

        tilt_series.grid_volume_warp_y = tilt_series.grid_volume_warp_y.resize(
            new_size=setting
        ).to(device)
        leaf_variable_y = tilt_series.grid_volume_warp_y.values.requires_grad_(True)
        tilt_series.grid_volume_warp_y = CubicGrid(setting, leaf_variable_y)

        tilt_series.grid_volume_warp_z = tilt_series.grid_volume_warp_z.resize(
            new_size=setting
        ).to(device)
        leaf_variable_z = tilt_series.grid_volume_warp_z.values.requires_grad_(True)
        tilt_series.grid_volume_warp_z = CubicGrid(setting, leaf_variable_z)

        parameters = [
            leaf_variable_x,
            leaf_variable_y,
            leaf_variable_z,
        ]
    else:
        raise ValueError(f"Invalid setting for alignment optimization: {setting}")

    alignment_optimizer = torch.optim.LBFGS(
        parameters,
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # update the alignments
        if setting == "global":
            tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y
            tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x

        batches = int(math.ceil(positions.shape[0] / batch_size))
        total_samples = positions.shape[0]
        total_loss = 0.0

        # Use gradient accumulation: process each batch separately
        for b in range(batches):
            if b == batches - 1:
                batch_positions = positions[b * batch_size :]
            else:
                batch_positions = positions[b * batch_size : (b + 1) * batch_size]

            current_batch_size = batch_positions.shape[0]

            # reconstruct subvolumes for this batch
            subvolumes = tilt_series.reconstruct_subvolumes_single(
                tilt_data=images,
                coords=batch_positions.to(device),
                pixel_size=pixel_size,
                size=patch_size,
                apply_ctf=apply_ctf,
                oversampling=2.0,
            )

            # ensure normalization per subvolume
            mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
            std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
            subvolumes = (subvolumes - mean) / std

            # change channel to batch dimension
            subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

            # Get loss for this batch and compute backward pass
            batch_loss = model(subvolumes)
            batch_loss = batch_loss.mean()

            # Weight by batch size for proper gradient accumulation
            weighted_loss = batch_loss * (current_batch_size / total_samples)

            # Backward pass for this batch (gradients accumulate)
            weighted_loss.backward()

            # Accumulate total loss for logging (unweighted for interpretability)
            total_loss += batch_loss.item() * current_batch_size

        # Average loss across all samples
        avg_loss = total_loss / total_samples
        loss_values.append(avg_loss)

        return avg_loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        alignment_optimizer.step(closure)

    if setting == "global":
        # remove gradients and finalize global shifts
        tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y.detach()
        tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x.detach()
    elif len(setting) == 3:
        # remove gradients
        tilt_series.grid_movement_x.values = tilt_series.grid_movement_x.values.detach()
        tilt_series.grid_movement_y.values = tilt_series.grid_movement_y.values.detach()
    elif len(setting) == 4:
        # remove gradients
        tilt_series.grid_volume_warp_x.values = (
            tilt_series.grid_volume_warp_x.values.detach()
        )
        tilt_series.grid_volume_warp_y.values = (
            tilt_series.grid_volume_warp_y.values.detach()
        )
        tilt_series.grid_volume_warp_z.values = (
            tilt_series.grid_volume_warp_z.values.detach()
        )
    # move back because there were modified in-place
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def run_iterative_anchoring(
    tilt_series: TiltSeries,
    optimize_fn: Callable[[TiltSeries], tuple[TiltSeries, list[float]]],
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[TiltSeries, list[float]]:
    """Core iterative anchoring logic for global shift optimization.

    Addresses high-tilt images getting stuck in local minima by:
    1. Treating only the central fraction of tilts as "reliable"
    2. Storing relative offsets of unreliable (high angle) tilts
    3. Running the optimizer
    4. Restoring relative structure of unreliable tilts while
       propagating the movement of boundary reliable tilts
    5. Gradually expanding the reliable set (1 tilt per side per iteration)

    Parameters
    ----------
    tilt_series : TiltSeries
        Tilt series to optimize.
    optimize_fn : Callable[[TiltSeries], tuple[TiltSeries, list[float]]]
        Optimization function that takes a TiltSeries and returns
        (optimized_tilt_series, loss_values).
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """
    n_tilts = tilt_series.n_tilts
    sorted_indices = tilt_series.indices_sorted_angle()

    # Calculate initial number of unreliable tilts per side
    n_reliable = int(n_tilts * initial_reliable_fraction)
    n_reliable = max(1, min(n_reliable, n_tilts))
    n_unreliable_total = n_tilts - n_reliable
    n_unreliable_per_side = n_unreliable_total // 2

    all_loss_values = []

    while n_unreliable_per_side > 0:
        print(f"n_unreliable_per_side: {n_unreliable_per_side}")
        # Define boundaries in sorted index space
        # sorted_indices[0] is most negative angle, sorted_indices[-1] most positive
        neg_boundary_sorted_idx = n_unreliable_per_side
        pos_boundary_sorted_idx = n_tilts - 1 - n_unreliable_per_side

        # Store relative offsets for unreliable tilts (chain-like, toward center)
        # Negative side: sorted indices 0 to neg_boundary_sorted_idx-1
        neg_relative_offsets_x = []
        neg_relative_offsets_y = []
        for i in range(n_unreliable_per_side):
            curr_tilt_idx = sorted_indices[i]
            next_tilt_idx = sorted_indices[i + 1]  # neighbor toward center
            rel_x = (
                tilt_series.tilt_axis_offset_x[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_x[next_tilt_idx]
            ).clone()
            rel_y = (
                tilt_series.tilt_axis_offset_y[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_y[next_tilt_idx]
            ).clone()
            neg_relative_offsets_x.append(rel_x)
            neg_relative_offsets_y.append(rel_y)

        # Positive side: sorted indices pos_boundary_sorted_idx+1 to n_tilts-1
        pos_relative_offsets_x = []
        pos_relative_offsets_y = []
        for i in range(pos_boundary_sorted_idx + 1, n_tilts):
            curr_tilt_idx = sorted_indices[i]
            prev_tilt_idx = sorted_indices[i - 1]  # neighbor toward center
            rel_x = (
                tilt_series.tilt_axis_offset_x[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_x[prev_tilt_idx]
            ).clone()
            rel_y = (
                tilt_series.tilt_axis_offset_y[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_y[prev_tilt_idx]
            ).clone()
            pos_relative_offsets_x.append(rel_x)
            pos_relative_offsets_y.append(rel_y)

        # Run optimization
        tilt_series, loss_values = optimize_fn(tilt_series)
        all_loss_values.extend(loss_values)
        print(f"Loss went from {loss_values[0]} to {loss_values[-1]}")

        # Restore unreliable tilts with chain-like relative offsets
        # Negative side: propagate from boundary outward (high sorted idx to low)
        for i in range(n_unreliable_per_side - 1, -1, -1):
            curr_tilt_idx = sorted_indices[i]
            next_tilt_idx = sorted_indices[i + 1]
            tilt_series.tilt_axis_offset_x[curr_tilt_idx] = (
                tilt_series.tilt_axis_offset_x[next_tilt_idx] + neg_relative_offsets_x[i]
            )
            tilt_series.tilt_axis_offset_y[curr_tilt_idx] = (
                tilt_series.tilt_axis_offset_y[next_tilt_idx] + neg_relative_offsets_y[i]
            )

        # Positive side: propagate from boundary outward (low sorted idx to high)
        for i, sorted_i in enumerate(range(pos_boundary_sorted_idx + 1, n_tilts)):
            curr_tilt_idx = sorted_indices[sorted_i]
            prev_tilt_idx = sorted_indices[sorted_i - 1]
            tilt_series.tilt_axis_offset_x[curr_tilt_idx] = (
                tilt_series.tilt_axis_offset_x[prev_tilt_idx] + pos_relative_offsets_x[i]
            )
            tilt_series.tilt_axis_offset_y[curr_tilt_idx] = (
                tilt_series.tilt_axis_offset_y[prev_tilt_idx] + pos_relative_offsets_y[i]
            )

        # Expand reliable set by 1 tilt per side
        n_unreliable_per_side -= 1

    # Final optimization with all tilts reliable
    tilt_series, loss_values = optimize_fn(tilt_series)
    all_loss_values.extend(loss_values)
    print(f"Last one: loss went from {loss_values[0]} to {loss_values[-1]}")

    return tilt_series, all_loss_values


def optimize_shifts_iterative(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[TiltSeries, list[float]]:
    """Iterative anchored optimization for global shifts.

    Convenience wrapper around run_iterative_anchoring that uses
    optimize_shifts as the optimizer function.

    Parameters
    ----------
    model : MissAlignment
        Trained model for scoring reconstructions.
    tilt_series : TiltSeries
        Tilt series to optimize.
    images : torch.Tensor
        Preprocessed tilt images.
    pixel_size : float
        Pixel size in Angstroms.
    positions : torch.Tensor
        3D positions to reconstruct and evaluate.
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """

    def optimize_fn(ts: TiltSeries) -> tuple[TiltSeries, list[float]]:
        return optimize_shifts(
            model=model,
            tilt_series=ts,
            images=images,
            pixel_size=pixel_size,
            positions=positions,
            setting="global",
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
        )

    return run_iterative_anchoring(
        tilt_series=tilt_series,
        optimize_fn=optimize_fn,
        initial_reliable_fraction=initial_reliable_fraction,
    )


def evaluate_tilt_series(
    model_checkpoint_path: Path,
    tilt_series_path: Path,
    output_directory: Path,
    setting: str | tuple[int, int, int] | tuple[int, int, int, int] = "global",
    patch_size: int = 96,
    patch_overlap: float = 0.1,
    batch_size: int = 16,
    apply_ctf: bool = True,
    downsample: int = 1,
    device: str = "cpu",
    iterative: bool = True,
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[Path, list[float]]:
    # load the best model and run alignment optimization
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path,
        map_location="cpu",
    )

    # load tilt_series and set its name for output
    tilt_series_data = TiltSeriesData.from_json(tilt_series_path)
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
        downsample=downsample
    )
    # run the preprocessing from warp for consistency
    images = preprocess_tilt_data(
        tilt_data=images,
        normalize=True,
        invert=False,
        subvolume_size=patch_size,
    )

    # generate a position grid to optimize over
    position_grid = generate_position_grid(
        volume_dimensions_physical=tilt_series.volume_dimensions_physical,
        pixel_size=pixel_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # run alignment optimization with the model
    if iterative and setting == "global":
        tilt_series, loss = optimize_shifts_iterative(
            model=model,
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            positions=position_grid,
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
            initial_reliable_fraction=initial_reliable_fraction,
        )
    else:
        tilt_series, loss = optimize_shifts(
            model,
            tilt_series,
            images,
            pixel_size,
            position_grid,
            setting=setting,
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
        )

    # write all necessary output
    xml_out = (output_directory / (tilt_series_data.xml_filename + ".xml")).absolute()
    json_out = (output_directory / (tilt_series_data.xml_filename + ".json")).absolute()
    new_tilt_series_data = tilt_series_data.replace(
        xml_metadata_path=xml_out,
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_out)

    return json_out, loss
