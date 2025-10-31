from pathlib import Path

import math
import functools
import torch
import einops
from warpylib.cubic_grid import CubicGrid
from warpylib import TiltSeries

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
    zs = ((torch.arange(pz) + .5) / pz) * z
    ys = ((torch.arange(py) + .5) / py) * y
    xs = ((torch.arange(px) + .5) / px) * x

    # merge positions in tensor
    position_grid = (
        torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
    )
    # ensure we get a list of xyz positions!!
    position_grid = einops.rearrange(position_grid, 'd h w xyz -> (d h w) xyz')
    return position_grid


def optimize_shifts(
        model: MissAlignment,
        tilt_series: TiltSeries,
        images: torch.Tensor,
        pixel_size: float,
        positions: torch.Tensor,
        setting: str |
                 tuple[int, int, int] |
                 tuple[int, int, int, int] = 'global',
        patch_size: int = 96,
        batch_size: int = 16,
        apply_ctf: bool = True,
        device: str | torch.device = 'cpu',
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
    if setting == 'global':
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
    elif len(setting) == 3:  #TODO add case of starting from existent grid
        # movement grids - these should receive gradients
        n_params = int(functools.reduce(lambda x, y: x * y, setting))
        movement_x_leaf = (
            torch.zeros(n_params, requires_grad=True, device=device)
        )
        tilt_series.grid_movement_x = CubicGrid(setting, movement_x_leaf)
        movement_y_leaf = (
            torch.zeros(n_params, requires_grad=True, device=device)
        )
        tilt_series.grid_movement_y = CubicGrid(setting, movement_y_leaf)
        parameters = [movement_x_leaf, movement_y_leaf]
    elif len(setting) == 4:  #TODO add case of starting from existent grid
        n_params = int(functools.reduce(lambda x, y: x * y, setting))
        movement_x_leaf = (
            torch.zeros(n_params, requires_grad=True, device=device)
        )
        tilt_series.grid_volume_warp_x = CubicGrid(setting, movement_x_leaf)
        movement_y_leaf = (
            torch.zeros(n_params, requires_grad=True, device=device)
        )
        tilt_series.grid_volume_warp_y = CubicGrid(setting, movement_y_leaf)
        movement_z_leaf = (
            torch.zeros(n_params, requires_grad=True, device=device)
        )
        tilt_series.grid_volume_warp_z = CubicGrid(setting, movement_z_leaf)
        parameters = [movement_x_leaf, movement_y_leaf, movement_z_leaf]
    else:
        raise ValueError(f'Invalid setting for alignment optimization: {setting}')

    alignment_optimizer = torch.optim.LBFGS(
        parameters,
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # update the alignments
        if setting == 'global':
            tilt_series.tilt_axis_offset_y = (
                initial_tilt_axis_offset_y + shifts_y
            )
            tilt_series.tilt_axis_offset_x = (
                initial_tilt_axis_offset_x + shifts_x
            )

        batches = int(math.ceil(positions.shape[0] / batch_size))
        subvolumes = []
        for b in range(batches):
            if b == batches - 1:
                batch_positions = positions[b * batch_size:]
            else:
                batch_positions = positions[b * batch_size: (b+1) * batch_size]
            # reconstruct subvolumes
            subvolumes.append(tilt_series.reconstruct_subvolumes_single(
                tilt_data=images,
                coords=batch_positions.to(device),
                pixel_size=pixel_size,
                size=patch_size,
                apply_ctf=apply_ctf,
            ))
        subvolumes = torch.cat(subvolumes, dim=0)
        # ensure normalization per tilt
        mean = einops.reduce(
            subvolumes, "n d h w -> n 1 1 1", reduction="mean"
        )
        std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
        subvolumes = (subvolumes - mean) / std

        # change channel to batch dimension
        subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

        # Get loss and compute backward pass
        loss = model(subvolumes)
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        return loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        alignment_optimizer.step(closure)

    if setting == 'global':
        # remove gradients and finalize global shifts
        tilt_series.tilt_axis_offset_y = (
            initial_tilt_axis_offset_y + shifts_y.detach()
        )
        tilt_series.tilt_axis_offset_x = (
            initial_tilt_axis_offset_x + shifts_x.detach()
        )
    elif len(setting) == 3:
        # remove gradients
        tilt_series.grid_movement_x.values = (
            tilt_series.grid_movement_x.values.detach()
        )
        tilt_series.grid_movement_y.values = (
            tilt_series.grid_movement_y.values.detach()
        )
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


def evaluate_tilt_series(
        model_checkpoint_path: Path,
        tilt_series_path: Path,
        output_directory: Path,
        setting: str |
                 tuple[int, int, int] |
                 tuple[int, int, int, int] = 'global',
        patch_size: int = 96,
        patch_overlap: float = 0.1,
        batch_size: int = 16,
        apply_ctf: bool = True,
        device: str = "cpu",
) -> tuple[Path, list[float]]:
    # load the best model and run alignment optimization
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path, map_location='cpu',
    )

    # load tilt_series and set its name for output
    tilt_series_name = tilt_series_path.stem
    tilt_series_data = TiltSeriesData.from_json(tilt_series_path)
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack()

    # generate a position grid to optimize over
    position_grid = generate_position_grid(
        volume_dimensions_physical=tilt_series.volume_dimensions_physical,
        pixel_size=pixel_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # run alignment optimization with the model
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
    xml_out = output_directory / (tilt_series_data.xml_filename + '.xml')
    json_out = output_directory / (tilt_series_data.xml_filename + '.json')
    new_tilt_series_data = (
        tilt_series_data.replace(xml_metadata_path=xml_out,)
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_out)

    return json_out, loss
