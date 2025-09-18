import pathlib
import torch
import einops
from torch_tomogram import Tomogram
from miss_alignment.models import MissAlignment
from miss_alignment.data.io import read_tomogram_from_pickle
from miss_alignment.alignment import evaluate_tilt_series
import matplotlib.pyplot as plt


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


torch.set_printoptions(precision=3, sci_mode=False)

# config
single_index = 0  # only interpolate this tilt index

model_checkpoint = pathlib.Path(
    "/home/marten/Downloads/epoch=7--step=12496--loss_epoch=0.00.ckpt"
)
model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
model.freeze()

path_xcorr = pathlib.Path("/home/marten/data/datasets/shrec_v2/run1/iter3/")
path_gt = pathlib.Path("/home/marten/data/datasets/shrec_v2/run1/ground_truth/")
output_dir = pathlib.Path("/home/marten/data/datasets/shrec_v2/grid_search/")

xcorr_files = list(path_xcorr.glob("*.pickle"))
gt_files = list(path_gt.glob("*.pickle"))

for gt, xcorr in zip(gt_files, xcorr_files):
    if "model_9" not in str(gt):
        continue
    data_gt = read_tomogram_from_pickle(gt)
    images = data_gt.images
    images -= einops.reduce(images, "tilt h w -> tilt 1 1", reduction="mean")
    images /= torch.std(images, dim=(-2, -1), keepdim=True)

    data_xcorr = read_tomogram_from_pickle(xcorr)

    tilt_angles = data_xcorr.tilt_angles
    tilt_axis_angle = data_xcorr.tilt_axis_angle

    ground_truth_alignment = data_gt.sample_translations
    xcorr_alignment = data_xcorr.sample_translations

    tomogram = Tomogram(
        images=images,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=xcorr_alignment,
    )
    tomogram._pad_factor = 1.5

    tomogram_gt = Tomogram(
        images=images,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=ground_truth_alignment,
    )
    tomogram_gt._pad_factor = 1.5

    # fig, ax = plt.subplots(figsize=(5, 5))

    # viewer = napari.Viewer()
    # make interpolated series between alignments
    # grid_search = torch.linspace(-10, 10, 2)
    scores = list()
    for y in range(5):
        # interpolate translation and set in object
        # if single_index is not None:
        #     interpolated_alignment = xcorr_alignment.clone()
        #     interpolated_alignment[single_index][1] += y_grid
        # only interpolate x shift
        # interpolated_alignment[single_index][1] = xcorr_alignment[
        # single_index][1] * (
        #     1 - factor
        # ) + (ground_truth_alignment[single_index][1] * factor)
        # else:
        #     interpolated_alignment = (
        #         xcorr_alignment * (1 - factor) + ground_truth_alignment * factor
        #     )
        tomogram.sample_translations = xcorr_alignment + torch.normal(
            0, 3, xcorr_alignment.shape
        )

        # volume = tomogram.reconstruct_tomogram((128, 512, 512), 128)
        # mean, std = torch.mean(volume), torch.std(volume)
        # volume = (volume - mean) / std
        # with torch.no_grad():
        #     score = model(einops.rearrange(volume, "d h w -> 1 1 d h w"))
        # print(y_grid, score.squeeze())
        # scores += [score.item()]
        # abs_diff = torch.abs(ground_truth_alignment - tomogram.sample_translations)
        # ax.plot(
        #     abs_diff[:, 1],
        #     # color="tab:blue",
        #     # alpha=(factor.item() + 0.1) / 1.1,
        #     label=f"{y_grid.item():.2f}: {score.item():.3f}",
        # )

        ts, loss = evaluate_tilt_series(
            model,
            "model_0_" + str(y),
            tomogram,
            (1, 4, 4),
            128,
            (180, 512, 512),
            output_dir,
            tilt_series_ground_truth=tomogram_gt,
        )
        scores.append(loss.item())
        # viewer.add_image(volume.numpy(), name=str(factor))
    # napari.run()
    fig, ax = plt.subplots()
    ax.plot(scores)
    ax.set_xlabel("anneal index")
    ax.set_ylabel("loss after optimization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "loss_per_perturbation.png",
        dpi=300,
        bbox_inches="tight",
    )

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(grid_search, scores)
    # plt.show()
    #
    # idx = argmin(scores)
    # best_shift = grid_search[idx]
    # interpolated_alignment = xcorr_alignment.clone()
    # interpolated_alignment[single_index] += best_shift
    # tomogram.sample_translations = interpolated_alignment
    # tomogram.sample_translations = xcorr_alignment + torch.normal(
    #     0, 3, xcorr_alignment.shape
    # )
    #
    # tomogram_gt = Tomogram(
    #     images=images,
    #     tilt_angles=tilt_angles,
    #     tilt_axis_angle=tilt_axis_angle,
    #     sample_translations=ground_truth_alignment,
    # )
    #
    # ts, loss = evaluate_tilt_series(
    #     model,
    #     "model_0",
    #     tomogram,
    #     (1, 4, 4),
    #     128,
    #     (180, 512, 512),
    #     output_dir,
    #     tilt_series_ground_truth=tomogram_gt,
    # )
