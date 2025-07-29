import pickle
import pathlib
import mrcfile
import torch
import einops
import napari
from torch_tomogram import Tomogram
from miss_alignment.models import MissAlignment

torch.set_printoptions(precision=3, sci_mode=False)

model_checkpoint = pathlib.Path(
    "/home/marten/data/experiments/shrec/006_training/lightning_logs"
    "/version_2/checkpoints/epoch=239-step=52320.ckpt"
)
model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
model.eval()

path_xcorr = pathlib.Path(
    "/home/marten/coding/miss-alignment/dataset_prep/shrec/rescaled/train"
)
path_gt = pathlib.Path(
    "/home/marten/coding/miss-alignment/dataset_prep/shrec/rescaled/ground_truth/train"
)

xcorr_files = list(path_xcorr.glob("*.pickle"))
gt_files = list(path_gt.glob("*.pickle"))

for gt, xcorr in zip(gt_files, xcorr_files):
    with open(gt, "rb") as fgt:
        data = pickle.load(fgt)
    images = mrcfile.read(path_xcorr / data["tilt_series"])
    images = torch.from_numpy(images)
    images -= einops.reduce(images, "tilt h w -> tilt 1 1", reduction="mean")
    images /= torch.std(images, dim=(-2, -1), keepdim=True)

    tilt_angles = data["tilt_angles"]
    tilt_axis_angle = data["tilt_axis_angle"]

    ground_truth_alignment = torch.tensor(data["sample_translations"])
    with open(xcorr, "rb") as fxcorr:
        data = pickle.load(fxcorr)
    xcorr_alignment = torch.tensor(data["sample_translations"])
    print(xcorr_alignment)

    viewer = napari.Viewer()
    # make interpolated series between alignments
    for factor in torch.linspace(0, 1, 10):
        interpolated_alignment = (
            (xcorr_alignment * (1 - factor)) + (ground_truth_alignment * factor)
        ) / 2

        tomogram = Tomogram(
            images=images,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=interpolated_alignment,
        )
        tomogram._pad_factor = 1.5
        # reconstruction = tomogram.reconstruct_tomogram((180, 512, 512), 512)
        volume = tomogram.reconstruct_tomogram((128,) * 3, 128)
        mean, std = torch.mean(volume), torch.std(volume)
        volume = (volume - mean) / std
        with torch.no_grad():
            score = model(einops.rearrange(volume, "d h w -> 1 1 d h w"))
        print(factor, score.squeeze())
        viewer.add_image(volume.numpy(), name=str(factor))
    napari.run()
