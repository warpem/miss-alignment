from pathlib import Path
import pickle

import napari
import einops
import mrcfile
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_tiltxcorr import tiltxcorr

from torch_tomogram import Tomogram


def _read_shrec_alignment(file: Path) -> tuple[list]:
    with open(model_folder / "alignment_simulated.txt") as aln:
        lines = [x.split() for x in aln.readlines() if not x.startswith("#")]
        data = zip(*[(float(x[0]), float(x[1]), float(x[2])) for x in lines])
    return data


def _save_tomo_to_folder(data: Tomogram, folder: Path, name: str) -> None:
    tilt_series_mrc = name + ".mrc"
    metadata = name + ".pickle"
    with mrcfile.new(folder / tilt_series_mrc, overwrite=True) as mrc:
        mrc.set_data(data.images.numpy())
        mrc.voxel_size = 10
    data_dict = {
        "tilt_series": tilt_series_mrc,
        "tilt_angles": data.tilt_angles.tolist(),
        "tilt_axis_angle": data.tilt_axis_angle.tolist(),
        "sample_translations": data.sample_translations.tolist(),
    }
    with open(folder / metadata, "wb") as out:
        pickle.dump(data_dict, out)


if __name__ == "__main__":
    train_folder = Path("rescaled/train")
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder = Path("rescaled/test")
    test_folder.mkdir(parents=True, exist_ok=True)
    reconstruct_and_show = False

    for i in range(10):
        model_folder = Path(  # downloaded from DataverseNL
            f"shrec21_full_dataset_no_mirroring/model_{i}"
        )
        with mrcfile.open(
            model_folder / "projections_unbinned.mrc"
            if i != 9
            else model_folder / "projections_unbinne.mrc",
            permissive=True,
        ) as mrc:
            tilt_series = torch.tensor(mrc.data)
        tilt_series, _ = fourier_rescale_2d(tilt_series, 5.0, 10.0)
        tilt_series -= einops.reduce(
            tilt_series, "tilt h w -> tilt 1 1", reduction="mean"
        )
        tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
        x_shift, y_shift, tilt_angles = _read_shrec_alignment(
            model_folder / "alignment_simulated.txt"
        )
        n_tilts = len(tilt_angles)
        tilt_axis_angle = 0.0
        # invert to match reconstruction with SHREC grand_model
        tilt_angles = list(reversed(tilt_angles))
        # divide by two for 2x downsampling of tilt-series
        # invert the shifts because we employ a forward projection model!
        shifts = -1 * torch.tensor([y_shift, x_shift]).T / 2

        # also multiply with -1 for forward-projection model
        xcorr_shifts = -1 * tiltxcorr(
            tilt_series,
            tilt_angles,
            tilt_axis_angle,
            low_pass_cutoff=0.5,
        )
        xcorr_tomogram = Tomogram(
            images=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=xcorr_shifts,
        )  # invert the shifts because we employ a forward projection model!

        with mrcfile.open(model_folder / "grandmodel.mrc", permissive=True) as mrc:
            ground_truth = mrc.data

        _save_tomo_to_folder(
            xcorr_tomogram, test_folder if i > 6 else train_folder, f"model_{i}"
        )

        if reconstruct_and_show:
            # 180 is the box size of the grand model
            xcorr_volume = xcorr_tomogram.reconstruct_tomogram((180, 512, 512), 128)

            tomogram = Tomogram(
                images=tilt_series,
                tilt_angles=tilt_angles,
                tilt_axis_angle=tilt_axis_angle,
                sample_translations=shifts,
            )

            # 180 is the box size of the grand model
            volume = tomogram.reconstruct_tomogram((180, 512, 512), 128)

            viewer = napari.Viewer()
            viewer.add_image(ground_truth, name="ground_truth")
            viewer.add_image(volume.numpy(), name="reconstruction")
            viewer.add_image(xcorr_volume.numpy(), name="reconstruction xcorr")
            napari.run()
