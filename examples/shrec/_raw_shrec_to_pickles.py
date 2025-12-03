"""
This script describes the conversion of the SHREC dataset as present on
DataverseNL (https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA),
specifically shrec21_full_dataset_no_mirroring.zip .

The following was done to the tilt-series:
* downsample from 10A to 5A with Fourier cropping
* coarse cross-correlation based alignment with torch-tiltxcorr
* saving the data as .pickle files (a deprecated format of miss-alignment)
  - both a ground_truth file and a tiltxcorr file

The converted data were uploaded to Zenodo at: https://zenodo.org/records/16574872
how-to-run.md describes processing from the Zenodo archive
"""
from pathlib import Path

import napari
import einops
import mrcfile
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_tiltxcorr import tiltxcorr
from torch_tomogram import Tomogram

from miss_alignment.data.io import save_tomogram_to_pickle


def _read_shrec_alignment(file: Path) -> tuple[list]:
    with open(model_folder / "alignment_simulated.txt") as aln:
        lines = [x.split() for x in aln.readlines() if not x.startswith("#")]
        data = zip(*[(float(x[0]), float(x[1]), float(x[2])) for x in lines])
    return data


if __name__ == "__main__":
    ground_truth_folder = Path(
        "../../dataset_prep/shrec/rescaled/ground_truth")
    ground_truth_folder.mkdir(parents=True, exist_ok=True)
    xcorr_alignment_folder = Path(
        "../../dataset_prep/shrec/rescaled/xcorr_alignment/")
    xcorr_alignment_folder.mkdir(parents=True, exist_ok=True)
    reconstruct_and_show = False

    for i in range(10):
        print(f"running shrec model {i}")
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
            low_pass_cutoff=0.25,
        )
        xcorr_tomogram = Tomogram(
            images=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=xcorr_shifts,
        )  # invert the shifts because we employ a forward projection model!

        save_tomogram_to_pickle(
            xcorr_tomogram, xcorr_alignment_folder / f"model_{i}.pickle"
        )

        ground_truth_tomogram = Tomogram(
            images=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=shifts,
        )

        save_tomogram_to_pickle(
            ground_truth_tomogram, ground_truth_folder / f"model_{i}.pickle"
        )

        if reconstruct_and_show:
            with mrcfile.open(model_folder / "grandmodel.mrc", permissive=True) as mrc:
                ground_truth = mrc.data

            # 180 is the box size of the grand model
            xcorr_volume = xcorr_tomogram.reconstruct_tomogram((180, 512, 512), 128)

            # 180 is the box size of the grand model
            volume = ground_truth_tomogram.reconstruct_tomogram((180, 512, 512), 128)
            mrcfile.write(
                ground_truth_folder / f"model_{i}_reconstruction.mrc",
                volume.numpy(),
                voxel_size=10,
                overwrite=True,
            )

            viewer = napari.Viewer()
            viewer.add_image(ground_truth, name="grand model")
            viewer.add_image(volume.numpy() * -1, name="reconstruction ground_truth")
            viewer.add_image(
                xcorr_volume.numpy() * -1, name="reconstruction torch-tiltxcorr"
            )
            napari.run()
