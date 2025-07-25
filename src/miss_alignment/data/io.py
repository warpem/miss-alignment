import pickle
from torch_tomogram import Tomogram
from pathlib import Path


def save_tomogram(data: Tomogram, save_path: Path) -> None:
    if save_path.suffix != ".pickle":
        raise ValueError("save_path must end with .pickle")
    if 'cpu' != str(data.device):
        raise ValueError("the Tomogram data should be on CPU for saving")
    data_dict = {
        "tilt_series": data.images,
        "tilt_angles": data.tilt_angles,
        "tilt_axis_angle": data.tilt_axis_angle,
        "sample_translations": data.sample_translations,
    }
    with open(save_path, "wb") as outfile:
        pickle.dump(data_dict, outfile)


def read_tomogram(save_path: Path) -> Tomogram:
    with open(save_path, "rb") as infile:
        data_dict = pickle.load(infile)
    tomogram = Tomogram(
        tilt_angles=data_dict["tilt_angles"],
        tilt_axis_angle=data_dict["tilt_axis_angle"],
        sample_translations=data_dict["sample_translations"],
        images=data_dict["tilt_series"],
    )
    return tomogram
