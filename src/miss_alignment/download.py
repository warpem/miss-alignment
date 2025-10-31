from pathlib import Path
import subprocess
import shutil
import os


_zenodo_archive = "16574872"

def _download_to_dir(dataset_directory: Path):
    subprocess.run(
        [
            "zenodo_get",
            _zenodo_archive,  # update value
            "--output-dir",
            str(dataset_directory),
        ]
    )
    for archive in ("torch_tiltxcorr.zip", "ground_truth.zip"):
        zipped_archive = dataset_directory / archive
        shutil.unpack_archive(
            zipped_archive,
            extract_dir=dataset_directory,
        )
        os.remove(zipped_archive)