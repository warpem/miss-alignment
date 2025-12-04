"""
Preprocessing script for SHREC data.

This script:
1. Downloads SHREC data from Zenodo
2. Unzips the archives
3. Converts pickle files to JSON format for miss-alignment
4. Extracts tilt angles from XML metadata and writes .rawtlt files (with inverted angles)
"""

from pathlib import Path
import subprocess
import shutil
import os
import argparse

from miss_alignment.data.io import convert_pickle_to_json_helper
from warpylib import TiltSeries


# Configuration
ZENODO_ARCHIVE = "16574872"

# SHREC data parameters (based on rescaling from 5.0A to 10.0A)
STACK_PIXEL_SIZE = 10.0  # Angstroms
ORIGINAL_PIXEL_SIZE = 10.0  # Angstroms (after rescaling)
ORIGINAL_STACK_SHAPE = (512, 512)  # width, height (adjust based on actual data)
VOLUME_SHAPE = (512, 512, 180)  # x, y, z


def download_and_unzip(download_dir: Path):
    """Download SHREC data from Zenodo and unzip archives."""
    print(f"Creating download directory: {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    ground_truth_dir = download_dir / "ground_truth"
    iter0_dir = download_dir / "iter0"
    if ground_truth_dir.exists() and iter0_dir.exists():
        print("Data already downloaded and extracted, skipping download step")
        return

    print(f"Downloading from Zenodo archive {ZENODO_ARCHIVE}...")
    subprocess.run(
        [
            "zenodo_get",
            ZENODO_ARCHIVE,
            "--output-dir",
            str(download_dir),
        ],
        check=True,
    )

    print("Unzipping archives...")
    for archive in ("torch_tiltxcorr.zip", "ground_truth.zip"):
        zipped_archive = download_dir / archive
        if zipped_archive.exists():
            print(f"  Extracting {archive}...")
            shutil.unpack_archive(
                zipped_archive,
                extract_dir=download_dir,
            )
            os.remove(zipped_archive)
            print(f"  Removed {archive}")
        else:
            print(f"  Warning: {archive} not found, skipping")


def convert_pickles_to_json(download_dir: Path):
    """Convert pickle files to JSON format for both ground_truth and torch_tiltxcorr."""

    # Process ground_truth pickles
    ground_truth_dir = download_dir / "ground_truth"
    if ground_truth_dir.exists():
        print(f"\nConverting ground_truth pickles from {ground_truth_dir}...")
        pickle_files = sorted(ground_truth_dir.glob("*.pickle"))
        print(f"Found {len(pickle_files)} pickle files")

        for pickle_file in pickle_files:
            print(f"  Converting {pickle_file.name}...")
            try:
                tilt_series_data, json_path = convert_pickle_to_json_helper(
                    pickle_data_path=pickle_file,
                    stack_pixel_size=STACK_PIXEL_SIZE,
                    original_pixel_size=ORIGINAL_PIXEL_SIZE,
                    original_stack_shape=ORIGINAL_STACK_SHAPE,
                    volume_shape=VOLUME_SHAPE,
                )
                print(f"    Created {json_path.name}")

                # Load XML metadata and extract angles
                tilt_series = TiltSeries(tilt_series_data.xml_metadata_path)
                angles = tilt_series.angles.cpu().numpy()

                # Write .rawtlt file with inverted angles
                rawtlt_path = tilt_series_data.xml_metadata_path.with_suffix('.rawtlt')
                with open(rawtlt_path, 'w') as f:
                    for angle in angles:
                        f.write(f"{angle}\n")
                print(f"    Created {rawtlt_path.name}")
            except Exception as e:
                print(f"    Error converting {pickle_file.name}: {e}")
    else:
        print(f"Warning: ground_truth directory not found at {ground_truth_dir}")

    # Process torch_tiltxcorr pickles (extracts to iter0 directory)
    tiltxcorr_dir = download_dir / "iter0"
    if tiltxcorr_dir.exists():
        print(f"\nConverting iter0 pickles from {tiltxcorr_dir}...")
        pickle_files = sorted(tiltxcorr_dir.glob("*.pickle"))
        print(f"Found {len(pickle_files)} pickle files")

        for pickle_file in pickle_files:
            print(f"  Converting {pickle_file.name}...")
            try:
                tilt_series_data, json_path = convert_pickle_to_json_helper(
                    pickle_data_path=pickle_file,
                    stack_pixel_size=STACK_PIXEL_SIZE,
                    original_pixel_size=ORIGINAL_PIXEL_SIZE,
                    original_stack_shape=ORIGINAL_STACK_SHAPE,
                    volume_shape=VOLUME_SHAPE,
                )
                print(f"    Created {json_path.name}")

                # Load XML metadata and extract angles
                tilt_series = TiltSeries(tilt_series_data.xml_metadata_path)
                angles = tilt_series.angles.cpu().numpy()

                # Write .rawtlt file with inverted angles
                rawtlt_path = tilt_series_data.xml_metadata_path.with_suffix('.rawtlt')
                with open(rawtlt_path, 'w') as f:
                    for angle in angles:
                        f.write(f"{angle}\n")
                print(f"    Created {rawtlt_path.name}")
            except Exception as e:
                print(f"    Error converting {pickle_file.name}: {e}")
    else:
        print(f"Warning: iter0 directory not found at {tiltxcorr_dir}")


def main():
    """Run the full preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess SHREC data: download from Zenodo, unzip, and convert to JSON."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        required=True,
        help="Directory to download and extract SHREC data",
    )
    args = parser.parse_args()

    download_dir = args.download_dir

    print("=" * 70)
    print("SHREC Data Preprocessing")
    print("=" * 70)

    # Step 1: Download and unzip
    print("\nStep 1: Downloading and unzipping data...")
    download_and_unzip(download_dir)

    # Step 2: Convert pickles to JSON
    print("\nStep 2: Converting pickles to JSON format...")
    convert_pickles_to_json(download_dir)

    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print(f"Data location: {download_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
