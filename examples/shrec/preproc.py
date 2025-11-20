"""
Preprocessing script for SHREC data.

This script:
1. Downloads SHREC data from Zenodo
2. Unzips the archives
3. Converts pickle files to JSON format for miss-alignment
"""

from pathlib import Path
import subprocess
import shutil
import os

from miss_alignment.data.io import convert_pickle_to_json_helper


# Configuration
DOWNLOAD_DIR = Path("/home/marten/data/download/shrec_zenodo")
ZENODO_ARCHIVE = "16574872"

# SHREC data parameters (based on rescaling from 5.0A to 10.0A)
STACK_PIXEL_SIZE = 10.0  # Angstroms
ORIGINAL_PIXEL_SIZE = 10.0  # Angstroms (after rescaling)
ORIGINAL_STACK_SHAPE = (512, 512)  # width, height (adjust based on actual data)
VOLUME_SHAPE = (512, 512, 180)  # x, y, z


def download_and_unzip():
    """Download SHREC data from Zenodo and unzip archives."""
    print(f"Creating download directory: {DOWNLOAD_DIR}")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    ground_truth_dir = DOWNLOAD_DIR / "ground_truth"
    iter0_dir = DOWNLOAD_DIR / "iter0"
    if ground_truth_dir.exists() and iter0_dir.exists():
        print("Data already downloaded and extracted, skipping download step")
        return

    print(f"Downloading from Zenodo archive {ZENODO_ARCHIVE}...")
    subprocess.run(
        [
            "zenodo_get",
            ZENODO_ARCHIVE,
            "--output-dir",
            str(DOWNLOAD_DIR),
        ],
        check=True,
    )

    print("Unzipping archives...")
    for archive in ("torch_tiltxcorr.zip", "ground_truth.zip"):
        zipped_archive = DOWNLOAD_DIR / archive
        if zipped_archive.exists():
            print(f"  Extracting {archive}...")
            shutil.unpack_archive(
                zipped_archive,
                extract_dir=DOWNLOAD_DIR,
            )
            os.remove(zipped_archive)
            print(f"  Removed {archive}")
        else:
            print(f"  Warning: {archive} not found, skipping")


def convert_pickles_to_json():
    """Convert pickle files to JSON format for both ground_truth and torch_tiltxcorr."""

    # Process ground_truth pickles
    ground_truth_dir = DOWNLOAD_DIR / "ground_truth"
    if ground_truth_dir.exists():
        print(f"\nConverting ground_truth pickles from {ground_truth_dir}...")
        pickle_files = sorted(ground_truth_dir.glob("*.pickle"))
        print(f"Found {len(pickle_files)} pickle files")

        for pickle_file in pickle_files:
            print(f"  Converting {pickle_file.name}...")
            try:
                _, json_path = convert_pickle_to_json_helper(
                    pickle_data_path=pickle_file,
                    stack_pixel_size=STACK_PIXEL_SIZE,
                    original_pixel_size=ORIGINAL_PIXEL_SIZE,
                    original_stack_shape=ORIGINAL_STACK_SHAPE,
                    volume_shape=VOLUME_SHAPE,
                )
                print(f"    Created {json_path.name}")
            except Exception as e:
                print(f"    Error converting {pickle_file.name}: {e}")
    else:
        print(f"Warning: ground_truth directory not found at {ground_truth_dir}")

    # Process torch_tiltxcorr pickles (extracts to iter0 directory)
    tiltxcorr_dir = DOWNLOAD_DIR / "iter0"
    if tiltxcorr_dir.exists():
        print(f"\nConverting iter0 pickles from {tiltxcorr_dir}...")
        pickle_files = sorted(tiltxcorr_dir.glob("*.pickle"))
        print(f"Found {len(pickle_files)} pickle files")

        for pickle_file in pickle_files:
            print(f"  Converting {pickle_file.name}...")
            try:
                _, json_path = convert_pickle_to_json_helper(
                    pickle_data_path=pickle_file,
                    stack_pixel_size=STACK_PIXEL_SIZE,
                    original_pixel_size=ORIGINAL_PIXEL_SIZE,
                    original_stack_shape=ORIGINAL_STACK_SHAPE,
                    volume_shape=VOLUME_SHAPE,
                )
                print(f"    Created {json_path.name}")
            except Exception as e:
                print(f"    Error converting {pickle_file.name}: {e}")
    else:
        print(f"Warning: iter0 directory not found at {tiltxcorr_dir}")


def main():
    """Run the full preprocessing pipeline."""
    print("=" * 70)
    print("SHREC Data Preprocessing")
    print("=" * 70)

    # Step 1: Download and unzip
    print("\nStep 1: Downloading and unzipping data...")
    download_and_unzip()

    # Step 2: Convert pickles to JSON
    print("\nStep 2: Converting pickles to JSON format...")
    convert_pickles_to_json()

    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print(f"Data location: {DOWNLOAD_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
