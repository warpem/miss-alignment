# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

miss-alignment is a deep learning package for tilt-series alignment in cryo-electron tomography. It uses a contrastive learning approach with PyTorch and PyTorch Lightning to iteratively train 3D convolutional neural networks that optimize tilt-series alignment by minimizing shift artifacts in reconstructions.

**Key concept**: The system alternates between (1) training a model to score reconstruction quality and (2) using that model to optimize tilt-series alignment parameters through gradient descent.

### Scientific publication

This software is a research project. It will be written up in a manuscript. Therefore, consider for anything you write in the paper/ folder that it should adhere to scientific principles with an emphasis on clarity and simplicity.

## Environment Setup

This project has specific CUDA and PyTorch dependencies. Use the conda environment setup from README.md:

```bash
conda create -n miss-alignment -c conda-forge python=3.11 cuda-toolkit=12.9 -y
conda activate miss-alignment
```

Install with: `python -m pip install -e .[dev,test]`

## Common Commands

### Testing
```bash
# Run all tests with coverage
pytest --color=yes --cov --cov-report=xml --cov-report=term-missing

# Run tests in a specific directory
pytest tests/alignment/

# Run a single test file
pytest tests/data/test_training_dataset.py

# Run a specific test function
pytest tests/test_miss_alignment.py::test_something
```

### Linting
```bash
# Run ruff linter with auto-fix
ruff check --fix

# Run ruff formatter
ruff format

# Run pre-commit hooks manually
pre-commit run --all-files
```

### CLI Usage
The package provides a `miss-alignment` CLI command:
```bash
# Train a model (primary workflow)
miss-alignment train --config-file config.yaml

# Download training data
miss-alignment download-training-data --dataset-directory /path/to/data
```

### Development
```bash
# Build the package
python -m build

# Install in editable mode with dependencies
python -m pip install -e .[dev,test]
```

## Architecture

### Core Modules

1. **`models/`** - Neural network architectures
   - `MissAlignment`: Main PyTorch Lightning module that wraps the 3D CNN
   - Model variants: `Compact3DConvNet` (default), `Compact3DConvNetGELU`, `Compact3DConvNetWide`, `Compact3DConvNetDeep`, `CompactResNet3D`
   - Uses `TripletMarginRankingLoss` for contrastive learning (comparing aligned vs. misaligned reconstructions)

2. **`data/`** - Data loading and synthetic shift generation
   - `MissAlignmentDataModule`: PyTorch Lightning data module managing the reconstruction pool
   - `ReconstructionPoolDataset`: Consumes from a pool of pre-computed 3D patches
   - `shift_generation.py`: Creates synthetic alignment errors (trajectories, jitter, outliers, fractures) for training data
   - `_reconstruction_worker.py`: Multiprocessing workers that generate 3D reconstruction patches in parallel
   - **Architecture note**: Uses a producer-consumer pattern where reconstruction workers populate a temporary pool directory that the dataloader consumes from

3. **`alignment/`** - Tilt-series alignment optimization
   - `tilt_series.py`: Core optimization logic using gradient descent on shift parameters
   - `parallel.py`: Distributes alignment across multiple GPU devices
   - `correlation.py`: Traditional correlation-based alignment methods
   - **Key functions**: `optimize_shifts()` supports three modes:
     - `"global"`: Single shift per image
     - `(3, 3, 41)`: 2D warping field per image
     - `(3, 3, 2, 10)`: 3D volume warp grid

4. **`train.py`** - Iterative training loop
   - Alternates between model training and tilt-series alignment
   - Each iteration: train model → align tilt-series → use aligned data for next iteration
   - Configured via YAML file (see `config_template.yaml`)

### Dependencies on External Libraries

- **`warpylib`**: Provides `TiltSeries` and `CubicGrid` for tilt-series geometry and warping
- **`torch-fourier-*`**: Suite of PyTorch-based Fourier transform utilities (rescale, slice, shift, filter)
- **`torch-tiltxcorr`**: Cross-correlation utilities for tilt-series
- **`torch-grid-utils`** and **`torch-cubic-spline-grids`**: Grid manipulation utilities

### Configuration System

Training is configured via YAML files (template: `src/miss_alignment/config_template.yaml`):
- **`general`**: Training directory, CTF settings, iteration-specific parameters (downsample, alignment mode)
- **`model_training`**: Architecture selection, learning rate, loss margin, weight decay, scheduler
- **`data_loading`**: Batch size, patch size, steps per epoch
- **`shift_generation`**: Probabilities and magnitudes for synthetic shifts (trajectories, jitter, outliers, fractures)
- **`tilt_series_alignment`**: Patch size, overlap, batch size for alignment

### Data Flow

1. **Training**: Dataset JSON files (`.json`) → Reconstruction workers → Pool directory (`.pickle` patches triplet) → DataLoader → Model
2. **Alignment**: Trained model checkpoint → Load tilt-series data → Gradient-based optimization → Output aligned parameters (`.json`)

## Working with Tilt-Series Data

This project uses `warpylib.TiltSeries` as the primary representation of tilt-series metadata. Data is stored in JSON files that reference both the metadata (`.xml`) and image stack (`.st` or `.mrc`).

**Note**: `torch-tomogram` is a dependency only for backward compatibility. Use `warpylib` for all tilt-series operations.

### Loading Tilt-Series Data

```python
from pathlib import Path
from miss_alignment.data.io import TiltSeriesData

# Load metadata from JSON file
tilt_series_data = TiltSeriesData.from_json(Path("path/to/data.json"))

# Load the actual TiltSeries object, images, and pixel size
tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
    downsample=1  # optional downsampling factor
)

# Access tilt-series metadata (all torch tensors)
angles = tilt_series.angles  # tilt angles in degrees
tilt_axis_angles = tilt_series.tilt_axis_angles  # tilt axis rotation per image
tilt_axis_offset_x = tilt_series.tilt_axis_offset_x  # X shifts in Angstroms
tilt_axis_offset_y = tilt_series.tilt_axis_offset_y  # Y shifts in Angstroms
```

### TiltSeries Key Attributes

The `warpylib.TiltSeries` object contains:
- **`angles`**: Tilt angles in degrees (shape: `[n_tilts]`)
- **`tilt_axis_angles`**: Rotation of tilt axis for each image in degrees (shape: `[n_tilts]`)
- **`tilt_axis_offset_x`**: X-axis shifts in Angstroms (shape: `[n_tilts]`)
- **`tilt_axis_offset_y`**: Y-axis shifts in Angstroms (shape: `[n_tilts]`)
- **`image_dimensions_physical`**: Physical image dimensions in Angstroms (shape: `[2]`)
- **`volume_dimensions_physical`**: Physical volume dimensions in Angstroms (shape: `[3]`)

All attributes are `torch.Tensor` objects.

### Performing Reconstructions

```python
import torch
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data

# Preprocess images (normalize, optionally invert)
images = preprocess_tilt_data(
    tilt_data=images,
    normalize=True,
    invert=False,
    subvolume_size=64,  # patch size for filter calculation
)

# Define reconstruction position (in Angstroms, in volume coordinate system)
reconstruction_location = torch.tensor([[x, y, z]], device="cuda")

# Perform reconstruction at a specific location
patch_size = 64  # size of output volume (cubic)
reconstruction = tilt_series.reconstruct_subvolumes_single(
    tilt_data=images,
    coords=reconstruction_location,  # shape: (1, 3) in Angstroms
    pixel_size=pixel_size,
    size=patch_size,
    apply_ctf=True,  # apply CTF correction
    angles=torch.tensor([0.0, 0.0, 0.0]),  # optional rotation (ZYZ Euler)
    oversampling=2.0,  # reconstruction oversampling factor
)
# Output shape: (1, patch_size, patch_size, patch_size)
```

### Modifying Alignment Parameters

```python
# Modify shifts (all operations in Angstroms)
tilt_series.tilt_axis_offset_x += shift_x  # add X shifts
tilt_series.tilt_axis_offset_y += shift_y  # add Y shifts

# Modify angles
tilt_series.angles = new_angles  # set new tilt angles
tilt_series.tilt_axis_angles = new_tilt_axis_angles  # set tilt axis rotation
```

### Saving Updated Metadata

```python
# Save updated TiltSeries metadata back to XML
tilt_series_data.save_metadata_to_xml(tilt_series)

# Or save to a new JSON file
new_tilt_series_data = tilt_series_data.replace(
    xml_metadata_path=Path("path/to/new_metadata.xml")
)
new_tilt_series_data.save_metadata_to_xml(tilt_series)
new_tilt_series_data.to_json(Path("path/to/new_data.json"))
```

### Adding Synthetic Shifts for Testing

```python
from miss_alignment.data.shift_generation import (
    JitterGenerator,
    TrajectoryGenerator,
    OutlierGenerator,
    FractureGenerator,
)

# Create shift generators
n_tilts = len(tilt_series.angles)
device = tilt_series.angles.device

# Generate shifts (returned in pixels, shape: [n_tilts, 3] for ZYX)
jitter_shifts = JitterGenerator(jitter_max_std=2.0)(n_tilts, device)
trajectory_shifts = TrajectoryGenerator(trajectory_max_shift=10.0)(n_tilts, device)
outlier_shifts = OutlierGenerator(outlier_max_shift=20, max_sequence_length=3)(n_tilts, device)
fracture_shifts = FractureGenerator(fracture_max_shift=30)(n_tilts, device)

# Convert to 2D shifts and apply (need projection matrices)
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from torch_affine_utils.transforms_3d import Ry, Rz

r0 = Ry(-tilt_series.angles, zyx=True)
r1 = Rz(tilt_series.tilt_axis_angles, zyx=True)
rotation_matrices = r1 @ r0
projection_matrices = rotation_matrices[..., 1:3, :3]

# Project 3D shifts to 2D (shape: [n_tilts, 2] for YX)
shifts_2d = project_shifts_3d_to_2d(jitter_shifts, projection_matrices)
shifts_angstrom = shifts_2d * pixel_size

# Apply shifts
tilt_series.tilt_axis_offset_y += shifts_angstrom[:, 0]
tilt_series.tilt_axis_offset_x += shifts_angstrom[:, 1]
```

## Development Notes

- **Python version**: Requires Python ≥3.10 (tested primarily on 3.11)
- **Linting**: Uses `ruff` with line length 88, ignores E712 (comparisons with False for readability)
- **Pre-commit**: Configured with ruff linter and formatter
- **Testing**: Uses `pytest` with coverage reporting; warnings treated as errors
- **Multiprocessing**: Reconstruction workers use multiprocessing with device assignment via `reconstruction_accelerators` parameter
- **Temporary files**: `MissAlignmentDataModule` creates a temporary pool directory (`RECON_POOL_SIZE` env var controls size, default 1000)
- **Package manager**: Currently uses conda/pip for environment installation -> would like to move to uv in the future

## Bug Reporting and Fixes

When working on this codebase, if you encounter potential bugs or implementation issues:

1. **Report the issue**: Clearly explain what you found and why it appears to be a bug
2. **Suggest a fix**: Provide a proposed solution with reasoning
3. **Implement if appropriate**: For clear bugs (e.g., missing conversions, type inconsistencies, logic errors), implement the fix
4. **Let the maintainer decide**: The maintainer will evaluate whether it's truly a bug or was intentional design

**Examples of bugs to report and fix:**
- Missing data type conversions (e.g., forgetting to convert a list back to tuple in deserialization)
- Inconsistent behavior between related functions (e.g., `to_dict()` and `from_dict()` not being symmetric)
- Logic errors or off-by-one errors
- Missing error handling

**Don't just write tests to conform to buggy behavior** - if the implementation looks wrong, report it and suggest the fix rather than masking it with adjusted test expectations.

## File Organization

```
miss-alignment/
├── src/miss_alignment/
│   ├── models/          # Neural network architectures
│   ├── data/            # Data loading, shift generation, reconstruction workers
│   ├── alignment/       # Alignment optimization algorithms
│   ├── gradcam/         # Gradient-weighted Class Activation Mapping utilities
│   ├── train.py         # Main training loop
│   ├── _cli.py          # CLI setup with typer
│   └── config_template.yaml  # Configuration template
├── tests/               # Test suite (mirrors src structure)
├── examples/            # Example scripts for data processing and visualization
├── dataset_prep/        # Dataset preparation scripts
└── paper/               # Paper-related notebooks and evaluation
```

## Important Patterns

1. **Configuration-driven**: All training parameters are specified in YAML files
2. **Iterative refinement**: Training alternates between model training and alignment steps
3. **Multi-GPU support**: Both training and alignment can use multiple GPUs
4. **Synthetic data augmentation**: Training uses procedurally generated alignment errors
5. **Pool-based data loading**: Pre-computed reconstructions in a pool improve training throughput
