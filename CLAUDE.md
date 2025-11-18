# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

miss-alignment is a deep learning package for tilt-series alignment in cryo-electron tomography. It uses a contrastive learning approach with PyTorch and PyTorch Lightning to iteratively train 3D convolutional neural networks that optimize tilt-series alignment by minimizing shift artifacts in reconstructions.

**Key concept**: The system alternates between (1) training a model to score reconstruction quality and (2) using that model to optimize tilt-series alignment parameters through gradient descent.

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
- **`torch-tomogram`**: Used for tomographic reconstruction operations
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
