# Alignment Optimization Visualization

This document describes how to visualize the L-BFGS optimization process during tilt-series alignment.

## Overview

The visualization tools capture intermediate states during alignment optimization, including:
- **Loss values** at each L-BFGS step
- **Precision values** from the model
- **Reconstructed subvolumes** at selected positions
- **Alignment parameters** (shifts) evolution
- **Score distributions** across the volume

This is useful for:
- Creating presentation-quality figures
- Understanding optimization behavior
- Debugging alignment issues
- Publishing scientific results

## Quick Start

### 1. Run Alignment with Tracking

Use the provided example script:

```bash
python examples/visualize_alignment_optimization.py \
    --model-checkpoint /path/to/model.ckpt \
    --tilt-series /path/to/series.xml \
    --output-dir /path/to/output \
    --device cuda:0
```

This will:
- Run alignment optimization
- Save step-by-step data to disk
- Generate basic visualization plots
- Create a text summary

### 2. Explore Results Interactively

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook examples/visualize_optimization_results.ipynb
```

The notebook provides:
- Publication-quality figure generation
- Interactive data exploration
- Subvolume comparison across steps
- Precision distribution analysis
- Data export to CSV

## Programmatic Usage

### Basic Example

```python
from pathlib import Path
from miss_alignment.alignment import (
    OptimizationTracker,
    optimize_shifts_with_tracking,
)
from miss_alignment.data.io import TiltSeriesData
from miss_alignment.models import MissAlignment

# Load model and data
model = MissAlignment.load_from_checkpoint("model.ckpt")
tilt_series_data = TiltSeriesData(xml_metadata_path=Path("series.xml"))
tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack()

# Create tracker
tracker = OptimizationTracker(
    output_dir=Path("optimization_tracking"),
    capture_frequency=1,  # Capture every step
    max_subvolumes_per_step=32,  # Save max 32 subvolumes per step
    save_subvolumes=True,  # Save subvolume data
)

# Run optimization with tracking
tilt_series, losses = optimize_shifts_with_tracking(
    model=model,
    tilt_series=tilt_series,
    images=images,
    pixel_size=pixel_size,
    positions=position_grid,
    tracker=tracker,
    setting="global",
    device="cuda:0",
)

# Summary is automatically saved
print(f"Captured {tracker.current_step} optimization steps")
```

### Loading and Analyzing Results

```python
from miss_alignment.alignment import load_optimization_data

# Load captured data
step_data = load_optimization_data(
    Path("optimization_tracking"),
    load_subvolumes=True,
)

# Access data from each step
for step in step_data:
    print(f"Step {step.step}:")
    print(f"  Loss: {step.loss}")
    print(f"  Mean precision: {step.mean_precision}")
    if step.subvolumes is not None:
        print(f"  Subvolumes shape: {step.subvolumes.shape}")
    if step.shifts_x is not None:
        print(f"  Shifts X: {step.shifts_x}")
```

## Output Files

The tracker saves data to the specified `output_dir`:

```
output_dir/
├── step_0000.pt          # Data from step 0
├── step_0001.pt          # Data from step 1
├── ...
├── summary.pt            # Summary statistics (losses, precisions)
└── summary.txt           # Human-readable summary
```

Each `step_XXXX.pt` file contains:
- `step`: Step number
- `loss`: Loss value
- `mean_precision`: Mean precision across volume
- `total_precision`: Sum of precisions
- `subvolumes`: Sampled reconstructions (shape: [n, d, h, w])
- `precisions`: Precision values (shape: [n])
- `scores`: Score values (shape: [n])
- `shifts_x`: X shifts in Angstroms (shape: [n_tilts])
- `shifts_y`: Y shifts in Angstroms (shape: [n_tilts])
- `positions`: 3D positions of subvolumes (shape: [n, 3])

## Configuration Options

### OptimizationTracker Parameters

- **`output_dir`**: Directory to save tracking data
- **`capture_frequency`** (default: 1): Capture detailed data every N steps
  - Set to 1 to capture every step
  - Set to higher values to save disk space
- **`max_subvolumes_per_step`** (default: 32): Maximum subvolumes to save per step
  - Reduces memory usage for large volumes
  - Samples uniformly across the volume
- **`save_subvolumes`** (default: True): Whether to save subvolume data
  - Set to False to only track losses/precisions (smaller file size)

### Example: Minimal Tracking (for large datasets)

```python
# Only track loss and precision, no subvolumes
tracker = OptimizationTracker(
    output_dir=Path("tracking"),
    capture_frequency=1,
    save_subvolumes=False,  # Don't save subvolumes
)
```

### Example: Sparse Tracking

```python
# Capture detailed data every 5 steps
tracker = OptimizationTracker(
    output_dir=Path("tracking"),
    capture_frequency=5,  # Only save subvolumes every 5 steps
    max_subvolumes_per_step=16,  # Fewer subvolumes
)
```

## Visualization Examples

### Loss and Precision Curves

```python
import matplotlib.pyplot as plt

steps = [s.step for s in step_data]
losses = [s.loss for s in step_data]
precisions = [s.mean_precision for s in step_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(steps, losses, 'o-')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Evolution')

ax2.plot(steps, precisions, 's-')
ax2.set_xlabel('Step')
ax2.set_ylabel('Mean Precision')
ax2.set_title('Precision Evolution')

plt.tight_layout()
plt.savefig('optimization_curves.png', dpi=300)
```

### Subvolume Comparison

```python
# Compare first and last step
initial = step_data[0]
final = step_data[-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Middle Z-slice of first subvolume
z_slice = 48
ax1.imshow(initial.subvolumes[0, z_slice, :, :], cmap='gray')
ax1.set_title(f'Initial (Loss: {initial.loss:.4f})')
ax1.axis('off')

ax2.imshow(final.subvolumes[0, z_slice, :, :], cmap='gray')
ax2.set_title(f'Final (Loss: {final.loss:.4f})')
ax2.axis('off')

plt.tight_layout()
plt.savefig('subvolume_comparison.png', dpi=300)
```

### Shift Evolution

```python
import torch

# Extract shifts from all steps
shifts_x = torch.stack([s.shifts_x for s in step_data], dim=0)
shifts_y = torch.stack([s.shifts_y for s in step_data], dim=0)

# Compute magnitude
shift_magnitude = torch.sqrt(shifts_x**2 + shifts_y**2)

# Plot mean shift evolution
plt.figure(figsize=(10, 5))
plt.plot(steps, shift_magnitude.mean(dim=1), linewidth=2)
plt.xlabel('Step')
plt.ylabel('Mean Shift Magnitude (Angstroms)')
plt.title('Alignment Convergence')
plt.grid(True, alpha=0.3)
plt.savefig('shift_evolution.png', dpi=300)
```

## Performance Considerations

### Disk Space

Each step file size depends on:
- Number of subvolumes saved (`max_subvolumes_per_step`)
- Patch size (e.g., 96³ = ~3.5 MB per subvolume in float32)
- Whether subvolumes are saved (`save_subvolumes`)

**Example:** With default settings (32 subvolumes, 96³ patches):
- ~112 MB per step with subvolumes
- ~1 KB per step without subvolumes

For a typical optimization (10-30 steps), expect:
- With subvolumes: 1-3 GB
- Without subvolumes: <100 KB

### Memory Usage

Subvolumes are saved to disk incrementally, so memory usage is minimal. The tracker only holds one step's data in memory at a time.

### I/O Performance

To minimize I/O overhead:
- Use `capture_frequency > 1` for very large volumes
- Set `save_subvolumes=False` if you only need loss/precision tracking
- Save to fast storage (SSD) when possible

## Integration with Existing Code

The visualization tools are designed to be non-invasive. The standard `optimize_shifts` function remains unchanged. Use `optimize_shifts_with_tracking` when you want visualization.

```python
# Standard alignment (no visualization)
from miss_alignment.alignment.optimize_global import optimize_shifts

tilt_series, losses = optimize_shifts(
    model, tilt_series, images, pixel_size, positions, device="cuda:0"
)

# With visualization
from miss_alignment.alignment import (
    OptimizationTracker,
    optimize_shifts_with_tracking,
)

tracker = OptimizationTracker(output_dir=Path("tracking"))
tilt_series, losses = optimize_shifts_with_tracking(
    model, tilt_series, images, pixel_size, positions, tracker, device="cuda:0"
)
```

## For Paper Figures

The provided Jupyter notebook (`examples/visualize_optimization_results.ipynb`) includes a publication-quality figure template that combines:
- Loss evolution
- Precision evolution
- Precision distribution
- Subvolume comparison (initial, middle, final)
- Shift evolution

This figure is designed to be:
- High resolution (300 DPI)
- Properly labeled with subfigures (A, B, C, ...)
- Suitable for scientific publications
- Customizable for your specific needs

## Troubleshooting

### "No step data found"

Make sure you ran `optimize_shifts_with_tracking` and the `output_dir` contains `step_*.pt` files.

### "Out of memory"

Reduce `max_subvolumes_per_step` or set `save_subvolumes=False`.

### "Files too large"

Increase `capture_frequency` to save data less frequently, or disable subvolume saving.

### "Different number of steps than expected"

L-BFGS calls the closure multiple times per iteration (for line search). The number of steps = number of closure calls, which can be 5-20 per L-BFGS iteration.

## Citation

If you use this visualization functionality in your research, please cite the miss-alignment paper (when published).

## Contributing

If you create useful visualization utilities, consider contributing them back to the project!
