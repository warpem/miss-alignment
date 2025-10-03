import torch
from miss_alignment.data.io import read_tomogram_from_pickle
from miss_alignment.data._reconstruction_worker import _create_pool_reconstruction
from miss_alignment.data.shift_generation import create_default_generator

tilt_series = read_tomogram_from_pickle('/data/mchaillet/model_training/shrec/run2/iter0/model_0.pickle')
tilt_series.to('cuda')
tomogram_shape = (180, 512, 512)
patch_size = 96
#shift_generator = create_default_generator()
def generate_shifts(n_tilts: int, device: torch.device = 'cuda') -> torch.Tensor:
    # Create directly on target device
    return torch.randn(n_tilts, 3, device=device)
shift_generator = generate_shifts

# warm up module loading before actual profiling
_create_pool_reconstruction(tilt_series, tomogram_shape, patch_size, shift_generator)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
        _create_pool_reconstruction(tilt_series, tomogram_shape, patch_size, shift_generator)

print(prof.key_averages().table(sort_by="self_cuda_time_total", max_name_column_width=40))


# Enable detailed logging
torch.cuda.memory._record_memory_history()

# Run your code
_create_pool_reconstruction(tilt_series, tomogram_shape, patch_size, shift_generator)

# Check allocations
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze it
from torch.cuda._memory_viz import profile_plot

# This creates an HTML visualization
profile_plot(snapshot_file="memory_snapshot.pickle", output="memory_profile.html")
