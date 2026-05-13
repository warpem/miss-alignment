## Running with WarpTools

### 1. Initial alignment

miss-alignment works best when started from an initial patch-tracking alignment produced by eTomo. The WarpTools commands for this are available as a [gist](https://gist.github.com/McHaillet/74596b3bea760001fd253de933baafe6). Patch tracking followed by the `autolevel` command gives solid starting alignments. You may need to adjust the patch size depending on your pixel size — a value of 1000 works well at 1.7 Å/px, while the default of 500 suits 1.0 Å/px data.

### 2. Update Warp XML attributes

Before running miss-alignment, two attributes of the Warp XML files need to be updated (this step may become unnecessary in future Warp releases). A helper script is available as a [gist](https://gist.github.com/McHaillet/117b321f504ac54d2f082bbe9bb01f16). Copy it into your `warp_tiltseries/` folder and update the tomogram shape, image shape, and pixel size at the top to match your dataset. The tomogram shape should tightly fit your sample to avoid training on empty regions — when samples vary in thickness, use the thickest one as the reference (similar to AreTomo).

Then run:
```
conda activate miss-alignment
cd /path/to/warp_tiltseries/
python update_warp_xml.py
```

### 3. Configure miss-alignment

Place a miss-alignment config file in the `warp_tiltseries/` directory — use [config_template.yaml](config_template.yaml) as a starting point. Key settings to update:

- **`training_directory`**: set to `/path/to/your/warp/project/warp_tiltseries/`
- **`batch_size`** (in the `tilt_series_alignment` section): controls how many patches are reconstructed simultaneously during alignment. A value of 32 works well for 24 GB cards; reduce it for smaller cards or increase it for larger ones to improve throughput.

### 4. Run miss-alignment

With 4 GPUs (12–24 GB VRAM each, e.g. RTX 3080/3090/4090):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 miss-alignment --config-file /path/to/config.yaml --training-devices 0,1 --reconstruction-devices 2,2,2,3,3,3 --dataloaders-per-trainer 5 --start-at-iteration 0 --prepare-stacks 10.0
```
With a single large GPU (≥40 GB VRAM, e.g. A100 40 GB), since training and reconstruction workers share the same device:
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 miss-alignment --config-file /path/to/config.yaml --training-devices 0 --reconstruction-devices 0,0,0 --dataloaders-per-trainer 5 --start-at-iteration 0 --prepare-stacks 10.0
```
These are just two examples — `--training-devices` and `--reconstruction-devices` can be freely mixed and matched to make the best use of whatever GPUs are available on your system.

If the run is interrupted, it can be resumed at any iteration with `--start-at-iteration N` (counting from 0).

### 5. Post-processing

After miss-alignment finishes, update the CTF parameters in WarpTools (`ts_ctf`) and then reconstruct the tomograms (`ts_reconstruct`) to evaluate the results.

## Running on SLURM

### Single-node requirement

miss-alignment uses `LOCAL_RANK` (set by Lightning's process launcher) to identify
the main process. This means **all GPUs must be on a single node**. Multi-node
distributed training is not supported. Your submission script must request all GPUs
on one node.

### Example submission script

```bash
#!/bin/bash
#SBATCH --job-name=miss-alignment
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=miss-alignment_%j.log
#SBATCH --error=miss-alignment_%j.err

# Activate environment (adjust to your cluster's setup)
module purge
module load miniforge  # or: ml miss-alignment
conda activate miss-alignment

# Avoid thread oversubscription from OpenMP/MKL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Point TMPDIR to a local scratch with sufficient space (the reconstruction pool
# writes up to pool_size pickle files; at default settings ~1000 × ~2 MB ≈ 2 GB).
# Uncomment and adjust if /tmp is too small on your cluster:
# export TMPDIR=/scratch/$SLURM_JOB_ID

miss-alignment \
    --config-file config.yaml \
    --training-devices 0,1 \
    --reconstruction-devices 2,2,2,3,3,3 \
    --dataloaders-per-trainer 5 \
    --start-at-iteration 0 \
    --prepare-stacks 10.0
```

### Key SLURM settings

| Setting | Why |
|---|---|
| `--ntasks=1` | Required. miss-alignment manages its own subprocesses; SLURM should only start one task. |
| `--nodes=1` | Required. All GPUs must be on a single node (see above). |
| `--cpus-per-task` | Set to `len(--reconstruction-devices) + n_training_devices × dataloaders_per_trainer`. Each reconstruction worker and each DataLoader worker (spawned per DDP rank) needs a CPU. For the example above: 6 + 2×5 = 16. |
| `--gres=gpu:N` | Request all GPUs you intend to use for training + reconstruction. |

### Notes

- **Lightning srun warning**: Lightning may warn that `srun` is available but not used.
  This warning can be safely ignored — do not prepend `srun` to the command, as that
  would conflict with miss-alignment's own process management.

- **Temporary storage**: The reconstruction pool is written to `$TMPDIR` (defaulting to
  `/tmp`). On clusters where `/tmp` is shared or small, set `TMPDIR` to a local scratch
  directory with at least a few GB of free space before running.

- **Resuming**: If a job times out mid-run, restart with `--start-at-iteration N` where
  `N` is the last completed iteration (counting from 0).
