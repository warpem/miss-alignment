## Running with WarpTools

### 1. Initial alignment

miss-alignment starts from an initially coarse aligned dataset. WarpTools provides some wrappers that simultaneously run coarse alignment and average the tilt axis angle over the whole dataset. Some WarpTools commands for this are available as a [gist](https://gist.github.com/McHaillet/74596b3bea760001fd253de933baafe6) using patch tracking in etomo (you can also use AreTomo). You may need to adjust the patch size depending on your pixel size — a value of 1000 works well at 1.7 Å/px, while the default of 500 suits 1.0 Å/px data. Usually, patch tracking followed by the `autolevel` command give solid starting alignments for MissAlignment to refine further. 

### 2. Update Warp XML attributes

Before running miss-alignment, two attributes of the Warp XML files need to be updated (this step may become unnecessary in future Warp releases). A helper script is available as a [gist](https://gist.github.com/McHaillet/117b321f504ac54d2f082bbe9bb01f16). Copy it into your `warp_tiltseries/` folder and update the tomogram shape, image shape, and pixel size at the top to match your dataset. The tomogram shape should tightly fit your sample in all three dimensions (X, Y, Z) to avoid training on empty regions — when samples vary in thickness, use the thickest one as the reference (similar to AreTomo).

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

miss-alignment launches its own per-GPU training processes (via
`torch.multiprocessing.spawn`) and exports `LOCAL_RANK` so Lightning attaches to that
process group instead of starting its own. These processes rendezvous over
`localhost`, so **all GPUs must be on a single node**. Multi-node distributed training
is not supported. Your submission script must request all GPUs on one node.

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
conda activate miss-alignment

# Avoid thread oversubscription from OpenMP/MKL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
| `--ntasks=1` | Required. miss-alignment is its own launcher — it spawns one training process per `--training-devices` entry. SLURM must start exactly **one** task. Do **not** use `srun --ntasks=N`, or SLURM will start N copies of the program that each spawn their own workers. |
| `--nodes=1` | Required. All GPUs must be on a single node (see above). |
| `--cpus-per-task` | Set to `len(--reconstruction-devices) + n_training_devices × dataloaders_per_trainer`. Each reconstruction worker and each DataLoader worker (spawned per DDP rank) needs a CPU. For the example above: 6 + 2×5 = 16. |
| `--gres=gpu:N` | Request all GPUs you intend to use for training + reconstruction. |

### Notes

- **Do not launch with `srun --ntasks=N`**: miss-alignment spawns its own per-GPU
  training processes, so the job must run as a single task (`--ntasks=1`). Launching
  multiple tasks would start independent copies that each spawn their own workers. Run
  the command directly — no `srun` prefix is needed. The cluster environment is pinned
  to Lightning's `LightningEnvironment`, so SLURM's `SLURM_PROCID`/`SLURM_NTASKS` are
  ignored and the "srun available but not used" warning is suppressed.

- **Temporary storage**: The reconstruction pool is written to `$TMPDIR` (defaulting to
  `/tmp`). On clusters where `/tmp` is shared or small, set `TMPDIR` to a local scratch
  directory with at least a few GB of free space before running.

- **Resuming**: If a job times out mid-run, restart with `--start-at-iteration N` where
  `N` is the last completed iteration (counting from 0).

- **Log verbosity**: miss-alignment logs at `WARNING` by default. Set
  `MISS_ALIGNMENT_LOG_LEVEL=DEBUG` (or `INFO`) to see its own diagnostic output,
  including from the spawned reconstruction workers. PyTorch Lightning's INFO startup
  banners stay suppressed regardless of this setting. This controls text logging only;
  the tqdm progress bars are always shown (they write to stdout independently).

- **`torch.compile` workers**: each training rank runs its own TorchInductor compile
  pool (default `min(32, n_cpus)` processes *per rank*), so multi-GPU runs can spawn a
  lot of idle compile workers. miss-alignment defaults `TORCHINDUCTOR_COMPILE_THREADS`
  to `available_cpus // n_training_devices` (respecting the SLURM `--cpus-per-task`
  cpuset via `sched_getaffinity`). Override it by exporting `TORCHINDUCTOR_COMPILE_THREADS`
  yourself.

## Troubleshooting

### NCCL hang with multiple GPUs on HPC servers

**Symptom**: training hangs and eventually crashes with a message like:

```
ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress
in monitoring enqueued collectives.
...
ProcessExitedException: process 1 terminated with signal SIGABRT
```

**Cause**: On servers with datacenter GPUs (L40S, A100, H100, etc.), multiple GPUs are
often split across two PCIe switches connected to different NUMA nodes. When two training
GPUs sit on different switches, data between them has to cross the CPU's PCIe root
complex — a path NCCL tries to use for peer-to-peer transfers, but which can stall
depending on the driver and NCCL version. Consumer workstations (RTX 3080/3090/4090)
typically put all GPUs on the same switch, so this problem rarely appears there.

**Diagnose**: run the following on the node where you run miss-alignment:

```bash
nvidia-smi topo -m
```

Look at the cell for your two training GPUs (e.g. GPU0 × GPU1). A value of `SYS` or
`NODE` (rather than `PIX` — same switch) means P2P transfers cross a PCIe bridge and
may trigger the hang.

**Fix**: add `NCCL_P2P_DISABLE=1` to your command to force NCCL to use the
shared-memory transport instead:

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    miss-alignment --config-file config.yaml \
    --training-devices 0,1 \
    --reconstruction-devices 2,2,2,3,3,3 \
    --dataloaders-per-trainer 5 \
    --start-at-iteration 0
```

The shared-memory path is slightly lower bandwidth than direct P2P, but is stable across
all PCIe topologies and typically has negligible impact on overall training time.
