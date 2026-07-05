# Cluster distribution example

This directory contains example configuration for distributing miss-alignment
inference across a SLURM cluster. The same mechanism works for stack preparation
(`--prepare-stacks`) and cross-correlation pre-alignment (`--preprocess`) phases too.

## Files

- **`cluster_config.json`** — describes how to submit, identify, and cancel cluster jobs.
  Adapt the commands for your scheduler (SLURM shown; PBS/Torque and others work the
  same way with different commands).
- **`worker.sh`** — SLURM submission script template. The `{{command}}` placeholder
  is filled in automatically with the `miss-alignment worker` invocation. Edit resource
  requests (`--mem`, `--time`, `--gres`, `--partition`) to match your cluster.

## Setup

1. Point two environment variables at the files in this directory (or copies of them):

   ```bash
   export MISS_CLUSTER_CONFIG=/path/to/cluster_config.json
   export MISS_CLUSTER_SCRIPT=/path/to/worker.sh
   ```

2. Set `MISS_CLUSTER_VAR_partition` to fill in the `{{partition}}` placeholder in
   `worker.sh` (or hard-code your partition name directly in the script):

   ```bash
   export MISS_CLUSTER_VAR_partition=gpu
   ```

3. Add `--n-cluster-workers N` to your `miss-alignment train` or `miss-alignment infer`
   command. `N` controls how many simultaneous cluster jobs are submitted. A good
   starting point is one job per GPU node you want to use — each job claims and processes
   tilt series from the shared queue until the queue is drained.

   ```bash
   miss-alignment train --config-file config.yaml --n-cluster-workers 8
   ```

## How it works

The head node (where you run `miss-alignment train/infer`) writes one task JSON file
per tilt series into `<training_dir>/tasks/pending/`. It then submits `N` cluster
jobs, each running `miss-alignment worker --queue-dir <training_dir>/tasks --device 0`.
Workers race to claim tasks by atomic file rename — no scheduler or lock files needed —
and process series until the queue is empty. The head node blocks until all tasks are
in `done/` or `failed/`, then continues to the next macro-iteration.

The queue directory (`<training_dir>/tasks/`) must be on a shared filesystem visible
to all worker nodes (Lustre, GPFS, NFS, etc.). Since the training directory already
needs to hold the XML metadata and MRC stacks that workers read, this is satisfied
automatically.

## Adapting for other schedulers

Replace the three fields in `cluster_config.json`:

| Field | Purpose | PBS/Torque example |
|---|---|---|
| `submit` | Command to submit `{{script_path}}` | `"qsub {{script_path}}"` |
| `submit_job_id_regex` | Regex capturing the job ID from submit stdout | `"(\\d+)\\..*"` |
| `cancel` | Command to cancel `{{job_id}}` | `"qdel {{job_id}}"` |

Update `worker.sh` with the corresponding scheduler directives (`#PBS` instead of
`#SBATCH`, etc.).
