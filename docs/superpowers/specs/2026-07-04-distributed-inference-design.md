# Distributed Inference Design

**Date:** 2026-07-04  
**Status:** Approved (revised)  
**Scope:** Cluster distribution of the per-tilt-series alignment/inference phase

---

## Problem

With 300 tilt-series, the alignment phase (inference) takes 3× longer than the preceding training phase. Each series is independently optimizable — no data exchange between series — but the current implementation is limited to GPUs on the local machine. The head node has to finish all series sequentially across its local GPU pool before the next macro-iteration can begin.

## Goal

Distribute the per-series alignment tasks across a compute cluster so the head node fans out work, blocks until all series are done, and then continues to the next macro-iteration. Cluster mode is opt-in via a new `--n-cluster-workers` CLI argument; without it the system runs exactly as today (local multi-GPU pool).

---

## Architecture

A new `miss_alignment/distributed/` module sits between the existing `alignment/parallel.py` public API and the underlying `evaluate_tilt_series` function.

`run_alignment_parallel` accepts a new `n_cluster_workers: int | None` parameter. When it is `None`, `LocalProvisioner` is used (one subprocess per GPU, unchanged behavior). When it is set, `ClusterProvisioner` is used and submits exactly `n_cluster_workers` jobs; each worker runs the claim loop and processes as many series as it can grab until the queue drains.

`evaluate_tilt_series` gains an optional `model: MissAlignment | None = None` parameter. When provided, the worker's resident model is used directly instead of loading from disk — this amortizes checkpoint loading across all series a worker processes in one macro-iteration. Callers that omit the parameter are unaffected.

### Components

| File | Role |
|---|---|
| `distributed/queue.py` | Queue directory layout, task JSON read/write, atomic rename claim |
| `distributed/manager.py` | Head-node coordinator: writes tasks, runs scheduler thread, blocks until done |
| `distributed/provisioner.py` | `WorkerProvisioner` interface + `LocalProvisioner` + `ClusterProvisioner` |
| `distributed/worker.py` | `miss-alignment worker` subcommand: claims tasks, runs inference, writes results |
| `distributed/config.py` | Reads env vars, returns `ClusterConfig` dataclass or raises if missing |

---

## Task Format

One JSON file per tilt-series, written to `<training_dir>/tasks/pending/<index>-<series-name>.json`:

```json
{
  "task_id": "0000003-tilt_series_01",
  "model_checkpoint_path": "/data/project/iter2/model.ckpt",
  "tilt_series_path": "/data/project/tilt_series_01.xml",
  "output_directory": "/data/project/",
  "setting": "anchoring",
  "patch_size": 96,
  "patch_overlap": 0.1,
  "batch_size": 32,
  "apply_ctf": false,
  "downsample": 2,
  "init_fingerprint": "<sha256 of checkpoint path + alignment settings>"
}
```

`init_fingerprint` covers the model checkpoint path and all alignment parameters. A worker skips reloading the model when consecutive tasks share the same fingerprint, amortizing checkpoint loading across the many series each worker processes in one macro-iteration.

`setting` is serialized to JSON as a string or array. Workers convert array back to `tuple` before passing to `evaluate_tilt_series`.

On completion, result fields are appended before writing to `done/`:
```json
{ "final_loss": 0.0312, "device": "cuda:0" }
```

On failure, error fields are appended before writing to `failed/`:
```json
{ "error": "CUDA out of memory...", "worker_id": "cluster-node01-12345-0" }
```

---

## Queue Directory Layout

```
<training_dir>/tasks/
├── pending/                    # one JSON per queued task
├── running/
│   └── <worker_id>/            # per-worker subdir
│       ├── <task_id>.json      # claimed task lives here during execution
│       └── hb-<seq>            # worker heartbeat tick files (latest only)
├── done/                       # completed task JSONs
├── failed/                     # failed task JSONs
├── manager/
│   └── hb-<seq>                # manager heartbeat tick files (latest only)
└── cluster/
    └── worker-<i>.sh           # rendered cluster submission scripts
```

The queue directory is always `<training_dir>/tasks/`. Since the training directory must already be on a shared filesystem (workers need the XML/MRC files), no additional configuration is needed for cluster nodes to access it.

---

## Claim Protocol

1. Worker lists `pending/`, **shuffles randomly** (avoids thundering herd on the same lexicographically-first file)
2. Attempts `os.rename(pending/<id>.json, running/<wid>/<id>.json)`
3. `FileNotFoundError` means another worker won the race — try the next candidate
4. On task completion: write `done/<id>.json`, then delete `running/<wid>/<id>.json` (publish-before-delete: a crash mid-step leaves an orphan in `running/` that the scheduler sweeps, not a lost task)
5. On task failure: write `failed/<id>.json` with error, delete running copy, continue

No lock files. The OS rename is the coordination primitive.

---

## Worker Lifecycle

**Startup** (`miss-alignment worker --queue-dir <path> --device <id> [--worker-id <id>]`):
1. Derive `worker_id` from `--worker-id` or `local-<pid>-gpu<device>`
2. Create `tasks/running/<worker_id>/`
3. Start background heartbeat thread: write `tasks/running/<worker_id>/hb-<seq>` every 5s, keep only the latest tick file
4. Enter claim loop

**Claim loop:**
- Check manager heartbeat (`tasks/manager/hb-*`): if latest tick is >120s old → exit cleanly (manager is dead)
- List `pending/`, shuffle, attempt rename
- On empty queue: exit cleanly
- On claim: check if `init_fingerprint` matches last loaded model; if not, load checkpoint from `model_checkpoint_path` and cache it
- Call `evaluate_tilt_series(..., model=cached_model, device=f"cuda:{device}")` — all internal LBFGS retries and optimization passes run unchanged
- Write result to `done/` or `failed/`, delete running copy, loop back

Each worker processes **many series** per run. The checkpoint is loaded only once per macro-iteration (all tasks in a phase share the same `init_fingerprint`).

**No task-level retries.** Failures are deterministic (bad data, corrupt file, config error); the manager hard-fails after all remaining tasks complete.

---

## Manager Lifecycle

**Startup:**
1. Clear stale queue state from any prior run: first delete `pending/`, `done/`, `failed/` contents; then recover any `running/` orphans back to `pending/`
2. Write all task JSONs to `pending/`
3. Write the first manager heartbeat tick immediately (before starting workers, to avoid a race where workers start and find no heartbeat)
4. Start scheduler background thread
5. Start `ClusterProvisioner` or `LocalProvisioner`, call `ensure_workers(n_workers)`
6. Block in poll loop (500ms interval) until all tasks are in `done/` or `failed/`

**Scheduler thread** (runs every ~10s):
- Write manager heartbeat to `tasks/manager/hb-<seq>`
- Sweep stalled workers: for each `running/<wid>/`, check age of latest `hb-<seq>` file; if >120s stale → move task back to `pending/`, clean up dir
- Call `provisioner.ensure_workers()` to respawn any dead local workers

**Shutdown** (on completion or `KeyboardInterrupt`):
- Call `provisioner.shutdown()` (SIGTERM local children / cancel SLURM job IDs)
- Delete `tasks/` directory (clean state for next macro-iteration)
- Return `dict[series_name → final_loss]` on full success, or raise `RuntimeError` listing all failed series if any task is in `failed/`

**Failure policy:** if any series ends in `failed/`, the manager raises after all other tasks complete. `train.py` and `infer.py` propagate this as a hard failure — training stops.

---

## Provisioners

### `LocalProvisioner`

Activated when `n_cluster_workers` is `None` (default).

- `ensure_workers(n_workers)`: spawns `miss-alignment worker --queue-dir <tasks_dir> --device <gpu_id>` via `subprocess.Popen`, one per GPU in `devices_alignment` (same list as today: `list(range(torch.cuda.device_count()))`, respecting `CUDA_VISIBLE_DEVICES`). `n_workers` is ignored; the number of local workers is always equal to the number of devices.
- Respawns any that have exited prematurely (checked each scheduler tick)
- `shutdown()`: SIGTERM all children, short timeout, SIGKILL if needed

This replaces `_parallel.py`'s `run_device_pool` / `mp.spawn` internals with no behavioral change for the local case.

### `ClusterProvisioner`

Activated when `n_cluster_workers` is set.

Requires both `MISS_CLUSTER_CONFIG` and `MISS_CLUSTER_SCRIPT` env vars to be set; raises `RuntimeError` immediately if either is absent, so the user gets a clear error rather than a silent fallback to local mode.

**`MISS_CLUSTER_CONFIG`** — path to a JSON file:
```json
{
  "submit": "sbatch {{script_path}}",
  "submit_job_id_regex": "Submitted batch job (\\d+)",
  "cancel": "scancel {{job_id}}"
}
```

**`MISS_CLUSTER_SCRIPT`** — path to a `.sh` template:
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

conda activate miss-alignment
{{command}}
```

- `{{command}}` is filled by the provisioner with the `miss-alignment worker` invocation
- All cluster-specific settings (partition, memory, time limit, environment setup) are the user's responsibility in the template
- Additional `{{custom_var}}` placeholders filled via `MISS_CLUSTER_VAR_<name>=<value>` env vars
- Rendered scripts are written to `tasks/cluster/worker-<i>.sh`
- The injected `{{command}}` is: `miss-alignment worker --queue-dir <tasks_dir> --device 0 --worker-id "$(hostname)-$$-<i>"` — `$(hostname)` and `$$` are expanded by the compute node's shell at runtime, guaranteeing unique, stable worker IDs
- `ensure_workers(n_workers)` submits exactly `n_workers` jobs; each worker pulls tasks from the shared queue until it drains
- `shutdown()`: runs configured `cancel` command for each stored job ID; registers SIGINT/SIGTERM handlers so Ctrl-C on the head node cancels the cluster pool

---

## Configuration

### CLI changes

`miss-alignment train` and `miss-alignment infer` each gain one new optional argument:

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--n-cluster-workers` | `int \| None` | `None` | Number of cluster jobs to submit. When set, activates cluster mode. When absent, local multi-GPU mode is used (unchanged). |

### Environment variables (cluster mode only)

| Env var | Purpose |
|---|---|
| `MISS_CLUSTER_CONFIG` | Path to cluster scheduler JSON config. **Required** when `--n-cluster-workers` is set. |
| `MISS_CLUSTER_SCRIPT` | Path to job submission shell script template. **Required** when `--n-cluster-workers` is set. |
| `MISS_CLUSTER_VAR_<name>` | Additional template variables, e.g. `MISS_CLUSTER_VAR_partition=gpu`. |

`config.py` is called only when `n_cluster_workers` is not `None`. It raises `RuntimeError` if either env var is absent, with a message naming which one is missing.

---

## Changes to `evaluate_tilt_series`

A single optional parameter is added:

```python
def evaluate_tilt_series(
    model_checkpoint_path: Path,
    tilt_series_path: Path,
    output_directory: Path,
    setting: str | tuple[int, int] | tuple[int, int, int, int] = "anchoring",
    patch_size: int = 96,
    patch_overlap: float = 0.1,
    batch_size: int = 16,
    apply_ctf: bool = True,
    downsample: int = 1,
    device: str = "cpu",
    initial_reliable_fraction: float = 1 / 2,
    n_control_points: int = 7,
    model: MissAlignment | None = None,   # NEW
) -> tuple[Path, list[float]]:
```

When `model` is provided, the function uses it directly instead of loading from `model_checkpoint_path`. This is the only change to the function. All existing callers (which omit `model`) are unaffected.

---

## What Does Not Change

- `evaluate_tilt_series` internal logic — only the signature gains one optional parameter
- LBFGS retries, anchoring iterations, coarse-to-fine spline passes — all internal to `evaluate_tilt_series`
- `CUDA_VISIBLE_DEVICES` still controls which local GPUs are used
- Single-GPU local behavior is identical

---

## File Changes Summary

**New files:**
- `src/miss_alignment/distributed/__init__.py`
- `src/miss_alignment/distributed/queue.py`
- `src/miss_alignment/distributed/manager.py`
- `src/miss_alignment/distributed/provisioner.py`
- `src/miss_alignment/distributed/worker.py`
- `src/miss_alignment/distributed/config.py`
- `src/miss_alignment/__main__.py` — enables `python -m miss_alignment` for `LocalProvisioner` subprocess launch

**Modified files:**
- `src/miss_alignment/alignment/tilt_series.py` — add optional `model` parameter to `evaluate_tilt_series`
- `src/miss_alignment/alignment/parallel.py` — replace `run_device_pool` call with distributed manager; add `n_cluster_workers` parameter
- `src/miss_alignment/train.py` — accept and pass through `--n-cluster-workers`
- `src/miss_alignment/infer.py` — accept and pass through `--n-cluster-workers`
- `src/miss_alignment/_cli.py` — register `worker` subcommand
- `src/miss_alignment/__init__.py` — export `worker_miss_align`

**Deleted files:**
- `src/miss_alignment/_parallel.py` — superseded by `LocalProvisioner`; deleted once the new path is validated in tests
