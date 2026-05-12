"""Utility functions for miss-alignment."""

import os

import torch


def is_rank_zero() -> bool:
    """Check if we're on the main process (rank 0) in DDP.

    Check order:
    1. torch.distributed (always correct when initialized)
    2. LOCAL_RANK (set by Lightning's LightningEnvironment for each spawned process)

    SLURM_PROCID is intentionally not checked: without srun it is 0 for all
    processes spawned from a single SLURM task, making it actively misleading.
    With srun, torch.distributed is initialized before this function is called
    in any meaningful context, so SLURM_PROCID is never needed.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0

    return True


def rank_zero_print(*args, **kwargs) -> None:
    """Print only on rank 0 in DDP training.

    All arguments are passed directly to the built-in print function.
    """
    if is_rank_zero():
        print(*args, **kwargs)


def distributed_barrier() -> None:
    """Synchronize all processes in distributed training.

    This is a no-op if distributed training is not initialized (single GPU).
    Use this to ensure all ranks wait before/after non-distributed operations.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
