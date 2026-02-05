"""Utility functions for miss-alignment."""

import os

import torch


def is_rank_zero() -> bool:
    """Check if we're on the main process (rank 0) in DDP.

    This checks torch.distributed (if initialized), then global rank environment
    variables (RANK, SLURM_PROCID), and finally LOCAL_RANK as a fallback.

    Note: LOCAL_RANK is per-node, so on multi-node SLURM jobs each node would
    have a LOCAL_RANK=0. We prioritize global rank variables to avoid multiple
    processes thinking they're rank 0.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # Check global rank env vars (set by various launchers including SLURM)
    for var in ("RANK", "SLURM_PROCID"):
        rank = os.environ.get(var)
        if rank is not None:
            return int(rank) == 0

    # LOCAL_RANK is per-node, only use as last resort for single-node
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
