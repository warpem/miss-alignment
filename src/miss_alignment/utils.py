"""Utility functions for miss-alignment."""

import os

import torch


def is_rank_zero() -> bool:
    """Check if we're on the main process (rank 0) in DDP.

    This checks both torch.distributed (if initialized) and the LOCAL_RANK
    environment variable (set by DDP launchers before script execution).
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Check LOCAL_RANK env var (set by DDP launcher before torch.distributed init)
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
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print(f"[Rank {rank}] Entering barrier (world_size={world_size})")
        torch.distributed.barrier()
        print(f"[Rank {rank}] Exited barrier")
