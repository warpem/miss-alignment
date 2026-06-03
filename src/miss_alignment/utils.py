"""Utility functions for miss-alignment."""

import os

import torch


def is_rank_zero() -> bool:
    """Check if we're on the main process (rank 0) in DDP.

    Check order:
    1. torch.distributed (always correct when initialized)
    2. LOCAL_RANK env var, set by the training worker before the process group
       exists (e.g. at datamodule __enter__) and read by Lightning's
       LightningEnvironment to detect the external launcher

    Single-GPU training sets neither, so the final ``return True`` covers the
    in-process (rank 0) case.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0

    return True
