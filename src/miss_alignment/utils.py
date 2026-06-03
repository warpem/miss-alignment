"""Utility functions for miss-alignment."""

import logging
import os

import torch

# Env var controlling miss-alignment's own log verbosity (DEBUG/INFO/WARNING/...).
LOG_LEVEL_ENV_VAR = "MISS_ALIGNMENT_LOG_LEVEL"


def configure_logging() -> None:
    """Configure miss-alignment logging from ``MISS_ALIGNMENT_LOG_LEVEL``.

    Applies the env var's level (default ``WARNING``) to the ``miss_alignment``
    package logger and attaches a stream handler once. Must be called at the
    start of every process — including spawned training and reconstruction
    workers — because env vars cross the spawn boundary while runtime logging
    configuration does not.

    Lightning's INFO startup banners are always silenced (they repeat per rank
    every macro-iteration and are not controlled by this env var).
    """
    level = os.environ.get(LOG_LEVEL_ENV_VAR, "WARNING").upper()

    pkg_logger = logging.getLogger("miss_alignment")
    pkg_logger.setLevel(level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        )
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False  # avoid double emission via the root logger

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


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
