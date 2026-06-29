"""Utility functions for miss-alignment."""

import logging
import os
from pathlib import Path
from shutil import copyfile

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

    # Mute Lightning's INFO startup banners. Both lightning.pytorch and
    # lightning.fabric give their own logger an explicit INFO level (with a
    # handler and propagate=False) at import, so the level must be set on each
    # directly -- setting the parent "lightning" logger has no effect.
    for name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_device_list(value: str) -> list[int]:
    """Parse comma-separated device list like '0,1,2' into [0, 1, 2]."""
    return [int(x.strip()) for x in value.split(",")]


def sync_start_iteration_xmls(start_iter: int, training_directory: Path) -> None:
    """Align the working XMLs with the ``iter{start_iter}/`` snapshot.

    iter 0 (fresh run): back up the original alignments from the training
    directory into ``iter0/`` as the baseline.

    Resuming (``start_iter > 0``): restore the working alignments *from*
    ``iter{start_iter}/`` (written at the end of the previous iteration) back
    into the training directory, so the run continues from the correct state
    even if a previous attempt crashed partway through this iteration.
    """
    iteration_directory = training_directory / f"iter{start_iter}"

    if start_iter == 0:
        iteration_directory.mkdir(parents=True, exist_ok=True)
        for xml_file in training_directory.glob("*.xml"):
            copyfile(xml_file, iteration_directory / xml_file.name)
    else:
        if not iteration_directory.is_dir():
            raise FileNotFoundError(
                f"Cannot resume at iteration {start_iter}: "
                f"{iteration_directory} does not exist."
            )
        for xml_file in iteration_directory.glob("*.xml"):
            copyfile(xml_file, training_directory / xml_file.name)


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
