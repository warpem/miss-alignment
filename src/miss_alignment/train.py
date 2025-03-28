from pathlib import Path

import typer
import torch
from pytorch_lightning import Trainer, seed_everything

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import EMDBDataModule
from .models import MissAlignment


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    dataset_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option("training", **OPTION_PROMPT_KWARGS),
    batch_size: int = 1,
    epochs: int = 100,
    n_workers: int = 1,
    learning_rate: float = 1e-5,
    seed: int = 45132,
) -> None:
    """Train MissAlignment on a dataset."""
    torch.set_float32_matmul_precision("medium")
    seed_everything(seed, workers=True)

    model = MissAlignment(learning_rate=learning_rate)
    data_module = EMDBDataModule(
        dataset_directory,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision="32",
        default_root_dir=output_directory,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        enable_checkpointing=True,
        deterministic=False,  # setting to True breaks on max_pool_3d
    )
    trainer.fit(model, datamodule=data_module)
    return None
