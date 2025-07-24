from pathlib import Path
import yaml

import typer
import torch
from pytorch_lightning import Trainer, seed_everything

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .models import MissAlignment


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    dataset_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
) -> None:
    """Train MissAlignment on a dataset using configuration from a YAML file."""
    # Load configuration from YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = training_config.get("seed", 45132)
    seed_everything(seed, workers=True)

    # Initialize model with parameters from config
    model_params = {
        "learning_rate": model_config.get("learning_rate", 1e-5),
        "margin": model_config.get("margin", 0.5),
        "weight_decay": model_config.get("weight_decay", 0)
    }

    # Add learning rate scheduler if specified in config
    if "lr_scheduler" in model_config:
        model_params["lr_scheduler"] = model_config["lr_scheduler"]

    model = MissAlignment(**model_params)

    # Initialize data module with parameters from config
    data_module = MissAlignmentDataModule(
        dataset_directory,
        dataset_type=data_config.get("dataset_type", "SHREC"),
        batch_size=data_config.get("batch_size", 1),
        target_size=data_config.get("patch_size", 64),
        train_val_split=tuple(data_config.get("train_val_split", [0.7, 0.3])),
        num_workers=data_config.get("num_workers", 1),
    )

    # Set up trainer with parameters from config
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision=training_config.get("precision", "32"),
        default_root_dir=training_config.get("output_directory", "training"),
        max_epochs=training_config.get("epochs", 100),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 5),
        enable_checkpointing=True,
        deterministic=False,  # setting to True breaks on max_pool_3d
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    return None
