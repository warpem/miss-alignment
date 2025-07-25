from pathlib import Path
import yaml

from functools import partial
import typer
import torch
from pytorch_lightning import Trainer, seed_everything

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import SHRECDataModule
from .data.shift_generation import generate_shifts
from .models import MissAlignment

data_module_dict = {
    "SHREC": SHRECDataModule,
}


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    num_workers: int = 8,
) -> None:
    """Train MissAlignment on a dataset using configuration from a YAML file."""
    # Load configuration from YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    general_config = config["general"]
    model_training_config = config["model_training"]
    data_loading_config = config["data_loading"]
    shift_generation_config = config["shift_generation"]

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = general_config["seed"]
    seed_everything(seed, workers=True)

    # Initialize data module with parameters from config
    data_module = data_module_dict[data_loading_config["dataset_type"]](
        data_loading_config["dataset_directory"],
        partial(generate_shifts, **shift_generation_config),
        num_workers=num_workers,
        batch_size=data_loading_config["batch_size"],
        target_size=data_loading_config["patch_size"],
        patches_per_tomogram=data_loading_config["patches_per_tomogram"],
        training_iteration=general_config["training_iteration"],
    )

    # Set up trainer with parameters from config
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        default_root_dir=model_training_config["output_directory"],
        max_epochs=model_training_config["epochs"],
        log_every_n_steps=model_training_config["log_every_n_steps"],
        enable_checkpointing=True,
        deterministic=False,  # setting to True breaks on max_pool_3d
        limit_val_batches=0,  # turn on validation steps
        num_sanity_val_steps=0,
    )

    # Train the model
    if general_config["resume_training_from_checkpoint"]:
        model = MissAlignment()
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=model_training_config["checkpoint_path"],
        )
    else:
        # Initialize model with parameters from config
        model_params = {
            "learning_rate": model_training_config["learning_rate"],
            "margin": model_training_config["margin"],
            "weight_decay": model_training_config["weight_decay"],
        }

        # Add learning rate scheduler if specified in config
        if "lr_scheduler" in model_training_config:
            model_params["lr_scheduler"] = model_training_config["lr_scheduler"]

        model = MissAlignment()
        model.load_from_checkpoint(
            model_training_config["checkpoint_path"],
            **model_params
        )

        trainer.fit(model, datamodule=data_module)
    return None
