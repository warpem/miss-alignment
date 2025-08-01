from pathlib import Path
import yaml

from functools import partial
import typer
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import SHRECDataModule
from .data.shift_generation import generate_shifts
from .models import MissAlignment, MAEarlyStopping

data_module_dict = {
    "SHREC": SHRECDataModule,
}


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    num_workers: int = 8,
    iterations: int = 3,
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
    alignment_config = config["tilt_series_alignment"]

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
        training_iteration=general_config["start_at_iteration"],
    )

    for _ in range(iterations):  # iterations of MissAlignment to run
        # Define the early stopping callback
        early_stopping_config = model_training_config["early_stopping"]
        early_stopping = MAEarlyStopping(
            # steps with no improvement
            patience=early_stopping_config["patience"],
            # minimum change to qualify as an improvement
            min_delta=early_stopping_config["min_delta"],
        )

        # save checkpoints based on training loss performance
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            mode="min",  # 'min' for loss, 'max' for accuracy
            save_top_k=3,  # Keep 5 best checkpoints
            filename="{epoch}--{step}--{train_loss:.2f}",
            every_n_train_steps=model_training_config["loss_metric_steps"],
        )

        # Set up trainer with parameters from config
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            default_root_dir=model_training_config["output_directory"],
            max_epochs=model_training_config["max_epochs_per_iteration"],
            log_every_n_steps=50,
            enable_checkpointing=True,
            deterministic=False,  # setting to True breaks on max_pool_3d
            limit_val_batches=0,  # turn on validation steps
            num_sanity_val_steps=0,
            callbacks=[early_stopping, checkpoint_callback],
        )

        # Train the model
        if general_config["resume_training_from_checkpoint"]:
            model = MissAlignment()
            trainer.fit(
                model,
                datamodule=data_module,
                ckpt_path=model_training_config["model_checkpoint"],
            )
        else:
            # Initialize model with parameters from config
            model_params = {
                "learning_rate": model_training_config["learning_rate"],
                "margin": model_training_config["loss_margin"],
                "weight_decay": float(model_training_config["weight_decay"]),
                "warmup_steps": model_training_config["warmup_steps"],
                "loss_metric_steps": model_training_config["loss_metric_steps"],
            }

            # Add learning rate scheduler if specified in config
            if "lr_scheduler" in model_training_config:
                model_params["lr_scheduler"] = model_training_config["lr_scheduler"]

            model = MissAlignment.load_from_checkpoint(
                model_training_config["model_checkpoint"], **model_params
            )

            trainer.fit(model, datamodule=data_module)

        print(trainer.checkpoint_callback.best_model_path)
        # update the config with the trained model
        model_training_config["model_checkpoint"] = (
            trainer.checkpoint_callback.best_model_path
        )

        # run alignment optimization
        model.freeze()  # freeze model to perform alignments
        data_module.align_dataset(
            model,
            alignment_config["patches_per_dim"],
            alignment_config["patch_size"],
            Path(alignment_config["ground_truth_fetch_directory"]),
        )  # data_module will update automatically to point to new files

    return None
