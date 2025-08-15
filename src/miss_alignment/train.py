from pathlib import Path
import yaml

from functools import partial
import typer
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import SHRECDataModule
from .data.shift_generation import generate_shifts
from .data.io import read_tomogram_from_pickle
from .models import MissAlignment, MAEarlyStopping
from .alignment import evaluate_tilt_series

data_module_dict = {
    "SHREC": SHRECDataModule,
}


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    reconstruction_workers: int = 4,
    dataloader_workers: int = 4,
    iterations: int = 3,
) -> None:
    """Train MissAlignment on a dataset using configuration from a YAML file."""
    # Load configuration from YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    general_config = config["general"]
    model_training_config = config["model_training"]
    data_module_config = config["data_module"]
    shift_generation_config = config["shift_generation"]
    alignment_config = config["tilt_series_alignment"]

    # track the path to the dataset
    training_directory = Path(data_module_config["training_directory"])
    ground_truth_directory = Path(alignment_config["ground_truth_directory"])

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = general_config["seed"]
    seed_everything(seed, workers=True)

    for x in range(iterations):  # iterations of MissAlignment to run
        # Define the early stopping callback
        early_stopping = MAEarlyStopping(
            patience=5,  # cycles with no improvement
            min_delta=0.001,  # minimum change to qualify as an improvement
            wait_for_scheduler=True,
        )

        # save checkpoints based on training loss performance
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            mode="min",  # 'min' for loss, 'max' for accuracy
            save_top_k=3,  # Keep 5 best checkpoints
            filename=str(x) + "_{epoch}--{step}--{train_loss:.3f}",
            every_n_train_steps=model_training_config["n_steps_per_cycle"],
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
            plugins=[SLURMEnvironment(auto_requeue=False)],
        )

        # Initialize model with parameters from config
        model_params = {
            "learning_rate": model_training_config["learning_rate"],
            "margin": model_training_config["loss_margin"],
            "weight_decay": float(model_training_config["weight_decay"]),
            "warmup_steps": model_training_config["warmup_steps"],
            "loss_metric_steps": model_training_config[
                "steps_per_epoch"],
            "multistep_lr_scheduler": model_training_config[
                "multistep_lr_scheduler"]
        }

        model = MissAlignment.load_from_checkpoint(
            model_training_config["model_checkpoint"], **model_params
        )
        # Initialize data module with parameters from config
        with SHRECDataModule(
            training_directory / ('iter' + str(x)),
            partial(generate_shifts, **shift_generation_config),
            reconstruction_workers=reconstruction_workers,
            dataloader_workers=dataloader_workers,
            batch_size=data_module_config["batch_size"],
            patch_size=data_module_config["patch_size"],
            steps_per_epoch=data_module_config["steps_per_epoch"],
            download_data=data_module_config["download_data"],
        ) as dm:
            # enter datamodule context to start the reconstruction worker pool
            trainer.fit(model, datamodule=dm)

        print(
            f'Best model after '
            f'training iteration {x}:',
            trainer.checkpoint_callback.best_model_path
        )
        # update the config with the trained model
        model_training_config["model_checkpoint"] = (
            trainer.checkpoint_callback.best_model_path
        )

        # load the best model and run alignment optimization
        model = MissAlignment.load_from_checkpoint(
            model_training_config["model_checkpoint"],
        )
        model.freeze()  # freeze model to perform alignments
        input_directory = training_directory / ('iter' + str(x))
        output_directory = training_directory / ('iter' + str(x + 1))
        output_directory.mkdir(parents=True, exist_ok=True)
        for file_path in input_directory.iterdir():
            if file_path.suffix != '.pickle':
                continue
            tilt_series = read_tomogram_from_pickle(file_path)
            tilt_series_name = file_path.stem
            tilt_series_ground_truth = read_tomogram_from_pickle(
                ground_truth_directory / f"{tilt_series_name}.pickle"
            )
            # results are written to the output_directory
            evaluate_tilt_series(
                model,
                tilt_series_name,
                tilt_series,
                alignment_config["patches_per_dim"],
                alignment_config["patch_size"],
                (180, 512, 512),
                output_directory,
                tilt_series_ground_truth=tilt_series_ground_truth,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    return None
