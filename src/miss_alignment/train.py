from pathlib import Path
import yaml
import subprocess
import shutil
import os

from functools import partial
import typer
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .data.shift_generation import create_default_generator
from .data.io import read_tomogram_from_pickle
from .models import MissAlignment, MAEarlyStopping
from .alignment import evaluate_tilt_series
from .data._pool_monitor import SimplePoolMonitor

_zenodo_archive = "16574872"


def _download_to_dir(dataset_directory: Path):
    subprocess.run(
        [
            "zenodo_get",
            _zenodo_archive,  # update value
            "--output-dir",
            str(dataset_directory),
        ]
    )
    for archive in ("torch_tiltxcorr.zip", "ground_truth.zip"):
        zipped_archive = dataset_directory / archive
        shutil.unpack_archive(
            zipped_archive,
            extract_dir=dataset_directory,
        )
        os.remove(zipped_archive)


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    reconstruction_workers: int = 4,
    dataloader_workers: int = 4,
    iterations: int = 3,
    monitor_production_and_consumption: bool = False,
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
    training_directory.mkdir(exist_ok=True, parents=True)
    ground_truth_directory = (
        Path(alignment_config["ground_truth_directory"])
        if alignment_config["ground_truth_directory"] is not None
        else None
    )
    if general_config['download_data']:
        _download_to_dir(ground_truth_directory)
    #TODO we would like to estimate the sample thickness
    TOMOGRAM_SHAPE = general_config["tomogram_shape"]

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = general_config["seed"]
    seed_everything(seed, workers=True)

    start_iter = general_config['start_at_iteration']
    end_iter = start_iter + iterations
    if not isinstance(model_training_config['learning_rate'], list):
        learning_rates = [model_training_config["learning_rate"],] * iterations
    else:
        learning_rates = model_training_config["learning_rate"]

    for x, lr in zip(range(start_iter, end_iter), learning_rates):
        iteration_directory = training_directory / ('iter' + str(x))
        iteration_directory.mkdir(parents=True, exist_ok=True)

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
            save_on_train_epoch_end=True,
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

        # initialize the monitor for production consumption rate
        if monitor_production_and_consumption:
            monitor = SimplePoolMonitor()
        else:
            monitor = None

        # Initialize model with parameters from config
        model_params = {
            "learning_rate": lr,
            "margin": model_training_config["loss_margin"],
            "weight_decay": float(model_training_config["weight_decay"]),
            "warmup_steps": model_training_config["warmup_steps"],
            "multistep_lr_scheduler": model_training_config[
                "multistep_lr_scheduler"],
            "monitor": monitor,
        }

        if model_training_config["model_checkpoint"] is not None:
            model = MissAlignment.load_from_checkpoint(
                model_training_config["model_checkpoint"], **model_params
            )
        else:
            model = MissAlignment(**model_params)

        # Initialize data module with parameters from config
        torch.set_num_threads(1)
        with MissAlignmentDataModule(
            iteration_directory,
            create_default_generator(**shift_generation_config),
            reconstruction_workers=reconstruction_workers,
            dataloader_workers=dataloader_workers,
            batch_size=data_module_config["batch_size"],
            patch_size=data_module_config["patch_size"],
            tomogram_shape=TOMOGRAM_SHAPE,
            steps_per_epoch=data_module_config["steps_per_epoch"],
            monitor=monitor,
        ) as dm:
            dm.wait_for_pool_to_fill()
            training_data = dm.train_dataloader()
            # enter datamodule context to start the reconstruction worker pool
            trainer.fit(model, train_dataloaders=training_data)

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

        # set input and output dirs
        output_directory = training_directory / ('iter' + str(x + 1))
        output_directory.mkdir(parents=True, exist_ok=True)
        # set multithread for cpu to max 30 cores
        torch.set_num_threads(min(reconstruction_workers, 30))
        for file_path in iteration_directory.iterdir():
            if file_path.suffix != '.pickle':
                continue
            tilt_series = read_tomogram_from_pickle(file_path)
            tilt_series_name = file_path.stem
            tilt_series_ground_truth = (
                read_tomogram_from_pickle(
                    ground_truth_directory / f"{tilt_series_name}.pickle"
                ) if ground_truth_directory is not None
                else None
            )
            # results are written to the output_directory
            evaluate_tilt_series(
                model,
                tilt_series_name,
                tilt_series,
                alignment_config["patches_per_dim"],
                alignment_config["patch_size"],
                TOMOGRAM_SHAPE,
                output_directory,
                device="cuda" if torch.cuda.is_available() else "cpu",
                tilt_series_ground_truth=tilt_series_ground_truth,
            )

    return None
