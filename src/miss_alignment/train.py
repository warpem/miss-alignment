from pathlib import Path
import yaml

import typer
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .data.shift_generation import create_default_generator
from .models import MissAlignment, MAEarlyStopping
from .alignment import run_alignment_parallel
from .data._pool_monitor import SimplePoolMonitor


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    n_workers: int = 4,
    n_devices: int = 1,
    pool_size: int = 1000,
    start_at_iteration: int = 0,
    monitor_production_and_consumption: bool = False,
) -> None:
    """Train MissAlignment on a dataset using configuration from a YAML file."""

    # check hardware settings, we assume gpu's are available and limit
    # cpu multithreading
    torch.set_num_threads(1)

    # gpu devices is a number of available devices for processing
    # the devices should be limited by CUDA_VISIBLE_DEVICES
    if n_devices < 1:
        raise ValueError("MissAlignment needs at least 1 GPU")
    devices_list = list(range(n_devices))

    devices_training = devices_list[0:1]
    print(f"Using device {devices_training[0]} for training")

    if len(devices_list) == 1:
        devices_reconstruction = devices_list * n_workers
    else:
        # distribute remaining devices equally among reconstruction workers,
        # cycling through devices if there are fewer devices than workers
        devices_remaining = devices_list[1:]
        devices_reconstruction = [
            devices_remaining[i % len(devices_remaining)]
            for i in range(n_workers)
        ]
    print(f"Using devices {devices_reconstruction} for reconstruction workers")

    # Load configuration from YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    general_config = config["general"]
    model_training_config = config["model_training"]
    data_module_config = config["data_loading"]
    shift_generation_config = config["shift_generation"]
    alignment_config = config["tilt_series_alignment"]

    # track the path to the dataset
    training_directory = Path(general_config["training_directory"])
    training_directory.mkdir(exist_ok=True, parents=True)

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = general_config["seed"]
    seed_everything(seed, workers=True)

    start_iter = start_at_iteration
    end_iter = len(general_config["iteration_settings"])

    for x in range(start_iter, end_iter):
        # ============================================================
        # ================= model training step ======================
        # ============================================================
        iteration_directory = training_directory / ("iter" + str(x))
        iteration_directory.mkdir(parents=True, exist_ok=True)
        iteration_settings = general_config["iteration_settings"][x]

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
            accelerator="gpu",
            devices=devices_training,  # use the 0 device
            default_root_dir=training_directory / "models",
            max_epochs=model_training_config["max_epochs_per_iteration"],
            log_every_n_steps=50,
            enable_checkpointing=True,
            deterministic=False,  # setting to True breaks on max_pool_3d
            limit_val_batches=0,  # turn on validation steps
            num_sanity_val_steps=0,
            callbacks=[early_stopping, checkpoint_callback],
            plugins=[SLURMEnvironment(auto_requeue=False)],
            precision="16-mixed",  # Enable automatic mixed precision
        )

        # initialize the monitor for production consumption rate
        if monitor_production_and_consumption:
            monitor = SimplePoolMonitor()
        else:
            monitor = None

        # Initialize model with parameters from config
        model_params = {
            "learning_rate": model_training_config["learning_rate"],
            "margin": model_training_config["loss_margin"],
            "weight_decay": float(model_training_config["weight_decay"]),
            "warmup_steps": model_training_config["warmup_steps"],
            "multistep_lr_scheduler": model_training_config["multistep_lr_scheduler"],
            "monitor": monitor,
            "model_architecture": model_training_config["model_architecture"],
        }

        if model_training_config["model_checkpoint"] is not None:
            model = MissAlignment.load_from_checkpoint(
                model_training_config["model_checkpoint"], **model_params
            )
        else:
            model = MissAlignment(**model_params)

        # Initialize data module with parameters from config
        with MissAlignmentDataModule(
            iteration_directory,
            create_default_generator(**shift_generation_config),
            n_workers=n_workers,
            reconstruction_accelerators=devices_reconstruction,
            batch_size=data_module_config["batch_size"],
            patch_size=data_module_config["patch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            steps_per_epoch=data_module_config["steps_per_epoch"],
            pool_size=pool_size,
        ) as dm:
            training_data = dm.train_dataloader()
            # enter datamodule context to start the reconstruction worker pool
            trainer.fit(model, train_dataloaders=training_data)

        print(
            f"Best model after training iteration {x}:",
            trainer.checkpoint_callback.best_model_path,
        )
        # update the config with the trained model
        model_training_config["model_checkpoint"] = (
            trainer.checkpoint_callback.best_model_path
        )

        # ============================================================
        # =============== tilt-series alignment step =================
        # ============================================================
        # get the output directory ready
        output_directory = training_directory / ("iter" + str(x + 1))
        output_directory.mkdir(parents=True, exist_ok=True)

        # get list of all files to process for alignment
        tilt_series_list = list(iteration_directory.glob("*.json"))

        # run alignment in parallel over all available devices
        run_alignment_parallel(
            model_checkpoint=model_training_config["model_checkpoint"],
            tilt_series_list=tilt_series_list,
            output_directory=output_directory,
            setting=iteration_settings["alignment"],
            patch_size=alignment_config["patch_size"],
            patch_overlap=alignment_config["patch_overlap"],
            batch_size=alignment_config["batch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            devices_list=devices_list,
        )

    return None
