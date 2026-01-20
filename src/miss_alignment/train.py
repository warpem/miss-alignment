from pathlib import Path
from shutil import copyfile
from typing import Optional
import yaml

import typer
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .data.shift_generation import create_default_generator
from .models import MissAlignment, MAEarlyStopping, MAProgressBar
from .alignment import run_alignment_parallel
from .alignment.statistics import (
    identify_outliers,
    plot_loss_distribution,
    filter_outlier_xml_files,
)
from .prepare_stacks import prepare_stacks_parallel
from .preprocessing import run_cross_correlation_alignment


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    n_workers: int = typer.Option(
        2,
        help="Number of workers that feed subtomogram reconstructions "
        "to the data pool for training the CNN. Workers will be "
        "divided across the available GPU devices, where multiple "
        "workers per GPU are allowed.",
    ),
    n_devices: int = typer.Option(
        1,
        help="Number of GPU devices available for miss-alignment. "
        "GPU's are index from 0 to n and the system GPU's should be "
        "restricted with CUDA_VISIBLE_DEVICES before the command. "
        "One GPU's is always used to train the CNN, the others are "
        "split across the workers. If a single GPU is provided both "
        "the model and workers will use the same GPU.",
    ),
    pool_size: int = typer.Option(
        1000,
        help="Maximum number of subtomgram reconstructions "
        "to write to temporary storage during training of the "
        "misalignment model.",
    ),
    start_at_iteration: int = typer.Option(
        0,
        help="Continue from a specific iteration in the config.",
    ),
    prepare_stacks: Optional[float] = typer.Option(
        None,
        help="Pixel size (in Angstroms) for preprocessing tilt stacks. "
        "If provided, loads raw tilt images, rescales to this pixel size, "
        "and creates tilt stacks with thumbnails before training.",
    ),
    preprocess: bool = typer.Option(
        False,
        help="Run cross-correlation based alignment before training iterations. "
        "This performs coarse alignment with pretilt estimation.",
    ),
    filter_outliers: bool = typer.Option(
        False,
        help="[Experimental option] Filter out tilt-series "
        "with alignment loss > mean + 3*std. "
        "Outliers are moved to iter{N}_outliers/ directory.",
    ),
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
            devices_remaining[i % len(devices_remaining)] for i in range(n_workers)
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

    # Prepare tilt stacks if requested
    if prepare_stacks is not None:
        if prepare_stacks <= 0:
            raise ValueError("--prepare-stacks must be a positive non-zero number")
        prepare_stacks_parallel(
            training_directory=training_directory,
            desired_pixel_size=prepare_stacks,
            n_processes=4,
            devices=devices_list,
        )

    # Set up training environment
    torch.set_float32_matmul_precision("medium")
    seed = general_config["seed"]
    seed_everything(seed, workers=True)

    # Run preprocessing if requested
    if preprocess:
        if start_at_iteration != 0:
            raise ValueError(
                "Running preprocessing at while "
                "not starting at iteration 0. This "
                "is likely not desirable behaviour."
            )
        else:
            # Back up original data to pre-iter directory
            preiter_directory = training_directory / "pre-iter"
            preiter_directory.mkdir(parents=True, exist_ok=True)
            for xml_file in training_directory.glob("*.xml"):
                destination = preiter_directory / xml_file.name
                copyfile(xml_file, destination)
            print(f"Backed up original metadata to {preiter_directory}")

            # Run cross-correlation alignment on the training directory
            run_cross_correlation_alignment(
                training_directory=training_directory,
                device=devices_training[0],
            )

    start_iter = start_at_iteration
    end_iter = len(general_config["iteration_settings"])

    # make copies of the xml files and (model) we're starting from
    iteration_directory = training_directory / ("iter" + str(start_iter))
    iteration_directory.mkdir(parents=True, exist_ok=True)
    for xml_file in training_directory.glob("*.xml"):
        destination = iteration_directory / xml_file.name
        copyfile(xml_file, destination)

    if model_training_config["model_checkpoint"] is not None:
        source = model_training_config["model_checkpoint"]
        destination = iteration_directory / Path(source).name
        copyfile(source, destination)

    for x in range(start_iter, end_iter):
        # ============================================================
        # ================= model training step ======================
        # ============================================================
        iteration_settings = general_config["iteration_settings"][x]
        alignment_mode = iteration_settings["alignment"]
        print(f"\n{'=' * 60}")
        print(f"Iteration {x + 1}/{end_iter} - Alignment mode: {alignment_mode}")
        print(f"{'=' * 60}\n")

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

        # Progress bar showing total progress across all epochs
        max_epochs = model_training_config["max_epochs_per_iteration"]
        steps_per_epoch = data_module_config["steps_per_epoch"]
        progress_bar = MAProgressBar(
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            refresh_rate=10,
        )

        # Set up trainer with parameters from config
        trainer = Trainer(
            accelerator="gpu",
            devices=devices_training,  # use the 0 device
            default_root_dir=training_directory / "models",
            max_epochs=max_epochs,
            log_every_n_steps=50,
            enable_checkpointing=True,
            enable_progress_bar=False,  # disable default progress bar
            deterministic=False,  # setting to True breaks on max_pool_3d
            limit_val_batches=0,  # turn on validation steps
            num_sanity_val_steps=0,
            callbacks=[early_stopping, checkpoint_callback, progress_bar],
            plugins=[SLURMEnvironment(auto_requeue=False)],
            precision="16-mixed",  # Enable automatic mixed precision
        )

        # Initialize model with parameters from config
        model_params = {
            "learning_rate": model_training_config["learning_rate"],
            "margin": model_training_config["loss_margin"],
            "weight_decay": float(model_training_config["weight_decay"]),
            "warmup_steps": model_training_config["warmup_steps"],
            "multistep_lr_scheduler": model_training_config["multistep_lr_scheduler"],
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
            training_directory,
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

        # copy best model to training directory as model.ckpt
        best_model_path = Path(trainer.checkpoint_callback.best_model_path)
        training_model_path = training_directory / "model.ckpt"
        copyfile(best_model_path, training_model_path)
        print(f"Copied best model to {training_model_path}")

        # ============================================================
        # =============== tilt-series alignment step =================
        # ============================================================

        # get list of all files to process for alignment
        tilt_series_list = list(training_directory.glob("*.xml"))

        # run alignment in parallel over all available devices
        alignment_losses = run_alignment_parallel(
            model_checkpoint=model_training_config["model_checkpoint"],
            tilt_series_list=tilt_series_list,
            output_directory=training_directory,
            setting=iteration_settings["alignment"],
            patch_size=alignment_config["patch_size"],
            patch_overlap=alignment_config["patch_overlap"],
            batch_size=alignment_config["batch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            devices_list=devices_list,
        )

        # Plot loss distribution
        plot_path = training_directory / f"iter{x + 1}_alignment_losses.png"
        plot_loss_distribution(
            losses=alignment_losses,
            output_path=plot_path,
            n_std=3.0,
            iteration=x + 1,
        )

        # Filter outliers if requested
        if filter_outliers:
            outliers, mean_loss, std_loss = identify_outliers(
                alignment_losses, n_std=3.0
            )
            if outliers:
                print(f"\nFiltering {len(outliers)} outlier tilt-series:")
                filter_outlier_xml_files(
                    training_directory=training_directory,
                    outliers=outliers,
                    iteration=x + 1,
                )

        # make copies of the xml files and model after alignment
        iteration_directory = training_directory / ("iter" + str(x + 1))
        iteration_directory.mkdir(parents=True, exist_ok=True)

        for xml_file in training_directory.glob("*.xml"):
            destination_xml = iteration_directory / xml_file.name
            copyfile(xml_file, destination_xml)

            # copy the file with the alignment loss
            loss_json = xml_file.stem + "_alignment_loss.json"
            destination_json = iteration_directory / loss_json
            copyfile(training_directory / loss_json, destination_json)

        training_model_path = iteration_directory / "model.ckpt"
        copyfile(best_model_path, training_model_path)
        print(f"Copied best model to {training_model_path}")

    return None
