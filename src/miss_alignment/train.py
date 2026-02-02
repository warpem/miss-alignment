from pathlib import Path
from shutil import copyfile
from typing import Optional
import yaml

import typer
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .data.shift_generation import create_default_generator
from .models import MissAlignment, MAEarlyStopping, MAProgressBar
from .alignment import run_alignment_parallel
from .prepare_stacks import prepare_stacks_parallel
from .preprocessing import run_cross_correlation_alignment
from .utils import distributed_barrier, is_rank_zero, rank_zero_print


def parse_device_list(value: str) -> list[int]:
    """Parse comma-separated device list like '0,1,2' into [0, 1, 2]."""
    return [int(x.strip()) for x in value.split(",")]


@cli.command(name="train", no_args_is_help=True)
def train_miss_align(
    config_file: Path = typer.Option("config_template.yaml", **OPTION_PROMPT_KWARGS),
    training_devices: str = typer.Option(
        "0",
        help="Comma-separated list of GPU device indices for model training. "
        "For multi-GPU training, specify multiple devices (e.g., '0,1'). "
        "The batch size from config will be divided by the number of "
        "training devices to maintain the same effective batch size.",
    ),
    reconstruction_devices: str = typer.Option(
        "0",
        help="Comma-separated list of GPU device indices for reconstruction workers. "
        "Each entry spawns one worker on that device. Devices can be repeated "
        "to run multiple workers per GPU (e.g., '0,0,1,1' runs 2 workers each "
        "on GPU 0 and GPU 1). Can overlap with training devices if needed.",
    ),
    dataloaders_per_trainer: int = typer.Option(
        2,
        help="Number of CPU DataLoader workers per training device. "
        "Total DataLoader workers = this value × number of training devices.",
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
) -> None:
    """Train MissAlignment on a dataset using configuration from a YAML file."""

    # check hardware settings, we assume gpu's are available and limit
    # cpu multithreading
    torch.set_num_threads(1)

    # Parse device lists from comma-separated strings
    devices_training = parse_device_list(training_devices)
    devices_reconstruction = parse_device_list(reconstruction_devices)

    if len(devices_training) < 1:
        raise ValueError("MissAlignment needs at least 1 training device")
    if len(devices_reconstruction) < 1:
        raise ValueError("MissAlignment needs at least 1 reconstruction worker")

    # For alignment stage, use all visible GPUs
    n_visible_gpus = torch.cuda.device_count()
    devices_alignment = list(range(n_visible_gpus))

    rank_zero_print(f"Using devices {devices_training} for training")
    rank_zero_print(f"Using devices {devices_reconstruction} for reconstruction")
    rank_zero_print(f"Using devices {devices_alignment} for alignment")

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
            devices=devices_alignment,
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
            rank_zero_print(f"Backed up original metadata to {preiter_directory}")

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
        rank_zero_print(f"\n{'=' * 60}")
        rank_zero_print(f"Iteration {x + 1}/{end_iter} - Alignment: {alignment_mode}")
        rank_zero_print(f"{'=' * 60}\n")

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
        # Use DDP strategy for multi-GPU training
        strategy = "ddp" if len(devices_training) > 1 else "auto"
        trainer = Trainer(
            accelerator="gpu",
            devices=devices_training,
            strategy=strategy,
            default_root_dir=training_directory / "models",
            max_epochs=max_epochs,
            log_every_n_steps=50,
            enable_checkpointing=True,
            enable_progress_bar=False,  # disable default progress bar
            deterministic=False,  # setting to True breaks on max_pool_3d
            limit_val_batches=0,  # turn on validation steps
            num_sanity_val_steps=0,
            callbacks=[early_stopping, checkpoint_callback, progress_bar],
            precision="16-mixed",  # Enable automatic mixed precision
        )

        # Initialize model with parameters from config
        # Scale learning rate by number of GPUs (linear scaling rule)
        scaled_learning_rate = model_training_config["learning_rate"] * len(
            devices_training
        )
        model_params = {
            "learning_rate": scaled_learning_rate,
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

        # Convert BatchNorm to SyncBatchNorm for DDP training
        # This synchronizes batch statistics across all GPUs
        if len(devices_training) > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Initialize data module with parameters from config
        # Each GPU uses the full batch size (effective batch size scales with GPUs)
        batch_size_per_device = data_module_config["batch_size"]

        with MissAlignmentDataModule(
            training_directory,
            create_default_generator(**shift_generation_config),
            reconstruction_accelerators=devices_reconstruction,
            n_training_devices=len(devices_training),
            batch_size=batch_size_per_device,
            patch_size=data_module_config["patch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            steps_per_epoch=data_module_config["steps_per_epoch"],
            pool_size=pool_size,
            dataloader_workers=dataloaders_per_trainer,
        ) as dm:
            # enter datamodule context to start the reconstruction worker pool
            # passing datamodule lets Lightning apply DistributedSampler
            trainer.fit(model, datamodule=dm)

            # Synchronize all ranks after training, while DDP is still active
            # This must be BEFORE the context manager exits, as Lightning's DDP
            # process group may become invalid during/after teardown
            distributed_barrier()

        # Only rank 0 performs alignment and file operations
        # Other ranks wait at the barrier at the end of the iteration
        if is_rank_zero():
            best_model_path = Path(trainer.checkpoint_callback.best_model_path)
            rank_zero_print(f"Best model after iteration {x}:", best_model_path)

            # copy best model to training directory as model.ckpt
            # This known path is used by all ranks for the next iteration
            training_model_path = training_directory / "model.ckpt"
            copyfile(best_model_path, training_model_path)
            rank_zero_print(f"Copied best model to {training_model_path}")

            # ============================================================
            # =============== tilt-series alignment step =================
            # ============================================================

            # get list of all files to process for alignment
            tilt_series_list = list(training_directory.glob("*.xml"))

            # run alignment in parallel over all available devices
            run_alignment_parallel(
                model_checkpoint=str(training_model_path),
                tilt_series_list=tilt_series_list,
                output_directory=training_directory,
                setting=iteration_settings["alignment"],
                patch_size=alignment_config["patch_size"],
                patch_overlap=alignment_config["patch_overlap"],
                batch_size=alignment_config["batch_size"],
                apply_ctf=general_config["apply_ctf"],
                downsample=iteration_settings["downsample"],
                devices_list=devices_alignment,
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

            iteration_model_path = iteration_directory / "model.ckpt"
            copyfile(best_model_path, iteration_model_path)
            rank_zero_print(f"Copied best model to {iteration_model_path}")

        # All ranks use the known model.ckpt path for the next iteration
        # (rank 0 wrote it above, barrier below ensures it exists)
        model_training_config["model_checkpoint"] = str(
            training_directory / "model.ckpt"
        )

        # Synchronize all ranks before starting the next iteration
        distributed_barrier()

    return None
