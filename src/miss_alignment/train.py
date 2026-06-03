import os
import socket
from pathlib import Path
from shutil import copyfile
from typing import Optional
import yaml

import typer
import torch
import torch.multiprocessing as torch_mp

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy

from ._cli import OPTION_PROMPT_KWARGS, cli
from .data import MissAlignmentDataModule
from .data.shift_generation import create_default_generator
from .models import MissAlignment, MAEarlyStopping, MAProgressBar
from .alignment import run_alignment_parallel
from .prepare_stacks import prepare_stacks_parallel
from .preprocessing import run_cross_correlation_alignment_parallel


def parse_device_list(value: str) -> list[int]:
    """Parse comma-separated device list like '0,1,2' into [0, 1, 2]."""
    return [int(x.strip()) for x in value.split(",")]


def _find_free_port() -> int:
    """Find a free TCP port on localhost for the DDP rendezvous.

    A fresh port is chosen for every training phase so that re-creating the
    process group each iteration cannot collide with a socket still lingering
    in TIME_WAIT from the previous iteration.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _set_ddp_env(rank: int, world_size: int, master_port: int) -> None:
    """Export the rendezvous variables so we act as the DDP process launcher.

    Lightning's ``LightningEnvironment`` keys on ``LOCAL_RANK`` being present to
    detect an external launcher (torchrun/SLURM-style) and connect to this
    existing process group instead of spawning a fresh copy of the program.
    Single-node only, so the master address is always localhost.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = "0"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)


def _resolve_start_checkpoint(
    start_iter: int,
    training_directory: Path,
    config_checkpoint: Optional[str],
) -> Optional[Path]:
    """Pick the checkpoint to start the first training iteration from.

    iter 0 uses the config's ``model_checkpoint`` (or None for random weights).
    When resuming (``start_iter > 0``), load the model saved at the end of the
    previous iteration (``iter{start_iter}/model.ckpt``), falling back to
    ``training_directory/model.ckpt``; the config's checkpoint is irrelevant in
    that case. Returns None (train from scratch) if no checkpoint is found.
    """
    if start_iter == 0:
        return Path(config_checkpoint) if config_checkpoint else None

    iter_checkpoint = training_directory / f"iter{start_iter}" / "model.ckpt"
    fallback_checkpoint = training_directory / "model.ckpt"
    if iter_checkpoint.exists():
        print(
            f"Resuming at iteration {start_iter}: loading model from {iter_checkpoint}"
        )
        return iter_checkpoint
    if fallback_checkpoint.exists():
        print(
            f"Resuming at iteration {start_iter}: "
            f"loading model from {fallback_checkpoint}"
        )
        return fallback_checkpoint
    print(
        f"Warning: resuming at iteration {start_iter} but no model checkpoint "
        "found; training will start from random weights."
    )
    return None


def _training_worker(
    rank: int,
    world_size: int,
    master_port: int,
    iteration: int,
    model_checkpoint: Optional[Path],
    training_directory: Path,
    devices_training: list[int],
    devices_reconstruction: list[int],
    general_config: dict,
    model_training_config: dict,
    data_module_config: dict,
    shift_generation_config: dict,
    iteration_settings: dict,
    dataloaders_per_trainer: int,
    pool_size: int,
) -> None:
    """Run a single model-training phase to completion.

    This is the unit of work that is parallelised across the training GPUs. For
    multi-GPU training it is launched once per GPU via ``mp.spawn`` and acts as
    its own process launcher: by exporting ``LOCAL_RANK`` (and the other
    rendezvous variables) Lightning's ``LightningEnvironment`` detects an
    external launcher and connects to the existing process group instead of
    spawning a fresh copy of the whole program. For single-GPU training it is
    called directly in the main process with ``world_size == 1`` and no DDP.

    Rank 0 publishes the best checkpoint to ``training_directory / model.ckpt``
    before returning; the main process picks it up from there for alignment.
    """
    multi_gpu = world_size > 1

    if multi_gpu:
        _set_ddp_env(rank, world_size, master_port)

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("medium")
    seed_everything(general_config["seed"], workers=True)

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
        filename=str(iteration) + "_{epoch}--{step}--{train_loss:.3f}",
        save_on_train_epoch_end=True,
    )

    # Progress bar showing total progress across all epochs
    max_epochs = model_training_config["max_epochs_per_iteration"]
    progress_bar = MAProgressBar(
        max_epochs=max_epochs,
        steps_per_epoch=data_module_config["steps_per_epoch"],
        refresh_rate=10,
    )

    # Use DDP only for multi-GPU; the workers live only for this fit() call, so
    # the default DDP timeout is fine (no rank sits idle during alignment).
    # Pin the cluster environment so that we (the mp.spawn launcher) stay in
    # control of the ranks. Without this, Lightning auto-detects SLURMEnvironment
    # inside any SLURM allocation and reads SLURM_PROCID/SLURM_NTASKS instead of
    # the RANK/WORLD_SIZE/LOCAL_RANK we export, which would collapse the spawned
    # workers into world_size=1. LightningEnvironment keys off LOCAL_RANK.
    strategy = (
        DDPStrategy(
            cluster_environment=LightningEnvironment(),
            find_unused_parameters=False,  # feed-forward model
        )
        if multi_gpu
        else "auto"
    )
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
        benchmark=True,  # cuDNN caches fastest conv algo for fixed input shape
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

    if model_checkpoint is not None:
        model = MissAlignment.load_from_checkpoint(model_checkpoint, **model_params)
    else:
        model = MissAlignment(**model_params)

    # Each GPU uses the full batch size (effective batch size scales with GPUs)
    with MissAlignmentDataModule(
        training_directory,
        create_default_generator(**shift_generation_config),
        reconstruction_accelerators=devices_reconstruction,
        n_training_devices=len(devices_training),
        batch_size=data_module_config["batch_size"],
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

    # Rank 0 publishes the best checkpoint at a known path for the main process.
    if trainer.is_global_zero:
        best_model_path = Path(trainer.checkpoint_callback.best_model_path)
        training_model_path = training_directory / "model.ckpt"
        copyfile(best_model_path, training_model_path)
        print(f"Best model after iteration {iteration}: {best_model_path}")
        print(f"Copied best model to {training_model_path}")


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

    # Each macro-iteration runs two phases; report which devices each one uses.
    print("Macro-iteration phase 1 (model training):")
    print(f"  training devices:       {devices_training}")
    print(f"  reconstruction devices: {devices_reconstruction}")
    print("Macro-iteration phase 2 (tilt-series alignment):")
    print(f"  alignment devices:      {devices_alignment}")

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
            run_cross_correlation_alignment_parallel(
                training_directory=training_directory,
                devices=devices_alignment,
                n_processes=4,
            )

    start_iter = start_at_iteration
    end_iter = len(general_config["iteration_settings"])

    # Determine the model checkpoint to start the first training iteration from.
    training_model_path = _resolve_start_checkpoint(
        start_iter, training_directory, model_training_config["model_checkpoint"]
    )

    # make copies of the xml files and model we're starting from
    iteration_directory = training_directory / ("iter" + str(start_iter))
    iteration_directory.mkdir(parents=True, exist_ok=True)
    for xml_file in training_directory.glob("*.xml"):
        destination = iteration_directory / xml_file.name
        copyfile(xml_file, destination)

    if training_model_path is not None:
        copyfile(training_model_path, iteration_directory / "model.ckpt")

    for x in range(start_iter, end_iter):
        # ============================================================
        # ================= model training step ======================
        # ============================================================
        iteration_settings = general_config["iteration_settings"][x]
        alignment_mode = iteration_settings["alignment"]
        print(f"\n{'=' * 60}")
        print(f"Iteration {x + 1}/{end_iter} - Alignment: {alignment_mode}")
        print(f"{'=' * 60}\n")

        # Run the training phase. Multi-GPU spawns one DDP worker per training
        # device and joins; single-GPU runs in-process. Either way the main
        # process is single-process again once this returns, with the best
        # checkpoint published at training_directory / model.ckpt.
        world_size = len(devices_training)
        worker_args = (
            world_size,
            _find_free_port(),
            x,
            training_model_path,
            training_directory,
            devices_training,
            devices_reconstruction,
            general_config,
            model_training_config,
            data_module_config,
            shift_generation_config,
            iteration_settings,
            dataloaders_per_trainer,
            pool_size,
        )
        if world_size > 1:
            torch_mp.spawn(
                _training_worker, args=worker_args, nprocs=world_size, join=True
            )
        else:
            _training_worker(0, *worker_args)

        # This known path was written by rank 0 and is used for alignment and
        # as the starting checkpoint for the next iteration.
        training_model_path = training_directory / "model.ckpt"

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
        copyfile(training_model_path, iteration_model_path)
        print(f"Copied best model to {iteration_model_path}")

        # training_model_path now points at the model just trained
        # (set after the worker above); the next iteration starts from it.

    return None
