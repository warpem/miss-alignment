import logging
from pathlib import Path
from shutil import copyfile
from typing import Optional

import yaml

import typer
import torch

from lightning.pytorch import seed_everything

from ._cli import OPTION_PROMPT_KWARGS, cli
from .utils import configure_logging, sync_start_iteration_xmls
from .data.io import exclude_nonfinite_alignment_tilt_series
from .alignment import run_alignment_parallel
from .prepare_stacks import prepare_stacks_parallel
from .preprocessing import run_cross_correlation_alignment_parallel

logger = logging.getLogger(__name__)


@cli.command(name="infer", no_args_is_help=True)
def infer_miss_align(
    config_file: Path = typer.Option(
        "inference_config_template.yaml", **OPTION_PROMPT_KWARGS
    ),
    start_at_iteration: int = typer.Option(
        0,
        help="Continue from a specific iteration in the config.",
    ),
    prepare_stacks: Optional[float] = typer.Option(
        None,
        help="Pixel size (in Angstroms) for preprocessing tilt stacks. "
        "If provided, loads raw tilt images, rescales to this pixel size, "
        "and creates tilt stacks with thumbnails before alignment.",
    ),
    preprocess: bool = typer.Option(
        False,
        help="Run cross-correlation based alignment before the inference "
        "iterations. This performs coarse alignment.",
    ),
) -> None:
    """Align a dataset by applying models from a previous training run.

    Inference mode skips the per-iteration model training entirely. For each
    iteration it loads the model that a previous `train` run saved at
    `<model_run_directory>/iter{N}/model.ckpt` and runs only the alignment
    phase, following the `iteration_settings` schedule from the config.

    Alignment is automatically spread over all visible GPUs; to limit this,
    set `CUDA_VISIBLE_DEVICES` to the subset of devices that should be used
    (not needed under a SLURM submission, which already scopes visible GPUs).
    """

    # honor MISS_ALIGNMENT_LOG_LEVEL and quiet Lightning's banners
    configure_logging()

    # check hardware settings, limit cpu multithreading
    torch.set_num_threads(1)

    # Load configuration from YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    general_config = config["general"]
    alignment_config = config["tilt_series_alignment"]

    # the dataset to align (and where aligned output is written)
    data_directory = Path(general_config["data_directory"])
    data_directory.mkdir(exist_ok=True, parents=True)

    # the finished run holding iter1..iterN/model.ckpt
    model_run_directory = Path(general_config["model_run_directory"])

    # Set up environment
    torch.set_float32_matmul_precision("medium")
    seed_everything(general_config["seed"], workers=True)

    # For alignment, use all visible GPUs
    n_visible_gpus = torch.cuda.device_count()
    devices_alignment = list(range(n_visible_gpus))

    print("Inference mode (apply models from a previous run):")
    print(f"  data directory:      {data_directory}")
    print(f"  model run directory: {model_run_directory}")
    print(f"  alignment devices:   {devices_alignment}")

    # Prepare tilt stacks if requested
    if prepare_stacks is not None:
        if prepare_stacks <= 0:
            raise ValueError("--prepare-stacks must be a positive non-zero number")
        prepare_stacks_parallel(
            training_directory=data_directory,
            desired_pixel_size=prepare_stacks,
            devices=devices_alignment,
        )

    # Run preprocessing if requested
    if preprocess:
        if start_at_iteration != 0:
            raise ValueError(
                "Running preprocessing while not starting at iteration 0. "
                "This is likely not desirable behaviour."
            )
        # Back up original data to pre-iter directory
        preiter_directory = data_directory / "pre-iter"
        preiter_directory.mkdir(parents=True, exist_ok=True)
        for xml_file in data_directory.glob("*.xml"):
            copyfile(xml_file, preiter_directory / xml_file.name)
        print(f"Backed up original metadata to {preiter_directory}")

        run_cross_correlation_alignment_parallel(
            training_directory=data_directory,
            devices=devices_alignment,
        )

    exclude_nonfinite_alignment_tilt_series(data_directory)

    start_iter = start_at_iteration
    end_iter = len(general_config["iteration_settings"])

    # Resolve and validate every model up front so a missing checkpoint fails
    # before any alignment work is done.
    model_paths: dict[int, Path] = {}
    for x in range(start_iter, end_iter):
        model_path = model_run_directory / f"iter{x + 1}" / "model.ckpt"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Inference needs a model for iteration {x + 1} but "
                f"{model_path} does not exist. The model run directory must "
                f"contain iter1..iter{end_iter}/model.ckpt from a previous run."
            )
        model_paths[x] = model_path

    # iter 0 snapshots the input alignments as a baseline; resuming restores the
    # working alignments from that iteration's snapshot.
    sync_start_iteration_xmls(start_iter, data_directory)

    for x in range(start_iter, end_iter):
        iteration_settings = general_config["iteration_settings"][x]
        alignment_mode = iteration_settings["alignment"]
        model_checkpoint = model_paths[x]

        print(f"\n{'=' * 60}")
        print(f"Iteration {x + 1}/{end_iter} - Alignment: {alignment_mode}")
        print(f"  model: {model_checkpoint}")
        print(f"{'=' * 60}\n")

        # get list of all files to process for alignment
        tilt_series_list = list(data_directory.glob("*.xml"))

        # run alignment in parallel over all available devices
        run_alignment_parallel(
            model_checkpoint=str(model_checkpoint),
            tilt_series_list=tilt_series_list,
            output_directory=data_directory,
            setting=iteration_settings["alignment"],
            patch_size=alignment_config["patch_size"],
            patch_overlap=alignment_config["patch_overlap"],
            batch_size=alignment_config["batch_size"],
            apply_ctf=general_config["apply_ctf"],
            downsample=iteration_settings["downsample"],
            devices_list=devices_alignment,
        )

        # make copies of the xml files after alignment
        iteration_directory = data_directory / ("iter" + str(x + 1))
        iteration_directory.mkdir(parents=True, exist_ok=True)

        for xml_file in data_directory.glob("*.xml"):
            destination_xml = iteration_directory / xml_file.name
            copyfile(xml_file, destination_xml)

            # copy the file with the alignment loss
            loss_json = xml_file.stem + "_alignment_loss.json"
            destination_json = iteration_directory / loss_json
            copyfile(data_directory / loss_json, destination_json)

        # record which model produced this iteration's alignment (provenance)
        (iteration_directory / "model_source.txt").write_text(
            f"{model_checkpoint.resolve()}\n"
        )

    return None
