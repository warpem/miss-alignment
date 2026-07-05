from pathlib import Path

from ..distributed.manager import run_distributed


def run_alignment_parallel(
    model_checkpoint: Path,
    tilt_series_list: list[Path],
    output_directory: Path,
    setting: str | tuple[int, int] | tuple[int, int, int, int],
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    devices_list: list[int],
    n_cluster_workers: int | None = None,
) -> dict[str, float]:
    """Distribute per-tilt-series alignment across local GPUs or a cluster.

    Without --n-cluster-workers, one worker subprocess is spawned per GPU in
    devices_list (local mode, unchanged behaviour). Set --n-cluster-workers N
    to submit N cluster jobs instead; requires MISS_CLUSTER_CONFIG and
    MISS_CLUSTER_SCRIPT to be set.

    Returns dict mapping tilt-series stem names to their final loss values.
    """
    queue_root = output_directory / "tasks"

    return run_distributed(
        tilt_series_list=tilt_series_list,
        model_checkpoint=model_checkpoint,
        output_directory=output_directory,
        setting=setting,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        downsample=downsample,
        devices=devices_list,
        n_cluster_workers=n_cluster_workers,
        queue_root=queue_root,
    )
