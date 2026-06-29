"""Tests for the scoped-DDP launcher mechanism in ``train.py``.

These do not require a GPU. They exercise the part of the training refactor that
is independent of CUDA: running one worker per rank, acting as the external
process launcher (exporting ``LOCAL_RANK``) so Lightning attaches to the existing
process group instead of re-spawning the program, and rank-gating with
``is_rank_zero()``. The multi-rank case uses ``accelerator="cpu"`` + gloo so it
runs on a single-GPU (or no-GPU) machine.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy

from miss_alignment.train import (
    _find_free_port,
    _resolve_start_checkpoint,
    _set_ddp_env,
)
from miss_alignment.utils import is_rank_zero, sync_start_iteration_xmls


class _TinyDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return torch.randn(4), torch.randn(1)


class _TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def training_step(self, batch, _):
        x, y = batch
        loss = ((self.lin(x) - y) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _launch_and_train(rank: int, world_size: int, master_port: int, tmp: Path) -> None:
    """Stand-in for ``_training_worker`` reusing its real launcher setup on CPU.

    Writes one marker file per rank: ``<pid> <pre_dist_rank0> <global_rank>
    <world_size>``. ``pre_dist_rank0`` is the value of ``is_rank_zero()`` before
    the process group exists, which is what gates the reconstruction pool in the
    real datamodule.
    """
    multi = world_size > 1
    if multi:
        # exercise the real rendezvous setup from train.py (not a copy)
        _set_ddp_env(rank, world_size, master_port)

        # Simulate a single-task, non-interactive SLURM allocation: every
        # spawned worker inherits the SAME SLURM_PROCID=0 / SLURM_NTASKS=1. If
        # the strategy let Lightning auto-detect SLURMEnvironment, all workers
        # would collapse to rank 0 of a world of size 1. Pinning
        # LightningEnvironment (below) must keep our exported ranks authoritative.
        os.environ["SLURM_NTASKS"] = "1"
        os.environ["SLURM_PROCID"] = "0"
        os.environ["SLURM_LOCALID"] = "0"
        os.environ["SLURM_JOB_NAME"] = "ddp-regression"  # not bash/interactive
        os.environ["SLURM_JOB_ID"] = "123456"

    pre_dist_rank_zero = is_rank_zero()

    strategy = (
        DDPStrategy(
            cluster_environment=LightningEnvironment(),
            process_group_backend="gloo",
            find_unused_parameters=False,
        )
        if multi
        else "auto"
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=world_size,
        strategy=strategy,
        max_epochs=1,
        limit_train_batches=2,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_TinyModel(), DataLoader(_TinyDataset(), batch_size=4))

    (tmp / f"rank_{rank}.txt").write_text(
        f"{os.getpid()} {pre_dist_rank_zero} {trainer.global_rank} {trainer.world_size}"
    )


@pytest.mark.filterwarnings("ignore")
def test_single_device_runs_in_process(tmp_path):
    """``world_size == 1`` runs in the calling process with no DDP or spawn."""
    _launch_and_train(0, 1, _find_free_port(), tmp_path)

    pid, pre_zero, global_rank, world = (tmp_path / "rank_0.txt").read_text().split()
    assert int(pid) == os.getpid()  # stayed in this process
    assert pre_zero == "True"  # rank-0 gate -> would spawn the recon pool
    assert (global_rank, world) == ("0", "1")  # single-process Lightning view


@pytest.mark.filterwarnings("ignore")
def test_external_launcher_overrides_slurm_detection(tmp_path):
    """``mp.spawn`` + pinned ``LightningEnvironment`` keeps our ranks.

    The workers run with single-task SLURM env vars set, so this also guards the
    regression where Lightning would auto-detect ``SLURMEnvironment`` and collapse
    every spawned worker to rank 0 of a world of size 1.
    """
    world_size = 2
    torch_mp.spawn(
        _launch_and_train,
        args=(world_size, _find_free_port(), tmp_path),
        nprocs=world_size,
        join=True,
    )

    # back in the single parent process; inspect what the workers recorded
    rows = [m.read_text().split() for m in sorted(tmp_path.glob("rank_*.txt"))]

    assert len(rows) == world_size  # one worker per rank
    assert str(os.getpid()) not in {r[0] for r in rows}  # no re-exec of program
    assert sum(r[1] == "True" for r in rows) == 1  # exactly one recon-pool gate
    assert {r[2] for r in rows} == {"0", "1"}  # our ranks won over SLURM_PROCID=0
    assert {r[3] for r in rows} == {"2"}  # world_size 2, not SLURM_NTASKS=1


def test_resolve_start_checkpoint_iter0_uses_config(tmp_path):
    """iter 0 starts from the config's checkpoint when one is given."""
    ckpt = tmp_path / "pretrained.ckpt"
    ckpt.touch()
    assert _resolve_start_checkpoint(0, tmp_path, str(ckpt)) == ckpt


def test_resolve_start_checkpoint_iter0_none(tmp_path):
    """iter 0 with no config checkpoint trains from scratch."""
    assert _resolve_start_checkpoint(0, tmp_path, None) is None


def test_resolve_start_checkpoint_resume_prefers_iter_dir(tmp_path):
    """Resuming loads iter{N}/model.ckpt, ignoring the config and the fallback."""
    (tmp_path / "iter2").mkdir()
    iter_ckpt = tmp_path / "iter2" / "model.ckpt"
    iter_ckpt.touch()
    (tmp_path / "model.ckpt").touch()  # fallback also present
    assert _resolve_start_checkpoint(2, tmp_path, "ignored.ckpt") == iter_ckpt


def test_resolve_start_checkpoint_resume_falls_back(tmp_path):
    """Resuming uses training_directory/model.ckpt when iter{N}/ is absent."""
    fallback = tmp_path / "model.ckpt"
    fallback.touch()
    assert _resolve_start_checkpoint(3, tmp_path, None) == fallback


def test_resolve_start_checkpoint_resume_missing_returns_none(tmp_path):
    """Resuming with no checkpoint anywhere returns None (regression for #111)."""
    assert _resolve_start_checkpoint(3, tmp_path, None) is None


def test_sync_start_iteration_xmls_iter0_backs_up(tmp_path):
    """iter 0 snapshots the training-directory XMLs into iter0/ (originals kept)."""
    (tmp_path / "a.xml").write_text("orig-a")
    (tmp_path / "b.xml").write_text("orig-b")

    sync_start_iteration_xmls(0, tmp_path)

    assert (tmp_path / "iter0" / "a.xml").read_text() == "orig-a"
    assert (tmp_path / "iter0" / "b.xml").read_text() == "orig-b"
    # the working copies are untouched
    assert (tmp_path / "a.xml").read_text() == "orig-a"


def test_sync_start_iteration_xmls_resume_restores(tmp_path):
    """Resuming overwrites the working XMLs with the iter{N}/ snapshot."""
    iter_dir = tmp_path / "iter2"
    iter_dir.mkdir()
    (iter_dir / "a.xml").write_text("iter2-a")
    # the training directory holds stale/partial state from a crashed attempt
    (tmp_path / "a.xml").write_text("stale-a")

    sync_start_iteration_xmls(2, tmp_path)

    assert (tmp_path / "a.xml").read_text() == "iter2-a"
    # the snapshot itself is not modified
    assert (iter_dir / "a.xml").read_text() == "iter2-a"


def test_sync_start_iteration_xmls_resume_missing_raises(tmp_path):
    """Resuming at an iteration with no snapshot directory is an error."""
    with pytest.raises(FileNotFoundError, match="Cannot resume at iteration 3"):
        sync_start_iteration_xmls(3, tmp_path)
