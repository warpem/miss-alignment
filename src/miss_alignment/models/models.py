from typing import Optional, Callable, Any

import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, SequentialLR
)

from ._resnet import resnet3d_18
from ._compact import Compact3DConvNet


LAMBDA = .1  # earlier .01


def loss_l2(s_a: torch.Tensor, s_m: torch.Tensor) -> torch.Tensor:
    """Contrastive loss with L2-normalization.

    loss = s_a - s_m + LAMBDA * (s_m ** 2 + s_a ** 2)

    The loss is minimized, so a higher value for the misaligned volume is encouraged.

    Parameters
    ----------
    s_a: torch.Tensor
        Score assigned to aligned volume. (b,)
    s_m: torch.Tensor
        Score assigned to misaligned volume.  (b,)

    Returns
    -------
    loss: torch.Tensor
    """
    return torch.mean(s_a - s_m + LAMBDA * (s_m ** 2 + s_a ** 2))  # (b,)


def margin_loss(s_a, s_m, margin=.5):
    return torch.mean(torch.clamp(margin - (s_m - s_a), min=0.))


class MissAlignment(pl.LightningModule):
    in_channels: int = 1
    num_classes: int = 1

    def __init__(
            self, learning_rate: float = 1e-04, warmup_epochs=10, margin=1.
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.warmup_epochs = warmup_epochs
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self.net = Compact3DConvNet()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        return out

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        img1, img2, target = batch
        score1 = self(img1)
        score2 = self(img2)
        loss = self.criterion(score1, score2, target)
        # aligned, misaligned = batch["aligned"], batch["misaligned"]
        # s_a, s_m = self(aligned), self(misaligned)
        # loss = loss_l2(s_a, s_m)

        ind = target > 0
        s_a = (score1[ind].mean() + score2[torch.logical_not(ind)].mean()) / 2
        s_m = (score2[ind].mean() + score1[torch.logical_not(ind)].mean()) / 2
        batch_size = target.shape[0]
        self.log(
            name="train loss",
            value=loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name="train s_a",
            value=s_a,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name="train s_m",
            value=s_m,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log(
            name='learning_rate',
            value=current_lr,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        img1, img2, target = batch
        score1 = self(img1)
        score2 = self(img2)
        loss = self.criterion(score1, score2, target)
        ind = target > 0
        s_a = (score1[ind].mean() + score2[torch.logical_not(ind)].mean()) / 2
        s_m = (score2[ind].mean() + score1[torch.logical_not(ind)].mean()) / 2
        batch_size = target.shape[0]
        self.log(
            name="val loss",
            value=loss,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            name="val s_a",
            value=s_a,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            name="val s_m",
            value=s_m,
            batch_size=batch_size,
            prog_bar=True,
        )
        return loss

    def predict_step(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,
        )

        # Warm-up scheduler (linear warm-up)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of self.lr
            end_factor=1.0,  # End at 100% of self.lr
            total_iters=self.warmup_epochs
        )

        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Number of epochs/iterations for one cycle
            # eta_min=1e-06. learning rate at the end of the cycle (default 0)
        )

        sequential_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]  # Switch at the end of warm-up
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
