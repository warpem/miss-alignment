import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, SequentialLR
)
from torch.nn import MarginRankingLoss

from ._resnet import resnet3d_18
from ._compact import Compact3DConvNet


class RegularizedMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_reg=0.1):
        """
        Parameters
        ----------
        margin : float, optional
            Margin for ranking, by default 0.1
        lambda_reg : float, optional
            L2 regularization strength, by default 0.01
        """
        super().__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, scores1, scores2, targets):
        """
        Parameters
        ----------
        scores1 : torch.Tensor
            Model predictions for first images
        scores2 : torch.Tensor
            Model predictions for second images
        targets : torch.Tensor
            1 if scores1 should be higher than scores2, -1 if vice versa

        Returns
        -------
        torch.Tensor
            Computed loss value with regularization
        """
        # Apply ranking loss
        ranking_loss = self.ranking_loss(scores1, scores2, targets)

        # Add L2 regularization on scores to prevent inflation
        reg_loss = torch.mean(scores1 ** 2) + torch.mean(scores2 ** 2)

        # Combined loss
        total_loss = ranking_loss + self.lambda_reg * reg_loss

        return total_loss


class MissAlignment(pl.LightningModule):
    in_channels: int = 1
    num_classes: int = 1

    def __init__(
            self, learning_rate: float = 1e-04, warmup_epochs=10, margin=.5
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.warmup_epochs = warmup_epochs
        self.criterion = MarginRankingLoss(margin=margin)
        self.net = Compact3DConvNet()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        return out

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        # get the batch data
        img1, img2, target = batch
        batch_size = target.shape[0]

        # calculate scores and loss
        score1 = self(img1)
        score2 = self(img2)
        loss = self.criterion(score1, score2, target)

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        ind = target < 0  # if target==-1 the first input is aligned
        s_a = torch.mean(  # if target==1 the first input is misaligned
            torch.cat((score1[ind], score2[torch.logical_not(ind)]))
        )
        s_m = torch.mean(
            torch.cat((score2[ind], score1[torch.logical_not(ind)]))
        )

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
        # get the batch data
        img1, img2, target = batch
        batch_size = target.shape[0]

        # calculate scores and loss
        score1 = self(img1)
        score2 = self(img2)
        loss = self.criterion(score1, score2, target)

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        ind = target < 0  # if target==-1 the first input is aligned
        s_a = torch.mean(  # if target==1 the first input is misaligned
            torch.cat((score1[ind], score2[torch.logical_not(ind)]))
        )
        s_m = torch.mean(
            torch.cat((score2[ind], score1[torch.logical_not(ind)]))
        )

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
            T_mult=2,
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
