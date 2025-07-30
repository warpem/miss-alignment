import pytorch_lightning as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from typing import Optional

# from miss_alignment.models import resnet3d_18
from miss_alignment.models import Compact3DConvNet


class TripletMarginRankingLoss(nn.Module):
    """
    Triplet Margin Loss with explicit triplet ordering based on indices.

    Parameters
    ----------
    margin : float, optional
        Margin for the triplet ranking loss. Default: 1.0
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(self, margin=1.0, reduction="mean"):
        super(TripletMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, triplet_indices):
        """
        Forward pass using triplet indices to organize samples.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor of shape (batch_size, 3)
        triplet_indices : torch.Tensor
            Tensor of shape (batch_size, 3) where each row contains indices
            for [anchor_idx, positive_idx, negative_idx]

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        example_type = einops.rearrange(triplet_indices.sum(dim=-1), "b -> b 1")
        close = embeddings[triplet_indices == example_type]  # (b, 2)
        close = einops.rearrange(close, "(b n) -> b n", n=2)
        distant = embeddings[triplet_indices != example_type]  # (b, 1)
        distant = einops.rearrange(distant, "b -> b 1")

        dist_pos = torch.abs(close[..., 0] - close[..., 1])
        dist_neg = torch.min((close - distant) * example_type, dim=-1).values

        # Compute triplet loss with margin
        losses = F.relu(dist_pos + dist_neg + self.margin)

        # Apply reduction
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:  # 'none'
            return losses


class MissAlignment(pl.LightningModule):
    in_channels: int = 1  # configuration for resnet
    num_classes: int = 1

    def __init__(
        self,
        learning_rate: float = 1e-04,
        warmup_steps: int = 500,
        weight_decay: float = 0,
        margin: float = 0.5,
        lr_scheduler: Optional[dict] = None,
    ):
        super().__init__()

        # save hyperparams to self.hparams
        self.save_hyperparameters()

        # store for convenience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.criterion = TripletMarginRankingLoss(margin=margin)
        self.net = Compact3DConvNet()  # resnet3d_18()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        return out

    def _common_step(self, batch):
        # get the batch data
        img1, img2, img3, target = batch
        batch_size = target.shape[0]

        # calculate scores and loss
        scores = torch.stack((self(img1), self(img2), self(img3)))
        scores = einops.rearrange(scores, "d b 1 -> b d")
        loss = self.criterion(scores, target)

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        score_aligned = torch.mean(scores[target == 1])
        score_misaligned = torch.mean(scores[target == -1])

        return loss, batch_size, score_aligned, score_misaligned

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        loss, batch_size, score_aligned, score_misaligned = self._common_step(batch)

        self.log(
            name="loss_epoch",
            value=loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name="train_score_aligned",
            value=score_aligned,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name="train_score_misaligned",
            value=score_misaligned,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            name="learning_rate",
            value=current_lr,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def predict_step(self):
        pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate schedulers.

        Returns
        -------
        dict or optimizer
            If lr_scheduler is configured, returns dict with optimizer and schedulers.
            Otherwise returns optimizer only.
        """
        # Use AdamW if weight_decay is specified, otherwise use Adam
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.learning_rate,
            )

        # Warm-up scheduler (linear warm-up) - tracks steps
        warmup_scheduler = {
            "scheduler": LinearLR(
                optimizer,
                start_factor=0.001,  # Start at 0.1% of self.lr
                end_factor=1.0,  # End at 100% of self.lr
                total_iters=self.warmup_steps,
            ),
            "interval": "step",  # track steps, not epochs
            "frequency": 1,
        }

        # Check if lr_scheduler is defined in hparams
        if (
            hasattr(self.hparams, "lr_scheduler")
            and self.hparams.lr_scheduler is not None
        ):
            from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR

            scheduler_config = self.hparams.lr_scheduler
            plateau_scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config["mode"],
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
                cooldown=3,  # use some cooldown before continuing tracking
            )

            scheduler = {
                "scheduler": SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, plateau_scheduler],
                    milestones=[5],  # switch after 5 epochs
                ),
                "monitor": "loss_epoch",
                "interval": "epoch",
                # tracks epochs
                "frequency": 1,
            }
            return [optimizer], [scheduler]

        # Return only warmup scheduler if no plateau scheduler configured
        return [optimizer], [warmup_scheduler]
