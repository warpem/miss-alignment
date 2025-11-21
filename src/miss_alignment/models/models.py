import lightning.pytorch as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional
from lightning.pytorch.callbacks import Callback

# from miss_alignment.models import resnet3d_18
from miss_alignment.models import (
    Compact3DConvNet,
    Compact3DConvNetGELU,
    Compact3DConvNetSpread,
    Compact3DConvNetWide,
    Compact3DConvNetDeep,
    CompactResNet3D,
)
from ..data._pool_monitor import SimplePoolMonitor


model_map = {
    "default": Compact3DConvNet,
    "gelu": Compact3DConvNetGELU,
    "spread": Compact3DConvNetSpread,
    "wide": Compact3DConvNetWide,
    "deep": Compact3DConvNetDeep,
    "resnet": CompactResNet3D,
}


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
        # will be saved as a hyperparameter
        multistep_lr_scheduler: Optional[dict] = None,
        monitor: Optional[SimplePoolMonitor] = None,
        model_architecture: str = "default",
    ):
        super().__init__()

        # save hyperparams to self.hparams
        self.save_hyperparameters()

        # store for convenience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.criterion = TripletMarginRankingLoss(margin=margin)

        model = model_map[model_architecture]
        self.net = model()  # resnet3d_18()

        self.monitor = monitor

    def configure_model(self):
        """Configure model with torch.compile for faster training."""
        # Only compile if not already compiled
        if not isinstance(self.net, torch._dynamo.eval_frame.OptimizedModule):
            self.net = torch.compile(self.net)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        return out

    def _common_step(self, batch, batch_idx):
        # get the batch data
        img1, img2, img3, target = batch
        batch_size = target.shape[0]

        # Track consumption
        if self.monitor is not None:
            self.monitor.record_consumption(batch_size)

            # Print occasionally
            if batch_idx % 50 == 0 and batch_idx > 0:
                self.monitor.print_stats()

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
        (loss, batch_size, score_aligned, score_misaligned) = self._common_step(
            batch, batch_idx
        )

        self.log(  # add it to progress bar
            name="train_loss",
            value=loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # log actual assigned score of the model
        self.log(
            name="train_score_aligned",
            value=score_aligned.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            name="train_score_misaligned",
            value=score_misaligned.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            name="learning_rate",
            value=current_lr,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

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

        scheduler_config = self.hparams.multistep_lr_scheduler
        multistep = MultiStepLR(
            optimizer,
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config["gamma"],
        )

        scheduler = {
            "scheduler": multistep,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class MAEarlyStopping(Callback):
    """Early stopping callback that waits until MultiStepLR reaches its lowest point.

    Parameters
    ----------
    patience : int, default=4
        Number of steps to wait after last improvement before stopping.
    min_delta : float, default=0.0
        Minimum change to qualify as improvement.
    wait_for_scheduler : bool, default=True
        If True, waits for MultiStepLR to complete all milestones before starting.
    """

    def __init__(self, patience=5, min_delta=0.001, wait_for_scheduler=True):
        self.patience = patience
        self.min_delta = min_delta
        self.wait_for_scheduler = wait_for_scheduler
        self.best_score = None
        self.wait_count = 0
        self.scheduler_complete = False

    def _check_scheduler_complete(self, trainer, pl_module):
        """Check if MultiStepLR has completed all milestones."""
        if not self.wait_for_scheduler or self.scheduler_complete:
            return True

        # Check if module has MultiStepLR configured
        if not (
            hasattr(pl_module.hparams, "multistep_lr_scheduler")
            and pl_module.hparams.multistep_lr_scheduler is not None
        ):
            # No scheduler configured, can start early stopping immediately
            self.scheduler_complete = True
            return True

        scheduler_config = pl_module.hparams.multistep_lr_scheduler
        milestones = scheduler_config["milestones"]

        # Check if we've passed all milestones
        if trainer.current_epoch >= max(milestones):
            self.scheduler_complete = True
            return True

        return False

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training batch."""
        # Check if we should start early stopping yet
        if not self._check_scheduler_complete(trainer, pl_module):
            return

        logged_metrics = trainer.logged_metrics

        if "train_loss" in logged_metrics:
            current_score = logged_metrics["train_loss"]
        else:
            raise ValueError("Couldn't find train_loss in pl_module")

        if self.best_score is None or current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            trainer.should_stop = True
