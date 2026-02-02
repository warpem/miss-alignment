import lightning.pytorch as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional
from lightning.pytorch.callbacks import Callback

from miss_alignment.models import (
    Compact3DConvNet,
    Compact3DConvNetGELU,
    Compact3DConvNetSpread,
    Compact3DConvNetWide,
    Compact3DConvNetDeep,
    CompactResNet3D,
)


model_map = {
    "default": Compact3DConvNet,
    "simple": Compact3DConvNet,
    "gelu": Compact3DConvNetGELU,
    "spread": Compact3DConvNetSpread,
    "wide": Compact3DConvNetWide,
    "deep": Compact3DConvNetDeep,
    "resnet": CompactResNet3D,
}


class TripletMarginRankingLoss(nn.Module):
    """
    Precision-weighted Triplet Margin Loss with explicit triplet ordering.

    The loss is weighted by the predicted precision (inverse variance) of each
    sample, allowing the model to express uncertainty about uninformative regions.

    Parameters
    ----------
    margin : float, optional
        Margin for the triplet ranking loss. Default: 1.0
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'
    precision_reg_weight : float, optional
        Weight for precision regularization to prevent collapse. Default: 0.01
    """

    def __init__(self, margin=1.0, reduction="mean", precision_reg_weight=0.01):
        super(TripletMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.precision_reg_weight = precision_reg_weight

    def forward(self, scores, log_precisions, triplet_indices):
        """
        Forward pass using triplet indices to organize samples.

        Parameters
        ----------
        scores : torch.Tensor
            Tensor of shape (batch_size, 3) - alignment scores
        log_precisions : torch.Tensor
            Tensor of shape (batch_size, 3) - log precision for each score
        triplet_indices : torch.Tensor
            Tensor of shape (batch_size, 3) where each row contains indices
            for [anchor_idx, positive_idx, negative_idx]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (triplet_loss, precision_reg) - weighted triplet loss and precision
            regularization term
        """
        # Convert log precision to precision (always positive)
        precisions = log_precisions.exp()

        # Identify example types and compute distances
        example_type = einops.rearrange(triplet_indices.sum(dim=-1), "b -> b 1")
        close_mask = triplet_indices == example_type
        distant_mask = triplet_indices != example_type

        # Extract scores for close pair (aligned) and distant (misaligned)
        close_scores = scores[close_mask]  # (b * 2,)
        close_scores = einops.rearrange(close_scores, "(b n) -> b n", n=2)
        distant_scores = scores[distant_mask]  # (b,)
        distant_scores = einops.rearrange(distant_scores, "b -> b 1")

        # Extract precisions for weighting
        close_precisions = precisions[close_mask]
        close_precisions = einops.rearrange(close_precisions, "(b n) -> b n", n=2)
        distant_precisions = precisions[distant_mask]
        distant_precisions = einops.rearrange(distant_precisions, "b -> b 1")

        # Compute distances
        dist_pos = torch.abs(close_scores[..., 0] - close_scores[..., 1])
        dist_neg = torch.min(
            (close_scores - distant_scores) * example_type, dim=-1
        ).values

        # Compute triplet loss with margin
        triplet_losses = F.relu(dist_pos + dist_neg + self.margin)

        # Weight by geometric mean of precisions in the triplet
        # This downweights triplets where any member has low precision
        all_precisions = torch.cat([close_precisions, distant_precisions], dim=-1)
        triplet_precision = all_precisions.prod(dim=-1).pow(1 / 3)  # geometric mean
        weighted_losses = triplet_losses * triplet_precision

        # Precision regularization: penalize low precision to prevent collapse
        # Using mean of log_precisions (equivalent to log of geometric mean)
        precision_reg = -log_precisions.mean() * self.precision_reg_weight

        # Apply reduction to weighted losses
        if self.reduction == "mean":
            loss = weighted_losses.mean()
        elif self.reduction == "sum":
            loss = weighted_losses.sum()
        else:  # 'none'
            loss = weighted_losses

        return loss, precision_reg


class MissAlignment(pl.LightningModule):
    in_channels: int = 1  # configuration for resnet
    num_classes: int = 1

    def __init__(
        self,
        learning_rate: float = 1e-04,
        warmup_steps: int = 500,
        weight_decay: float = 0,
        margin: float = 0.5,
        precision_reg_weight: float = 0.01,
        # will be saved as a hyperparameter
        multistep_lr_scheduler: Optional[dict] = None,
        model_architecture: str = "default",
    ):
        super().__init__()

        # save hyperparams to self.hparams
        self.save_hyperparameters()

        # store for convenience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.criterion = TripletMarginRankingLoss(
            margin=margin, precision_reg_weight=precision_reg_weight
        )

        model = model_map[model_architecture]
        self.net = model()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning score and log_precision."""
        score, log_precision = self.net(image)
        return score, log_precision

    def _common_step(self, batch, batch_idx):
        # get the batch data
        img1, img2, img3, target = batch
        batch_size = target.shape[0]

        # calculate scores and log_precisions for all three images
        score1, log_prec1 = self(img1)
        score2, log_prec2 = self(img2)
        score3, log_prec3 = self(img3)

        # Stack into (batch, 3) format
        scores = torch.stack((score1, score2, score3), dim=1)
        scores = einops.rearrange(scores, "b d 1 -> b d")
        log_precisions = torch.stack((log_prec1, log_prec2, log_prec3), dim=1)
        log_precisions = einops.rearrange(log_precisions, "b d 1 -> b d")

        # Compute precision-weighted triplet loss
        loss, precision_reg = self.criterion(scores, log_precisions, target)
        total_loss = loss + precision_reg

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        score_aligned = torch.mean(scores[target == 1])
        score_misaligned = torch.mean(scores[target == -1])

        # Compute mean precision for logging
        mean_precision = log_precisions.exp().mean()

        return (
            total_loss,
            loss,
            precision_reg,
            batch_size,
            score_aligned,
            score_misaligned,
            mean_precision,
        )

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        (
            total_loss,
            triplet_loss,
            precision_reg,
            batch_size,
            score_aligned,
            score_misaligned,
            mean_precision,
        ) = self._common_step(batch, batch_idx)

        # Fail fast on NaN loss to prevent corrupted training
        if torch.isnan(total_loss):
            raise ValueError(
                f"NaN loss detected at batch {batch_idx}, epoch {self.current_epoch}. "
                "This may indicate numerical instability. Check your data, learning "
                "rate, or consider disabling AMP."
            )

        self.log(
            name="train_loss",
            value=total_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            name="triplet_loss",
            value=triplet_loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            name="precision_reg",
            value=precision_reg.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # log actual assigned score of the model
        self.log(
            name="train_score_aligned",
            value=score_aligned.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            name="train_score_misaligned",
            value=score_misaligned.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Log mean precision to monitor uncertainty learning
        self.log(
            name="mean_precision",
            value=mean_precision.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
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

        return total_loss

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

    def configure_model(self):
        """Configure model with torch.compile for faster training."""
        # Only compile if not already compiled
        if not isinstance(self.net, torch._dynamo.eval_frame.OptimizedModule):
            self.net = torch.compile(self.net)


class MAProgressBar(Callback):
    """Progress bar showing progress across the entire macro-iteration.

    Instead of showing per-epoch progress, this shows total progress across
    all epochs in the macro-iteration with estimated time remaining.

    Parameters
    ----------
    max_epochs : int
        Maximum number of epochs in the macro-iteration.
    steps_per_epoch : int
        Number of steps per epoch.
    refresh_rate : int, default=10
        How often to update the progress bar (in steps).
    """

    def __init__(self, max_epochs: int, steps_per_epoch: int, refresh_rate: int = 10):
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.refresh_rate = refresh_rate
        # total_steps will be adjusted for world_size in on_train_start
        self.total_steps = max_epochs * steps_per_epoch
        self.pbar = None
        self.start_time = None

    def _is_rank_zero(self, trainer) -> bool:
        """Check if we're on the main process (rank 0)."""
        return trainer.global_rank == 0

    def on_train_start(self, trainer, pl_module):
        # Only create progress bar on rank 0
        if not self._is_rank_zero(trainer):
            return

        import sys
        import time

        from tqdm import tqdm

        # Adjust total_steps for distributed training
        world_size = trainer.world_size
        self.total_steps = self.total_steps // world_size

        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only update progress bar on rank 0
        if not self._is_rank_zero(trainer) or self.pbar is None:
            return

        # Use trainer.global_step which correctly tracks optimizer steps
        # regardless of DDP world size
        global_step = trainer.global_step

        if global_step % self.refresh_rate == 0 or global_step == self.total_steps:
            # Update progress bar to current position
            self.pbar.n = global_step
            self.pbar.refresh()

    def on_train_epoch_end(self, trainer, pl_module):
        # Only update progress bar on rank 0
        if not self._is_rank_zero(trainer) or self.pbar is None:
            return

        # Update postfix with epoch metrics
        metrics = trainer.logged_metrics
        postfix = {}
        if "train_loss" in metrics:
            postfix["loss"] = f"{metrics['train_loss']:.4f}"
        if "train_score_aligned" in metrics:
            postfix["aligned"] = f"{metrics['train_score_aligned']:.3f}"
        if "train_score_misaligned" in metrics:
            postfix["misaligned"] = f"{metrics['train_score_misaligned']:.3f}"

        epoch_info = f"epoch {trainer.current_epoch + 1}/{self.max_epochs}"
        postfix["epoch"] = epoch_info

        if postfix:
            self.pbar.set_postfix(postfix)

    def on_train_end(self, trainer, pl_module):
        # Only close progress bar on rank 0
        if self._is_rank_zero(trainer) and self.pbar is not None:
            self.pbar.close()


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
        """Called at the end of each training epoch.

        Only rank 0 evaluates early stopping logic. The decision is then
        synchronized using Lightning's strategy.reduce_boolean_decision,
        which is designed to work with Lightning's internal DDP operations.
        """
        should_stop = False

        # Only rank 0 evaluates early stopping
        if trainer.global_rank == 0:
            # Check if we should start early stopping yet
            if self._check_scheduler_complete(trainer, pl_module):
                logged_metrics = trainer.logged_metrics

                if "train_loss" not in logged_metrics:
                    raise ValueError("Couldn't find train_loss in logged_metrics")

                current_score = float(logged_metrics["train_loss"])

                if (
                    self.best_score is None
                    or current_score < self.best_score - self.min_delta
                ):
                    self.best_score = current_score
                    self.wait_count = 0
                else:
                    self.wait_count += 1

                if self.wait_count >= self.patience:
                    should_stop = True

        # Use Lightning's strategy to sync the decision across ranks.
        # This is the Lightning-approved way to broadcast boolean decisions
        # and avoids conflicts with internal DDP operations.
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = should_stop
