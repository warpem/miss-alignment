import pytorch_lightning as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
from pytorch_lightning.callbacks import Callback

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
        loss_metric_steps: int = 1000,
        lr_scheduler: Optional[dict] = None,
    ):
        super().__init__()

        # save hyperparams to self.hparams
        self.save_hyperparameters()

        # store for convenience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # initialize buffer for loss values
        self.loss_metric_steps = loss_metric_steps
        self.loss_buffer = deque(maxlen=loss_metric_steps)

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

        self.loss_buffer.append(loss.item())

        if len(self.loss_buffer) == self.loss_metric_steps:
            avg_loss = sum(self.loss_buffer) / self.loss_metric_steps
            self.loss_buffer.clear()

            # Force immediate logging regardless of log_every_n_steps
            self.logger.log_metrics(
                {
                    "train_loss": avg_loss,
                    "train_score_aligned": score_aligned,
                    "train_score_misaligned": score_misaligned,
                },
                step=self.global_step,
            )
            self.log(  # add it to progress bar, this only works if
                name="train_loss",  # log_every_n_steps has a frequency
                value=avg_loss,  # that matches loss_metric_steps
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=False,
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

        # # Warm-up scheduler (linear warm-up) - tracks steps
        # warmup = LinearLR(
        #     optimizer,
        #     start_factor=0.001,  # Start at 0.1% of self.lr
        #     end_factor=1.0,  # End at 100% of self.lr
        #     total_iters=self.warmup_steps,
        # )
        # warmup_scheduler = {
        #     "scheduler": warmup,
        #     "interval": "step",  # track steps, not epochs
        #     "frequency": 1,
        # }

        # Check if lr_scheduler is defined in hparams
        if (
            hasattr(self.hparams, "lr_scheduler")
            and self.hparams.lr_scheduler is not None
        ):
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler_config = self.hparams.lr_scheduler
            plateau = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
                min_lr=1e-6,
            )

            scheduler = {
                "scheduler": plateau,
                "monitor": "train_loss",
                "interval": "step",
                # tracks n_steps
                "frequency": self.loss_metric_steps,
            }
            return [optimizer], [scheduler]

        # Return only warmup scheduler if no plateau scheduler configured
        return [optimizer]  # , [warmup_scheduler]


class MAEarlyStopping(Callback):
    def __init__(self, patience=4, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if the custom loss buffer is empty the metric has just been logged
        if len(pl_module.loss_buffer) == 0:
            current_score = pl_module.train_loss

            if (
                self.best_score is None
                or current_score > self.best_score + self.min_delta
            ):
                self.best_score = current_score
                self.wait_count = 0
            else:
                self.wait_count += 1

            if self.wait_count >= self.patience:
                trainer.should_stop = True
