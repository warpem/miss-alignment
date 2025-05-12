import pytorch_lightning as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

# from ._resnet import resnet3d_18
from ._compact import Compact3DConvNet


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

    def __init__(self, margin=1.0, reduction='mean'):
        super(TripletMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, triplet_indices):
        """
        Forward pass using triplet indices to organize samples.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor of shape (n, embedding_dim)
        triplet_indices : torch.Tensor
            Tensor of shape (batch_size, 3) where each row contains indices
            for [anchor_idx, positive_idx, negative_idx]

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        example_type = einops.rearrange(
            triplet_indices.sum(dim=-1), 'b -> b 1'
        )
        close = embeddings[triplet_indices == example_type]  # (b, 2)
        close = einops.rearrange(close, '(b n) -> b n', n=2)
        distant = embeddings[triplet_indices != example_type]  # (b, 1)
        distant = einops.rearrange(distant, 'b -> b 1')

        dist_pos = torch.abs(close[..., 0] - close[..., 1])
        dist_neg = torch.min((close - distant) * example_type, dim=-1).values

        # Compute triplet loss with margin
        losses = F.relu(dist_pos + dist_neg + self.margin)

        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


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
        self.criterion = TripletMarginRankingLoss(margin=margin)
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
        img1, img2, img3, target = batch
        batch_size = target.shape[0]

        # calculate scores and loss
        scores = torch.stack((self(img1), self(img2), self(img3)))
        scores = einops.rearrange(scores, 'd b 1 -> b d')
        loss = self.criterion(scores, target)

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        example_type = target.sum(dim=-1)
        s_a = torch.mean(scores[example_type == 1])
        s_m = torch.mean(scores[example_type == -1])

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
        img1, img2, img3, target = batch
        batch_size = target.shape[0]

        # calculate scores and loss
        scores = torch.stack((self(img1), self(img2), self(img3)))
        scores = einops.rearrange(scores, 'd b 1 -> b d')
        loss = self.criterion(scores, target)

        # find back the actual assigned scores to aligned/misaligned volume
        # to report back in the logs
        example_type = target.sum(dim=-1)
        s_a = torch.mean(scores[example_type == 1])
        s_m = torch.mean(scores[example_type == -1])

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

        # Add ReduceLROnPlateau scheduler that monitors validation loss
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            ),
            'monitor': 'val loss',  # Metric to monitor
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]
