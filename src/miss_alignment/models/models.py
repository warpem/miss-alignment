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
from ..align_backend import get_alignment_optimization_metrics


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


class TripletRatioLoss(nn.Module):
    """
    Triplet Ratio Loss with explicit triplet ordering based on indices.

    This loss learns embeddings such that δ−/δ+ → ∞ by using the formulation:
    λ(δ+,δ−) = (e^δ+ / (e^δ+ + e^δ−))^2 + (1 - e^δ− / (e^δ+ + e^δ−))^2

    The softplus function is applied to distances to ensure positivity while
    preserving the meaning of relative distances.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'
    beta : float, optional
        Beta parameter for softplus. Default: 1.0
    eps : float, optional
        Small value to ensure numerical stability. Default: 1e-6
    """

    def __init__(self, reduction='mean', beta=1.0, eps=1e-6):
        super(TripletRatioLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        self.eps = eps

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

        # Calculate raw distances, preserving sign
        raw_dist_pos = torch.abs(close[..., 0] - close[..., 1])
        raw_dist_neg = torch.min((close - distant) * example_type,
                                 dim=-1).values

        # Apply softplus to ensure positive values while preserving relative magnitudes
        dist_pos = F.softplus(raw_dist_pos, beta=self.beta) + self.eps
        dist_neg = F.softplus(raw_dist_neg, beta=self.beta) + self.eps

        # Apply exponential to the processed distances
        exp_pos = torch.exp(dist_pos)  # e^δ+
        exp_neg = torch.exp(dist_neg)  # e^δ−

        # Compute denominator
        denom = exp_pos + exp_neg + self.eps

        # Compute the ratio loss according to the formula
        term1 = torch.pow(exp_pos / denom, 2)
        term2 = torch.pow(1 - (exp_neg / denom), 2)
        losses = term1 + term2

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
            self, learning_rate: float = 1e-04, margin=.5
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        # self.criterion = TripletMarginRankingLoss(margin=margin)
        self.criterion = TripletRatioLoss()
        self.net = Compact3DConvNet()  # resnet3d_18()

        # hard code this for now
        self.optimize_flag = False
        self.test_data_directory = "/home/marten/data/datasets/emdb/test"

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

        self.optimize_flag = True

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

        # run alignment optimization and report the metric back!
        if self.optimize_flag:
            metrics = get_alignment_optimization_metrics(
                self.net,
                self.test_data_directory,
            )
            for metric_name, metric_value in metrics.items():
                self.log(
                    name=f"val {metric_name}",
                    value=metric_value,
                )
            self.optimize_flag = False

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
                patience=2,
            ),
            'monitor': 'val loss',  # Metric to monitor
            'interval': 'epoch',
            'frequency': 5
        }

        return [optimizer], [scheduler]
