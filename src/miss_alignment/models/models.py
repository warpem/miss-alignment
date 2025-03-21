from typing import Optional, Callable, Any

import pytorch_lightning as pl
import torch

from ._resnet import resnet3d_18


LAMBDA = .01  # earlier .1


def loss_l2(s_m: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
    """Contrastive loss with L2-normalization.

    loss = s_a - s_m + LAMBDA * (s_m ** 2 + s_a ** 2)

    The loss is minimized, so a higher value for the misaligned volume is encouraged.

    Parameters
    ----------
    s_m: torch.Tensor
        Score assigned to misaligned volume. (b,)
    s_a: torch.Tensor
        Score assigned to aligned volume.  (b,)

    Returns
    -------
    loss: torch.Tensor
    """
    return s_a - s_m + LAMBDA * (s_m ** 2 + s_a ** 2)  # (b,)


class MissAlignment(pl.LightningModule):
    in_channels: int = 1
    num_classes: int = 1

    def __init__(self, learning_rate: float = 1e-04):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.net = resnet3d_18(self.in_channels, self.num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        return out

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        aligned, misaligned = batch["aligned"], batch["misaligned"]
        s_a, s_m = self(aligned), self(misaligned)
        loss = loss_l2(s_m, s_a)
        loss = torch.mean(loss)
        batch_size = aligned.shape[0]
        self.log(
            name="train loss", value=loss, batch_size=batch_size, prog_bar=True
        )
        self.log(
            name="train s_m",
            value=s_m.mean(),
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            name="train s_a",
            value=s_a.mean(),
            batch_size=batch_size,
            prog_bar=True,
        )
        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        aligned, misaligned = batch["aligned"], batch["misaligned"]
        s_a, s_m = self(aligned), self(misaligned)
        loss = loss_l2(s_m, s_a)
        loss = torch.mean(loss)
        batch_size = aligned.shape[0]
        self.log(
            name="val loss",
            value=loss,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            name="val s_m",
            value=s_m.mean(),
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            name="val s_a",
            value=s_a.mean(),
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
        return optimizer

    def optimizer_step(
        self,
        epoch_idx: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> None:
        self.log("learning rate", optimizer.param_groups[0]["lr"])
        super().optimizer_step(
            epoch_idx, batch_idx, optimizer, optimizer_closure, **kwargs
        )
