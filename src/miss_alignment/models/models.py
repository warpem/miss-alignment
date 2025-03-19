from typing import Optional, Callable, Any

import pytorch_lightning as pl
import torch
from torch import Tensor, optim

from ._resnet import resnet3d_18


class MissAlignment(pl.LightningModule):
    in_channels: int = 1
    num_classes: int = 1

    def __init__(self, batch_size: int = 4, learning_rate: float = 1e-04):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.validation_epoch_loss = 0
        self.validation_step_outputs = []

        self.net = resnet3d_18(self.in_channels, self.num_classes)

    def forward(self, image: Tensor) -> Tensor:
        out = self.net(image)
        return out

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        """The network should assign a higher score to the aligned box (s_a) than to
        the misaligned box (s_m). Minimize this loss:

            loss = (s_m - s_a)

        (s_a and s_m fall to 0 and 1 range by applying a sigmoid as final layer)
        """
        aligned, misaligned = batch["aligned"], batch["misaligned"]
        s_a, s_m = self(aligned), self(misaligned)
        loss = s_m - s_a
        loss = 1.0 - torch.mean(loss)  # the closer to 0 the better
        self.log(
            name="training loss", value=loss, batch_size=aligned.shape[0], prog_bar=True
        )
        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        aligned, misaligned = batch["aligned"], batch["misaligned"]
        s_a, s_m = self(aligned), self(misaligned)
        loss = s_m - s_a
        loss = 1.0 - torch.mean(loss)
        self.log(
            name="validation loss",
            value=loss,
            batch_size=aligned.shape[0],
            prog_bar=True,
        )
        self.validation_step_outputs.append(loss)
        return loss

    def predict_step(self):
        pass

    def on_validation_epoch_end(self):
        mean_epoch_loss = torch.mean(torch.as_tensor(self.validation_step_outputs))
        self.validation_epoch_loss = mean_epoch_loss
        self.log(name="validation epoch loss", value=mean_epoch_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
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
