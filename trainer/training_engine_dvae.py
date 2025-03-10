import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from models.modelling_dvae import DVAE, DVAEConfig, EncoderConfig, DecoderConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


def compute_kl_loss_with_freebits(logits, num_embeddings, free_bits_threshold=0.1):
    log_probs = torch.log_softmax(logits, dim=-1)
    log_uniform = torch.log(torch.ones_like(log_probs) / num_embeddings)
    kl_div = F.kl_div(log_probs, log_uniform, log_target=True, reduction="none")

    kl_div = kl_div.view(kl_div.shape[0], -1)
    kl_div_adjusted = torch.clamp(kl_div - free_bits_threshold, min=0)
    kl_loss = kl_div_adjusted.sum(dim=-1).mean()
    return kl_loss


class DVAETrainer(pl.LightningModule):
    def __init__(
        self, dvae_model, dvae_config, temp_scheduler, kl_div_weight_scheduler
    ):
        super().__init__()
        self.model = dvae_model
        self.global_configs = dvae_config
        self.temp_scheduler = temp_scheduler
        self.kl_div_weight_scheduler = kl_div_weight_scheduler

        self._reset_current_distribution()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.global_configs.learning_rate)
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.global_configs.learning_rate_scheduler_min,
            total_iters=self.global_configs.max_epochs,
        )
        return [optimizer], [scheduler]

    def _step(self, batch: torch.Tensor, mode: str):

        if isinstance(batch, list):
            batch = batch[0]

        temperature = self.temp_scheduler.get_value()
        kl_div_weight = self.kl_div_weight_scheduler.get_value()

        encoded, logits = self.model.encode(batch, temperature=temperature)
        decoded = self.model.decode(encoded)

        reconstruction_loss = F.mse_loss(decoded, batch)

        elbo_loss = compute_kl_loss_with_freebits(
            logits, self.global_configs.num_embeddings
        )
        loss = reconstruction_loss + 0.01 * elbo_loss

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_elbo_loss", elbo_loss, prog_bar=True)
        self.log(f"{mode}_reconstruction_loss", reconstruction_loss, prog_bar=True)

        return loss

    def training_step(self, batch: torch.Tensor):
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor):
        return self._step(batch, "val")

    def on_train_epoch_end(self) -> None:  # noqa: D102
        self._reset_current_distribution()

        self.temp_scheduler.step()
        self.kl_div_weight_scheduler.step()

        self.log("temperature", self.temp_scheduler.get_value())
        self.log("kl_div_weight", self.kl_div_weight_scheduler.get_value())

    def on_validation_epoch_start(self) -> None:  # noqa: D102
        self._reset_current_distribution()

    def on_test_epoch_start(self) -> None:  # noqa: D102
        self._reset_current_distribution()

    def _reset_current_distribution(self) -> None:
        self._current_epoch_softmax_sum = None
        self._current_epoch_softmax_count = 0

    def test_step(self, image: torch.Tensor):
        encoded, logits = self.model.encode(image)
        decoded = self.model.decode(encoded)

        return decoded
