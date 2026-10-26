from math import isfinite
from numbers import Real

import torch
from torch import nn

from .config import AutoencoderConfig


def add_latent_noise(
    latent: torch.Tensor,
    scale: float,
    generator: torch.Generator | None = None,
    latent_std: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(scale, bool) or not isinstance(scale, Real) or not isfinite(scale) or scale < 0:
        raise ValueError("noise_scale must be a finite nonnegative number.")
    if scale == 0:
        return latent
    spread = latent.std(dim=0, correction=0).detach() if latent_std is None else latent_std
    noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype, device=latent.device)
    return latent + noise * spread * scale


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        previous = input_dim
        for width in config.hidden:
            encoder_layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)])
            previous = width
        encoder_layers.append(nn.Linear(previous, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        previous = config.latent_dim
        for width in reversed(config.hidden):
            decoder_layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)])
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self, values: torch.Tensor, noise_scale: float = 0.0,
        generator: torch.Generator | None = None, latent_std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = add_latent_noise(self.encoder(values), noise_scale, generator, latent_std)
        return self.decoder(latent)


def reconstruction_loss(
    output: torch.Tensor, values: torch.Tensor, numeric_count: int, category_sizes: list[int],
) -> torch.Tensor:
    loss = output.sum() * 0.0
    if numeric_count:
        loss = nn.functional.mse_loss(output[:, :numeric_count], values[:, :numeric_count])
    start = numeric_count
    for size in category_sizes:
        target = values[:, start:start + size].argmax(dim=1)
        loss = loss + nn.functional.cross_entropy(output[:, start:start + size], target)
        start += size
    return loss