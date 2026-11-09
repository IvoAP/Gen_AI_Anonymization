from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 16
    hidden_dim: int = 64
    anchors: int = 16
    kl_weight: float = 0.01
    geometry_weight: float = 0.1
    prior_separation: float = 1.0
    seed: int = 42


class LatentGenerator(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        if min(input_dim, config.latent_dim, config.hidden_dim, config.anchors) < 1:
            raise ValueError("Model dimensions and anchor count must be positive.")
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        rng = torch.Generator().manual_seed(config.seed)
        input_anchors = torch.rand(config.anchors, input_dim, generator=rng) * 2 - 1
        projection = torch.randn(input_dim, config.latent_dim, generator=rng) / input_dim**0.5
        self.register_buffer("input_anchors", input_anchors)
        self.register_buffer("latent_anchors", input_anchors @ projection)
        prior_means = torch.zeros(2, config.latent_dim)
        prior_means[0, 0] = -config.prior_separation
        prior_means[1, 0] = config.prior_separation
        self.register_buffer("prior_means", prior_means)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.mean_head(hidden), self.logvar_head(hidden).clamp(-6, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(features)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.classifier(latent).squeeze(-1), mean, logvar

    def losses(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits, mean, logvar = outputs
        classification = nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        prior_mean = self.prior_means[labels.long()]
        kl = 0.5 * (logvar.exp() + (mean - prior_mean).square() - 1 - logvar).mean(dim=1)
        input_distances = (
            (features[:, None, :] - self.input_anchors[None, :, :]).square().mean(dim=2) + 1e-8
        ).sqrt()
        latent_distances = (
            (mean[:, None, :] - self.latent_anchors[None, :, :]).square().mean(dim=2) + 1e-8
        ).sqrt()
        geometry = (input_distances - latent_distances).square().mean(dim=1)
        total = classification + self.config.kl_weight * kl + self.config.geometry_weight * geometry
        return {"total": total, "classification": classification, "kl": kl, "geometry": geometry}

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        labels = labels.to(device=self.prior_means.device, dtype=torch.long)
        noise = torch.randn(
            len(labels), self.config.latent_dim, device=labels.device, generator=generator
        )
        return self.prior_means[labels] + noise