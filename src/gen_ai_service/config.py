from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int = 8
    hidden: tuple[int, ...] = (64, 32)
    max_epochs: int = 30
    patience: int = 15
    val_fraction: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    train_noise: float = 0.0
    max_encoded_features: int = 4096

    def __post_init__(self) -> None:
        for name in ("latent_dim", "max_epochs", "patience", "batch_size", "max_encoded_features"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.hidden or any(isinstance(width, bool) or not isinstance(width, Integral) or width < 1 for width in self.hidden):
            raise ValueError("hidden must contain positive integer widths.")
        for name in ("val_fraction", "learning_rate", "weight_decay", "dropout", "train_noise"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if not 0 < self.val_fraction < 1 or not 0 <= self.dropout < 1 or self.learning_rate == 0:
            raise ValueError("Require 0 < val_fraction < 1, 0 <= dropout < 1, and learning_rate > 0.")