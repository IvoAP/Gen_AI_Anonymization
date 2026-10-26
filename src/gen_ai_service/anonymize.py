"""Perturb tabular records using an MLP autoencoder and latent Gaussian noise.

This is not differential privacy or a guarantee of anonymization. Reconstruction
preserves row correspondence and can retain identifying information. Scaling,
categories, bounds, and model weights are learned from private training data.
"""

from numbers import Integral

import numpy as np
import pandas as pd
import torch

from .config import AutoencoderConfig
from .encoding import TabularEncoder
from .model import Autoencoder, add_latent_noise, reconstruction_loss
from .training import train_autoencoder

__all__ = [
    "Anonymizer", "AutoencoderConfig", "TabularEncoder", "Autoencoder",
    "add_latent_noise", "reconstruction_loss", "anonymize_dataset",
]


class Anonymizer:
    """Train once on selected columns; reconstruct the same rows with latent noise.

    CPU implementation. This provides no epsilon/delta or device-level privacy.
    Select columns explicitly before fitting; identifiers are not auto-detected.
    """

    def __init__(self, data: pd.DataFrame, seed: int = 42, config: AutoencoderConfig | None = None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if len(data) < 2 or len(data.columns) == 0 or not data.columns.is_unique:
            raise ValueError("data must have at least two rows and unique, nonempty columns.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        self.config = config if config is not None else AutoencoderConfig()
        self.data = data.copy(deep=True)
        self.seed = int(seed)
        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(len(data), generator=generator)
        validation_size = max(1, int(len(data) * self.config.val_fraction))
        training_indices = permutation[validation_size:]
        self.encoder = TabularEncoder().fit(
            data.iloc[training_indices.numpy()], self.config.max_encoded_features,
        )
        self.values = torch.from_numpy(self.encoder.transform(data))
        training = self.values[training_indices]
        validation = self.values[permutation[:validation_size]]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = Autoencoder(self.encoder.input_dim, self.config)
            self.history, self.best_epoch = train_autoencoder(
                self.model, training, validation,
                config=self.config,
                numeric_count=len(self.encoder.num_cols),
                category_sizes=self.encoder.cat_sizes,
                seed=self.seed,
                generator=generator,
            )
        self.model.eval()
        with torch.no_grad():
            latent = torch.cat([
                self.model.encoder(batch) for batch in training.split(self.config.batch_size)
            ])
            self.latent_std = latent.std(dim=0, correction=0)

    @torch.no_grad()
    def generate(self, noise_scale: float = 1.0, seed: int | None = None) -> pd.DataFrame:
        """Reconstruct fitted rows, preserving index/order; zero noise is still lossy.

        Noise uses training latent standard deviations, independently of inference
        batch size. A fixed seed reproduces perturbations; it is not a DP release.
        """
        generation_seed = self.seed if seed is None else seed
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, Integral) or not 0 <= generation_seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        generator = torch.Generator().manual_seed(int(generation_seed))
        self.model.eval()
        output = np.empty((len(self.values), self.encoder.input_dim), dtype=np.float32)
        offset = 0
        for batch in self.values.split(self.config.batch_size):
            decoded = self.model(batch, noise_scale, generator, self.latent_std)
            output[offset:offset + len(batch)] = decoded.numpy()
            offset += len(batch)
        if not np.isfinite(output).all():
            raise ValueError("Generation produced non-finite values; reduce noise_scale.")
        result = self.encoder.inverse_transform(output)
        result.index = self.data.index.copy()
        return result


def anonymize_dataset(
    data: pd.DataFrame, noise_scale: float = 1.0, seed: int = 42,
    config: AutoencoderConfig | None = None,
) -> pd.DataFrame:
    """Train and perturb selected columns; no formal privacy guarantee is provided."""
    return Anonymizer(data, seed=seed, config=config).generate(noise_scale=noise_scale)