import copy
from functools import partial
from math import isfinite

import pandas as pd
import torch

from .config import AutoencoderConfig
from .model import Autoencoder, reconstruction_loss


def train_autoencoder(
    model: Autoencoder,
    training: torch.Tensor,
    validation: torch.Tensor,
    *,
    config: AutoencoderConfig,
    numeric_count: int,
    category_sizes: list[int],
    seed: int,
    generator: torch.Generator,
) -> tuple[pd.DataFrame, int]:
    """Restore the best validation weights and return history and best epoch."""
    loss_function = partial(reconstruction_loss, numeric_count=numeric_count, category_sizes=category_sizes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(training), generator=generator)
        train_total = 0.0
        for batch in training[permutation].split(config.batch_size):
            loss = loss_function(model(batch, config.train_noise, generator), batch)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite; check feature scales and learning_rate.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch)
        model.eval()
        validation_generator = torch.Generator().manual_seed(seed)
        validation_total = 0.0
        with torch.no_grad():
            for batch in validation.split(config.batch_size):
                loss = loss_function(model(batch, config.train_noise, validation_generator), batch)
                validation_total += loss.item() * len(batch)
        validation_loss = validation_total / len(validation)
        if not isfinite(validation_loss):
            raise ValueError("Validation loss is not finite.")
        history.append({"epoch": epoch, "train_loss": train_total / len(training), "val_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch