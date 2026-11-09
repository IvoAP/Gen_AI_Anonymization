from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import LatentGenerator, ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    noise_multiplier: float = 0.5
    max_grad_norm: float = 1.0
    delta: float = 1e-6
    device: str = "cpu"
    seed: int = 42
    verbose: int = 1


def train_generator(
    features: np.ndarray, labels: np.ndarray, model_config: ModelConfig, config: TrainingConfig
) -> tuple[LatentGenerator, dict]:
    if config.verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0 (silent), 1 (epochs), or 2 (epochs and batches).")
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive.")
    if config.noise_multiplier < 0 or config.max_grad_norm <= 0:
        raise ValueError("Noise must be nonnegative and clipping norm must be positive.")
    if not 0 < config.delta < 1 / len(features):
        raise ValueError("delta must be positive and smaller than 1 / training records.")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = LatentGenerator(features.shape[1], model_config).to(config.device)
    ModuleValidator.validate(model, strict=True)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.verbose:
        tqdm.write(
            f"[Setup] Records: {len(features)} | Features: {features.shape[1]} | "
            f"Latent dimensions: {model_config.latent_dim} | Device: {config.device}"
        )
        tqdm.write(
            f"[Privacy] Noise multiplier: {config.noise_multiplier:g} | "
            f"Max gradient norm: {config.max_grad_norm:g} | "
            f"Mode: {'DP-SGD' if config.noise_multiplier > 0 else 'non-private baseline'}"
        )
        tqdm.write("[Warning] Training losses are private diagnostics, not DP-protected releases.")
    start = perf_counter()
    engine = None
    if config.noise_multiplier > 0:
        engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        wrapped, optimizer, loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=config.noise_multiplier, max_grad_norm=config.max_grad_norm,
            poisson_sampling=True,
        )
    else:
        wrapped = model
    history = []
    for epoch in tqdm(range(config.epochs), desc="Generator training", unit="epoch", disable=not config.verbose):
        epoch_start = perf_counter()
        wrapped.train()
        sums = {key: 0.0 for key in ("total", "classification", "kl", "geometry")}
        records = 0
        batches = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch",
            leave=False, disable=config.verbose < 2,
        )
        for batch_features, batch_labels in batches:
            batch_features = batch_features.to(config.device)
            batch_labels = batch_labels.to(config.device)
            optimizer.zero_grad()
            losses = model.losses(batch_features, batch_labels, wrapped(batch_features))
            (losses["total"].sum() / max(len(batch_labels), 1)).backward()
            optimizer.step()
            records += len(batch_labels)
            for key, values in losses.items():
                sums[key] += values.detach().sum().item()
            if config.verbose >= 2:
                batches.set_postfix(loss=f"{sums['total'] / max(records, 1):.4f}", refresh=False)
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        epoch_seconds = perf_counter() - epoch_start
        epoch_losses = {key: value / max(records, 1) for key, value in sums.items()}
        history.append({"epoch": epoch + 1, "seconds": epoch_seconds, **epoch_losses})
        if config.verbose:
            tqdm.write(
                f"Epoch {epoch + 1}/{config.epochs} | Loss: {epoch_losses['total']:.6f} | "
                f"BCE: {epoch_losses['classification']:.6f} | KL: {epoch_losses['kl']:.6f} | "
                f"Geometry: {epoch_losses['geometry']:.6f} | Time: {epoch_seconds:.2f}s"
            )
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    epsilon = engine.get_epsilon(config.delta) if engine is not None else None
    if config.verbose:
        tqdm.write(f"[Complete] Generator training time: {elapsed:.2f}s")
        if epsilon is not None:
            tqdm.write(f"[Privacy] Epsilon: {epsilon:.4f} | Delta: {config.delta:g}")
    if engine is not None:
        model = wrapped.to_standard_module()
    model.eval()
    return model, {
        "training_seconds": elapsed, "epsilon": epsilon,
        "delta": config.delta if engine is not None else None,
        "private_training": engine is not None,
        "privacy_scope": "record-level training mechanism only; diagnostics are not DP releases",
        "secure_rng": False, "model_config": asdict(model_config),
        "training_config": asdict(config), "history": history,
    }


@torch.no_grad()
def encode_features(model: LatentGenerator, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = [
        model.encode(torch.from_numpy(features[start:start + batch_size]).to(device))[0].cpu().numpy()
        for start in range(0, len(features), batch_size)
    ]
    return np.concatenate(chunks)


@torch.no_grad()
def generate_latents(model: LatentGenerator, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    if n_samples < 2:
        raise ValueError("At least two synthetic samples are required.")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(n_samples, device=device) % 2
    labels = labels[torch.randperm(n_samples, device=device, generator=rng)]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    latent = model.sample(labels, generator=rng).cpu().numpy()
    elapsed = perf_counter() - start
    return latent, labels.cpu().numpy(), elapsed