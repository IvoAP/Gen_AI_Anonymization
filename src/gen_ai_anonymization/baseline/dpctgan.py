import math
from typing import Any

from .base import SmartNoiseBaseline


class DPCTGANBaseline(SmartNoiseBaseline):
    def __init__(
        self, epsilon: float, *, delta: float = 1e-6, preprocessor_eps: float = 0.0,
        sigma: float = 5.0, max_per_sample_grad_norm: float = 1.0,
        epochs: int = 10, batch_size: int = 500, cuda: bool = False,
        verbose: bool = True, **parameters: Any,
    ):
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("DPCTGAN delta must be between zero and one.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("DPCTGAN sigma must be finite and positive.")
        if not math.isfinite(max_per_sample_grad_norm) or max_per_sample_grad_norm <= 0:
            raise ValueError("DPCTGAN gradient clipping norm must be finite and positive.")
        if epochs < 1 or batch_size < 2 or batch_size % 2:
            raise ValueError("DPCTGAN requires positive epochs and an even batch size of at least two.")
        if parameters.pop("disabled_dp", False):
            raise ValueError("This baseline does not allow disabling differential privacy.")
        if parameters.pop("pac", 1) != 1:
            raise ValueError("DPCTGAN baseline uses pac=1 for per-record gradients.")
        super().__init__(
            "dpctgan", epsilon, preprocessor_eps=preprocessor_eps, delta=delta,
            sigma=sigma, max_per_sample_grad_norm=max_per_sample_grad_norm,
            epochs=epochs, batch_size=batch_size, cuda=cuda, verbose=verbose,
            disabled_dp=False, pac=1, **parameters,
        )