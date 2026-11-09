import math
from typing import Any

from .base import SmartNoiseBaseline


class AIMBaseline(SmartNoiseBaseline):
    def __init__(
        self, epsilon: float, *, delta: float = 1e-6, preprocessor_eps: float = 0.0,
        degree: int = 2, rounds: int | None = None, max_model_size: int = 80,
        verbose: bool = False, **parameters: Any,
    ):
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("AIM delta must be between zero and one.")
        if degree < 1 or max_model_size < 1 or (rounds is not None and rounds < 1):
            raise ValueError("AIM degree, model size and explicit round count must be positive.")
        super().__init__("aim", epsilon, preprocessor_eps=preprocessor_eps,
                         delta=delta, degree=degree, rounds=rounds,
                         max_model_size=max_model_size, verbose=verbose, **parameters)