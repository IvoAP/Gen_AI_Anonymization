import math
from typing import Any

from .base import SmartNoiseBaseline


class MSTBaseline(SmartNoiseBaseline):
    def __init__(
        self, epsilon: float, *, delta: float = 1e-6, preprocessor_eps: float = 0.0,
        verbose: bool = False, **parameters: Any,
    ):
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("MST delta must be between zero and one.")
        super().__init__("mst", epsilon, preprocessor_eps=preprocessor_eps,
                         delta=delta, verbose=verbose, **parameters)