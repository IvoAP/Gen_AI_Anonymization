from typing import Any

from .base import SmartNoiseBaseline


class MWEMBaseline(SmartNoiseBaseline):
    def __init__(
        self, epsilon: float, *, preprocessor_eps: float = 0.0,
        iterations: int | None = None, split_factor: int | None = None,
        verbose: bool = False, **parameters: Any,
    ):
        if iterations is not None:
            if iterations < 1:
                raise ValueError("MWEM iterations must be positive.")
            parameters["iterations"] = iterations
        if split_factor is not None:
            if split_factor < 1:
                raise ValueError("MWEM split_factor must be positive.")
            parameters["split_factor"] = split_factor
        super().__init__("mwem", epsilon, preprocessor_eps=preprocessor_eps, verbose=verbose, **parameters)