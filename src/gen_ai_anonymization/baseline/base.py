from importlib.metadata import PackageNotFoundError, version
import math
from time import perf_counter
from typing import Any
import warnings

import pandas as pd


class SmartNoiseBaseline:
    def __init__(
        self, method: str, epsilon: float, *, preprocessor_eps: float = 0.0, **parameters: Any,
    ):
        if method not in ("mwem", "mst", "aim", "dpctgan"):
            raise ValueError("Unsupported SmartNoise baseline.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        if not math.isfinite(preprocessor_eps) or not 0 <= preprocessor_eps < epsilon:
            raise ValueError("preprocessor_eps must be nonnegative and smaller than epsilon.")
        self.method = method
        self.epsilon = float(epsilon)
        self.preprocessor_eps = float(preprocessor_eps)
        self.parameters = dict(parameters)
        self._synthesizer = None
        self._columns: list[str] = []
        self.training_seconds: float | None = None
        self.sampling_seconds: float | None = None
        self._backend_warnings: list[str] = []

    def _create_synthesizer(self):
        try:
            import snsynth
        except ImportError as error:
            raise ImportError(
                "SmartNoise baselines require a separate environment with smartnoise-synth. "
                "See environments/baselines/pyproject.toml; do not downgrade DP-TAL's Opacus."
            ) from error
        try:
            opacus_version = version("opacus")
        except PackageNotFoundError:
            opacus_version = "missing"
        if not opacus_version.startswith("0.14."):
            raise ImportError(
                f"The selected SmartNoise baseline environment requires Opacus 0.14.x, found {opacus_version}. "
                "Run baselines separately from the DP-TAL environment."
            )
        return snsynth.Synthesizer.create(self.method, epsilon=self.epsilon, **self.parameters)

    def fit(
        self, data: pd.DataFrame, *, transformer: Any = None,
        categorical_columns: list[str] | None = None,
        ordinal_columns: list[str] | None = None,
        continuous_columns: list[str] | None = None, nullable: bool = False,
    ) -> "SmartNoiseBaseline":
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("fit requires a nonempty pandas DataFrame.")
        if data.columns.duplicated().any():
            raise ValueError("Column names must be unique.")
        if not all(isinstance(column, str) for column in data.columns):
            raise ValueError("Column names must be strings.")
        rounds = self.parameters.get("rounds")
        if self.method == "aim" and rounds is not None and rounds < len(data.columns):
            raise ValueError("AIM rounds must be at least the number of columns to retain budget after initial marginals.")
        hints = [categorical_columns or [], ordinal_columns or [], continuous_columns or []]
        named_columns = [column for group in hints for column in group]
        if len(named_columns) != len(set(named_columns)) or set(named_columns) - set(data.columns):
            raise ValueError("Column hints must be disjoint and refer to existing columns.")
        self._synthesizer = None
        self.training_seconds = None
        self.sampling_seconds = None
        self._backend_warnings = []
        synthesizer = self._create_synthesizer()
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            synthesizer.fit(
                data, transformer=transformer, categorical_columns=list(hints[0]),
                ordinal_columns=list(hints[1]), continuous_columns=list(hints[2]),
                preprocessor_eps=self.preprocessor_eps, nullable=nullable,
            )
        self.training_seconds = perf_counter() - start
        self._backend_warnings = list(dict.fromkeys(str(item.message) for item in captured))
        for item in captured:
            warnings.warn(str(item.message), item.category, stacklevel=2)
        self._columns = list(data.columns)
        self._synthesizer = synthesizer
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("Fit the baseline before sampling.")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        start = perf_counter()
        samples = self._synthesizer.sample(n_samples)
        self.sampling_seconds = perf_counter() - start
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples, columns=self._columns)
        if len(samples) != n_samples or list(samples.columns) != self._columns:
            raise RuntimeError("SmartNoise returned an unexpected sample size or schema.")
        return samples

    def fit_sample(self, data: pd.DataFrame, n_samples: int, **fit_options: Any) -> pd.DataFrame:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        return self.fit(data, **fit_options).sample(n_samples)

    def get_report(self) -> dict:
        return {
            "method": self.method, "epsilon_requested": self.epsilon,
            "preprocessor_epsilon_allocated": self.preprocessor_eps,
            "delta_requested": self.parameters.get("delta"),
            "parameters": dict(self.parameters), "fitted": self._synthesizer is not None,
            "training_seconds": self.training_seconds, "sampling_seconds": self.sampling_seconds,
            "backend_warnings": list(self._backend_warnings), "privacy_budget_verified": False,
            "privacy_note": "Requested budgets are not independently verified guarantees; account for preprocessing and repeated fits.",
        }