from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd
from diffprivlib.accountant import BudgetAccountant


def validate_privacy(epsilon: float, delta: float = 0.0) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")
    if isinstance(delta, bool) or not isinstance(delta, Real) or not isfinite(delta) or not 0 <= delta < 1:
        raise ValueError("delta must be finite and in [0, 1).")


def validate_size(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def validate_bounds(bounds: Mapping[str, tuple[float, float]]) -> None:
    if not bounds:
        raise ValueError("Provide public bounds for at least one numeric column.")
    for column, limits in bounds.items():
        if len(limits) != 2:
            raise ValueError(f"Bounds for {column} must contain lower and upper limits.")
        lower, upper = limits
        if not all(isinstance(value, Real) and not isinstance(value, bool) and isfinite(value) for value in limits):
            raise ValueError(f"Bounds for {column} must be finite numbers.")
        if lower >= upper or not isfinite(upper - lower):
            raise ValueError(f"Bounds for {column} must have a finite, positive width.")


def validate_frame(data: pd.DataFrame, columns: list[str]) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty or not data.columns.is_unique:
        raise ValueError("data must be nonempty and have unique column names.")
    if not set(columns).issubset(data.columns):
        raise ValueError("Some schema columns are missing from data.")


def bounded_column(values: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    midpoint = lower + (upper - lower) / 2
    return numeric.clip(lower, upper).fillna(midpoint)


def private_means(
    data: pd.DataFrame,
    *,
    bounds: Mapping[str, tuple[float, float]],
    epsilon: float,
    delta: float,
    gaussian: bool,
    accountant: BudgetAccountant | None,
) -> pd.Series:
    from diffprivlib.mechanisms import GaussianAnalytic, Laplace

    validate_privacy(epsilon, delta)
    if gaussian and delta == 0:
        raise ValueError("Gaussian privacy requires 0 < delta < 1.")
    validate_bounds(bounds)
    validate_frame(data, list(bounds))
    budget = BudgetAccountant.load_default(accountant)
    budget.check(epsilon, delta)
    column_count = len(bounds)
    mechanisms = {}
    for column, (lower, upper) in bounds.items():
        parameters = {
            "epsilon": epsilon / column_count,
            "delta": delta / column_count,
            "sensitivity": (upper - lower) / len(data),
        }
        mechanisms[column] = (GaussianAnalytic if gaussian else Laplace)(**parameters)
    budget.spend(epsilon, delta)
    released = {}
    for column, (lower, upper) in bounds.items():
        values = bounded_column(data[column], lower, upper)
        true_mean = float((values / len(data)).sum())
        released[column] = float(np.clip(mechanisms[column].randomise(true_mean), lower, upper))
    return pd.Series(released, name="private_mean", dtype="float64")