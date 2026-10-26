from collections.abc import Mapping, Sequence
from math import exp
from typing import Literal
import pandas as pd
from diffprivlib.accountant import BudgetAccountant

from ._common import bounded_column, validate_bounds, validate_frame, validate_privacy, validate_size


def synthesize(
    algorithm: Literal["aim", "mst"],
    data: pd.DataFrame,
    *,
    epsilon: float,
    delta: float,
    n_rows: int,
    categorical_domains: Mapping[str, Sequence[str | int]],
    numeric_bounds: Mapping[str, tuple[float, float]],
    bins: int,
    accountant: BudgetAccountant | None,
) -> pd.DataFrame:
    from snsynth import Synthesizer
    from snsynth.transform import BinTransformer, LabelTransformer, TableTransformer

    validate_privacy(epsilon, delta)
    if delta == 0:
        raise ValueError("AIM and MST require 0 < delta < 1.")
    validate_size(n_rows, "n_rows")
    validate_size(bins, "bins", minimum=2)
    internal_epsilon = epsilon / 2
    attenuation = exp(-internal_epsilon)
    internal_delta = delta * attenuation / (1 + attenuation)
    if internal_delta == 0:
        raise ValueError("epsilon/delta cannot be calibrated at floating-point precision.")
    if set(categorical_domains) & set(numeric_bounds):
        raise ValueError("A column cannot have both categorical and numeric domains.")
    if numeric_bounds:
        validate_bounds(numeric_bounds)
    columns = list(categorical_domains) + list(numeric_bounds)
    if len(columns) < 2:
        raise ValueError("Provide public domains for at least two columns.")
    validate_frame(data, columns)

    public_values = {}
    transformers = []
    for column, domain in categorical_domains.items():
        if isinstance(domain, (str, bytes)) or len(domain) < 2:
            raise ValueError(f"The public domain for {column} must contain at least two categories.")
        categories = list(domain)
        if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in categories):
            raise ValueError("Public categories must be strings or integers.")
        if len(set(categories)) != len(categories) or len({type(value) for value in categories}) != 1:
            raise ValueError("Public categories must be unique and have the same type.")
        public_values[column] = categories
        transformers.append(LabelTransformer(nullable=False))
    for column, (lower, upper) in numeric_bounds.items():
        public_values[column] = [float(lower)]
        transformers.append(BinTransformer(bins=bins, lower=lower, upper=upper))

    public_size = max(len(values) for values in public_values.values())
    public_frame = pd.DataFrame({
        column: [values[index % len(values)] for index in range(public_size)]
        for column, values in public_values.items()
    })
    transformer = TableTransformer(transformers)
    transformer.fit(public_frame, epsilon=0.0)
    prepared = pd.DataFrame(index=pd.RangeIndex(len(data)))
    for column, categories in categorical_domains.items():
        values = data[column].reset_index(drop=True)
        prepared[column] = values.where(values.isin(categories), categories[0]).fillna(categories[0])
    for column, (lower, upper) in numeric_bounds.items():
        prepared[column] = bounded_column(data[column], lower, upper).to_numpy()

    budget = BudgetAccountant.load_default(accountant)
    budget.check(epsilon, delta)
    model = Synthesizer.create(algorithm, epsilon=float(internal_epsilon), delta=float(internal_delta))
    budget.spend(epsilon, delta)
    model.fit(prepared, transformer=transformer, preprocessor_eps=0.0)
    return pd.DataFrame(model.sample(int(n_rows)), columns=columns).reset_index(drop=True)