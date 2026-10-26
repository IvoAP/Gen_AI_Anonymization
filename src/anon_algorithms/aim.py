from collections.abc import Mapping, Sequence

import pandas as pd
from diffprivlib.accountant import BudgetAccountant

from ._smartnoise import synthesize


def synthesize_aim(
	data: pd.DataFrame,
	*,
	epsilon: float,
	delta: float,
	n_rows: int,
	categorical_domains: Mapping[str, Sequence[str | int]] | None = None,
	numeric_bounds: Mapping[str, tuple[float, float]] | None = None,
	bins: int = 16,
	accountant: BudgetAccountant | None = None,
) -> pd.DataFrame:
	"""Fit SmartNoise AIM and sample records using explicit public domains.

	Only schema columns are released. Numeric columns are binned; unknown or
	missing categories map to the first public category. Row count is public.
	Privacy is record-level, not device-level. Every fit consumes a new budget.
	"""
	return synthesize(
		"aim", data, epsilon=epsilon, delta=delta, n_rows=n_rows,
		categorical_domains=categorical_domains if categorical_domains is not None else {},
		numeric_bounds=numeric_bounds if numeric_bounds is not None else {},
		bins=bins, accountant=accountant,
	)
