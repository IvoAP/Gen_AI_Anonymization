from collections.abc import Mapping

import pandas as pd
from diffprivlib.accountant import BudgetAccountant

from ._common import private_means


def laplace_means(
	data: pd.DataFrame,
	*,
	bounds: Mapping[str, tuple[float, float]],
	epsilon: float,
	accountant: BudgetAccountant | None = None,
) -> pd.Series:
	"""Release bounded means under replacement adjacency and a public row count.

	The total epsilon is split across selected columns. Bounds must be public.
	Missing/non-numeric values use the public midpoint. Other columns are not
	released. Repeated releases compose through the supplied/default accountant.
	"""
	return private_means(
		data, bounds=bounds, epsilon=epsilon, delta=0.0,
		gaussian=False, accountant=accountant,
	)
