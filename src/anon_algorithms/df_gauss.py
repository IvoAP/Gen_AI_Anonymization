from collections.abc import Mapping

import pandas as pd
from diffprivlib.accountant import BudgetAccountant

from ._common import private_means


def gaussian_means(
    data: pd.DataFrame,
    *,
    bounds: Mapping[str, tuple[float, float]],
    epsilon: float,
    delta: float,
    accountant: BudgetAccountant | None = None,
) -> pd.Series:
    """Release bounded means using the analytic Gaussian mechanism.

    Assumes replacement adjacency and a public row count. Total epsilon/delta
    are split across columns. Missing/non-numeric values use public midpoints.
    Only the selected means are released, never unchanged input records.
    """
    return private_means(
        data, bounds=bounds, epsilon=epsilon, delta=delta,
        gaussian=True, accountant=accountant,
    )