"""EWMA volatility (RiskMetrics) with normal VaR and ES, zero mean."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskcheck.models.base import normal_risk, to_frame

LAMBDA = 0.94
WARMUP = 250  # first forecast after this many returns, matching historical simulation


def ewma_variance(r: np.ndarray, lam: float = LAMBDA, init_n: int = 60) -> np.ndarray:
    """Return s2 where s2[i] is the variance forecast for r[i] using r[:i] only.

    Seeded with the mean square of the first ``init_n`` returns; s2[i] for i < init_n
    is NaN because the seed itself would use those returns.
    """
    n = len(r)
    s2 = np.full(n, np.nan)
    v = float(np.mean(r[:init_n] ** 2))
    for i in range(init_n, n):
        s2[i] = v
        v = lam * v + (1.0 - lam) * r[i] ** 2
    return s2


def forecast(returns: pd.Series, lam: float = LAMBDA, warmup: int = WARMUP) -> pd.DataFrame:
    s2 = ewma_variance(returns.to_numpy(dtype=float), lam)
    sigma = np.sqrt(s2[warmup:])
    return to_frame(returns.index[warmup:], normal_risk(0.0, sigma))
