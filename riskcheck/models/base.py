"""Shared conventions for all risk models.

Every model returns a DataFrame indexed by the *forecast date* d with columns:

- ``vol``:   predicted standard deviation of the log return on day d
- ``var95``: 1-day 95% Value at Risk, as a positive loss in log-return units
- ``var99``: 1-day 99% Value at Risk
- ``var975``: 1-day 97.5% Value at Risk (the threshold for the ES backtest)
- ``es975``: 1-day 97.5% Expected Shortfall (average loss beyond the 97.5% VaR)

The row for date d may only use returns dated strictly before d. A breach on day d
means ``-r_d > var`` for that row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

COLUMNS = ["vol", "var95", "var99", "var975", "es975"]


def normal_risk(mu: np.ndarray | float, sigma: np.ndarray | float) -> dict[str, np.ndarray]:
    """VaR and ES for a normal return distribution N(mu, sigma^2), as positive losses."""
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    z95, z99, z975 = stats.norm.ppf([0.95, 0.99, 0.975])
    es975 = sigma * stats.norm.pdf(z975) / 0.025 - mu
    return {
        "vol": sigma,
        "var95": z95 * sigma - mu,
        "var99": z99 * sigma - mu,
        "var975": z975 * sigma - mu,
        "es975": es975,
    }


def student_t_risk(
    mu: np.ndarray | float, sigma: np.ndarray | float, nu: np.ndarray | float
) -> dict[str, np.ndarray]:
    """VaR and ES when returns are mu + sigma * e, with e a *standardized* Student-t (var 1).

    For a standard t with nu dof, quantile q at level a, the ES of the lower tail is
    f(q) * (nu + q^2) / ((nu - 1) * (1 - a)); scaling by sqrt((nu - 2) / nu) gives unit variance.
    """
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nu = np.asarray(nu, dtype=float)
    scale = np.sqrt((nu - 2.0) / nu)
    out = {"vol": sigma}
    for name, a in (("var95", 0.95), ("var99", 0.99), ("var975", 0.975)):
        out[name] = stats.t.ppf(a, nu) * scale * sigma - mu
    q = stats.t.ppf(0.975, nu)
    es_std = stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1.0) * 0.025) * scale
    out["es975"] = es_std * sigma - mu
    return out


def to_frame(index: pd.Index, values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({c: values[c] for c in COLUMNS}, index=index)
