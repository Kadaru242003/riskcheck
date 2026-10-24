"""GARCH(1,1) with standardized Student-t errors and a constant mean.

Parameters are refit every ``REFIT_EVERY`` trading days on the trailing window
(up to ``MAX_WINDOW`` returns; an expanding window from ``MIN_WINDOW`` until then, so
the forecasts cover 2008). Between refits the variance is filtered forward with the
last fitted parameters, using returns strictly before each forecast date.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from riskcheck.models.base import student_t_risk, to_frame

MAX_WINDOW = 1000
MIN_WINDOW = 500
REFIT_EVERY = 5
SCALE = 100.0  # fit in percent units for numerical stability


def _fit(r_pct: np.ndarray, start: np.ndarray | None):
    model = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", starting_values=start, show_warning=False)
    if not np.isfinite(res.params).all() or res.params["nu"] <= 2.05:
        res = model.fit(disp="off", show_warning=False)
    return res


def forecast(
    returns: pd.Series,
    min_window: int = MIN_WINDOW,
    max_window: int = MAX_WINDOW,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    r = returns.to_numpy(dtype=float) * SCALE
    n = len(r)
    mu = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    nu = np.full(n, np.nan)
    params_log = []

    start = None
    i = min_window
    while i < n:
        lo = max(0, i - max_window)
        res = _fit(r[lo:i], start)  # data strictly before forecast date i
        p = res.params
        start = p.to_numpy()
        m, w, a, b, v = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"], p["nu"]
        params_log.append((returns.index[i], m, w, a, b, v))
        # variance for day i from the last in-sample day (i - 1)
        s2_prev = res.conditional_volatility[-1] ** 2
        e_prev = r[i - 1] - m
        for j in range(i, min(i + refit_every, n)):
            s2 = w + a * e_prev**2 + b * s2_prev
            mu[j], sig[j], nu[j] = m, np.sqrt(s2), v
            s2_prev, e_prev = s2, r[j] - m  # r[j] is only used for forecasts after j
        i += refit_every

    keep = slice(min_window, n)
    out = to_frame(
        returns.index[keep],
        student_t_risk(mu[keep] / SCALE, sig[keep] / SCALE, nu[keep]),
    )
    out.attrs["params"] = pd.DataFrame(
        params_log, columns=["fit_date", "mu", "omega", "alpha", "beta", "nu"]
    ).set_index("fit_date")
    return out
