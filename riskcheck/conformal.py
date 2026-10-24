"""Split-conformal correction for VaR forecasts.

For each forecast date d, the score on past days is s_t = loss_t / VaR_t (how far the
realized loss went relative to the model's VaR). The corrected VaR is

    VaR*_d = VaR_d * q_d,   q_d = the ceil((n + 1) * a) / n empirical quantile of the
                                  scores on the ``CALIB`` days strictly before d.

If scores are exchangeable over the calibration window, P(loss_d <= VaR*_d) >= a.
The correction only rescales the model's forecast, so it keeps its timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALIB = 500


def conformal_var(loss: pd.Series, var: pd.Series, level: float, calib: int = CALIB) -> pd.Series:
    """Corrected VaR for each date with ``calib`` prior outcomes; NaN before that."""
    if not loss.index.equals(var.index):
        raise ValueError("loss and var must share an index")
    s = (loss / var).to_numpy(dtype=float)
    v = var.to_numpy(dtype=float)
    k = int(np.ceil((calib + 1) * level))  # rank of the order statistic, 1-based
    out = np.full(len(s), np.nan)
    if k > calib:
        raise ValueError("calibration window too small for this level")
    for i in range(calib, len(s)):
        window = s[i - calib : i]
        q = np.partition(window, k - 1)[k - 1]
        out[i] = v[i] * q
    return pd.Series(out, index=var.index, name=f"{var.name}_conformal")
