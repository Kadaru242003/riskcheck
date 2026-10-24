"""Historical simulation: VaR and ES are empirical statistics of the last N returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskcheck.models.base import COLUMNS

WINDOW = 250


def forecast(returns: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    """Forecast for each date d from the ``window`` returns strictly before d."""
    r = returns.to_numpy(dtype=float)
    rows = []
    idx = returns.index[window:]
    for i in range(window, len(r)):
        past = r[i - window : i]
        q05, q01, q025 = np.quantile(past, [0.05, 0.01, 0.025])
        tail = past[past <= q025]
        rows.append((past.std(ddof=1), -q05, -q01, -q025, -tail.mean()))
    return pd.DataFrame(rows, index=idx, columns=COLUMNS)
