"""Six-factor risk model with risk attribution.

For forecast date d:
- B: each asset's exposures to the 6 factors, from an OLS (with intercept) of its
  daily log returns on factor returns over the ``WINDOW`` days strictly before d.
- D: each asset's residual (idiosyncratic) variance from the same regression.
- Sigma_f: EWMA covariance of factor returns using days strictly before d.

Portfolio variance: sigma_p^2 = w' (B Sigma_f B' + D) w, with w the weights held on d
(set at the previous close, so known before d). VaR and ES assume normal returns.

Attribution (Euler): factor k's share of variance is b_k (Sigma_f b)_k / sigma_p^2 with
b = B' w; the idiosyncratic share is w' D w / sigma_p^2. Shares sum to 1; a factor
share can be negative when it hedges the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskcheck.data import FACTORS
from riskcheck.models.base import normal_risk, to_frame

WINDOW = 250
LAMBDA = 0.94
COV_INIT = 60
IDIO = "idiosyncratic"


@dataclass
class FactorModel:
    dates: pd.DatetimeIndex  # forecast dates
    assets: list[str]
    B: np.ndarray  # (T, n_assets, n_factors)
    D: np.ndarray  # (T, n_assets)
    S: np.ndarray  # (T, n_factors, n_factors)

    def portfolio(self, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Forecasts and attribution for a weight history indexed by forecast date."""
        common = self.dates.intersection(weights.index)
        pos = self.dates.get_indexer(common)
        W = weights.reindex(index=common, columns=self.assets).fillna(0.0).to_numpy()
        B, D, S = self.B[pos], self.D[pos], self.S[pos]
        b = np.einsum("tij,ti->tj", B, W)  # portfolio factor exposures
        Sb = np.einsum("tjk,tk->tj", S, b)
        contrib = b * Sb  # (T, n_factors)
        idio = np.einsum("ti,ti->t", W**2, D)
        var = contrib.sum(axis=1) + idio
        sigma = np.sqrt(var)
        fc = to_frame(common, normal_risk(0.0, sigma))
        attr = pd.DataFrame(contrib / var[:, None], index=common, columns=FACTORS)
        attr[IDIO] = idio / var
        expo = pd.DataFrame(b, index=common, columns=[f"beta_{f}" for f in FACTORS])
        return fc, attr.join(expo)


def ewma_cov(F: np.ndarray, lam: float = LAMBDA, init_n: int = COV_INIT) -> np.ndarray:
    """S[i] = covariance forecast for row i using rows before i (NaN until init_n)."""
    n, k = F.shape
    S = np.full((n, k, k), np.nan)
    c = np.cov(F[:init_n].T)
    for i in range(init_n, n):
        S[i] = c
        x = F[i][:, None]
        c = lam * c + (1.0 - lam) * (x @ x.T)
    return S


def fit(asset_returns: pd.DataFrame, factors: pd.DataFrame, window: int = WINDOW) -> FactorModel:
    """Build exposures, residual variances and factor covariance for every forecast date."""
    if not asset_returns.index.equals(factors.index):
        raise ValueError("asset and factor returns must share the same dates")
    R = asset_returns.to_numpy(dtype=float)
    F = factors[FACTORS].to_numpy(dtype=float)
    n, k = F.shape
    S_all = ewma_cov(F)
    start = max(window, COV_INIT)
    T = n - start
    B = np.empty((T, R.shape[1], k))
    D = np.empty((T, R.shape[1]))
    for t, i in enumerate(range(start, n)):
        X = np.column_stack([np.ones(window), F[i - window : i]])
        Y = R[i - window : i]
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ coef
        B[t] = coef[1:].T
        D[t] = (resid**2).sum(axis=0) / (window - k - 1)
    return FactorModel(
        dates=asset_returns.index[start:],
        assets=list(asset_returns.columns),
        B=B,
        D=D,
        S=S_all[start:],
    )
