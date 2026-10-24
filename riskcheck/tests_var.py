"""Statistical tests for VaR and ES forecasts.

- Kupiec (1995) unconditional coverage: is the breach rate equal to the target?
- Christoffersen (1998) independence: do breaches cluster on consecutive days?
- Basel traffic light: 99% VaR breaches per rolling 250-day window.
- Expected Shortfall: realized/forecast loss ratio on 97.5% VaR breach days with a
  bootstrap interval, and the Acerbi-Szekely (2014) Z2 statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from riskcheck import SEED


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


@dataclass
class TestResult:
    stat: float
    pvalue: float


def kupiec(breaches: np.ndarray, p: float) -> TestResult:
    """Likelihood-ratio test that the breach probability equals ``p`` (chi-square, 1 dof)."""
    b = np.asarray(breaches, dtype=bool)
    n, x = len(b), int(b.sum())
    pi = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen(breaches: np.ndarray) -> TestResult:
    """Likelihood-ratio test of independence against first-order Markov breaches (1 dof)."""
    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def basel_zones(breaches99: pd.Series, window: int = 250) -> pd.DataFrame:
    """Rolling 250-day count of 99% breaches and its Basel zone (green 0-4, yellow 5-9, red 10+)."""
    count = breaches99.astype(int).rolling(window).sum().dropna()
    zone = pd.cut(count, bins=[-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
    return pd.DataFrame({"count": count, "zone": zone})


@dataclass
class ESResult:
    n_tail: int
    ratio: float  # mean realized loss / mean forecast ES on tail days (>1: ES too low)
    ci_low: float
    ci_high: float
    z2: float  # Acerbi-Szekely Z2; 0 if correct, negative if ES underestimates


def es_backtest(
    loss: np.ndarray,
    var975: np.ndarray,
    es975: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> ESResult:
    loss, var975, es975 = (np.asarray(a, dtype=float) for a in (loss, var975, es975))
    tail = loss > var975
    n = len(loss)
    z2 = 1.0 - float(np.sum(loss[tail] / es975[tail])) / (n * 0.025)
    L, E = loss[tail], es975[tail]
    if len(L) < 2:
        return ESResult(len(L), np.nan, np.nan, np.nan, z2)
    ratio = L.mean() / E.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(L), size=(n_boot, len(L)))
    boot = L[idx].mean(axis=1) / E[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return ESResult(len(L), float(ratio), float(lo), float(hi), z2)


def wilson_ci(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + conf / 2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)
