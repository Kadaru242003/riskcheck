"""VaR test functions against textbook values, conformal coverage, and leakage checks."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskcheck import SEED
from riskcheck.backtest import check_no_lookahead, evaluate
from riskcheck.conformal import conformal_var
from riskcheck.models import ewma, historical
from riskcheck.tests_var import basel_zones, christoffersen, es_backtest, kupiec, wilson_ci


def seq(x: int, n: int) -> np.ndarray:
    return np.r_[np.ones(x), np.zeros(n - x)].astype(bool)


# Kupiec (1995) 95% non-rejection regions, as tabulated in Jorion, Value at Risk, Table 6-3.
# (The table lists N < 7 for T=255, p=1%; the LR test itself rejects N=0 there, LR=5.13.)
JORION = [
    (0.01, 255, 1, 6),
    (0.01, 510, 2, 10),
    (0.01, 1000, 5, 16),
    (0.05, 255, 7, 20),
    (0.05, 510, 17, 35),
    (0.05, 1000, 38, 64),
]


@pytest.mark.parametrize("p,n,lo,hi", JORION)
def test_kupiec_matches_jorion_table(p, n, lo, hi):
    assert kupiec(seq(lo, n), p).pvalue > 0.05
    assert kupiec(seq(hi, n), p).pvalue > 0.05
    assert kupiec(seq(lo - 1, n), p).pvalue < 0.05
    assert kupiec(seq(hi + 1, n), p).pvalue < 0.05


def test_kupiec_closed_form():
    n, x, p = 250, 10, 0.01
    pi = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    assert kupiec(seq(x, n), p).stat == pytest.approx(lr)


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(SEED)
    iid = rng.random(2000) < 0.05
    clustered = np.zeros(2000, dtype=bool)
    for start in rng.choice(np.arange(0, 1990, 20), size=20, replace=False):
        clustered[start : start + 5] = True  # same 5% rate, in runs of 5
    assert christoffersen(iid).pvalue > 0.05
    assert christoffersen(clustered).pvalue < 1e-6


def test_christoffersen_closed_form():
    b = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    # transitions: n00=4, n01=2, n10=2, n11=1
    n00, n01, n10, n11 = 4, 2, 2, 1
    pi01, pi11, pi = n01 / 6, n11 / 3, 3 / 9
    l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    l1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    assert christoffersen(b).stat == pytest.approx(-2 * (l0 - l1))


def test_basel_zone_thresholds():
    idx = pd.bdate_range("2020-01-01", periods=250)
    for x, zone in [(4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")]:
        z = basel_zones(pd.Series(seq(x, 250), index=idx))
        assert len(z) == 1 and z["zone"].iloc[0] == zone


def test_es_backtest_correct_model_near_one():
    rng = np.random.default_rng(SEED)
    loss = rng.standard_normal(200_000)
    var = np.full_like(loss, stats.norm.ppf(0.975))
    es = np.full_like(loss, stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025)
    res = es_backtest(loss, var, es)
    assert res.ratio == pytest.approx(1.0, abs=0.02)
    assert res.ci_low < 1.0 < res.ci_high
    assert abs(res.z2) < 0.05


def test_es_backtest_flags_underestimate():
    rng = np.random.default_rng(SEED)
    loss = rng.standard_t(3, 50_000) * 1.5
    var = np.full_like(loss, stats.norm.ppf(0.975))
    es = np.full_like(loss, stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025)
    res = es_backtest(loss, var, es)
    assert res.ci_low > 1.0 and res.z2 < -0.7


def test_wilson_ci_textbook():
    lo, hi = wilson_ci(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-4) and hi == pytest.approx(0.1744, abs=1e-4)


def test_conformal_fixes_undercoverage():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=6000)
    loss = pd.Series(rng.standard_t(4, 6000) * 0.01, index=idx)
    bad_var = pd.Series(np.full(6000, 0.01), index=idx, name="var99")  # far too small
    fixed = conformal_var(loss, bad_var, 0.99)
    ok = fixed.notna()
    raw_cov = (loss[ok] <= bad_var[ok]).mean()
    conf_cov = (loss[ok] <= fixed[ok]).mean()
    assert raw_cov < 0.85
    assert conf_cov == pytest.approx(0.99, abs=0.004)


def test_conformal_no_lookahead():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=900)
    loss = pd.Series(rng.standard_normal(900), index=idx)
    var = pd.Series(np.full(900, 2.0), index=idx, name="var99")
    base = conformal_var(loss, var, 0.99)
    d = idx[700]
    loss2 = loss.copy()
    loss2[loss2.index >= d] += 50
    after = conformal_var(loss2, var, 0.99)
    pd.testing.assert_series_equal(base.loc[:d], after.loc[:d])


# ------------------------------------------------------------------ leakage checks


@pytest.fixture
def garch_like():
    """Simulated GARCH(1,1) returns: volatility clusters, so honest forecasts carry signal."""
    rng = np.random.default_rng(SEED)
    n = 3000
    r, s2 = np.zeros(n), np.full(n, 1e-4)
    for t in range(1, n):
        s2[t] = 2e-6 + 0.1 * r[t - 1] ** 2 + 0.88 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return pd.Series(r, index=pd.bdate_range("2005-01-03", periods=n))


def test_shuffled_dates_break_the_backtest(garch_like):
    """With returns shuffled against forecasts, the forecasts should lose all their signal."""
    fc = ewma.forecast(garch_like)
    actual = garch_like.reindex(fc.index)
    aligned = np.corrcoef(fc["vol"], actual.abs())[0, 1]
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(actual.to_numpy())
    broken = np.corrcoef(fc["vol"], np.abs(shuffled))[0, 1]
    assert aligned > 0.2
    assert abs(broken) < 0.05
    # Sanity: the aligned backtest itself is near its 1% target on this data.
    df = pd.DataFrame({"loss": -actual, **{c: fc[c] for c in fc}}, index=fc.index)
    assert abs(evaluate(df)["breach_rate_99"] - 0.01) < 0.006


def test_check_no_lookahead_catches_a_cheating_model(garch_like):
    def cheater(s: pd.Series) -> pd.DataFrame:
        fc = historical.forecast(s)
        fc["var99"] = -s.reindex(fc.index).clip(upper=0) + 1e-9  # peeks at today's return
        return fc

    dates = [garch_like.index[1500], garch_like.index[2200]]
    assert check_no_lookahead(historical.forecast, garch_like, dates)
    assert check_no_lookahead(ewma.forecast, garch_like, dates)
    assert not check_no_lookahead(cheater, garch_like, dates)
