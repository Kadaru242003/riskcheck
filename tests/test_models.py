"""Risk model checks: textbook values, and no reads on or after the forecast date."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from riskcheck import SEED
from riskcheck.data import FACTORS
from riskcheck.models import ewma, factor, garch, historical
from riskcheck.models.base import normal_risk, student_t_risk


@pytest.fixture
def series():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=700)
    vol = np.where(np.arange(700) < 400, 0.008, 0.02)  # a vol regime change
    return pd.Series(rng.standard_t(5, 700) * vol / np.sqrt(5 / 3), index=idx)


def shocked(s: pd.Series, d: pd.Timestamp) -> pd.Series:
    out = s.copy()
    out.loc[out.index >= d] = out.loc[out.index >= d] * 7.0 + 0.05
    return out


MODEL_FNS = {
    "historical": historical.forecast,
    "ewma": ewma.forecast,
    "garch": lambda s: garch.forecast(s, min_window=300, max_window=400, refit_every=5),
}


# ------------------------------------------------------------------ textbook values


def test_normal_var_es_textbook():
    out = normal_risk(0.0, 1.0)
    assert out["var95"] == pytest.approx(1.6449, abs=1e-4)
    assert out["var99"] == pytest.approx(2.3263, abs=1e-4)
    assert out["es975"] == pytest.approx(2.3378, abs=1e-4)  # Basel: ES 97.5 ~= VaR 99


def test_student_t_es_matches_numerical_integration():
    nu = 5.0
    scale = np.sqrt((nu - 2) / nu)
    out = student_t_risk(0.0, 1.0, nu)
    q = stats.t.ppf(0.025, nu)
    tail, _ = integrate.quad(lambda x: x * stats.t.pdf(x, nu), -np.inf, q)
    assert out["es975"] == pytest.approx(-tail / 0.025 * scale, rel=1e-6)
    assert out["var99"] == pytest.approx(-stats.t.ppf(0.01, nu) * scale, rel=1e-9)


def test_student_t_tends_to_normal():
    t = student_t_risk(0.0, 1.0, 1e6)
    n = normal_risk(0.0, 1.0)
    for k in ("var95", "var99", "es975"):
        assert t[k] == pytest.approx(n[k], rel=1e-4)


def test_historical_simple_case():
    idx = pd.bdate_range("2020-01-01", periods=101)
    s = pd.Series(np.r_[np.linspace(-0.05, 0.049, 100), 0.0], index=idx)
    fc = historical.forecast(s, window=100)
    assert len(fc) == 1
    assert fc["var95"].iloc[0] == pytest.approx(-np.quantile(s.iloc[:100], 0.05))
    worst = np.sort(s.iloc[:100].to_numpy())[:3]  # values at or below the 2.5% quantile
    assert fc["es975"].iloc[0] == pytest.approx(-worst.mean())


def test_ewma_recursion():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    s2 = ewma.ewma_variance(r, lam=0.9, init_n=2)
    v0 = np.mean(r[:2] ** 2)
    assert np.isnan(s2[:2]).all()
    assert s2[2] == pytest.approx(v0)
    assert s2[3] == pytest.approx(0.9 * v0 + 0.1 * 0.03**2)


def test_garch_reacts_to_vol_regime(series):
    fc = MODEL_FNS["garch"](series)
    calm = fc.loc[fc.index < series.index[400], "vol"].mean()
    stressed = fc.loc[fc.index > series.index[450], "vol"].mean()
    assert stressed > 1.8 * calm


# ------------------------------------------------------------------ no lookahead


@pytest.mark.parametrize("name", list(MODEL_FNS))
def test_no_lookahead(series, name):
    fn = MODEL_FNS[name]
    base = fn(series)
    for d in (series.index[420], series.index[555]):
        after = fn(shocked(series, d))
        keep = base.index <= d
        pd.testing.assert_frame_equal(base.loc[keep], after.loc[keep])
        assert not np.allclose(base.loc[~keep].iloc[1:3], after.loc[~keep].iloc[1:3])


@pytest.fixture
def factor_data():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=400)
    F = pd.DataFrame(rng.normal(0, 0.01, (400, 6)), index=idx, columns=FACTORS)
    B = rng.normal(0.5, 0.3, (3, 6))
    R = pd.DataFrame(
        F.to_numpy() @ B.T + rng.normal(0, 0.004, (400, 3)), index=idx, columns=["A", "B", "C"]
    )
    W = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=idx)
    return R, F, W, B


def test_factor_attribution_sums_to_one(factor_data):
    R, F, W, _ = factor_data
    fc, attr = factor.fit(R, F).portfolio(W)
    shares = attr[[*FACTORS, factor.IDIO]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    assert (fc["vol"] > 0).all()


def test_factor_recovers_exposures(factor_data):
    R, F, W, B_true = factor_data
    fm = factor.fit(R, F)
    assert np.allclose(fm.B[-1], B_true, atol=0.1)


def test_factor_no_lookahead(factor_data):
    R, F, W, _ = factor_data
    d = R.index[320]
    base, base_attr = factor.fit(R, F).portfolio(W)
    R2, F2 = R.copy(), F.copy()
    R2.loc[R2.index >= d] *= 9.0
    F2.loc[F2.index >= d] *= 9.0
    after, after_attr = factor.fit(R2, F2).portfolio(W)
    pd.testing.assert_frame_equal(base.loc[:d], after.loc[:d])
    pd.testing.assert_frame_equal(base_attr.loc[:d], after_attr.loc[:d])
