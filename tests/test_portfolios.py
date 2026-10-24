import numpy as np
import pandas as pd
import pytest

from riskcheck.portfolios import (
    PORTFOLIOS,
    RP_LOOKBACK,
    build_all,
    max_drawdown,
    rebalance_dates,
    risk_parity,
    simulate,
    target_weights,
)


def test_rebalance_dates_are_month_ends(synthetic_returns):
    rd = rebalance_dates(synthetic_returns.index)
    assert len(rd) == 48
    assert rd[0] == pd.Timestamp("2005-01-31")
    assert all(d in synthetic_returns.index for d in rd)


@pytest.mark.parametrize("name", list(PORTFOLIOS))
def test_weights_sum_to_one_and_long_only(synthetic_returns, name):
    tw = target_weights(synthetic_returns, PORTFOLIOS[name])
    assert np.allclose(tw.sum(axis=1), 1.0)
    assert (tw >= 0).all().all()


def test_risk_parity_needs_full_window(synthetic_returns):
    with pytest.raises(ValueError):
        risk_parity(synthetic_returns.iloc[: RP_LOOKBACK - 1])


def test_risk_parity_weights_ignore_future(synthetic_returns):
    """Weights at a rebalance date must not change when data after it changes."""
    d = rebalance_dates(synthetic_returns.index)[10]
    base = target_weights(synthetic_returns, risk_parity).loc[d]
    shocked = synthetic_returns.copy()
    shocked.loc[shocked.index > d] *= 25.0
    after = target_weights(shocked, risk_parity).loc[d]
    pd.testing.assert_series_equal(base, after)


def test_portfolio_return_on_day_ignores_same_day_rebalance(synthetic_returns):
    """The return on day t+1 uses weights set at the close of t, never weights from t+1."""
    port, _ = build_all(synthetic_returns)
    d = rebalance_dates(synthetic_returns.index)[12]
    shocked = synthetic_returns.copy()
    shocked.loc[shocked.index > d] *= 3.0
    port2, _ = build_all(shocked)
    pd.testing.assert_series_equal(port.loc[:d, "risk_parity"], port2.loc[:d, "risk_parity"])


def test_simulate_constant_mix_matches_manual():
    idx = pd.bdate_range("2020-01-01", periods=5)
    lr = pd.DataFrame({"A": np.log([1.0, 1.1, 0.9, 1.0, 1.05]), "B": 0.0}, index=idx)
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[idx[0]])
    res, w = simulate(lr, targets)
    # Day 2: 0.5*0.1 = 0.05; weights drift to A = 0.55/1.05
    assert res["ret"].iloc[0] == pytest.approx(0.05)
    assert w["A"].iloc[1] == pytest.approx(0.55 / 1.05)
    assert res["ret"].iloc[1] == pytest.approx((0.55 / 1.05) * -0.1)


def test_max_drawdown():
    lr = pd.Series(np.log([1.0, 1.2, 0.5, 1.5]))
    # wealth: 1, 1.2, 0.6, 0.9 -> peak 1.2, trough 0.6 -> -50%
    assert max_drawdown(lr) == pytest.approx(-0.5)
