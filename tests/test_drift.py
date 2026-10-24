import numpy as np
import pandas as pd
import pytest

from riskcheck import SEED
from riskcheck.drift import (
    COOLDOWN,
    breach_count,
    cusum,
    page_hinkley,
    regime_starts,
    score,
    vix_threshold,
)


def test_regime_starts_rule():
    idx = pd.bdate_range("2015-01-01", periods=300)
    v = pd.Series(15.0, index=idx)
    v.iloc[100:105] = 35.0  # start at 100 (calm before)
    v.iloc[130] = 32.0  # within 60 days of the last start: merged
    v.iloc[200:260] = 25.0  # prior mean > 20 ...
    v.iloc[260] = 31.0  # ... so no start here
    starts = regime_starts(v)
    assert list(starts) == [idx[100]]


def test_cusum_quiet_on_calibrated_errors_and_fires_on_jump():
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(600)
    assert cusum(z[:200]).sum() <= 2
    z[400:] *= 3
    a = cusum(z)
    first = np.flatnonzero(a[400:])
    assert len(first) and first[0] < 10


def test_cooldown_suppresses_repeat_alarms():
    z = np.full(100, 5.0)
    a = cusum(z)
    hits = np.flatnonzero(a)
    assert np.all(np.diff(hits) > COOLDOWN)


def test_page_hinkley_fires_on_variance_increase():
    rng = np.random.default_rng(SEED)
    z = np.r_[rng.standard_normal(300), 3 * rng.standard_normal(100)]
    hits = np.flatnonzero(page_hinkley(z))
    assert any(300 <= h < 330 for h in hits)


def test_breach_count_threshold():
    b = np.zeros(40, dtype=bool)
    b[[5, 8, 11]] = True
    assert not breach_count(b).any()
    b[15] = True  # 4th breach within 20 days
    assert np.flatnonzero(breach_count(b)).tolist() == [15]


def test_vix_threshold_upcross_only():
    v = np.array([20, 24, 26, 27, 24, 26] + [20] * 30 + [26])
    hits = np.flatnonzero(vix_threshold(v)).tolist()
    assert hits == [2, 36]  # day 5 recross is inside the cooldown


def test_score_delay_and_false_alarms():
    idx = pd.bdate_range("2019-01-01", periods=400)
    vix = pd.Series(15.0, index=idx)
    vix.iloc[200:240] = 35.0
    starts = pd.DatetimeIndex([idx[200]])
    alarms = pd.Series(False, index=idx)
    alarms.iloc[[50, 198, 210]] = True  # false alarm, early detection, repeat
    sc = score(alarms, starts, vix)
    assert sc.detected == 1 and sc.delays == [-2]
    assert sc.false_alarms == 1
    assert sc.precision == pytest.approx(2 / 3)
