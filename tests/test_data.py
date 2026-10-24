"""Data checks.

Unit tests on synthetic inputs always run. Tests on the real processed data run only
when data/processed exists (after ``make data``); CI skips them because raw data is
never committed.
"""

import numpy as np
import pandas as pd
import pytest

from riskcheck.data import (
    CORE_ASSETS,
    END,
    EXTREME_RETURN,
    FACTORS,
    PROCESSED_DIR,
    clean,
    load,
    parse_french_csv,
)

FRENCH_SAMPLE = """This file was created using the 202512 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20050103,  -0.95,  -0.37,   0.08,   0.21,   0.10,   0.009
20050104,  -1.29,  -0.40,   0.39,   0.12,   0.30,   0.009

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_csv():
    df = parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2005-01-03")
    assert df.loc["2005-01-03", "Mkt-RF"] == pytest.approx(-0.0095)


class TestCleanSynthetic:
    def test_drops_are_logged(self, synthetic_inputs):
        cd = clean(**synthetic_inputs)
        reasons = dict(zip(cd.dropped["date"], cd.dropped["reason"], strict=True))
        assert "TLT" in reasons[pd.Timestamp("2006-03-15")]
        assert reasons[pd.Timestamp("2007-02-01")] == "no factor row"
        assert pd.Timestamp("2006-03-15") not in cd.returns.index

    def test_alignment_and_no_gaps(self, synthetic_inputs):
        cd = clean(**synthetic_inputs)
        assert cd.factors.index.equals(cd.returns.index)
        assert cd.market.index.equals(cd.returns.index)
        assert not cd.returns[CORE_ASSETS].isna().any().any()
        assert not cd.factors[FACTORS].isna().any().any()
        assert not cd.market.isna().any().any()

    def test_log_returns(self, synthetic_inputs):
        cd = clean(**synthetic_inputs)
        p = cd.prices["SPY"]
        d0, d1 = cd.returns.index[5], cd.returns.index[6]
        assert cd.returns.at[d1, "SPY"] == pytest.approx(np.log(p[d1] / p[d0]))

    def test_vix_forward_fill_uses_past_only(self, synthetic_inputs):
        vix = synthetic_inputs["vix"]
        cd = clean(**synthetic_inputs)
        gap = vix.index[vix.isna()][3]
        prev = vix.loc[:gap].dropna().iloc[-1]
        assert cd.market.at[gap, "vix"] == prev

    def test_extreme_return_flagged(self, synthetic_inputs):
        p = synthetic_inputs["prices"].copy()
        p.loc["2007-06-01":, "XLE"] *= 1.5  # ~ +40% one-day jump
        cd = clean(
            p, synthetic_inputs["factors"], synthetic_inputs["vix"], synthetic_inputs["tbill"]
        )
        hit = cd.extremes[(cd.extremes["ticker"] == "XLE")]
        assert len(hit) == 1 and hit["reason"].iloc[0] == ""


real = pytest.mark.skipif(
    not (PROCESSED_DIR / "returns.parquet").exists(), reason="run `make data` first"
)


@real
class TestRealData:
    @pytest.fixture(scope="class")
    def cd(self):
        return load()

    def test_no_future_dates(self, cd):
        assert cd.returns.index.max() <= pd.Timestamp(END)
        assert cd.returns.index.max() <= pd.Timestamp.today()

    def test_no_duplicate_dates(self, cd):
        for frame in (cd.prices, cd.returns, cd.factors, cd.market):
            assert frame.index.is_unique
            assert frame.index.is_monotonic_increasing

    def test_extreme_returns_have_reasons(self, cd):
        r = cd.returns.stack()
        n_extreme = int((r.abs() > EXTREME_RETURN).sum())
        assert n_extreme == len(cd.extremes)
        unexplained = cd.extremes[cd.extremes["reason"].fillna("") == ""]
        assert unexplained.empty, f"unexplained extreme returns:\n{unexplained}"

    def test_factor_dates_match_price_dates(self, cd):
        assert cd.factors.index.equals(cd.returns.index)
        assert not cd.factors[FACTORS].isna().any().any()

    def test_full_history(self, cd):
        assert cd.returns.index.min() < pd.Timestamp("2005-01-15")
        assert cd.returns.index.max() > pd.Timestamp("2025-12-15")
        assert len(cd.returns) > 5000
