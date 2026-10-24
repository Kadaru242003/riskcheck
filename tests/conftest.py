import numpy as np
import pandas as pd
import pytest

from riskcheck import SEED
from riskcheck.data import ALL_ASSETS, CORE_ASSETS, FACTORS


@pytest.fixture
def synthetic_inputs():
    """Fake raw inputs shaped like the real downloads (prices, factors, VIX, T-bill)."""
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2005-01-03", "2007-12-31")
    rets = rng.normal(0.0003, 0.01, size=(len(days), len(ALL_ASSETS)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=ALL_ASSETS)
    prices.loc[:"2006-06-30", "XLRE"] = np.nan  # late-start ETF
    prices.loc["2006-03-15", "TLT"] = np.nan  # missing core price -> drop
    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(days), len(FACTORS))), index=days, columns=FACTORS
    )
    factors["RF"] = 0.0001
    factors = factors.drop(pd.Timestamp("2007-02-01"))  # missing factor row -> drop
    vix = pd.Series(rng.uniform(10, 40, len(days)), index=days, name="VIXCLS")
    vix.iloc[::17] = np.nan  # gaps to forward-fill
    tbill = pd.Series(3.0, index=days, name="DTB3")
    dgs10 = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.05, len(days))), index=days, name="DGS10")
    dgs10 = dgs10.drop(pd.Timestamp("2006-07-04"))  # FRED holiday gap -> forward-filled
    dgs10.loc["2006-11-10"] = np.nan  # FRED "." entry -> forward-filled
    return {"prices": prices, "factors": factors, "vix": vix, "tbill": tbill, "dgs10": dgs10}


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2005-01-03", "2008-12-31")
    cols = [*CORE_ASSETS]
    return pd.DataFrame(rng.normal(0.0002, 0.01, (len(days), len(cols))), index=days, columns=cols)
