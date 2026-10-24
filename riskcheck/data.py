"""Download, cache, clean and align all input data.

Sources (all free; raw files are cached in data/raw and never committed):
- ETF adjusted closes: Yahoo Finance via yfinance, falling back to Stooq CSV.
- Fama-French 5 factors and momentum, daily: Kenneth French Data Library.
- VIX (VIXCLS) and 3-month T-bill (DTB3): FRED.

Run ``python -m riskcheck.data`` to build data/processed/*.parquet.
Reruns read the cache and need no network.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START = "2005-01-01"
END = "2025-12-31"

SECTORS_LONG = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
SECTORS_LATE = ["XLRE", "XLC"]  # start 2015 and 2018; robustness checks only
CORE_ASSETS = ["SPY", *SECTORS_LONG, "TLT", "GLD"]
ALL_ASSETS = [*CORE_ASSETS, *SECTORS_LATE]

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Daily moves above this size need a documented reason.
EXTREME_RETURN = 0.25
# Known, verified extreme moves: (ticker, date) -> reason. Empty until one is found.
KNOWN_EXTREMES: dict[tuple[str, str], str] = {}

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- download


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "riskcheck/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _cached(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / name


def download_prices(tickers: list[str] = ALL_ASSETS, refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker. Cached per ticker."""
    cols = {}
    for t in tickers:
        path = _cached(f"price_{t}.csv")
        if refresh or not path.exists():
            s = _download_yahoo(t)
            if s is None or s.empty:
                s = _download_stooq(t)
            if s is None or s.empty:
                raise RuntimeError(f"Could not download prices for {t}")
            s.rename("close").to_csv(path, index_label="date")
        s = pd.read_csv(path, index_col="date", parse_dates=True)["close"]
        cols[t] = s
    return pd.DataFrame(cols).sort_index()


def _download_yahoo(ticker: str) -> pd.Series | None:
    try:
        import yfinance as yf

        df = yf.download(ticker, start=START, end="2026-01-01", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.astype(float)
    except Exception as e:  # network or API change: fall back to Stooq
        log.warning("yfinance failed for %s: %s", ticker, e)
        return None


def _download_stooq(ticker: str) -> pd.Series | None:
    try:
        raw = _fetch(f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d")
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"], index_col="Date")
        return df["Close"].astype(float)
    except Exception as e:
        log.warning("Stooq failed for %s: %s", ticker, e)
        return None


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Kenneth French daily CSV (header lines, then YYYYMMDD rows, then a footer).

    Returns decimal returns (the files are in percent).
    """
    lines = text.splitlines()

    def is_row(ln: str) -> bool:
        first = ln.split(",")[0].strip()
        return len(first) == 8 and first.isdigit()

    first_row = next(i for i, ln in enumerate(lines) if is_row(ln))
    rows = []
    for ln in lines[first_row:]:
        if not is_row(ln):
            break  # the daily block ends at the first non-date line
        rows.append(ln)
    header = [h.strip() for h in lines[first_row - 1].split(",")]
    header[0] = "date"
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=header)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    df.columns = [c.strip() for c in df.columns]
    return df


def download_factors(refresh: bool = False) -> pd.DataFrame:
    """Daily FF5 + momentum + RF, as decimals."""
    frames = []
    for key, fname in FRENCH_FILES.items():
        path = _cached(f"french_{key}.csv")
        if refresh or not path.exists():
            z = zipfile.ZipFile(io.BytesIO(_fetch(FRENCH_BASE + fname)))
            inner = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            path.write_bytes(z.read(inner))
        frames.append(parse_french_csv(path.read_text(encoding="latin-1")))
    df = frames[0].join(frames[1], how="inner")
    df = df.rename(columns={c: "Mom" for c in df.columns if c.lower().startswith("mom")})
    return df[[*FACTORS, "RF"]]


def download_fred(series: str, refresh: bool = False) -> pd.Series:
    path = _cached(f"fred_{series}.csv")
    if refresh or not path.exists():
        path.write_bytes(_fetch(FRED_URL.format(series=series)))
    df = pd.read_csv(path)
    date_col = df.columns[0]
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.rename(series)


# --------------------------------------------------------------------------- clean


@dataclass
class CleanData:
    """Aligned, lookahead-free inputs on a common trading-day calendar."""

    prices: pd.DataFrame  # adjusted closes
    returns: pd.DataFrame  # daily log returns, row t = log(P_t / P_{t-1})
    factors: pd.DataFrame  # daily factor returns (decimal), same index as returns
    market: pd.DataFrame  # vix level, daily risk-free rate; same index as returns
    dropped: pd.DataFrame  # every dropped day with its reason
    extremes: pd.DataFrame  # every |return| > EXTREME_RETURN with its reason


def clean(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    vix: pd.Series,
    tbill: pd.Series,
    start: str = START,
    end: str = END,
) -> CleanData:
    """Align everything on trading days and compute log returns.

    Rules:
    - The calendar is the set of days with a SPY price inside [start, end].
    - A day is dropped if any core asset price is missing, or the factor file has no row.
    - Late-start ETFs (XLRE, XLC) may be NaN before their inception; they are not core.
    - VIX and T-bill gaps are forward-filled from earlier days only (never backward).
    """
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)].sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.loc[prices["SPY"].notna()]

    dropped = []
    missing_core = prices[CORE_ASSETS].isna().any(axis=1)
    for d in prices.index[missing_core]:
        miss = [c for c in CORE_ASSETS if pd.isna(prices.at[d, c])]
        dropped.append((d, "missing core price: " + ",".join(miss)))
    prices = prices.loc[~missing_core]

    no_factor = ~prices.index.isin(factors.index)
    for d in prices.index[no_factor]:
        dropped.append((d, "no factor row"))
    prices = prices.loc[~no_factor]

    # Late-start ETFs get NaN returns before (and on) their first price, which is fine.
    returns = np.log(prices / prices.shift(1)).iloc[1:]

    fac = factors.reindex(returns.index)

    vix_a = vix.sort_index().reindex(vix.index.union(returns.index)).ffill()
    tb_a = tbill.sort_index().reindex(tbill.index.union(returns.index)).ffill()
    market = pd.DataFrame(
        {
            "vix": vix_a.reindex(returns.index),
            # DTB3 is an annualized percent yield; convert to a daily decimal rate.
            "rf_daily": tb_a.reindex(returns.index) / 100.0 / 252.0,
        }
    )

    ext = []
    stacked = returns.stack()
    for (d, t), r in stacked[stacked.abs() > EXTREME_RETURN].items():
        reason = KNOWN_EXTREMES.get((t, d.strftime("%Y-%m-%d")), "")
        ext.append((d, t, float(r), reason))

    return CleanData(
        prices=prices,
        returns=returns,
        factors=fac,
        market=market,
        dropped=pd.DataFrame(dropped, columns=["date", "reason"]),
        extremes=pd.DataFrame(ext, columns=["date", "ticker", "log_return", "reason"]),
    )


def save(cd: CleanData, out: Path = PROCESSED_DIR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cd.prices.to_parquet(out / "prices.parquet")
    cd.returns.to_parquet(out / "returns.parquet")
    cd.factors.to_parquet(out / "factors.parquet")
    cd.market.to_parquet(out / "market.parquet")
    cd.dropped.to_csv(out / "dropped_days.csv", index=False)
    cd.extremes.to_csv(out / "extreme_returns.csv", index=False)


def load(out: Path = PROCESSED_DIR) -> CleanData:
    return CleanData(
        prices=pd.read_parquet(out / "prices.parquet"),
        returns=pd.read_parquet(out / "returns.parquet"),
        factors=pd.read_parquet(out / "factors.parquet"),
        market=pd.read_parquet(out / "market.parquet"),
        dropped=pd.read_csv(out / "dropped_days.csv", parse_dates=["date"]),
        extremes=pd.read_csv(out / "extreme_returns.csv", parse_dates=["date"]),
    )


def build(refresh: bool = False) -> CleanData:
    prices = download_prices(refresh=refresh)
    factors = download_factors(refresh=refresh)
    vix = download_fred("VIXCLS", refresh=refresh)
    tbill = download_fred("DTB3", refresh=refresh)
    cd = clean(prices, factors, vix, tbill)
    save(cd)
    return cd


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cd = build()
    r = cd.returns
    print(
        f"returns: {r.shape[0]} days x {r.shape[1]} assets, {r.index.min():%Y-%m-%d} "
        f"to {r.index.max():%Y-%m-%d}"
    )
    print(f"dropped days: {len(cd.dropped)}  extreme returns: {len(cd.extremes)}")
    print(
        f"factor NaNs: {int(cd.factors.isna().sum().sum())}  "
        f"market NaNs: {int(cd.market.isna().sum().sum())}"
    )
