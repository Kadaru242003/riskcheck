"""Query layer over the risk engine's outputs, used by the MCP tools and the question set.

All numbers come from reports/forecasts.parquet and reports/attribution.parquet
(``make forecasts``). Risk numbers are returned as percent losses of portfolio value
(log-return units x 100), rounded for display; ground truth uses the same functions.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from riskcheck.portfolios import REPORTS_DIR

PORTFOLIOS = {
    "60_40": "60/40 (60% SPY, 40% TLT)",
    "equal_sector": "equal-weight sectors (9 SPDR sector ETFs)",
    "tech_heavy": "tech-heavy (50% XLK, 50% split across 8 other sectors)",
    "defensive": "defensive (equal XLU, XLP, XLV, TLT, GLD)",
    "risk_parity": "risk parity (11 ETFs weighted by inverse 60-day volatility)",
}
MODELS = {
    "historical": "historical simulation (250 days)",
    "ewma": "EWMA (lambda 0.94, normal)",
    "garch": "GARCH(1,1) with Student-t errors",
    "factor": "8-factor model (normal)",
}
SOURCES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "Rates", "Gold", "idiosyncratic"]
SOURCE_NAMES = {
    "Mkt-RF": "market",
    "SMB": "size",
    "HML": "value",
    "RMW": "profitability",
    "CMA": "investment",
    "Mom": "momentum",
    "Rates": "rates",
    "Gold": "gold",
    "idiosyncratic": "idiosyncratic",
}
DATA_END = pd.Timestamp("2025-12-31")
_ALIASES = {"60/40": "60_40", "60-40": "60_40", "6040": "60_40"}


class EngineError(ValueError):
    """A request the engine cannot answer (bad name, no data for that date, future date)."""


_override: dict[str, pd.DataFrame] = {}


@lru_cache(maxsize=1)
def _load_forecasts() -> pd.DataFrame:
    fc = pd.read_parquet(REPORTS_DIR / "forecasts.parquet")
    return fc.set_index(["portfolio", "model", "date"]).sort_index()


@lru_cache(maxsize=1)
def _load_attribution() -> pd.DataFrame:
    at = pd.read_parquet(REPORTS_DIR / "attribution.parquet")
    return at.set_index(["portfolio", "date"]).sort_index()


def _forecasts() -> pd.DataFrame:
    return _override["fc"] if "fc" in _override else _load_forecasts()


def _attribution() -> pd.DataFrame:
    return _override["at"] if "at" in _override else _load_attribution()


def set_data(forecasts: pd.DataFrame, attribution: pd.DataFrame) -> None:
    """Use these engine outputs instead of the files in reports/ (tests)."""
    _override["fc"] = forecasts.set_index(["portfolio", "model", "date"]).sort_index()
    _override["at"] = attribution.set_index(["portfolio", "date"]).sort_index()


def reset_data() -> None:
    _override.clear()


def _portfolio(p: str) -> str:
    key = _ALIASES.get(p.strip().lower(), p.strip().lower().replace("-", "_").replace(" ", "_"))
    if key not in PORTFOLIOS:
        raise EngineError(f"unknown portfolio '{p}'. Valid: {', '.join(PORTFOLIOS)}")
    return key


def _model(m: str) -> str:
    key = m.strip().lower()
    if key not in MODELS:
        raise EngineError(f"unknown model '{m}'. Valid: {', '.join(MODELS)}")
    return key


def _date(d: str) -> pd.Timestamp:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.strip()):
        raise EngineError(f"date '{d}' must be YYYY-MM-DD")
    ts = pd.Timestamp(d.strip())
    if ts > DATA_END:
        raise EngineError(
            f"no data for {d}: the engine's data ends on {DATA_END.date()}; "
            "it does not forecast beyond its data"
        )
    return ts


def _pct(x: float, nd: int = 3) -> float:
    return round(float(x) * 100.0, nd)


def var_exact(portfolio: str, date: str, model: str) -> dict[str, float]:
    """Unrounded forecast row (decimal log-return units) for one portfolio, model, date."""
    p, m, d = _portfolio(portfolio), _model(model), _date(date)
    try:
        row = _forecasts().loc[(p, m, d)]
    except KeyError:
        raise EngineError(
            f"no {m} forecast for {p} on {d.date()} (not a trading day, or before the "
            "model's first forecast)"
        ) from None
    return {k: float(row[k]) for k in ("vol", "var95", "var99", "es975", "actual")}


def get_var(portfolio: str, date: str, model: str) -> dict:
    """1-day risk forecast for ``date`` made with data up to the previous trading day."""
    r = var_exact(portfolio, date, model)
    return {
        "portfolio": _portfolio(portfolio),
        "date": date,
        "model": _model(model),
        "predicted_vol_pct": _pct(r["vol"]),
        "var95_pct": _pct(r["var95"]),
        "var99_pct": _pct(r["var99"]),
        "es975_pct": _pct(r["es975"]),
        "realized_return_pct": _pct(r["actual"]),
        "units": "percent of portfolio value; VaR and ES are losses (positive = loss)",
    }


def _range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    s, e = _date(start_date), _date(end_date)
    if s > e:
        raise EngineError("start_date must be on or before end_date")
    return s, e


def attribution_exact(portfolio: str, start_date: str, end_date: str) -> pd.Series:
    p = _portfolio(portfolio)
    s, e = _range(start_date, end_date)
    a = _attribution().loc[p]
    a = a.loc[(a.index >= s) & (a.index <= e), SOURCES]
    if a.empty:
        raise EngineError(f"no attribution data for {p} between {s.date()} and {e.date()}")
    return a.mean()


def get_attribution(portfolio: str, start_date: str, end_date: str) -> dict:
    """Average share of the portfolio's risk from each factor (factor model), in percent."""
    shares = attribution_exact(portfolio, start_date, end_date).sort_values(ascending=False)
    return {
        "portfolio": _portfolio(portfolio),
        "start_date": start_date,
        "end_date": end_date,
        "risk_share_pct": {SOURCE_NAMES[k]: round(float(v) * 100, 2) for k, v in shares.items()},
        "note": "Euler risk shares from the 8-factor model; shares sum to 100 and can be "
        "negative when a factor hedges the others.",
    }


def backtest_exact(
    portfolio: str, model: str, start_date: str, end_date: str, level: int
) -> tuple[int, int]:
    p, m = _portfolio(portfolio), _model(model)
    s, e = _range(start_date, end_date)
    if int(level) not in (95, 99):
        raise EngineError("level must be 95 or 99")
    g = _forecasts().loc[(p, m)]
    g = g.loc[(g.index >= s) & (g.index <= e)]
    if g.empty:
        raise EngineError(f"no {m} forecasts for {p} between {s.date()} and {e.date()}")
    breaches = int((-g["actual"] > g[f"var{int(level)}"]).sum())
    return len(g), breaches


def get_backtest_stats(portfolio: str, model: str, start_date: str, end_date: str, level: int):
    """How often the portfolio's loss exceeded the model's VaR over a date range."""
    n, b = backtest_exact(portfolio, model, start_date, end_date, level)
    return {
        "portfolio": _portfolio(portfolio),
        "model": _model(model),
        "start_date": start_date,
        "end_date": end_date,
        "level": int(level),
        "trading_days": n,
        "breaches": b,
        "breach_rate_pct": round(100 * b / n, 3),
        "expected_rate_pct": 100 - int(level),
    }


def get_returns(portfolio: str, start_date: str, end_date: str) -> dict:
    """Realized daily returns of the portfolio over a date range (percent)."""
    p = _portfolio(portfolio)
    s, e = _range(start_date, end_date)
    g = _forecasts().loc[(p, "historical")]["actual"]
    g = g.loc[(g.index >= s) & (g.index <= e)]
    if g.empty:
        raise EngineError(f"no returns for {p} between {s.date()} and {e.date()}")
    simple = np.expm1(g)
    out = {
        "portfolio": p,
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": len(g),
        "total_return_pct": _pct(np.expm1(g.sum())),
        "worst_day": str(simple.idxmin().date()),
        "worst_day_return_pct": _pct(simple.min()),
        "best_day": str(simple.idxmax().date()),
        "best_day_return_pct": _pct(simple.max()),
        "annualized_vol_pct": _pct(g.std(ddof=1) * np.sqrt(252)) if len(g) > 1 else None,
    }
    if len(g) <= 31:
        out["daily_return_pct"] = {str(d.date()): _pct(v) for d, v in simple.items()}
    return out
