"""Build the five study portfolios and their daily returns.

Timing convention (no lookahead):
- Rebalance dates are the last trading day of each month.
- Target weights for rebalance date t are computed from returns dated <= t only,
  set at the close of t, and earn returns from t+1 onward.
- Between rebalances weights drift with asset returns (buy and hold).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from riskcheck.data import ROOT, SECTORS_LONG  # noqa: E402

REPORTS_DIR = ROOT / "reports"
RP_LOOKBACK = 60
RP_UNIVERSE = [*SECTORS_LONG, "TLT", "GLD"]
TRADING_DAYS = 252

WeightFn = Callable[[pd.DataFrame], pd.Series]


def _fixed(weights: dict[str, float]) -> WeightFn:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    return lambda hist: w


def _tech_heavy() -> dict[str, float]:
    others = [s for s in SECTORS_LONG if s != "XLK"]
    return {"XLK": 0.5, **{s: 0.5 / len(others) for s in others}}


def risk_parity(hist: pd.DataFrame) -> pd.Series:
    """Weights inverse to trailing 60-day volatility, using rows of ``hist`` only."""
    window = hist[RP_UNIVERSE].iloc[-RP_LOOKBACK:]
    if len(window) < RP_LOOKBACK:
        raise ValueError("not enough history for risk parity")
    inv = 1.0 / window.std(ddof=1)
    return inv / inv.sum()


PORTFOLIOS: dict[str, WeightFn] = {
    "60_40": _fixed({"SPY": 0.6, "TLT": 0.4}),
    "equal_sector": _fixed({s: 1.0 for s in SECTORS_LONG}),
    "tech_heavy": _fixed(_tech_heavy()),
    "defensive": _fixed({"XLU": 1, "XLP": 1, "XLV": 1, "TLT": 1, "GLD": 1}),
    "risk_parity": risk_parity,
}


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def target_weights(log_returns: pd.DataFrame, fn: WeightFn) -> pd.DataFrame:
    """Target weights at each rebalance date, each computed from data up to that date."""
    rows = {}
    for d in rebalance_dates(log_returns.index):
        hist = log_returns.loc[:d]  # inclusive of d, nothing after
        try:
            rows[d] = fn(hist)
        except ValueError:
            continue
    return pd.DataFrame(rows).T.fillna(0.0)


def simulate(log_returns: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily portfolio returns from drifting weights reset at each rebalance.

    Returns a frame with simple return, log return, and the weights held during each day.
    """
    assets = list(targets.columns)
    simple = np.exp(log_returns[assets]) - 1.0
    first = targets.index[0]
    days = simple.index[simple.index > first]
    w = targets.loc[first].values.copy()
    out_r, out_w = [], []
    tgt = {d: targets.loc[d].values for d in targets.index}
    for d in days:
        r = simple.loc[d].values
        port = float(w @ r)
        out_r.append(port)
        out_w.append(w.copy())
        w = w * (1.0 + r) / (1.0 + port)
        if d in tgt:
            w = tgt[d].copy()
    res = pd.DataFrame({"ret": out_r}, index=days)
    res["log_ret"] = np.log1p(res["ret"])
    return res, pd.DataFrame(out_w, index=days, columns=assets)


def build_all(log_returns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Log returns for every portfolio on a common start date, plus weight histories."""
    tw = {name: target_weights(log_returns, fn) for name, fn in PORTFOLIOS.items()}
    common_start = max(t.index[0] for t in tw.values())
    rets, weights = {}, {}
    for name, t in tw.items():
        t = t.loc[common_start:]
        res, w = simulate(log_returns, t)
        rets[name] = res["log_ret"]
        weights[name] = w
    return pd.DataFrame(rets), weights


def max_drawdown(log_ret: pd.Series) -> float:
    wealth = np.exp(log_ret.cumsum())
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(port_log: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, lr in port_log.items():
        years = len(lr) / TRADING_DAYS
        simple = np.expm1(lr)
        rows[name] = {
            "ann_return": float(np.exp(lr.sum() / years) - 1.0),
            "ann_vol": float(lr.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "max_drawdown": max_drawdown(lr),
            "worst_day": float(simple.min()),
            "worst_day_date": simple.idxmin().strftime("%Y-%m-%d"),
        }
    return pd.DataFrame(rows).T


def plot_cumulative(port_log: pd.DataFrame, path: Path) -> None:
    from riskcheck import plotting

    plotting.style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, lr), color in zip(port_log.items(), plotting.SERIES, strict=False):
        ax.plot(lr.index, np.exp(lr.cumsum()), label=name, linewidth=1.2, color=color)
    for start, end in [
        ("2008-09-01", "2009-03-31"),
        ("2020-02-20", "2020-04-30"),
        ("2022-01-01", "2022-10-31"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Cumulative returns by portfolio (shaded: 2008, 2020, 2022 stress)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    from riskcheck.data import load

    cd = load()
    port_log, _ = build_all(cd.returns)
    REPORTS_DIR.mkdir(exist_ok=True)
    port_log.to_parquet(REPORTS_DIR / "portfolio_returns.parquet")
    tbl = summary(port_log)
    tbl.to_csv(REPORTS_DIR / "portfolio_summary.csv")
    plot_cumulative(port_log, REPORTS_DIR / "cumulative_returns.png")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(tbl)


if __name__ == "__main__":
    main()
