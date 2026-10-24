"""Rolling out-of-sample backtest of every model's VaR and ES forecasts.

The forecasts in reports/forecasts.parquet are already walk-forward (each row uses only
data before its date). This module compares them to realized returns from 2008 on.

Outputs (reports/):
- backtest_results.csv: per portfolio and model, breach rates, Kupiec and
  Christoffersen p-values, Basel zone days, ES ratio with bootstrap CI, Z2
- conformal_results.csv: coverage before and after the conformal correction
- regime_breakdown.csv: breach rates by regime with Wilson 95% intervals
- breach_timeline.png, basel_zones.png
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskcheck import plotting
from riskcheck.conformal import CALIB, conformal_var
from riskcheck.portfolios import REPORTS_DIR
from riskcheck.tests_var import basel_zones, christoffersen, es_backtest, kupiec, wilson_ci

BT_START = pd.Timestamp("2008-01-01")
BT_END = pd.Timestamp("2025-12-31")
MODELS = ["historical", "ewma", "garch", "factor"]
LEVELS = {"var95": 0.95, "var99": 0.99}
REGIMES = {
    "GFC 2008-2009": ("2008-01-01", "2009-12-31"),
    "COVID Mar-Apr 2020": ("2020-03-01", "2020-04-30"),
    "2022 rates shock": ("2022-01-01", "2022-12-31"),
}
CHART_PORTFOLIO = "60_40"


def load_forecasts() -> pd.DataFrame:
    fc = pd.read_parquet(REPORTS_DIR / "forecasts.parquet")
    fc["loss"] = -fc["actual"]
    return fc


def evaluate(df: pd.DataFrame) -> dict[str, float]:
    """All backtest statistics for one portfolio-model forecast series."""
    out: dict[str, float] = {"n_days": len(df)}
    for col, lvl in LEVELS.items():
        b = (df["loss"] > df[col]).to_numpy()
        tag = col[3:]
        out[f"breach_rate_{tag}"] = b.mean()
        out[f"expected_{tag}"] = 1 - lvl
        out[f"kupiec_p_{tag}"] = kupiec(b, 1 - lvl).pvalue
        out[f"christoffersen_p_{tag}"] = christoffersen(b).pvalue
    zones = basel_zones(pd.Series(df["loss"] > df["var99"], index=df.index))
    counts = zones["zone"].value_counts()
    for z in ("green", "yellow", "red"):
        out[f"basel_{z}_days"] = int(counts.get(z, 0))
    es = es_backtest(df["loss"], df["var975"], df["es975"])
    out.update(
        es_tail_days=es.n_tail,
        es_ratio=es.ratio,
        es_ratio_ci_low=es.ci_low,
        es_ratio_ci_high=es.ci_high,
        es_z2=es.z2,
    )
    return out


def main_table(fc: pd.DataFrame) -> pd.DataFrame:
    w = fc[(fc["date"] >= BT_START) & (fc["date"] <= BT_END)]
    rows = []
    for (p, m), g in w.groupby(["portfolio", "model"], sort=False):
        rows.append({"portfolio": p, "model": m, **evaluate(g.set_index("date"))})
    return pd.DataFrame(rows)


def add_conformal(fc: pd.DataFrame) -> pd.DataFrame:
    """Add var95_conformal and var99_conformal, each calibrated on the prior 500 days."""
    parts = []
    for _, g in fc.groupby(["portfolio", "model"], sort=False):
        g = g.sort_values("date").set_index("date").copy()
        for col, lvl in LEVELS.items():
            g[f"{col}_conformal"] = conformal_var(g["loss"], g[col], lvl).to_numpy()
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def conformal_table(fcc: pd.DataFrame) -> pd.DataFrame:
    """Coverage before/after on the common window where every corrected forecast exists."""
    has = fcc.dropna(subset=["var95_conformal", "var99_conformal"])
    start = max(has.groupby("model")["date"].min().max(), BT_START)
    w = has[(has["date"] >= start) & (has["date"] <= BT_END)]
    rows = []
    for (p, m), g in w.groupby(["portfolio", "model"], sort=False):
        row = {"portfolio": p, "model": m, "start": start.date(), "n_days": len(g)}
        for col, lvl in LEVELS.items():
            tag = col[3:]
            for kind, c in (("raw", col), ("conformal", f"{col}_conformal")):
                b = (g["loss"] > g[c]).to_numpy()
                row[f"coverage_{tag}_{kind}"] = 1 - b.mean()
                row[f"kupiec_p_{tag}_{kind}"] = kupiec(b, 1 - lvl).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def regime_mask(dates: pd.Series) -> dict[str, pd.Series]:
    masks = {k: dates.between(a, b) for k, (a, b) in REGIMES.items()}
    stress = np.logical_or.reduce(list(masks.values()))
    masks["Calm (all other days)"] = (~stress) & dates.between(BT_START, BT_END)
    return masks


def regime_table(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (p, m), g in fc.groupby(["portfolio", "model"], sort=False):
        for regime, mask in regime_mask(g["date"]).items():
            s = g[mask.to_numpy()]
            row = {"portfolio": p, "model": m, "regime": regime, "n_days": len(s)}
            for col in LEVELS:
                tag = col[3:]
                x = int((s["loss"] > s[col]).sum())
                lo, hi = wilson_ci(x, len(s))
                row.update(
                    {
                        f"breaches_{tag}": x,
                        f"breach_rate_{tag}": x / max(len(s), 1),
                        f"ci_low_{tag}": lo,
                        f"ci_high_{tag}": hi,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_breach_timeline(fc: pd.DataFrame, portfolio: str, path) -> None:
    plotting.style()
    w = fc[(fc["portfolio"] == portfolio) & (fc["date"] >= BT_START)]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS, strict=True):
        g = w[w["model"] == m].set_index("date")
        br = g["loss"] > g["var99"]
        ax.plot(g.index, g["loss"] * 100, color="#b9b8b2", linewidth=0.5, label="daily loss")
        ax.plot(
            g.index,
            g["var99"] * 100,
            color=plotting.MODEL_COLORS[m],
            linewidth=1.2,
            label="99% VaR",
        )
        ax.scatter(
            g.index[br], g["loss"][br] * 100, s=10, color=plotting.BREACH, zorder=3, label="breach"
        )
        ax.set_title(
            f"{m}: {int(br.sum())} breaches of 99% VaR ({br.mean():.2%} of days)", loc="left"
        )
        ax.set_ylabel("loss, %")
    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle(
        f"{portfolio}: daily loss vs 99% VaR, 2008 to 2025 (target 1% of days)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=plotting.TEXT,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_basel(fc: pd.DataFrame, portfolio: str, path) -> None:
    plotting.style()
    w = fc[(fc["portfolio"] == portfolio) & (fc["date"] >= BT_START)]
    fig, ax = plt.subplots(figsize=(11, 3.3))
    for i, m in enumerate(MODELS):
        g = w[w["model"] == m].set_index("date")
        z = basel_zones(g["loss"] > g["var99"])
        colors = z["zone"].astype(str).map(plotting.ZONE_COLORS)
        ax.scatter(
            z.index, np.full(len(z), i), c=colors.to_numpy(), marker="|", s=120, linewidths=1
        )
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.grid(False)
    for z, c in plotting.ZONE_COLORS.items():
        ax.scatter([], [], color=c, marker="s", label=z)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(
        f"{portfolio}: Basel traffic light, rolling 250-day 99% VaR breaches "
        "(green 0-4, yellow 5-9, red 10+)",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_no_lookahead(
    fn: Callable[[pd.Series], pd.DataFrame], series: pd.Series, dates: list[pd.Timestamp]
) -> bool:
    """True if forecasts up to each date are unchanged when data on/after it is altered."""
    base = fn(series)
    for d in dates:
        s2 = series.copy()
        s2[s2.index >= d] = s2[s2.index >= d] * 7.0 + 0.05
        after = fn(s2)
        keep = base.index <= d
        if not np.allclose(base.loc[keep].to_numpy(), after.loc[keep].to_numpy(), equal_nan=True):
            return False
    return True


def main() -> None:
    fc = load_forecasts()
    main_t = main_table(fc)
    fcc = add_conformal(fc)
    conf_t = conformal_table(fcc)
    reg_t = regime_table(fc[(fc["date"] >= BT_START) & (fc["date"] <= BT_END)])
    main_t.to_csv(REPORTS_DIR / "backtest_results.csv", index=False)
    conf_t.to_csv(REPORTS_DIR / "conformal_results.csv", index=False)
    reg_t.to_csv(REPORTS_DIR / "regime_breakdown.csv", index=False)
    plot_breach_timeline(fc, CHART_PORTFOLIO, REPORTS_DIR / "breach_timeline.png")
    plot_basel(fc, CHART_PORTFOLIO, REPORTS_DIR / "basel_zones.png")

    cols = [
        "portfolio",
        "model",
        "breach_rate_99",
        "kupiec_p_99",
        "christoffersen_p_99",
        "basel_red_days",
        "es_ratio",
        "es_z2",
    ]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(main_t[cols].to_string(index=False))
        print(f"\nConformal (calibration {CALIB} days):")
        print(
            conf_t[["portfolio", "model", "coverage_99_raw", "coverage_99_conformal"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
