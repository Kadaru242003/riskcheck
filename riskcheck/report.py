"""Build reports/RESULTS.md from the generated tables. Nothing in it is typed by hand.

Run: python -m riskcheck.report   (after make backtest, make drift; copilot section if present)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from riskcheck.portfolios import REPORTS_DIR

OUT = REPORTS_DIR / "RESULTS.md"
MODEL_ORDER = ["historical", "ewma", "garch", "factor"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _read(name: str) -> pd.DataFrame | None:
    p = REPORTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def section_portfolios() -> str:
    s = _read("portfolio_summary.csv").rename(columns={"Unnamed: 0": "portfolio"})
    t = pd.DataFrame(
        {
            "Portfolio": s["portfolio"],
            "Annual return": s["ann_return"].map(pct),
            "Annual volatility": s["ann_vol"].map(pct),
            "Max drawdown": s["max_drawdown"].map(pct),
            "Worst day": s["worst_day"].map(pct) + " (" + s["worst_day_date"] + ")",
        }
    )
    return (
        "## 1. Portfolios (2005 to 2025)\n\n"
        + md_table(t)
        + "\n\n![Cumulative returns](cumulative_returns.png)\n"
    )


def section_backtest() -> str:
    b = _read("backtest_results.csv")
    avg = (
        b.groupby("model")
        .agg(
            breach95=("breach_rate_95", "mean"),
            breach99=("breach_rate_99", "mean"),
            red=("basel_red_days", "mean"),
            es=("es_ratio", "mean"),
            rej99=("kupiec_p_99", lambda p: int((p < 0.05).sum())),
        )
        .reindex(MODEL_ORDER)
    )
    t1 = pd.DataFrame(
        {
            "Model": avg.index,
            "95% breach rate (target 5%)": avg["breach95"].map(pct),
            "99% breach rate (target 1%)": avg["breach99"].map(pct),
            "Kupiec rejects at 99% (of 5)": avg["rej99"].astype(int),
            "Basel red-zone days (avg)": avg["red"].map(lambda x: f"{x:.0f}"),
            "ES ratio (target 1.00)": avg["es"].map(lambda x: f"{x:.2f}"),
        }
    )
    t2 = pd.DataFrame(
        {
            "Portfolio": b["portfolio"],
            "Model": b["model"],
            "99% breach rate": b["breach_rate_99"].map(pct),
            "Kupiec p": b["kupiec_p_99"].map(pval),
            "Christoffersen p": b["christoffersen_p_99"].map(pval),
            "Basel red days": b["basel_red_days"],
            "ES ratio [95% CI]": b.apply(
                lambda r: f"{r.es_ratio:.2f} [{r.es_ratio_ci_low:.2f}, {r.es_ratio_ci_high:.2f}]",
                axis=1,
            ),
            "ES Z2": b["es_z2"].map(lambda x: f"{x:.2f}"),
        }
    )
    c = _read("conformal_results.csv")
    cm = (
        c.groupby("model")[
            ["coverage_99_raw", "coverage_99_conformal", "coverage_95_raw", "coverage_95_conformal"]
        ]
        .mean()
        .reindex(MODEL_ORDER)
    )
    t3 = pd.DataFrame(
        {
            "Model": cm.index,
            "99% coverage raw": cm["coverage_99_raw"].map(pct),
            "99% coverage conformal": cm["coverage_99_conformal"].map(pct),
            "95% coverage raw": cm["coverage_95_raw"].map(pct),
            "95% coverage conformal": cm["coverage_95_conformal"].map(pct),
        }
    )
    rej_raw = int((c["kupiec_p_99_raw"] < 0.05).sum())
    rej_conf = int((c["kupiec_p_99_conformal"] < 0.05).sum())
    r = _read("regime_breakdown.csv")
    rm = r.pivot_table(index="regime", columns="model", values="breach_rate_99", aggfunc="mean")[
        MODEL_ORDER
    ]
    t4 = rm.map(pct).reset_index().rename(columns={"regime": "Regime"})
    return (
        f"## 2. VaR backtest, {b['n_days'].iloc[0]:,} trading days from 2008 to 2025\n\n"
        "Averages over the 5 portfolios:\n\n"
        + md_table(t1)
        + "\n\nEvery portfolio and model:\n\n"
        + md_table(t2)
        + "\n\n![Breach timeline](breach_timeline.png)\n\n![Basel zones](basel_zones.png)\n\n"
        f"### Conformal VaR (from {c['start'].iloc[0]}, {c['n_days'].iloc[0]:,} days)\n\n"
        + md_table(t3)
        + f"\n\nKupiec rejections at 99%: {rej_raw} of {len(c)} before, {rej_conf} of {len(c)} "
        "after the correction.\n\n### 99% breach rate by regime (average over portfolios)\n\n"
        + md_table(t4)
        + "\n"
    )


def section_drift() -> str:
    d = _read("drift_results.csv")
    t = pd.DataFrame(
        {
            "Model": d["model"],
            "Detector": d["detector"],
            "Detected": d["detection_rate"].map(pct),
            "Median delay (days)": d["median_delay_days"].map(lambda x: f"{x:+.0f}"),
            "False alarms per calm year": d["false_alarms_per_year"].map(lambda x: f"{x:.2f}"),
            "Precision": d["precision"].map(lambda x: f"{x:.2f}"),
        }
    )
    e = _read("drift_exploratory.csv")
    te = pd.DataFrame(
        {
            "Model": e["model"],
            "Detector": e["detector"],
            "Alarms per year": e["alarms_per_year"].map(lambda x: f"{x:.2f}"),
            "Share near a regime start": e["share_in_windows"].map(pct),
        }
    )
    s = _read("drift_simulation.csv")
    ts = pd.DataFrame(
        {
            "Model": s["model"],
            "Detector": s["detector"],
            "Power": s["power"].map(pct),
            "Median delay (days)": s["median_delay_days"].map(lambda x: f"{x:.0f}"),
            "False alarms per year": s["false_alarms_per_year"].map(lambda x: f"{x:.2f}"),
        }
    )
    return (
        "## 3. Regime drift detection (pre-registered; see prereg.md)\n\n"
        + md_table(t)
        + "\n\nThe VIX baseline cannot register a false alarm under this scoring (see the "
        "amendment in prereg.md), so its false-alarm and precision columns are not "
        "comparable.\n\nExploratory (added after seeing results): share of all alarms that "
        "fall near a regime start.\n\n"
        + md_table(te)
        + "\n\nSimulation: volatility doubled at a random date, 1,000 runs.\n\n"
        + md_table(ts)
        + "\n\n![Drift alarms](drift_alarms.png)\n"
    )


def section_copilot() -> str:
    tests = _read("copilot/hypothesis_tests.csv")
    table = _read("copilot/results_table.csv")
    if tests is None or table is None:
        return (
            "## 4. AI risk copilot study\n\nIn progress. The design is pre-registered in "
            "prereg.md; results appear here after `make copilot-report`.\n"
        )
    prim = tests[tests["hypothesis"].str.startswith("PRIMARY")]
    tp = pd.DataFrame(
        {
            "Model": prim["model"],
            "Pairs": prim["n_pairs"],
            "Wrong-number rate, no tools": prim["rate_no_tools"].map(pct),
            "Wrong-number rate, tools": prim["rate_tools"].map(pct),
            "McNemar p": prim["p_value"].map(pval),
            "Bonferroni p": prim["p_bonferroni"].map(pval),
            "Significant": prim["significant"].map({True: "yes", False: "no"}),
        }
    )
    allq = (
        table.groupby(["model", "condition"])
        .apply(
            lambda g: pd.Series(
                {
                    "accuracy": (g["accuracy"] * g["n"]).sum() / g["n"].sum(),
                    "false_conf": (g["false_confidence_rate"] * g["n"]).sum() / g["n"].sum(),
                    "unsupported": (g["unsupported_claim_rate"] * g["n"]).sum() / g["n"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    ta = pd.DataFrame(
        {
            "Model": allq["model"],
            "Condition": allq["condition"],
            "Accuracy (all 200)": allq["accuracy"].map(pct),
            "False confidence": allq["false_conf"].map(pct),
            "Unsupported claims": allq["unsupported"].map(pct),
        }
    )
    by_type = (
        table.pivot_table(index=["model", "condition"], columns="type", values="accuracy")
        .map(pct)
        .reset_index()
    )
    expl = tests[~tests["hypothesis"].str.startswith("PRIMARY")]
    te = pd.DataFrame(
        {
            "Outcome": expl["hypothesis"],
            "Model": expl["model"],
            "No tools": expl["rate_no_tools"].map(pct),
            "Tools": expl["rate_tools"].map(pct),
            "Bonferroni p": expl["p_bonferroni"].map(pval),
        }
    )
    out = (
        "## 4. AI risk copilot study (pre-registered; see prereg.md)\n\n"
        "### Primary outcome: wrong-number rate, no tools vs tools\n\n"
        + md_table(tp)
        + "\n\n### Exploratory\n\n"
        + md_table(ta)
        + "\n\nAccuracy by question type:\n\n"
        + md_table(by_type)
        + "\n\n"
        + md_table(te)
    )
    v = _read("copilot/judge_validation.csv")
    if v is not None:
        tv = pd.DataFrame(
            {
                "Judge": v["judge_model"],
                "Hand-labelled": v["n_labelled"],
                "Agreement": v["agreement"].map(pct),
                "Cohen's kappa": v["cohen_kappa"].map(lambda x: f"{x:.2f}"),
                "Reliable (kappa >= 0.6)": v["reliable"].map({True: "yes", False: "no"}),
            }
        )
        out += "\n\n### Judge validation\n\n" + md_table(tv)
    base = _read("copilot/baselines.csv")
    if base is not None:
        out += "\n\nMajority-answer baselines: " + "; ".join(
            f"{r.subtype} {pct(r.majority_accuracy, 0)} ('{r.majority_answer}')"
            for r in base.itertuples()
        )
    if (REPORTS_DIR / "copilot" / "copilot_accuracy.png").exists():
        out += "\n\n![Copilot accuracy](copilot/copilot_accuracy.png)"
    return out + "\n"


def build() -> str:
    parts = [
        "# RiskCheck results\n",
        f"Generated by `make report` on {date.today()}. Every number below is read from "
        "files the pipeline writes; nothing is edited by hand.\n",
        section_portfolios(),
        section_backtest(),
        section_drift(),
        section_copilot(),
    ]
    text = "\n".join(parts)
    OUT.write_text(text)
    return text


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}")
