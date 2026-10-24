"""RiskCheck dashboard: explore any portfolio's risk forecasts, breaches and attribution.

Run:  streamlit run app/streamlit_app.py   (after `make forecasts`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskcheck import plotting  # noqa: E402
from riskcheck.copilot.engine import MODELS, PORTFOLIOS, SOURCE_NAMES  # noqa: E402
from riskcheck.portfolios import REPORTS_DIR  # noqa: E402
from riskcheck.tests_var import christoffersen, kupiec  # noqa: E402

st.set_page_config(page_title="RiskCheck", layout="wide")

FC_PATH = REPORTS_DIR / "forecasts.parquet"
AT_PATH = REPORTS_DIR / "attribution.parquet"
SOURCE_COLORS = dict(
    zip(SOURCE_NAMES.values(), [*plotting.SERIES, "#8a8984", "#b9b8b2"], strict=False)
)


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = pd.read_parquet(FC_PATH)
    fc["loss"] = -fc["actual"]
    return fc, pd.read_parquet(AT_PATH)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stats_for(g: pd.DataFrame, level: int) -> dict:
    b = (g["loss"] > g[f"var{level}"]).to_numpy()
    return {
        "days": len(g),
        "breaches": int(b.sum()),
        "rate": b.mean() if len(b) else np.nan,
        "kupiec_p": kupiec(b, 1 - level / 100).pvalue if len(b) else np.nan,
        "christoffersen_p": christoffersen(b).pvalue if len(b) > 1 else np.nan,
    }


if not (FC_PATH.exists() and AT_PATH.exists()):
    st.error("No engine outputs found. Run `make data` and `make forecasts` first.")
    st.stop()

fc, at = load()

st.title("RiskCheck")
st.caption(
    "Out-of-sample 1-day VaR and Expected Shortfall forecasts for five ETF portfolios, "
    "with breaches and factor risk attribution. Each forecast uses only data from "
    "before its date."
)

with st.sidebar:
    st.header("Settings")
    portfolio = st.selectbox("Portfolio", list(PORTFOLIOS), format_func=lambda p: PORTFOLIOS[p])
    model = st.selectbox("Risk model", list(MODELS), index=2, format_func=lambda m: MODELS[m])
    level = st.radio("VaR level", [99, 95], horizontal=True, format_func=lambda x: f"{x}%")
    years = st.slider("Years", 2008, 2025, (2008, 2025))

start, end = pd.Timestamp(f"{years[0]}-01-01"), pd.Timestamp(f"{years[1]}-12-31")
sel = fc[(fc["portfolio"] == portfolio) & fc["date"].between(start, end)]
g = sel[sel["model"] == model].set_index("date")
s = stats_for(g, level)
target = 1 - level / 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trading days", f"{s['days']:,}")
c2.metric("Breaches", f"{s['breaches']:,}", help=f"Days the loss exceeded the {level}% VaR")
c3.metric(
    "Breach rate",
    f"{s['rate']:.2%}",
    f"{(s['rate'] - target) * 100:+.2f} pts vs {target:.0%} target",
    delta_color="inverse",
)
c4.metric(
    "Kupiec p",
    fmt_p(s["kupiec_p"]),
    help="Is the breach rate equal to the target? Below 0.05: rejected.",
)
c5.metric(
    "Christoffersen p",
    fmt_p(s["christoffersen_p"]),
    help="Are breaches independent day to day? Below 0.05: they cluster.",
)

tab1, tab2, tab3 = st.tabs(["Forecasts and breaches", "Compare models", "Risk attribution"])

with tab1:
    br = g["loss"] > g[f"var{level}"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g.index,
            y=g["loss"] * 100,
            name="Daily loss",
            line=dict(color="#b9b8b2", width=0.7),
            hovertemplate="%{x|%Y-%m-%d}<br>loss %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g.index,
            y=g[f"var{level}"] * 100,
            name=f"{level}% VaR",
            line=dict(color=plotting.MODEL_COLORS[model], width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>VaR %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g.index[br],
            y=g["loss"][br] * 100,
            name="Breach",
            mode="markers",
            marker=dict(color=plotting.BREACH, size=7),
            hovertemplate="%{x|%Y-%m-%d}<br>breach: loss %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        yaxis_title="loss, % of portfolio (positive = loss)",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{PORTFOLIOS[portfolio]} · {MODELS[model]}. Expected breaches at {level}%: "
        f"about {target * s['days']:.0f} of {s['days']:,} days."
    )

with tab2:
    rows = []
    for m in MODELS:
        gm = sel[sel["model"] == m]
        r95, r99 = stats_for(gm, 95), stats_for(gm, 99)
        es_days = gm["loss"] > gm["var975"]
        es_ratio = (
            gm.loc[es_days, "loss"].mean() / gm.loc[es_days, "es975"].mean()
            if es_days.sum() > 1
            else np.nan
        )
        rows.append(
            {
                "Model": MODELS[m],
                "95% breach rate": f"{r95['rate']:.2%}",
                "95% Kupiec p": fmt_p(r95["kupiec_p"]),
                "99% breach rate": f"{r99['rate']:.2%}",
                "99% Kupiec p": fmt_p(r99["kupiec_p"]),
                "99% Christoffersen p": fmt_p(r99["christoffersen_p"]),
                "ES ratio (realized / forecast)": f"{es_ratio:.2f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Targets: 5% and 1% breach rates; ES ratio near 1.00. An ES ratio above 1 "
        "means losses beyond the 97.5% VaR were worse than the model expected."
    )

with tab3:
    a = at[(at["portfolio"] == portfolio) & at["date"].between(start, end)].copy()
    a["year"] = a["date"].dt.year
    shares = a.groupby("year")[list(SOURCE_NAMES)].mean().rename(columns=SOURCE_NAMES) * 100
    fig2 = go.Figure()
    for col in shares.columns:
        fig2.add_trace(
            go.Bar(
                x=shares.index,
                y=shares[col],
                name=col,
                marker_color=SOURCE_COLORS[col],
                hovertemplate=f"{col}: %{{y:.1f}}%<extra>%{{x}}</extra>",
            )
        )
    fig2.update_layout(
        barmode="relative",
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="share of portfolio risk, %",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Average yearly share of risk from each source in the 8-factor model (Euler "
        "attribution). Shares sum to 100%; a negative share means that source "
        "hedged the others that year."
    )
    st.dataframe(shares.round(1), use_container_width=True)

st.divider()
st.caption(
    "Data: free public sources, downloaded by the user. Inspired by industry risk "
    "platforms; not affiliated with any firm."
)
