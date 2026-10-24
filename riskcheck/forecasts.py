"""Run every risk model on every portfolio and save the forecasts.

Outputs (reports/):
- forecasts.parquet: long table (date, portfolio, model, vol, var95, var99, es975, actual)
- attribution.parquet: factor-model risk shares and exposures per portfolio and date
- garch_params.parquet: GARCH parameters at each refit
- forecast_checkpoint.csv: all models for the 60/40 portfolio on a few known days
"""

from __future__ import annotations

import pandas as pd

from riskcheck.data import CORE_ASSETS, load
from riskcheck.models import ewma, factor, garch, historical
from riskcheck.portfolios import REPORTS_DIR, build_all

MODELS = ["historical", "ewma", "garch", "factor"]
CHECK_DAYS = ["2008-09-29", "2008-10-15", "2017-06-15", "2020-03-12", "2020-03-16", "2022-06-13"]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    cd = load()
    port, weights = build_all(cd.returns)
    fm = factor.fit(cd.returns[CORE_ASSETS], cd.factors)

    frames, attrs, gparams = [], [], []
    for name in port.columns:
        r = port[name].dropna()
        per_model = {
            "historical": historical.forecast(r),
            "ewma": ewma.forecast(r),
            "garch": garch.forecast(r),
        }
        fc_f, attr = fm.portfolio(weights[name])
        per_model["factor"] = fc_f
        gp = per_model["garch"].attrs["params"].assign(portfolio=name)
        gparams.append(gp)
        for model, fc in per_model.items():
            fc = fc.assign(portfolio=name, model=model, actual=r.reindex(fc.index))
            frames.append(fc)
        attrs.append(attr.assign(portfolio=name))
        print(f"done: {name}")

    fc_all = pd.concat(frames).rename_axis("date").reset_index()
    attr_all = pd.concat(attrs).rename_axis("date").reset_index()
    REPORTS_DIR.mkdir(exist_ok=True)
    fc_all.to_parquet(REPORTS_DIR / "forecasts.parquet", index=False)
    attr_all.to_parquet(REPORTS_DIR / "attribution.parquet", index=False)
    pd.concat(gparams).to_parquet(REPORTS_DIR / "garch_params.parquet")

    chk = fc_all[(fc_all["portfolio"] == "60_40") & fc_all["date"].isin(pd.to_datetime(CHECK_DAYS))]
    chk = chk.pivot_table(index="date", columns="model", values=["vol", "var99", "actual"])
    chk.to_csv(REPORTS_DIR / "forecast_checkpoint.csv")
    return fc_all, attr_all


if __name__ == "__main__":
    fc, _ = run()
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(pd.read_csv(REPORTS_DIR / "forecast_checkpoint.csv", header=[0, 1], index_col=0))
