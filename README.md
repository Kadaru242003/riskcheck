# riskcheck

![CI](https://github.com/Kadaru242003/riskcheck/actions/workflows/ci.yml/badge.svg)

An open-source portfolio risk engine that forecasts Value at Risk (VaR) and Expected
Shortfall (ES) for five ETF portfolios, tests every forecast out of sample from 2008 to
2025, checks whether drift detectors can warn that a risk model is breaking, and measures
how often AI risk copilots misstate portfolio risk with and without tool access to the
engine.

## Headline findings

Backtest: 4 risk models x 5 portfolios, 4,529 trading days (2008 to 2025), each forecast
made only with data from before its date. Full tables: [reports/RESULTS.md](reports/RESULTS.md).

1. **Every model underestimates tail risk at 99%.** Breach rates averaged 1.50% to 2.13%
   against a 1% target, and the Kupiec test rejected correct coverage for all 20
   portfolio-model pairs.
2. **GARCH with fat tails was the most honest model.** It averaged 27 days in the Basel
   red zone (the other models: 206 to 237), and its Expected Shortfall was close to right
   (realized/forecast loss ratio 1.02, against 1.09 to 1.16 for the others).
3. **Historical simulation broke in the fastest crisis.** In March and April 2020 it
   breached its 99% VaR on 11.2% of days, against 3.3% for GARCH. Its breaches also
   clustered (Christoffersen p < 0.001 for every portfolio): it reacts too slowly.
4. **A conformal correction fixed coverage.** Calibrated on each model's previous 500
   days, it moved average 99% coverage from 97.9% to 98.7% raw to 99.0% to 99.1%, and
   Kupiec rejections from 18 of 20 to 0 of 20 (March 2009 to 2025).
5. **No drift detector beat a plain VIX threshold** at flagging regime starts. The best
   residual detector (CUSUM on historical simulation) caught 82% of 9 pre-registered
   regime starts, on the day of the shock rather than before it, with about 1 false alarm
   per calm year.
6. **AI copilot study: in progress.** The design, hypotheses and answer key are
   pre-registered in [prereg.md](prereg.md); results will be added here when the run
   finishes.

![Daily loss vs 99% VaR for the 60/40 portfolio, by model](reports/breach_timeline.png)

## What is in the repo

| Part | What it does | Code |
|---|---|---|
| Data | ETF prices, Fama-French factors, VIX, T-bill and 10-year yields; aligned on trading days, every dropped or filled day logged | `riskcheck/data.py` |
| Portfolios | 60/40, equal-weight sectors, tech-heavy, defensive, risk parity; monthly rebalancing with no lookahead | `riskcheck/portfolios.py` |
| Risk models | Historical simulation, EWMA, GARCH(1,1)-t, 8-factor model with risk attribution | `riskcheck/models/` |
| Backtests | Kupiec, Christoffersen, Basel traffic light, ES tests, conformal VaR, regime breakdown | `riskcheck/backtest.py`, `tests_var.py`, `conformal.py` |
| Drift | CUSUM, Page-Hinkley, breach count and VIX detectors; pre-registered scoring; simulation | `riskcheck/drift.py` |
| MCP server | The engine as 4 tools for AI agents | `riskcheck/mcp_server.py` |
| Copilot study | 200 questions, LLM runner with and without tools, graders, judge, statistics | `riskcheck/copilot/` |
| Dashboard | Explore any portfolio's forecasts, breaches and attribution | `app/streamlit_app.py` |

## Reproduce

Python 3.11 or newer. Data comes from free public sources and is downloaded by you; raw
price data is never republished in this repo.

```
python3 -m venv .venv && source .venv/bin/activate
make install      # or, for the exact versions used: pip install -r requirements.lock
make all          # data, portfolios, forecasts, backtests, drift study, reports/RESULTS.md
make test
```

`make all` rebuilds every number and chart in `reports/` from scratch. The AI copilot
study calls an LLM API, so it runs separately (it needs a Groq key in `.env`):

```
make copilot          # resumable; stops at free-tier daily limits, rerun to continue
make copilot-judge    # cross-family LLM judge for the unanswerable questions
make copilot-label    # hand-label sheet to validate the judge
make copilot-report   # tables, McNemar tests, chart
```

## Dashboard

```
streamlit run app/streamlit_app.py
```

Pick a portfolio, model, VaR level and years: daily losses against VaR with breaches
marked, a side-by-side model comparison, and yearly factor risk attribution.

## MCP server: the risk engine as tools for AI agents

After `make forecasts`, the engine is available as an MCP server over stdio with four
tools: `get_var`, `get_attribution`, `get_backtest_stats` and `get_returns`.

```
make mcp        # or: python -m riskcheck.mcp_server
```

To connect an MCP client (for example a desktop AI app), add a server entry like this,
with your own paths:

```json
{
  "mcpServers": {
    "riskcheck": {
      "command": "/path/to/riskcheck/.venv/bin/python",
      "args": ["-m", "riskcheck.mcp_server"],
      "cwd": "/path/to/riskcheck"
    }
  }
}
```

Unknown portfolio or model names, non-trading days and dates after 2025-12-31 come back
as tool errors, so an agent can tell when a question cannot be answered.

## How the results are kept honest

- **No lookahead.** Every model, the portfolio weights and the conformal correction have
  tests that change all data on or after a date and check that nothing up to that date
  moves. A deliberately cheating model is caught by the same check, and shuffling dates
  removes all forecasting signal.
- **Pre-registration.** The drift study and the AI copilot study were written down in
  [prereg.md](prereg.md) and committed before they were run. Every change made after
  that is listed there with its date and reason, and analyses added after seeing results
  are labelled exploratory.
- **Fixed answer key.** The copilot study grades against a file whose SHA-256 hash is
  recorded in prereg.md; the code refuses to grade if it changes.
- **Generated reports.** Every table and figure in `reports/` is written by code.

## Design change: 8-factor risk model

The original spec used the 6 Fama-French equity factors (market, size, value,
profitability, investment, momentum). Those left about 35% of the 60/40 portfolio's risk
and 46% of the defensive portfolio's risk unexplained, because bond and gold risk has no
equity factor to load on. Two factors were added before the AI copilot study was
pre-registered:

- Rates: the negative daily change in the 10-year Treasury yield (FRED DGS10), so a fall
  in yields is a positive factor return. It is built from yields, not from TLT returns,
  so the bond holding does not explain itself. FRED holiday gaps are forward-filled
  (never backward) and every filled day is logged in data/processed/filled_days.csv.
- Gold: GLD daily returns, used as a proxy.

With 8 factors the unexplained share falls to about 9% for 60/40 and 18% for defensive
(2006 to 2025 average).

## Limitations

- The gold factor is GLD's own return, and GLD is also a holding in the defensive and
  risk-parity portfolios, so GLD's exposure to gold is 1 by construction and gold risk
  is fully "explained" for that holding. A cleaner choice would be a gold price series
  that is not a portfolio holding.
- The rates factor uses the 10-year yield, while TLT holds 20+ year bonds, so part of
  TLT's risk stays in the leftover (idiosyncratic) share.
- About 15% of GARCH refits sit at the boundary where volatility shocks never fade
  (alpha + beta = 1), mostly in windows that include 2008. One-day forecasts stay valid.
- The regime rule marks 9 starts in 2008 to 2025 and does not mark 2008 or 2022, because
  VIX was already high before those crises. Nine events give wide uncertainty.
- The drift simulation covers historical simulation and EWMA only.
- Five ETF portfolios, daily data, one-day horizon: no liquidity, single stocks,
  options or multi-day risk.

This project is inspired by risk platforms of the kind used in industry. It is not
affiliated with any firm.
