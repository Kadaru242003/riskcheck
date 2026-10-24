# riskcheck

![CI](https://github.com/OWNER/riskcheck/actions/workflows/ci.yml/badge.svg)

An open-source portfolio risk engine that forecasts Value at Risk and Expected Shortfall,
tests those forecasts out of sample, detects regime breaks, and measures how often AI risk
copilots misstate portfolio risk with and without tool access.

Status: scaffolding. Headline findings will be added here after the final runs.

## Reproduce

```
python3.11 -m venv .venv && source .venv/bin/activate
make install
make data         # downloads to data/raw (cached; reruns are offline)
make portfolios   # reports/portfolio_summary.csv, reports/cumulative_returns.png
make test
```

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

Data is downloaded by the user from free public sources. Raw price data is never
republished in this repo.

This project is inspired by risk platforms of the kind used in industry. It is not
affiliated with any firm.
