# riskcheck

![CI](https://github.com/OWNER/riskcheck/actions/workflows/ci.yml/badge.svg)

An open-source portfolio risk engine that forecasts Value at Risk and Expected Shortfall,
tests those forecasts out of sample, detects regime breaks, and measures how often AI risk
copilots misstate portfolio risk with and without tool access.

Status: scaffolding. Headline findings will be added here after the final runs.

## Reproduce

```
make install
make test
```

Data is downloaded by the user from free public sources. Raw price data is never
republished in this repo.

This project is inspired by risk platforms of the kind used in industry. It is not
affiliated with any firm.
