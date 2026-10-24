# Pre-registration

Each section is committed before any run it covers. Changes after a commit are listed
under "Amendments" with the date and reason, and never made after seeing results.

## Part 4: regime drift detection

Committed: 2026-10-24, before any detector was scored on real data.

### Regime starts (defined from VIX only, never from model results)

- The spec's example rule (first VIX close above 30 after 60 straight days below 20)
  finds a single start in 2005 to 2025 (2018-02-05), because VIX usually rises above 20
  a few days before it crosses 30. One event cannot be scored.
- Rule used: the first day VIX closes above 30 (after closing at or below 30 the day
  before) while the mean VIX over the prior 60 trading days is below 20. Starts closer
  than 60 trading days to the previous start are merged into it.
- This rule was chosen by looking at VIX only, before any detector output existed.
- Starts in the 2008 to 2025 evaluation window (9): 2010-05-06, 2011-08-04, 2015-08-24,
  2018-02-05, 2018-12-21, 2020-02-27, 2021-12-01, 2024-08-05, 2025-04-03.
  (2007-08-15 also qualifies but is before the window.)
- Known gap: the rule does not mark 2008 or 2022, because VIX was already elevated
  before those crises. Those periods are neither detections nor false alarms (below).

### Signals

- Standardized forecast error z_t = r_t / vol_t from each model's forecast for day t.
- 95% VaR breaches from each model.
- VIX level.

### Detectors (all reset, then stay silent for 20 trading days after an alarm)

| Detector | Rule |
|---|---|
| CUSUM | S_t = max(0, S_{t-1} + z_t^2 - 1 - 0.5); alarm when S_t > h = 9.52 |
| Page-Hinkley | on x_t = z_t^2 vs its running mean since reset, delta = 0.1; alarm when m_t - min m > lambda = 15.26 |
| Breach count | 4 or more 95% VaR breaches in the trailing 20 days |
| VIX baseline | VIX closes above 25 after closing at or below 25 the day before |

h and lambda were set so the average time to a false alarm is 252 days on simulated
iid N(0,1) errors (400 runs, seed 42). No real data was used to set any threshold.

Caveat stated in advance: the VIX baseline uses the same series as the regime
definition, so it has a built-in advantage. A residual detector that beats it would be
strong evidence; one that loses to it is the expected outcome.

### Scoring

- Evaluation window: 2008-01-01 to 2025-12-31; all 5 portfolios x 4 models.
- Detection: the first alarm from 5 trading days before to 40 trading days after a
  start. Delay = alarm day minus start day (negative = early warning). No alarm = miss.
- Calm days: VIX below 20 and outside every detection window. An alarm on a calm day
  is a false alarm. Alarms on other days (elevated VIX, no defined start) are not scored.
- Metrics: detection rate, median delay (days), false alarms per calm year,
  precision = alarms in windows / (alarms in windows + false alarms).
- Pooled per model and detector across portfolios. All results are reported.

### Simulation check

- 1,000 runs, seed 42. Each run picks a portfolio and builds 750 days by resampling
  20-day blocks of its returns from days with VIX below 20.
- After a random jump day (between day 400 and day 650), returns are multiplied by 2.
- Forecasts: historical simulation and EWMA (GARCH and the factor model are too slow
  or need asset-level data; stated as a limitation). Detectors: CUSUM, Page-Hinkley,
  breach count (no VIX in simulated data).
- Metrics: power = detection within 40 days after the jump; median delay; false alarms
  per year before the jump.

## Part 5: AI copilot study (to be fixed before any full run)

- Hypotheses: TBD
- Metrics: wrong-number rate, unsupported-claim rate, false-confidence rate
- Numeric tolerance: TBD (spec suggests 5% relative)
- Tests: McNemar for paired comparisons, Bonferroni correction across models
- Models and versions: TBD, with dates logged

## Amendments

- 2026-10-24, Part 4, after seeing results: under the pre-registered scoring the VIX
  baseline cannot produce a false alarm (it fires only when VIX > 25, and false alarms
  are counted only on days with VIX < 20), so its 0 false alarms and precision of 1 are
  artifacts. The pre-registered scoring is kept and reported unchanged. An EXPLORATORY
  measure was added (reports/drift_exploratory.csv): the share of all alarms, on any
  day, that fall in a regime-start window. It is labelled exploratory everywhere.
- 2026-10-24, design change before Part 5 is pre-registered: the factor model now has 8
  factors (6 equity, rates from FRED DGS10, gold from GLD); see README. Parts 2 to 4 were
  rerun. In Part 4 only the factor model's residuals changed; the regime rule, detector
  thresholds and scoring are unchanged.
