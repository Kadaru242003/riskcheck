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

## Part 5: AI risk copilot study

Committed: 2026-10-24, before the pilot and before any full run.

### Answer key (fixed)

- The official answer key is the 200 questions as generated on the author's machine on
  2026-10-24: reports/copilot/answer_key.jsonl.
- SHA-256: 6893874f2864b22fa47a91259062914e95177f24a8b5457cb6c0b92e037ff4d4
- Every run and every grade uses this file. The code checks the hash before use and
  stops if it differs. Regenerating the questions (make copilot-questions) writes a
  different file (questions.jsonl) and never changes the key, so small numerical
  differences between machines (seen in the 6th decimal of some GARCH values) can never
  change a grade.

### Question set (seed 42, answers computed by the engine)

| Type | n | Graded by |
|---|---|---|
| Lookup: 1-day 95% VaR, 99% VaR or 97.5% ES for a portfolio, model, date (2008 to 2025) | 50 | number within 5% relative of the key value |
| Attribution: 20 "largest source overall", 30 "largest factor not counting market and idiosyncratic" (6 per portfolio) | 50 | exact match to the key's top source |
| Comparison: did portfolio A breach its VaR (model, 95% or 99%) on more days than B in a year (25 yes, 25 no; ties excluded) | 50 | yes/no |
| Unanswerable: 15 future dates, 15 dates with no data, 10 unknown portfolios, 10 measures or models the engine lacks | 50 | correct only if the response declines or flags that it cannot be answered (LLM judge) |

Attribution questions need a margin between the top source and the runner-up (2 points
overall, 1 point for the factor-only kind). Majority-answer baselines are reported.

### Conditions (paired: every model answers every question in both)

1. no_tools: system prompt with the portfolio holdings, model descriptions and answer
   format; the question. Nothing else.
2. tools: the same prompt plus one sentence that tools exist, and 4 MCP tools from
   riskcheck/mcp_server.py: get_var, get_attribution, get_backtest_stats, get_returns.
   Tool errors (unknown names, non-trading days, dates after 2025-12-31) are returned to
   the model as text. At most 6 tool rounds per question.

Every answer must end with "FINAL ANSWER: <number | source | yes | no | CANNOT ANSWER>".
Temperature 0, seed 42, max 2,048 completion tokens. Every prompt, tool call, tool result
and answer is logged with the API's model id and a timestamp.

### Models (confirmed available on the author's Groq key on 2026-10-24)

- Tested: openai/gpt-oss-120b and openai/gpt-oss-20b (OpenAI, open-weight; a size
  contrast within one family) and qwen/qwen3.8-27b (Alibaba, open-weight, Groq preview).
  These are the only tool-capable chat models on the key, and they span two families;
  llama-3.3-70b-versatile and minimaxai/minimax-m2.7 were planned but are not available.
- Judge (unanswerable set only), cross-family per answer: answers from the gpt-oss models
  are judged by qwen/qwen3.8-27b, answers from qwen/qwen3.8-27b by openai/gpt-oss-120b.
  No answer is judged by a model from its own family. The judges are also tested models;
  the hand-label validation below is the check on this.
- The exact model id returned by the API is logged with every answer.

### Pilot rule

- An 8-question pilot (2 per type, seed 7, none of them in the answer key) is run before
  the full study. Its only purpose is catching bugs: crashes, tool-call errors and
  grading errors.
- The pilot may lead to bug fixes in code. It may not change the hypotheses, metrics,
  tolerance, question set, answer key, models or prompts. Any bug fix is listed under
  Amendments with its date.
- Pilot answers are excluded from the final analysis and are not reported as results
  (they are stored separately in reports/copilot/pilot/, which is not committed).

### Metrics (per model x condition x question type)

- Wrong-number rate, over lookup + unanswerable questions (100 per model and condition):
  the final answer is a number and it is outside tolerance (lookup) or any number at all
  (unanswerable).
- Accuracy.
- Unsupported-claim rate, over all 200: a specific final answer (number, source, yes/no)
  that no successful tool result in the same conversation supports (a number within
  0.5% of a tool number; a source after a get_attribution call; yes/no after a
  get_backtest_stats call). Without tools every specific answer is unsupported.
- False-confidence rate, over all 200: a numeric final answer with no tool call and no
  hedging words (approximately, estimate, roughly, likely, uncertain, assume, ...).

### Primary outcome (the one confirmatory test)

- Outcome: the wrong-number rate, no_tools vs tools.
- Test: a paired, two-sided exact McNemar test for each of the 3 tested models (100 pairs
  each), Bonferroni-corrected across the 3 models: each p-value is multiplied by 3, and a
  result is significant if the corrected p-value is below 0.05.
- Expected direction: tools reduce the wrong-number rate. The result is reported
  whatever it shows, including a null or the opposite direction.

### Secondary outcomes (all exploratory)

Reported, but labelled exploratory, with no confirmatory claims:
- accuracy, no_tools vs tools (all 200 pairs; McNemar, Bonferroni across models);
- correct declines on unanswerable questions (50 pairs; same test; direction not
  predicted, since tool errors could help and having tools could also encourage answering);
- unsupported-claim and false-confidence rates;
- results by question type;
- the model size comparison (gpt-oss-120b vs gpt-oss-20b);
- the family comparison (gpt-oss vs qwen).

### Judge validation

- The judges grade all unanswerable answers (300: 50 questions x 2 conditions x 3 models).
- The author hand-labels 120 of them, without seeing any judge verdict: 60 drawn at
  random (seed 42) from the answers each judge grades, so that each judge has its own
  check (60 gpt-oss answers judged by qwen, 60 qwen answers judged by gpt-oss-120b).
- Agreement and Cohen's kappa are reported for each judge.
- If a judge's kappa against the hand labels is below 0.6, the hand labels replace that
  judge's labels for the labelled items, and that judge's other unanswerable results are
  reported as unreliable.

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
- 2026-10-24, Part 5 pilot, bug fix only: the first pilot attempt used a stale Makefile
  that still listed llama-3.3-70b-versatile; every call returned 404 and no answers were
  produced. The runner now stops when a model is not available instead of logging a
  failure for every question. Nothing in the pre-registered design changed.
- 2026-10-24, Part 5 pilot, grading bug fix only: one pilot answer ended with a bare
  "CANNOT ANSWER" line without the "FINAL ANSWER:" prefix, and the grader read it as no
  answer. Fix: if there is no FINAL ANSWER line, the grader reads the last non-empty line,
  but only when that whole line is exactly an answer (CANNOT ANSWER, yes, no, a number or
  a source name; at most 40 characters). Each graded answer records how it was read, and
  the results table reports the number of fallbacks and missing answers. The answer
  format in the prompt, the hypotheses, metrics, tolerance, questions, answer key, models
  and prompts are unchanged.

