# Pre-registration

Status: DRAFT. Nothing below is final until it is committed with a date, and it must be
committed before any full run it covers.

## Part 4: regime starts (to be fixed before any detector is scored)

- Rule: the first trading day the VIX closes above 30 after 60 trading days below 20.
- Source: FRED VIXCLS. Defined from outside data only, never from model results.

## Part 5: AI copilot study (to be fixed before any full run)

- Hypotheses: TBD
- Metrics: wrong-number rate, unsupported-claim rate, false-confidence rate
- Numeric tolerance: TBD (spec suggests 5% relative)
- Tests: McNemar for paired comparisons, Bonferroni correction across models
- Models and versions: TBD, with dates logged
