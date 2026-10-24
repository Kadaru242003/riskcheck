"""The results report builds from the committed tables and quotes them exactly."""

import pandas as pd

from riskcheck import report


def test_report_builds_from_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT", tmp_path / "RESULTS.md")
    text = report.build()
    for heading in (
        "## 1. Portfolios",
        "## 2. VaR backtest",
        "## 3. Regime drift",
        "## 4. AI risk copilot study",
    ):
        assert heading in text
    b = pd.read_csv(report.REPORTS_DIR / "backtest_results.csv")
    garch = b[b["model"] == "garch"]["breach_rate_99"].mean()
    assert report.pct(garch) in text
