"""Smoke test: the dashboard renders without errors on the real engine outputs."""

import pytest

from riskcheck.portfolios import REPORTS_DIR

pytestmark = pytest.mark.skipif(
    not (REPORTS_DIR / "forecasts.parquet").exists(), reason="run `make forecasts` first"
)


def test_dashboard_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 5
    at.selectbox[0].select("defensive").run()
    at.radio[0].set_value(95).run()
    assert not at.exception
