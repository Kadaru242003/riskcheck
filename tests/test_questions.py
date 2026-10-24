"""The 200-question set, checked against the real engine outputs (skipped without data)."""

import collections

import pytest

from riskcheck.copilot import engine
from riskcheck.copilot import questions as qmod
from riskcheck.portfolios import REPORTS_DIR

pytestmark = pytest.mark.skipif(
    not (REPORTS_DIR / "forecasts.parquet").exists(), reason="run `make forecasts` first"
)


@pytest.fixture(scope="module")
def qs():
    engine.reset_data()
    return qmod.build()


def test_counts_unique_and_deterministic(qs):
    assert collections.Counter(q["type"] for q in qs) == {
        "lookup": 50,
        "attribution": 50,
        "comparison": 50,
        "unanswerable": 50,
    }
    assert len({q["question"] for q in qs}) == 200
    assert [q["question"] for q in qmod.build()] == [q["question"] for q in qs]


def test_balance(qs):
    comp = collections.Counter(q["answer"] for q in qs if q["type"] == "comparison")
    assert comp == {"yes": 25, "no": 25}
    kinds = collections.Counter(q["meta"]["kind"] for q in qs if q["type"] == "attribution")
    assert kinds == {"overall": 20, "style": 30}


def test_answers_match_engine(qs):
    for q in qs:
        m = q["meta"]
        if q["type"] == "lookup":
            v = engine.var_exact(m["portfolio"], m["date"], m["model"])[m["measure"]] * 100
            assert q["answer"] == pytest.approx(v)
        elif q["type"] == "comparison":
            y = m["year"]
            _, a = engine.backtest_exact(m["a"], m["model"], f"{y}-01-01", f"{y}-12-31", m["level"])
            _, b = engine.backtest_exact(m["b"], m["model"], f"{y}-01-01", f"{y}-12-31", m["level"])
            assert q["answer"] == ("yes" if a > b else "no")


def test_unanswerable_really_fail_in_the_engine(qs):
    for q in qs:
        m = q["meta"]
        if q["type"] == "unanswerable" and m["kind"] in ("future", "no_data_date"):
            with pytest.raises(engine.EngineError):
                engine.var_exact(m["portfolio"], m["date"], m["model"])
