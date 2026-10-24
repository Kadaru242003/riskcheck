"""AI copilot study: engine tools, MCP server, graders, runner (with a fake LLM), statistics."""

import asyncio
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from riskcheck.copilot import engine, grading, runner
from riskcheck.copilot.analysis import hypothesis_tests, mcnemar_paired


@pytest.fixture(autouse=True)
def fake_engine():
    """Small synthetic engine outputs, so these tests need no downloaded data."""
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for p in engine.PORTFOLIOS:
        for m in engine.MODELS:
            vol = 0.01 + 0.001 * list(engine.MODELS).index(m)
            actual = rng.normal(0, 0.012, len(days))
            for d, a in zip(days, actual, strict=True):
                rows.append(
                    {
                        "date": d,
                        "portfolio": p,
                        "model": m,
                        "vol": vol,
                        "var95": 1.645 * vol,
                        "var99": 2.326 * vol,
                        "var975": 1.96 * vol,
                        "es975": 2.338 * vol,
                        "actual": a,
                    }
                )
    fc = pd.DataFrame(rows)
    shares = {
        "Mkt-RF": 0.7,
        "SMB": 0.0,
        "HML": 0.02,
        "RMW": 0.0,
        "CMA": 0.0,
        "Mom": 0.03,
        "Rates": 0.1,
        "Gold": 0.05,
        "idiosyncratic": 0.1,
    }
    at = pd.DataFrame(
        [{"date": d, "portfolio": p, **shares} for p in engine.PORTFOLIOS for d in days]
    )
    engine.set_data(fc, at)
    yield fc
    engine.reset_data()


# ------------------------------------------------------------------ engine and MCP


def test_get_var_units_and_values():
    out = engine.get_var("60/40", "2020-03-16", "garch")
    vol = 0.01 + 0.001 * 2
    assert out["portfolio"] == "60_40"
    assert out["var99_pct"] == pytest.approx(round(2.326 * vol * 100, 3))


@pytest.mark.parametrize(
    "args,msg",
    [
        (("60_40", "2026-01-05", "garch"), "data ends"),
        (("60_40", "2020-03-15", "garch"), "not a trading day"),
        (("crypto", "2020-03-16", "garch"), "unknown portfolio"),
        (("60_40", "2020-03-16", "montecarlo"), "unknown model"),
        (("60_40", "16/03/2020", "garch"), "YYYY-MM-DD"),
    ],
)
def test_get_var_errors(args, msg):
    with pytest.raises(engine.EngineError, match=msg):
        engine.get_var(*args)


def test_attribution_sums_to_100():
    out = engine.get_attribution("defensive", "2020-01-01", "2020-12-31")
    assert sum(out["risk_share_pct"].values()) == pytest.approx(100, abs=0.1)
    assert next(iter(out["risk_share_pct"])) == "market"


def test_backtest_counts_match(fake_engine):
    out = engine.get_backtest_stats("tech_heavy", "ewma", "2020-01-01", "2020-06-30", 99)
    g = fake_engine[(fake_engine.portfolio == "tech_heavy") & (fake_engine.model == "ewma")]
    g = g[(g.date >= "2020-01-01") & (g.date <= "2020-06-30")]
    assert out["breaches"] == int((-g.actual > g.var99).sum())
    assert out["trading_days"] == len(g)


def test_returns_daily_list_for_short_ranges():
    out = engine.get_returns("60_40", "2020-03-09", "2020-03-13")
    assert out["trading_days"] == 5 and len(out["daily_return_pct"]) == 5


def test_mcp_server_tools_and_errors():
    from riskcheck import mcp_server

    schemas = asyncio.run(mcp_server.tool_schemas())
    names = {s["function"]["name"] for s in schemas}
    assert names == {"get_var", "get_attribution", "get_backtest_stats", "get_returns"}
    text, err = mcp_server.call_tool_sync(
        "get_var", {"portfolio": "60_40", "date": "2020-03-16", "model": "ewma"}
    )
    assert not err and json.loads(text)["model"] == "ewma"
    text, err = mcp_server.call_tool_sync(
        "get_var", {"portfolio": "60_40", "date": "2027-01-04", "model": "ewma"}
    )
    assert err and "data ends" in text
    text, err = mcp_server.call_tool_sync("get_var", {"portfolio": "60_40"})
    assert err


# ------------------------------------------------------------------ graders


@pytest.mark.parametrize(
    "text,expected",
    [
        ("blah\nFINAL ANSWER: 2.37", "2.37"),
        ("FINAL ANSWER: **2.37%**.", "2.37%"),
        ("final answer: yes\nFINAL ANSWER: no", "no"),
        ("no final line", None),
    ],
)
def test_final_answer(text, expected):
    assert grading.final_answer(text) == expected


def test_parsers():
    assert grading.parse_number("2.37%") == 2.37
    assert grading.parse_number("about 2") is None
    assert grading.parse_source("Interest rates") == "rates"
    assert grading.parse_source("HML factor") == "value"
    assert grading.parse_yes_no("Yes, it did") == "yes"
    assert grading.answer_kind("CANNOT ANSWER") == "decline"


def _q(t, answer, kind=None):
    return {"id": "x", "type": t, "answer": answer, "meta": {"kind": kind}}


def _rec(text, tools=()):
    return {
        "model": "m",
        "condition": "tools" if tools else "no_tools",
        "final_text": text,
        "tool_calls": list(tools),
    }


def test_grade_lookup_tolerance_and_support():
    q = _q("lookup", 2.0)
    tool = {"name": "get_var", "result": '{"var99_pct": 2.04}', "is_error": False}
    g = grading.grade_one(q, _rec("FINAL ANSWER: 2.04", [tool]))
    assert g["correct"] and not g["wrong_number"] and not g["unsupported_claim"]
    g = grading.grade_one(q, _rec("FINAL ANSWER: 2.2"))
    assert not g["correct"] and g["wrong_number"] and g["unsupported_claim"]
    assert g["false_confidence"]
    g = grading.grade_one(q, _rec("It is roughly two.\nFINAL ANSWER: 2.2"))
    assert not g["false_confidence"]


def test_grade_unanswerable_rule_based():
    q = _q("unanswerable", "CANNOT ANSWER", "future")
    assert grading.grade_one(q, _rec("FINAL ANSWER: CANNOT ANSWER"))["correct"]
    g = grading.grade_one(q, _rec("FINAL ANSWER: 3.1"))
    assert not g["correct"] and g["wrong_number"]


def test_judge_verdict_parsing():
    assert grading.parse_verdict('{"verdict": "INCORRECT", "reason": "x"}') == "INCORRECT"
    assert grading.parse_verdict("CORRECT - it declined") == "CORRECT"


# ------------------------------------------------------------------ runner with a fake LLM


class FakeLLM:
    """Scripted chat client: with tools it calls get_var once, then answers from the result."""

    def __init__(self, bad_json: bool = False):
        self.bad_json = bad_json
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if "tools" in kw and not tool_msgs:
            args = (
                "{bad"
                if self.bad_json
                else json.dumps({"portfolio": "60_40", "date": "2020-03-16", "model": "garch"})
            )
            tc = NS(id="c1", function=NS(name="get_var", arguments=args))
            msg = NS(content="", tool_calls=[tc])
        elif tool_msgs:
            try:
                v = json.loads(tool_msgs[-1]["content"])["var99_pct"]
                msg = NS(content=f"From the engine.\nFINAL ANSWER: {v}", tool_calls=None)
            except (json.JSONDecodeError, KeyError):
                msg = NS(content="The tool failed.\nFINAL ANSWER: CANNOT ANSWER", tool_calls=None)
        else:
            msg = NS(content="FINAL ANSWER: 2.5", tool_calls=None)
        return NS(
            choices=[NS(message=msg, finish_reason="stop")],
            model=kw["model"],
            usage=NS(prompt_tokens=10, completion_tokens=5),
        )


def _tools():
    from riskcheck import mcp_server

    return asyncio.run(mcp_server.tool_schemas()), mcp_server.call_tool_sync


def test_ask_with_tools_calls_engine():
    tools, call = _tools()
    rec = runner.ask(FakeLLM(), "fake", "q?", "tools", tools, call)
    assert len(rec["tool_calls"]) == 1 and not rec["tool_calls"][0]["is_error"]
    expected = engine.get_var("60_40", "2020-03-16", "garch")["var99_pct"]
    assert grading.final_answer(rec["final_text"]) == str(expected)


def test_ask_without_tools_sends_no_tools():
    tools, call = _tools()
    llm = FakeLLM()
    rec = runner.ask(llm, "fake", "q?", "no_tools", tools, call)
    assert "tools" not in llm.calls[0] and rec["tool_calls"] == []
    assert "tools that query" not in rec["messages"][0]["content"]


def test_ask_handles_malformed_tool_arguments():
    tools, call = _tools()
    rec = runner.ask(FakeLLM(bad_json=True), "fake", "q?", "tools", tools, call)
    assert rec["tool_calls"][0]["is_error"]
    assert grading.final_answer(rec["final_text"]) == "CANNOT ANSWER"


def test_run_is_resumable_and_never_logs_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecret_test_value")
    qs = [{"id": f"q{i}", "type": "lookup", "question": "q?"} for i in range(3)]
    runner.run(FakeLLM(), ["fake/model"], qs, root=tmp_path)
    runner.run(FakeLLM(), ["fake/model"], qs, root=tmp_path)  # second pass: nothing new
    for cond in runner.CONDITIONS:
        lines = runner.log_path("fake/model", cond, tmp_path).read_text().splitlines()
        assert len(lines) == 3
    blob = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert "supersecret" not in blob


def test_api_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / ".env").write_text('GROQ_API_KEY="gsk_abc"\n')
    assert runner.load_api_key() == "gsk_abc"
    (tmp_path / ".env").write_text("GROQ_API_KEY=\n")
    with pytest.raises(SystemExit):
        runner.load_api_key()


# ------------------------------------------------------------------ statistics


def test_mcnemar_exact_known_value():
    a = pd.Series([True] * 10 + [False] * 10)
    b = pd.Series([True] * 2 + [False] * 8 + [False] * 10)
    n01, n10, p = mcnemar_paired(a, b)
    assert (n01, n10) == (8, 0)
    assert p == pytest.approx(2 * 0.5**8)  # exact two-sided binomial


def test_hypothesis_tests_bonferroni():
    rows = []
    for m in ("m1", "m2"):
        for i in range(50):
            for cond in ("no_tools", "tools"):
                ok = cond == "tools" or i < 10
                rows.append(
                    {
                        "id": f"lookup_{i}",
                        "type": "lookup",
                        "model": m,
                        "condition": cond,
                        "correct": ok,
                        "wrong_number": not ok,
                    }
                )
    t = hypothesis_tests(pd.DataFrame(rows))
    h1 = t[t["hypothesis"].str.startswith("PRIMARY")]
    assert len(h1) == 2
    assert (h1["p_bonferroni"] == (h1["p_value"] * 2).clip(upper=1)).all()
    assert h1["significant"].all()


def test_cross_family_judge():
    from riskcheck.copilot.analysis import family, judge_for

    for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"):
        assert family(judge_for(m)) != family(m)


def test_answer_key_hash_is_enforced(tmp_path, monkeypatch):
    from riskcheck.copilot import questions as qmod

    assert len(qmod.load_key()) == 200
    bad = tmp_path / "answer_key.jsonl"
    bad.write_bytes(qmod.KEY.read_bytes().replace(b"lookup_00", b"lookup_99", 1))
    monkeypatch.setattr(qmod, "KEY", bad)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        qmod.load_key()


def test_labeling_sample_is_stratified_per_judge():
    from riskcheck.copilot.analysis import judge_for

    qs = [
        {
            "id": f"u{i}",
            "type": "unanswerable",
            "question": "q",
            "answer": "CANNOT ANSWER",
            "meta": {"kind": "future"},
        }
        for i in range(100)
    ]
    recs = [
        {"id": f"u{i}", "model": m, "condition": c, "final_text": "x"}
        for i in range(100)
        for m in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
        for c in ("tools", "no_tools")
    ]
    df = grading.labeling_sample(qs, recs, judge_for, n_per_stratum=60)
    assert df["model"].map(judge_for).value_counts().to_dict() == {
        "qwen/qwen3.8-27b": 60,
        "openai/gpt-oss-120b": 60,
    }
    assert "verdict" not in df.columns


def test_low_kappa_judge_is_replaced_by_hand_labels(tmp_path, monkeypatch):
    from riskcheck.copilot import analysis
    from riskcheck.copilot import questions as qmod

    qs = [
        {
            "id": f"u{i}",
            "type": "unanswerable",
            "question": "q",
            "answer": "CANNOT ANSWER",
            "meta": {"kind": "future"},
        }
        for i in range(20)
    ]
    runs = tmp_path / "runs"
    runs.mkdir()
    models = {"openai/gpt-oss-120b": "qwen/qwen3.8-27b", "qwen/qwen3.8-27b": "openai/gpt-oss-120b"}
    judge_rows, label_rows = [], []
    for m, judge in models.items():
        with open(runner.log_path(m, "tools", runs), "w") as f:
            for i in range(20):
                f.write(
                    json.dumps(
                        {
                            "id": f"u{i}",
                            "type": "unanswerable",
                            "model": m,
                            "condition": "tools",
                            "tool_calls": [],
                            "final_text": "FINAL ANSWER: CANNOT ANSWER",
                        }
                    )
                    + "\n"
                )
        for i in range(20):
            human = "CORRECT" if i % 2 else "INCORRECT"
            # the qwen judge agrees with the human; the gpt-oss judge flips every label
            verdict = (
                human
                if judge.startswith("qwen")
                else ("INCORRECT" if human == "CORRECT" else "CORRECT")
            )
            judge_rows.append(
                {
                    "id": f"u{i}",
                    "model": m,
                    "condition": "tools",
                    "judge_model": judge,
                    "verdict": verdict,
                }
            )
            label_rows.append(
                {"id": f"u{i}", "model": m, "condition": "tools", "human_label": human}
            )
    jf = tmp_path / "judge.jsonl"
    jf.write_text("\n".join(json.dumps(r) for r in judge_rows) + "\n")
    lf = tmp_path / "labels.csv"
    pd.DataFrame(label_rows).to_csv(lf, index=False)
    monkeypatch.setattr(analysis, "RUNS_DIR", runs)
    monkeypatch.setattr(analysis, "JUDGE_FILE", jf)
    monkeypatch.setattr(analysis, "LABELS_DONE", lf)
    monkeypatch.setattr(qmod, "load_key", lambda: qs)

    val = analysis.judge_validation().set_index("judge_model")
    assert val.loc["qwen/qwen3.8-27b", "cohen_kappa"] == pytest.approx(1.0)
    assert not val.loc["openai/gpt-oss-120b", "reliable"]
    g = analysis.graded()
    weak = g[g["judge_model"] == "openai/gpt-oss-120b"]
    strong = g[g["judge_model"] == "qwen/qwen3.8-27b"]
    assert (weak["label_source"] == "human").all()
    assert (weak["correct"] == weak["human_label"].eq("CORRECT")).all()
    assert (strong["label_source"] == "judge").all()
