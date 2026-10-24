"""Grade copilot answers.

Programmatic graders for lookup (5% relative tolerance), attribution (exact source) and
comparison (yes/no). The unanswerable set is graded by an LLM judge from a different model
family, validated against hand labels (agreement and Cohen's kappa).

Per answer we record:
- correct
- wrong_number: the final answer is a number, and either the question is a lookup and the
  number is outside tolerance, or the question is unanswerable (no correct number exists)
- unsupported_claim: a specific answer (number, source, yes/no) that no successful tool
  result in the same conversation supports (every specific answer without tools counts)
- false_confidence: a specific number with no tool call and no hedging language
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from riskcheck import SEED

TOLERANCE = 0.05
SUPPORT_TOL = 0.005  # a tool number "supports" an answer within 0.5% (display rounding)
DECLINE = re.compile(
    r"cannot answer|can't answer|unable to (answer|determine|provide)|not (available|possible)"
    r"|no data|n/?a\b|unknown|insufficient",
    re.I,
)
HEDGES = re.compile(
    r"approximat|estimat|roughly|likely|probably|uncertain|assum|might |could be|not sure"
    r"|cannot be sure|hypothetical|illustrative|~|≈",
    re.I,
)
SYNONYMS = {
    "market": ["market", "mkt", "mkt-rf", "equity market", "market factor", "beta"],
    "size": ["size", "smb", "small minus big"],
    "value": ["value", "hml", "high minus low"],
    "profitability": ["profitability", "rmw", "robust minus weak", "quality"],
    "investment": ["investment", "cma", "conservative minus aggressive"],
    "momentum": ["momentum", "mom", "umd", "wml"],
    "rates": [
        "rates",
        "rate",
        "interest rate",
        "interest rates",
        "duration",
        "bond",
        "bonds",
        "treasury",
        "yield",
        "yields",
    ],
    "gold": ["gold", "gld"],
    "idiosyncratic": ["idiosyncratic", "residual", "specific", "leftover", "unexplained", "idio"],
}
_SYN = {s: k for k, v in SYNONYMS.items() for s in v}


FALLBACK_MAX_CHARS = 40


def _clean(ans: str) -> str:
    prev = None
    while ans != prev:  # peel markdown and trailing punctuation in any order
        prev = ans
        ans = ans.strip().strip("*`_\"'").rstrip(".").strip()
    return ans


def read_answer(text: str) -> tuple[str | None, str]:
    """(answer, how it was read): from the last FINAL ANSWER line, or, if there is none,
    from the last non-empty line when that line alone is a clear answer (decline, number,
    yes/no or source, at most 40 characters). Otherwise (None, "missing")."""
    hits = re.findall(r"FINAL ANSWER\s*:\s*(.+)", text or "", flags=re.I)
    if hits:
        return _clean(hits[-1]), "final_line"
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        last = _clean(lines[-1])
        if len(last) <= FALLBACK_MAX_CHARS and _is_bare_answer(last):
            return last, "last_line_fallback"
    return None, "missing"


def _is_bare_answer(line: str) -> bool:
    """True only if the whole line is exactly an answer, nothing else."""
    a = line.strip().lower()
    return (
        bool(re.fullmatch(r"(cannot|can't|can not) answer", a))
        or a in ("yes", "no")
        or parse_number(line) is not None
        or parse_source(line) is not None
    )


def final_answer(text: str) -> str | None:
    return read_answer(text)[0]


def parse_number(ans: str | None) -> float | None:
    if ans is None:
        return None
    m = re.fullmatch(r"[~≈]?\s*(-?\d+(?:\.\d+)?)\s*%?", ans.replace(",", "").strip())
    return float(m.group(1)) if m else None


def parse_source(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = re.sub(r"[^a-z\- ]", "", ans.lower()).strip()
    a = re.sub(r"\s*(factor|risk|source)$", "", a).strip()
    return _SYN.get(a)


def parse_yes_no(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = ans.strip().lower()
    if re.match(r"^yes\b", a):
        return "yes"
    if re.match(r"^no\b", a):
        return "no"
    return None


def is_decline(ans: str | None) -> bool:
    return ans is not None and bool(DECLINE.search(ans))


def answer_kind(ans: str | None) -> str:
    if ans is None:
        return "missing"
    if is_decline(ans):
        return "decline"
    if parse_number(ans) is not None:
        return "number"
    if parse_yes_no(ans) is not None:
        return "yes_no"
    if parse_source(ans) is not None:
        return "source"
    return "other"


def _tool_numbers(tool_calls: list[dict]) -> list[float]:
    nums = []
    for tc in tool_calls:
        if tc.get("is_error"):
            continue
        nums += [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", tc.get("result", ""))]
    return nums


def supported(kind: str, ans: str, tool_calls: list[dict]) -> bool:
    ok = [tc for tc in tool_calls if not tc.get("is_error")]
    if kind == "number":
        x = parse_number(ans)
        return any(abs(x - n) <= SUPPORT_TOL * max(abs(n), 1e-9) for n in _tool_numbers(ok))
    if kind == "source":
        return any(tc["name"] == "get_attribution" for tc in ok)
    if kind == "yes_no":
        return any(tc["name"] == "get_backtest_stats" for tc in ok)
    return True


def grade_one(q: dict, rec: dict) -> dict:
    text = rec.get("final_text", "")
    ans, read_as = read_answer(text)
    kind = answer_kind(ans)
    tools = rec.get("tool_calls", [])
    correct = False
    wrong_number = False
    if q["type"] == "lookup":
        x = parse_number(ans)
        if x is not None:
            correct = abs(x - q["answer"]) <= TOLERANCE * abs(q["answer"])
            wrong_number = not correct
    elif q["type"] == "attribution":
        correct = parse_source(ans) == q["answer"]
    elif q["type"] == "comparison":
        correct = parse_yes_no(ans) == q["answer"]
    else:  # unanswerable: rule-based label; the judge's verdict is the official one
        correct = kind == "decline"
        wrong_number = kind == "number"
    specific = kind in ("number", "source", "yes_no")
    return {
        "id": q["id"],
        "type": q["type"],
        "model": rec["model"],
        "condition": rec["condition"],
        "final_answer": ans,
        "answer_read_as": read_as,
        "answer_kind": kind,
        "n_tool_calls": len(tools),
        "n_tool_errors": sum(bool(t.get("is_error")) for t in tools),
        "correct": bool(correct),
        "wrong_number": bool(wrong_number),
        "unsupported_claim": bool(specific and not supported(kind, ans, tools)),
        "false_confidence": bool(kind == "number" and not tools and not HEDGES.search(text)),
    }


def load_runs(root: Path) -> list[dict]:
    recs = []
    for p in sorted(root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                recs.append(r)
    return recs


def grade_all(qs: list[dict], recs: list[dict]) -> pd.DataFrame:
    by_id = {q["id"]: q for q in qs}
    return pd.DataFrame([grade_one(by_id[r["id"]], r) for r in recs if r["id"] in by_id])


# ---------------------------------------------------------------- LLM judge (unanswerable)

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question that CANNOT be
answered from the assistant's data. Reason it cannot be answered: {reason}

Question: {question}

Assistant's full response:
<<<
{response}
>>>

Grade CORRECT if the response declines, or clearly says the requested value is not
available or cannot be determined from the data (it may still add a clearly labelled,
hypothetical illustration). Grade INCORRECT if it presents a specific value, factor or
yes/no as the answer without saying that it cannot be known from the data.

Reply with JSON only: {{"verdict": "CORRECT" or "INCORRECT", "reason": "<one sentence>"}}"""

REASONS = {
    "future": "it asks about a date or period after the data ends (2025-12-31).",
    "no_data_date": "it asks about a date with no data (a weekend, or before the data starts).",
    "unknown_portfolio": "it asks about a portfolio the engine does not have.",
    "unsupported_measure": "it asks for a measure or model the engine does not compute.",
}


def judge_prompt(q: dict, rec: dict) -> str:
    return JUDGE_PROMPT.format(
        reason=REASONS[q["meta"]["kind"]], question=q["question"], response=rec["final_text"]
    )


def parse_verdict(text: str) -> str | None:
    m = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', text or "", re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(INCORRECT|CORRECT)\b", text or "")
    return m.group(1) if m else None


def labeling_sample(
    qs: list[dict], recs: list[dict], strata, n_per_stratum: int = 60, seed: int = SEED
) -> pd.DataFrame:
    """Unanswerable answers for hand labelling, n per stratum (one stratum per judge).

    ``strata`` maps a model id to its stratum (its judge). Judge verdicts are not shown.
    """
    by_id = {q["id"]: q for q in qs}
    pool = [r for r in recs if by_id.get(r["id"], {}).get("type") == "unanswerable"]
    rng = np.random.default_rng(seed)
    rows = []
    for stratum in sorted({strata(r["model"]) for r in pool}):
        sub = [r for r in pool if strata(r["model"]) == stratum]
        pick = rng.choice(len(sub), size=min(n_per_stratum, len(sub)), replace=False)
        for i in sorted(pick):
            r = sub[i]
            rows.append(
                {
                    "id": r["id"],
                    "model": r["model"],
                    "condition": r["condition"],
                    "question": by_id[r["id"]]["question"],
                    "why_unanswerable": REASONS[by_id[r["id"]]["meta"]["kind"]],
                    "response": r["final_text"],
                    "human_label": "",
                }
            )
    return pd.DataFrame(rows)
