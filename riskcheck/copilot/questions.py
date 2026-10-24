"""Build the 200-question set (4 types x 50) with answers computed by the engine.

Run: python -m riskcheck.copilot.questions  ->  reports/copilot/questions.jsonl
Deterministic (seed 42). Every answer comes from riskcheck.copilot.engine.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from riskcheck import SEED
from riskcheck.copilot import engine
from riskcheck.portfolios import REPORTS_DIR

OUT = REPORTS_DIR / "copilot" / "questions.jsonl"
# The official answer key: the 200 questions as generated on the author's machine on
# 2026-10-24. Runs and grading always use this file, never a regenerated one.
KEY = REPORTS_DIR / "copilot" / "answer_key.jsonl"
KEY_SHA256 = "6893874f2864b22fa47a91259062914e95177f24a8b5457cb6c0b92e037ff4d4"
N_PER_TYPE = 50
FIRST_DATE, LAST_DATE = "2008-01-02", "2025-12-31"
YEARS = list(range(2008, 2026))
N_OVERALL = 20  # attribution questions about the largest source overall
MARGIN_PP = 2.0  # "overall": top source must beat the runner-up by >= 2 percentage points
MARGIN_STYLE_PP = 1.0  # "style": top factor must beat the runner-up by >= 1 point

NAMES = {
    "60_40": "60/40",
    "equal_sector": "equal-weight sectors",
    "tech_heavy": "tech-heavy",
    "defensive": "defensive",
    "risk_parity": "risk parity",
}
MODEL_NAMES = {
    "historical": "historical simulation",
    "ewma": "EWMA",
    "garch": "GARCH",
    "factor": "factor",
}
MEASURES = {
    "var95": ("1-day 95% VaR", "var95"),
    "var99": ("1-day 99% VaR", "var99"),
    "es975": ("1-day 97.5% Expected Shortfall", "es975"),
}
SOURCE_LIST = ", ".join(engine.SOURCE_NAMES.values())
STYLE_LIST = ", ".join(
    v for k, v in engine.SOURCE_NAMES.items() if k not in ("Mkt-RF", "idiosyncratic")
)


def _trading_days() -> pd.DatetimeIndex:
    fc = engine._forecasts()
    d = fc.loc[("60_40", "garch")].index
    return d[(d >= FIRST_DATE) & (d <= LAST_DATE)]


def lookup(rng: np.random.Generator) -> list[dict]:
    days = _trading_days()
    out = []
    for i in range(N_PER_TYPE):
        p = rng.choice(list(NAMES))
        m = rng.choice(list(MODEL_NAMES))
        key = rng.choice(list(MEASURES))
        d = days[rng.integers(len(days))].strftime("%Y-%m-%d")
        label, col = MEASURES[key]
        truth = engine.var_exact(p, d, m)[col] * 100
        out.append(
            {
                "id": f"lookup_{i:02d}",
                "type": "lookup",
                "question": f"According to the {MODEL_NAMES[m]} model, what was the "
                f"{NAMES[p]} portfolio's {label} on {d}? Give it as a percentage loss.",
                "answer": round(truth, 6),
                "meta": {"portfolio": p, "model": m, "measure": key, "date": d},
            }
        )
    return out


def attribution(rng: np.random.Generator) -> list[dict]:
    """20 'largest source overall' + 30 'largest factor besides market and idiosyncratic'.

    The second kind (6 per portfolio) is the discriminating one; the first is an easy
    check whose answer is usually the market factor.
    """
    out, seen = [], set()
    plan = [("overall", None)] * N_OVERALL + [
        ("style", p) for p in NAMES for _ in range((N_PER_TYPE - N_OVERALL) // len(NAMES))
    ]
    for kind, fixed_p in plan:
        while True:
            p = fixed_p or rng.choice(list(NAMES))
            y = int(rng.choice(YEARS))
            if (p, y, kind) in seen:
                continue
            shares = engine.attribution_exact(p, f"{y}-01-01", f"{y}-12-31") * 100
            margin = MARGIN_PP
            if kind == "style":
                shares = shares.drop(["Mkt-RF", "idiosyncratic"])
                margin = MARGIN_STYLE_PP
            top2 = shares.sort_values(ascending=False).iloc[:2]
            if top2.iloc[0] - top2.iloc[1] < margin:
                seen.add((p, y, kind))
                continue
            seen.add((p, y, kind))
            break
        top = engine.SOURCE_NAMES[top2.index[0]]
        if kind == "overall":
            q = (
                f"According to the factor model, which source contributed the largest share "
                f"of the {NAMES[p]} portfolio's risk in {y}? Choose one of: {SOURCE_LIST}."
            )
        else:
            q = (
                f"According to the factor model, which factor contributed the largest share "
                f"of the {NAMES[p]} portfolio's risk in {y}, not counting the market factor "
                f"and the idiosyncratic part? Choose one of: {STYLE_LIST}."
            )
        out.append(
            {
                "id": f"attribution_{len(out):02d}",
                "type": "attribution",
                "question": q,
                "answer": top,
                "meta": {
                    "portfolio": p,
                    "year": y,
                    "kind": kind,
                    "margin_pp": round(float(top2.iloc[0] - top2.iloc[1]), 2),
                },
            }
        )
    return out


def comparison(rng: np.random.Generator) -> list[dict]:
    want = {"yes": N_PER_TYPE // 2, "no": N_PER_TYPE - N_PER_TYPE // 2}
    out, seen = [], set()
    while len(out) < N_PER_TYPE:
        a, b = rng.choice(list(NAMES), size=2, replace=False)
        m = rng.choice(list(MODEL_NAMES))
        y = int(rng.choice(YEARS))
        lvl = int(rng.choice([95, 99]))
        if (a, b, m, y, lvl) in seen:
            continue
        seen.add((a, b, m, y, lvl))
        _, ba = engine.backtest_exact(a, m, f"{y}-01-01", f"{y}-12-31", lvl)
        _, bb = engine.backtest_exact(b, m, f"{y}-01-01", f"{y}-12-31", lvl)
        if ba == bb:
            continue
        ans = "yes" if ba > bb else "no"
        if want[ans] == 0:
            continue
        want[ans] -= 1
        out.append(
            {
                "id": f"comparison_{len(out):02d}",
                "type": "comparison",
                "question": f"Using the {MODEL_NAMES[m]} model's {lvl}% VaR, did the "
                f"{NAMES[a]} portfolio breach its VaR on more days than the {NAMES[b]} "
                f"portfolio in {y}? Answer yes or no.",
                "answer": ans,
                "meta": {
                    "a": a,
                    "b": b,
                    "model": m,
                    "year": y,
                    "level": lvl,
                    "breaches_a": ba,
                    "breaches_b": bb,
                },
            }
        )
    return out


def unanswerable(rng: np.random.Generator) -> list[dict]:
    """Four pre-registered kinds; correct only if the model declines or flags it."""
    qs: list[tuple[str, str, dict]] = []
    used: set[str] = set()

    def add(kind: str, q: str, meta: dict) -> None:
        if q not in used:
            used.add(q)
            qs.append((kind, q, meta))

    future = pd.bdate_range("2026-02-02", "2027-06-30")
    while len(qs) < 15:  # future dates / forecasts beyond the data
        p, m = rng.choice(list(NAMES)), rng.choice(list(MODEL_NAMES))
        d = future[rng.integers(len(future))].strftime("%Y-%m-%d")
        form = rng.integers(3)
        if form == 0:
            q = (
                f"According to the {MODEL_NAMES[m]} model, what will the {NAMES[p]} "
                f"portfolio's 1-day 99% VaR be on {d}?"
            )
        elif form == 1:
            q = (
                f"What will the {NAMES[p]} portfolio's 1-day 95% VaR be next month, "
                f"according to the {MODEL_NAMES[m]} model?"
            )
        else:
            q = (
                f"How many times will the {NAMES[p]} portfolio breach its 99% VaR "
                f"({MODEL_NAMES[m]} model) in 2027?"
            )
        add("future", q, {"portfolio": p, "model": m, "date": d})

    weekends = [d for d in pd.date_range(FIRST_DATE, LAST_DATE) if d.weekday() >= 5]
    while len(qs) < 30:  # dates with no data: 10 weekends, 5 before the data starts
        p, m = rng.choice(list(NAMES)), rng.choice(list(MODEL_NAMES))
        if len(qs) < 25:
            d = weekends[rng.integers(len(weekends))].strftime("%Y-%m-%d")
        else:
            d = f"{rng.integers(1995, 2004)}-{rng.integers(1, 13):02d}-15"
        q = (
            f"According to the {MODEL_NAMES[m]} model, what was the {NAMES[p]} portfolio's "
            f"1-day 99% VaR on {d}? Give it as a percentage loss."
        )
        add("no_data_date", q, {"portfolio": p, "model": m, "date": d})

    fake = ["emerging markets", "crypto", "high-yield bond", "small-cap value", "commodities"]
    while len(qs) < 40:  # portfolios the engine does not have
        f = fake[rng.integers(len(fake))]
        y = int(rng.choice(YEARS))
        q = (
            f"Which factor contributed the largest share of the {f} portfolio's risk in {y}, "
            f"according to the factor model?"
        )
        add("unknown_portfolio", q, {"portfolio": f, "year": y})

    unsupported = [
        "What was the {p} portfolio's 10-day 99% VaR on {d}?",
        "What was the {p} portfolio's 1-day 99.9% VaR on {d}?",
        "According to the Monte Carlo model, what was the {p} portfolio's 1-day 99% VaR on {d}?",
        "What was the {p} portfolio's liquidity-adjusted VaR on {d}?",
        "What was the {p} portfolio's 1-month 95% Expected Shortfall as of {d}?",
    ]
    days = _trading_days()
    while len(qs) < N_PER_TYPE:  # measures or models the engine does not compute
        p = rng.choice(list(NAMES))
        d = days[rng.integers(len(days))].strftime("%Y-%m-%d")
        t = unsupported[rng.integers(len(unsupported))]
        add("unsupported_measure", t.format(p=NAMES[p], d=d), {"portfolio": p, "date": d})

    return [
        {
            "id": f"unanswerable_{i:02d}",
            "type": "unanswerable",
            "question": q,
            "answer": "CANNOT ANSWER",
            "meta": {"kind": kind, **meta},
        }
        for i, (kind, q, meta) in enumerate(qs)
    ]


def build(seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    qs = lookup(rng) + attribution(rng) + comparison(rng) + unanswerable(rng)
    for q in qs:  # numpy scalars -> plain JSON types
        q["meta"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in q["meta"].items()}
    return qs


def save(qs: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")


def load() -> list[dict]:
    """The regenerated question file (for inspection; the study uses load_key)."""
    with open(OUT) as f:
        return [json.loads(line) for line in f]


def load_key() -> list[dict]:
    """The official answer key, after checking its SHA-256 against the pre-registered hash."""
    data = KEY.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != KEY_SHA256:
        raise RuntimeError(f"answer key hash mismatch: {digest} != {KEY_SHA256}")
    return [json.loads(line) for line in data.decode().splitlines()]


if __name__ == "__main__":
    qs = build()
    save(qs)
    print(pd.Series([q["type"] for q in qs]).value_counts().to_dict())
