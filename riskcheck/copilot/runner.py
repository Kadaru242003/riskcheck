"""Run LLM risk copilots on the question set, with and without the engine's MCP tools.

    python -m riskcheck.copilot.runner models                  # list models your key can use
    python -m riskcheck.copilot.runner pilot --models M1 M2    # 8 pilot questions (not in the set)
    python -m riskcheck.copilot.runner run --models M1 M2 M3   # the full study (resumable)

The Groq key is read from GROQ_API_KEY (or a .env file in the repo root). It is never
printed or written to any log. Every prompt, tool call, tool result and answer is
logged to reports/copilot/runs/<model>__<condition>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from riskcheck import SEED
from riskcheck.copilot import questions as qmod
from riskcheck.data import ROOT
from riskcheck.portfolios import REPORTS_DIR

RUNS_DIR = REPORTS_DIR / "copilot" / "runs"
PILOT_DIR = REPORTS_DIR / "copilot" / "pilot"
CONDITIONS = ["no_tools", "tools"]
MAX_TOOL_ROUNDS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 2048
MAX_RETRIES = 8
# Errors that will repeat on every question: stop instead of logging 200 failures.
FATAL_ERRORS = {"NotFoundError", "AuthenticationError", "PermissionDeniedError"}

HOLDINGS = """Portfolios (all rebalanced monthly; data 2006 to 2025):
- 60/40: 60% SPY, 40% TLT.
- equal-weight sectors: equal weights in XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU.
- tech-heavy: 50% XLK, 50% split equally across the other 8 sector ETFs above.
- defensive: equal weights in XLU, XLP, XLV, TLT, GLD.
- risk parity: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, TLT, GLD weighted by inverse
  trailing 60-day volatility.

Risk models: historical simulation (last 250 days), EWMA (lambda 0.94, normal), GARCH(1,1)
with Student-t errors, and an 8-factor model (market, size, value, profitability,
investment, momentum, rates, gold) with normal returns. VaR and Expected Shortfall are
1-day measures, stated as a percentage loss of portfolio value. A VaR breach is a day on
which the loss exceeded that day's VaR forecast."""

ANSWER_FORMAT = """Answer the user's question. If it cannot be answered from the information
available to you, say so. End your reply with one line in exactly this form:
FINAL ANSWER: <answer>
where <answer> is a number (a percentage, without the % sign), one factor name, yes, no,
or CANNOT ANSWER."""


def system_prompt(condition: str) -> str:
    head = "You are a portfolio risk assistant for RiskCheck, a portfolio risk engine."
    tools = (
        "\nYou have tools that query the RiskCheck engine's stored results."
        if condition == "tools"
        else ""
    )
    return f"{head}{tools}\n\n{HOLDINGS}\n\n{ANSWER_FORMAT}"


# ---------------------------------------------------------------- key handling


def load_api_key() -> str:
    """GROQ_API_KEY from the environment, else from ROOT/.env. Never printed."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*GROQ_API_KEY\s*=\s*(.*)\s*$", line)
            if m:
                key = m.group(1).strip().strip('"').strip("'")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set (put GROQ_API_KEY=... in .env)")
    return key


def make_client():
    from groq import Groq

    return Groq(api_key=load_api_key(), max_retries=0)


# ---------------------------------------------------------------- one conversation


def _with_retries(fn, *args, **kwargs):
    """Call the API, backing off on rate limits and transient errors."""
    delay = 5.0
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # groq.RateLimitError, APIConnectionError, 5xx
            name = type(e).__name__
            status = getattr(e, "status_code", None)
            retryable = name in ("RateLimitError", "APIConnectionError", "APITimeoutError") or (
                status is not None and status >= 500
            )
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            wait = delay
            resp = getattr(e, "response", None)
            if resp is not None:
                ra = resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = max(float(ra), 1.0)
                    except ValueError:
                        pass
            print(f"  {name}; waiting {wait:.0f}s (attempt {attempt + 1})")
            time.sleep(wait)
            delay = min(delay * 2, 120)


def _msg_to_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def ask(
    client, model: str, question: str, condition: str, tools: list[dict] | None, call_tool
) -> dict:
    """Run one question to a final answer; returns a full log record (no key inside)."""
    messages = [
        {"role": "system", "content": system_prompt(condition)},
        {"role": "user", "content": question},
    ]
    tool_log, usage, api_models = [], [], []
    final, finish = "", None
    for _ in range(MAX_TOOL_ROUNDS + 1):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            seed=SEED,
        )
        if condition == "tools":
            kwargs.update(tools=tools, tool_choice="auto")
        resp = _with_retries(client.chat.completions.create, **kwargs)
        choice = resp.choices[0]
        api_models.append(getattr(resp, "model", model))
        if getattr(resp, "usage", None) is not None:
            usage.append(
                {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
            )
        msg = choice.message
        finish = choice.finish_reason
        calls = getattr(msg, "tool_calls", None) or []
        if condition != "tools" or not calls:
            final = msg.content or ""
            messages.append(_msg_to_dict(msg))
            break
        messages.append(_msg_to_dict(msg))
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                text, is_error = call_tool(tc.function.name, args)
            except (json.JSONDecodeError, ValueError) as e:
                args, text, is_error = tc.function.arguments, f"invalid arguments: {e}", True
            tool_log.append(
                {"name": tc.function.name, "arguments": args, "result": text, "is_error": is_error}
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    else:
        final = messages[-1].get("content", "") if messages else ""
        finish = "max_tool_rounds"
    return {
        "model": model,
        "api_model": api_models[-1] if api_models else model,
        "condition": condition,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "final_text": final,
        "finish_reason": finish,
        "tool_calls": tool_log,
        "usage": usage,
        "messages": messages,
    }


# ---------------------------------------------------------------- batch runs


def _safe(model: str) -> str:
    return model.replace("/", "_")


def log_path(model: str, condition: str, root: Path = RUNS_DIR) -> Path:
    return root / f"{_safe(model)}__{condition}.jsonl"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if not rec.get("error"):
            ids.add(rec["id"])
    return ids


def check_models(client, models: list[str]) -> None:
    """Stop before any call if a requested model is not available to this key."""
    lister = getattr(getattr(client, "models", None), "list", None)
    if lister is None:  # test doubles
        return
    available = {m.id for m in lister().data}
    missing = [m for m in models if m not in available]
    if missing:
        raise SystemExit(f"not available to this key: {', '.join(missing)}")


def run(
    client,
    models: list[str],
    qs: list[dict],
    root: Path = RUNS_DIR,
    conditions: list[str] = CONDITIONS,
    pause: float = 0.0,
) -> None:
    import asyncio

    from riskcheck import mcp_server

    tools = asyncio.run(mcp_server.tool_schemas())
    check_models(client, models)
    root.mkdir(parents=True, exist_ok=True)
    for model in models:
        for cond in conditions:
            path = log_path(model, cond, root)
            skip = done_ids(path)
            todo = [q for q in qs if q["id"] not in skip]
            print(f"{model} / {cond}: {len(todo)} to run ({len(skip)} done)")
            for q in todo:
                fatal = False
                try:
                    rec = ask(client, model, q["question"], cond, tools, mcp_server.call_tool_sync)
                except Exception as e:  # log and move on; rerun picks it up later
                    rec = {
                        "model": model,
                        "condition": cond,
                        "error": f"{type(e).__name__}: {str(e)[:300]}",
                    }
                    print(f"  {q['id']}: {rec['error']}")
                    fatal = type(e).__name__ in FATAL_ERRORS  # e.g. model not on this key
                rec = {"id": q["id"], "type": q["type"], **rec}
                with open(path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                if fatal:
                    raise SystemExit(f"stopping: {model} is not usable; see {path.name}")
                if pause:
                    time.sleep(pause)


def pilot_questions() -> list[dict]:
    """8 questions (2 per type) from a different seed, never part of the study set."""
    main = {q["question"] for q in qmod.load_key()}
    qs = [q for q in qmod.build(seed=7) if q["question"] not in main]
    out = []
    for t in ("lookup", "attribution", "comparison", "unanswerable"):
        out += [dict(q, id=f"pilot_{q['id']}") for q in qs if q["type"] == t][:2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    for name in ("pilot", "run"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", required=True)
        p.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    args = ap.parse_args()
    client = make_client()
    if args.cmd == "models":
        for m in sorted(client.models.list().data, key=lambda m: m.id):
            print(m.id)
    elif args.cmd == "pilot":
        run(client, args.models, pilot_questions(), root=PILOT_DIR, pause=args.pause)
    else:
        run(client, args.models, qmod.load_key(), root=RUNS_DIR, pause=args.pause)


if __name__ == "__main__":
    main()
