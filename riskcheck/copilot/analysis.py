"""Judge the unanswerable set, validate the judge, and produce the study's tables and chart.

python -m riskcheck.copilot.analysis judge                   # cross-family LLM judge
python -m riskcheck.copilot.analysis label                   # writes labels_to_fill.csv
python -m riskcheck.copilot.analysis report                  # tables, tests, chart
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

from riskcheck import SEED, plotting
from riskcheck.copilot import grading
from riskcheck.copilot import questions as qmod
from riskcheck.copilot.runner import RUNS_DIR, _with_retries, make_client
from riskcheck.portfolios import REPORTS_DIR

OUT = REPORTS_DIR / "copilot"
JUDGE_FILE = OUT / "judge_verdicts.jsonl"
LABELS_TODO = OUT / "labels_to_fill.csv"
LABELS_DONE = OUT / "labels.csv"
ALPHA = 0.05
TYPES = ["lookup", "attribution", "comparison", "unanswerable"]
# Each answer is judged by a model from a different family than the one that wrote it.
JUDGES = {"openai": "qwen/qwen3.8-27b", "qwen": "openai/gpt-oss-120b"}


def family(model: str) -> str:
    return model.split("/")[0] if "/" in model else model.split("-")[0]


def judge_for(model: str) -> str:
    fam = family(model)
    if fam not in JUDGES:
        raise ValueError(f"no cross-family judge defined for {model}")
    return JUDGES[fam]


def run_judge(judge_model: str | None = None) -> None:
    qs = {q["id"]: q for q in qmod.load_key()}
    recs = [r for r in grading.load_runs(RUNS_DIR) if qs[r["id"]]["type"] == "unanswerable"]
    done = set()
    if JUDGE_FILE.exists():
        for line in JUDGE_FILE.read_text().splitlines():
            j = json.loads(line)
            done.add((j["id"], j["model"], j["condition"]))
    client = make_client()
    for r in recs:
        key = (r["id"], r["model"], r["condition"])
        if key in done:
            continue
        prompt = grading.judge_prompt(qs[r["id"]], r)
        jm = judge_model or judge_for(r["model"])
        resp = _with_retries(
            client.chat.completions.create,
            model=jm,
            temperature=0.0,
            seed=SEED,
            max_completion_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        out = {
            "id": r["id"],
            "model": r["model"],
            "condition": r["condition"],
            "judge_model": jm,
            "verdict": grading.parse_verdict(text),
            "raw": text,
        }
        with open(JUDGE_FILE, "a") as f:
            f.write(json.dumps(out) + "\n")


def load_judge() -> pd.DataFrame:
    if not JUDGE_FILE.exists():
        return pd.DataFrame(columns=["id", "model", "condition", "judge_model", "verdict"])
    rows = [json.loads(line) for line in JUDGE_FILE.read_text().splitlines()]
    return pd.DataFrame(rows)[["id", "model", "condition", "judge_model", "verdict"]]


KAPPA_MIN = 0.6


def load_labels() -> pd.DataFrame:
    if not LABELS_DONE.exists():
        return pd.DataFrame(columns=["id", "model", "condition", "human_label"])
    lab = pd.read_csv(LABELS_DONE)
    lab["human_label"] = lab["human_label"].astype(str).str.strip().str.upper()
    lab = lab[lab["human_label"].isin(["CORRECT", "INCORRECT"])]
    return lab[["id", "model", "condition", "human_label"]]


def judge_validation() -> pd.DataFrame:
    """Agreement and Cohen's kappa between each judge and the hand labels."""
    m = load_labels().merge(load_judge(), on=["id", "model", "condition"], how="inner")
    rows = []
    for judge, g in m.groupby("judge_model"):
        agree = float((g["human_label"] == g["verdict"]).mean())
        kappa = float(cohen_kappa_score(g["human_label"], g["verdict"]))
        rows.append(
            {
                "judge_model": judge,
                "n_labelled": len(g),
                "agreement": agree,
                "cohen_kappa": kappa,
                "reliable": kappa >= KAPPA_MIN,
            }
        )
    return pd.DataFrame(rows)


def graded() -> pd.DataFrame:
    """Graded answers. Unanswerable: judge verdict; hand label replaces it for a judge whose
    kappa against the hand labels is below 0.6 (labelled items only)."""
    qs = qmod.load_key()
    g = grading.grade_all(qs, grading.load_runs(RUNS_DIR))
    m = g["type"] == "unanswerable"
    g["label_source"] = np.where(m, "rule", "program")
    g["rule_correct"] = g["correct"]
    j = load_judge()
    if not j.empty:
        g = g.merge(j, on=["id", "model", "condition"], how="left")
        use = m & g["verdict"].notna()
        g.loc[use, "correct"] = g.loc[use, "verdict"].eq("CORRECT")
        g.loc[use, "label_source"] = "judge"
        lab = load_labels()
        if not lab.empty:
            val = judge_validation()
            weak = set(val.loc[~val["reliable"], "judge_model"])
            g = g.merge(lab, on=["id", "model", "condition"], how="left")
            swap = m & g["human_label"].notna() & g["judge_model"].isin(weak)
            g.loc[swap, "correct"] = g.loc[swap, "human_label"].eq("CORRECT")
            g.loc[swap, "label_source"] = "human"
    return g


def mcnemar_paired(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Exact McNemar on paired booleans: (a only, b only, p-value)."""
    a, b = a.astype(bool), b.astype(bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    table = [[int((a & b).sum()), n01], [n10, int((~a & ~b).sum())]]
    return n01, n10, float(mcnemar(table, exact=True).pvalue)


def hypothesis_tests(g: pd.DataFrame) -> pd.DataFrame:
    """Paired McNemar tests per model, no tools vs tools, Bonferroni across the 3 models.

    Only the PRIMARY row is confirmatory; the others are exploratory (see prereg.md).
    """
    models = sorted(g["model"].unique())
    k = len(models)
    specs = [
        (
            "PRIMARY: wrong-number rate (lookup + unanswerable)",
            "wrong_number",
            g["type"].isin(["lookup", "unanswerable"]),
        ),
        ("Exploratory: accuracy (all 200)", "correct", g["type"].notna()),
        ("Exploratory: correct declines (unanswerable)", "correct", g["type"] == "unanswerable"),
    ]
    rows = []
    for name, col, mask in specs:
        for m in models:
            s = g[mask & (g["model"] == m)].pivot(index="id", columns="condition", values=col)
            s = s.dropna()
            if s.empty or not {"no_tools", "tools"} <= set(s.columns):
                continue
            only_no, only_tools, p = mcnemar_paired(s["no_tools"], s["tools"])
            rows.append(
                {
                    "hypothesis": name,
                    "model": m,
                    "n_pairs": len(s),
                    "rate_no_tools": float(s["no_tools"].mean()),
                    "rate_tools": float(s["tools"].mean()),
                    "only_no_tools": only_no,
                    "only_tools": only_tools,
                    "p_value": p,
                    "p_bonferroni": min(1.0, p * k),
                    "significant": p * k < ALPHA,
                }
            )
    return pd.DataFrame(rows)


def results_table(g: pd.DataFrame) -> pd.DataFrame:
    agg = g.groupby(["model", "condition", "type"]).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        wrong_number_rate=("wrong_number", "mean"),
        unsupported_claim_rate=("unsupported_claim", "mean"),
        false_confidence_rate=("false_confidence", "mean"),
        mean_tool_calls=("n_tool_calls", "mean"),
        format_fallbacks=("answer_read_as", lambda x: int((x == "last_line_fallback").sum())),
        missing_answers=("answer_read_as", lambda x: int((x == "missing").sum())),
    )
    return agg.reset_index()


def baselines() -> pd.DataFrame:
    """Accuracy of always giving the most common answer, per question subtype."""
    qs = pd.DataFrame(qmod.load_key())
    rows = []
    for t in ("attribution", "comparison"):
        s = qs[qs["type"] == t]
        if t == "attribution":
            for kind, ss in s.groupby(s["meta"].map(lambda m: m["kind"])):
                vc = ss["answer"].value_counts()
                rows.append(
                    {
                        "subtype": f"attribution/{kind}",
                        "majority_answer": vc.index[0],
                        "majority_accuracy": vc.iloc[0] / len(ss),
                    }
                )
        else:
            vc = s["answer"].value_counts()
            rows.append(
                {
                    "subtype": t,
                    "majority_answer": vc.index[0],
                    "majority_accuracy": vc.iloc[0] / len(s),
                }
            )
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path) -> None:
    plotting.style()
    models = sorted(table["model"].unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 4), sharey=True, squeeze=False
    )
    x = np.arange(len(TYPES))
    for ax, m in zip(axes[0], models, strict=True):
        for i, (cond, color) in enumerate(
            (("no_tools", plotting.SERIES[1]), ("tools", plotting.SERIES[0]))
        ):
            t = table[(table["model"] == m) & (table["condition"] == cond)].set_index("type")
            vals = [t["accuracy"].get(ty, np.nan) * 100 for ty in TYPES]
            ax.bar(
                x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=cond.replace("_", " ")
            )
        ax.set_xticks(x, TYPES, rotation=20)
        ax.set_title(m, loc="left")
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("accuracy, %")
    axes[0][0].legend(loc="upper left")
    fig.suptitle(
        "AI risk copilots: accuracy without vs with engine tools",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=plotting.TEXT,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--judge-model", default=None, help="override; not used in the study")
    sub.add_parser("label")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "judge":
        run_judge(args.judge_model)
    elif args.cmd == "label":
        df = grading.labeling_sample(qmod.load_key(), grading.load_runs(RUNS_DIR), judge_for)
        df.to_csv(LABELS_TODO, index=False)
        print(
            f"wrote {len(df)} rows to {LABELS_TODO}; fill human_label with CORRECT or "
            f"INCORRECT and save as {LABELS_DONE}"
        )
    else:
        g = graded()
        g.to_csv(OUT / "graded_answers.csv", index=False)
        table = results_table(g)
        table.to_csv(OUT / "results_table.csv", index=False)
        tests = hypothesis_tests(g)
        tests.to_csv(OUT / "hypothesis_tests.csv", index=False)
        baselines().to_csv(OUT / "baselines.csv", index=False)
        plot(table, OUT / "copilot_accuracy.png")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
            print(table.to_string(index=False))
            print(tests.to_string(index=False))
        if LABELS_DONE.exists():
            val = judge_validation()
            val.to_csv(OUT / "judge_validation.csv", index=False)
            print(val.to_string(index=False))


if __name__ == "__main__":
    main()
