"""Regime drift detection: can a detector warn that a risk model is breaking?

All rules and settings here are fixed in prereg.md before any detector is scored.

Signals per portfolio and model:
- z_t = r_t / vol_t, the standardized forecast error (N(0,1)-like if the model is right)
- 95% VaR breaches
- the VIX level (market-wide)

Detectors (each resets and then stays silent for ``COOLDOWN`` days after an alarm):
- CUSUM on z^2 - 1 (one-sided, upward), threshold calibrated to ARL0 = 252 on iid N(0,1)
- Page-Hinkley on z^2 against its own running mean, threshold calibrated the same way
- Breach count: >= 4 breaches of 95% VaR in the trailing 20 days
- VIX baseline: VIX closes above 25 after closing at or below 25 the day before
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskcheck import SEED, plotting
from riskcheck.models import ewma, historical
from riskcheck.portfolios import REPORTS_DIR

# ---------------------------------------------------------------- pre-registered settings
REGIME_VIX_LEVEL = 30.0
REGIME_CALM_MEAN = 20.0
REGIME_LOOKBACK = 60
REGIME_MIN_GAP = 60
WINDOW_BEFORE = 5  # an alarm up to 5 days before a start counts as a detection
WINDOW_AFTER = 40  # ... and up to 40 trading days after
CALM_VIX = 20.0
COOLDOWN = 20
ARL0 = 252
CUSUM_K = 0.5
PH_DELTA = 0.1
BREACH_WINDOW = 20
BREACH_COUNT = 4
VIX_ALARM = 25.0
EVAL_START = pd.Timestamp("2008-01-01")
EVAL_END = pd.Timestamp("2025-12-31")
# Thresholds from calibrate() (iid N(0,1), seed 42); recorded in prereg.md.
CUSUM_H = 9.52
PH_LAMBDA = 15.26
SIM_RUNS = 1000
SIM_LENGTH = 750
SIM_WARMUP = 250
SIM_JUMP = 2.0
SIM_BLOCK = 20


# ---------------------------------------------------------------- regime starts


def regime_starts(vix: pd.Series) -> pd.DatetimeIndex:
    """First close above 30 while the prior 60-day mean VIX is below 20 (60-day min gap)."""
    prev = vix.shift(1)
    calm = vix.rolling(REGIME_LOOKBACK).mean().shift(1) < REGIME_CALM_MEAN
    hits = vix.index[calm & (vix > REGIME_VIX_LEVEL) & (prev <= REGIME_VIX_LEVEL)]
    out: list[pd.Timestamp] = []
    pos = {d: i for i, d in enumerate(vix.index)}
    for d in hits:
        if not out or pos[d] - pos[out[-1]] >= REGIME_MIN_GAP:
            out.append(d)
    return pd.DatetimeIndex(out)


# ---------------------------------------------------------------- detectors


def _run_with_cooldown(step, x: np.ndarray, cooldown: int = COOLDOWN) -> np.ndarray:
    """Generic loop: ``step(state, value) -> (state, fired)``; resets state after alarms."""
    alarms = np.zeros(len(x), dtype=bool)
    state, quiet = None, 0
    for i, xi in enumerate(x):
        if quiet > 0:
            quiet -= 1
            continue
        if not np.isfinite(xi):
            continue
        state, fired = step(state, xi)
        if fired:
            alarms[i] = True
            state, quiet = None, cooldown
    return alarms


def cusum(z: np.ndarray, h: float = CUSUM_H, k: float = CUSUM_K, cooldown: int = COOLDOWN):
    def step(s, zi):
        s = max(0.0, (s or 0.0) + zi * zi - 1.0 - k)
        return s, s > h

    return _run_with_cooldown(step, np.asarray(z, dtype=float), cooldown)


def page_hinkley(
    z: np.ndarray, lam: float = PH_LAMBDA, delta: float = PH_DELTA, cooldown: int = COOLDOWN
):
    """Page-Hinkley on x = z^2 versus its running mean since the last reset."""

    def step(st, zi):
        x = zi * zi
        n, mean, m, mmin = st or (0, 0.0, 0.0, 0.0)
        n += 1
        mean += (x - mean) / n
        m += x - mean - delta
        mmin = min(mmin, m)
        return (n, mean, m, mmin), (m - mmin) > lam

    return _run_with_cooldown(step, np.asarray(z, dtype=float), cooldown)


def breach_count(breaches: np.ndarray, cooldown: int = COOLDOWN) -> np.ndarray:
    b = np.asarray(breaches, dtype=float)

    def step(window, bi):
        window = ((window or []) + [bi])[-BREACH_WINDOW:]
        return window, sum(window) >= BREACH_COUNT

    return _run_with_cooldown(step, b, cooldown)


def vix_threshold(vix: np.ndarray, level: float = VIX_ALARM, cooldown: int = COOLDOWN):
    def step(prev, vi):
        fired = prev is not None and prev <= level < vi
        return vi, fired

    return _run_with_cooldown(step, np.asarray(vix, dtype=float), cooldown)


def calibrate(target: int = ARL0, runs: int = 400, seed: int = SEED) -> dict[str, float]:
    """Pick CUSUM h and PH lambda so the mean time to a false alarm on iid N(0,1) is ~252 days.

    Uses simulated data only, never the real returns.
    """
    rng = np.random.default_rng(seed)
    n = 5000

    def arl(fn, thr):
        times = []
        for _ in range(runs):
            z = rng.standard_normal(n)
            a = np.flatnonzero(fn(z, thr, cooldown=0))
            times.append(a[0] + 1 if len(a) else n)
        return float(np.mean(times))

    out = {}
    for name, fn, lo, hi in (("cusum_h", cusum, 1.0, 40.0), ("ph_lambda", page_hinkley, 1.0, 60.0)):
        for _ in range(18):  # bisection on the threshold
            mid = (lo + hi) / 2
            if arl(fn, mid) < target:
                lo = mid
            else:
                hi = mid
        out[name] = round((lo + hi) / 2, 2)
    return out


# ---------------------------------------------------------------- scoring


@dataclass
class Score:
    detected: int
    events: int
    delays: list[int]
    false_alarms: int
    calm_years: float
    true_alarms: int

    @property
    def detection_rate(self) -> float:
        return self.detected / self.events if self.events else np.nan

    @property
    def median_delay(self) -> float:
        return float(np.median(self.delays)) if self.delays else np.nan

    @property
    def fa_per_year(self) -> float:
        return self.false_alarms / self.calm_years if self.calm_years else np.nan

    @property
    def precision(self) -> float:
        tot = self.true_alarms + self.false_alarms
        return self.true_alarms / tot if tot else np.nan


def event_windows(dates: pd.DatetimeIndex, starts: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask of days within [start - 5, start + 40] trading days of any start."""
    mask = np.zeros(len(dates), dtype=bool)
    for s in starts:
        i = dates.get_indexer([s])[0]
        if i >= 0:
            mask[max(0, i - WINDOW_BEFORE) : i + WINDOW_AFTER + 1] = True
    return mask


def score(alarms: pd.Series, starts: pd.DatetimeIndex, vix: pd.Series) -> Score:
    dates = alarms.index
    a = alarms.to_numpy(dtype=bool)
    in_window = event_windows(dates, starts)
    calm = (vix.reindex(dates).to_numpy() < CALM_VIX) & ~in_window
    delays = []
    for s in starts:
        i = dates.get_indexer([s])[0]
        if i < 0:
            continue
        hits = np.flatnonzero(a[max(0, i - WINDOW_BEFORE) : i + WINDOW_AFTER + 1])
        if len(hits):
            delays.append(int(hits[0] + max(0, i - WINDOW_BEFORE) - i))
    n_events = int(sum(dates.get_indexer([s])[0] >= 0 for s in starts))
    return Score(
        detected=len(delays),
        events=n_events,
        delays=delays,
        false_alarms=int((a & calm).sum()),
        calm_years=calm.sum() / 252,
        true_alarms=int((a & in_window).sum()),
    )


def all_alarms(g: pd.DataFrame, vix: pd.Series) -> dict[str, pd.Series]:
    """Run every detector on one portfolio-model forecast series (indexed by date)."""
    z = (g["actual"] / g["vol"]).to_numpy()
    b = (-g["actual"] > g["var95"]).to_numpy()
    idx = g.index
    return {
        "cusum": pd.Series(cusum(z), idx),
        "page_hinkley": pd.Series(page_hinkley(z), idx),
        "breach_count": pd.Series(breach_count(b), idx),
        "vix_threshold": pd.Series(vix_threshold(vix.reindex(idx).to_numpy()), idx),
    }


def real_study(fc: pd.DataFrame, vix: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per portfolio-model-detector scores, and a pooled table per model and detector."""
    starts = regime_starts(vix)
    starts = starts[(starts >= EVAL_START) & (starts <= EVAL_END)]
    w = fc[(fc["date"] >= EVAL_START) & (fc["date"] <= EVAL_END)]
    rows, pooled = [], {}
    for (p, m), g in w.groupby(["portfolio", "model"], sort=False):
        for det, al in all_alarms(g.set_index("date"), vix).items():
            sc = score(al, starts, vix)
            rows.append(
                {
                    "portfolio": p,
                    "model": m,
                    "detector": det,
                    "detected": sc.detected,
                    "events": sc.events,
                    "median_delay": sc.median_delay,
                    "false_alarms_per_year": sc.fa_per_year,
                    "precision": sc.precision,
                    "delays": ",".join(map(str, sc.delays)),
                }
            )
            pooled.setdefault((m, det), []).append(sc)
    prow = []
    for (m, det), scs in pooled.items():
        delays = [d for s in scs for d in s.delays]
        det_n, ev_n = sum(s.detected for s in scs), sum(s.events for s in scs)
        fa, yrs = sum(s.false_alarms for s in scs), sum(s.calm_years for s in scs)
        ta = sum(s.true_alarms for s in scs)
        prow.append(
            {
                "model": m,
                "detector": det,
                "detection_rate": det_n / ev_n,
                "median_delay_days": float(np.median(delays)) if delays else np.nan,
                "false_alarms_per_year": fa / yrs,
                "precision": ta / (ta + fa) if ta + fa else np.nan,
                "events_scored": ev_n,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(prow)


def exploratory_share(fc: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """EXPLORATORY (added after seeing results; see prereg amendments).

    Share of *all* alarms that fall in a regime-start window, plus alarms per year. Unlike
    the pre-registered false-alarm count, this treats the VIX baseline like the others.
    """
    starts = regime_starts(vix)
    starts = starts[(starts >= EVAL_START) & (starts <= EVAL_END)]
    w = fc[(fc["date"] >= EVAL_START) & (fc["date"] <= EVAL_END)]
    rows = []
    for (_, m), g in w.groupby(["portfolio", "model"], sort=False):
        g = g.set_index("date")
        win = event_windows(g.index, starts)
        for det, al in all_alarms(g, vix).items():
            a = al.to_numpy()
            rows.append((m, det, int((a & win).sum()), int(a.sum()), a.sum() / (len(a) / 252)))
    r = pd.DataFrame(rows, columns=["model", "detector", "in_window", "total", "per_year"])
    out = r.groupby(["model", "detector"], sort=False).agg(
        alarms_in_windows=("in_window", "sum"),
        alarms_total=("total", "sum"),
        alarms_per_year=("per_year", "mean"),
    )
    out["share_in_windows"] = out["alarms_in_windows"] / out["alarms_total"]
    return out.reset_index()


# ---------------------------------------------------------------- simulation


def block_bootstrap(x: np.ndarray, n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, len(x) - block, size=n // block + 1)
    return np.concatenate([x[s : s + block] for s in starts])[:n]


def simulate(
    port_returns: pd.DataFrame, vix: pd.Series, runs: int = SIM_RUNS, seed: int = SEED
) -> pd.DataFrame:
    """Inject a volatility jump into resampled calm-period returns; score detectors.

    Each run: pick a portfolio, block-bootstrap 750 days from its calm-day returns
    (VIX < 20), multiply returns after a random jump day by 2, forecast with historical
    and EWMA, run the residual detectors after the 250-day warm-up.
    """
    rng = np.random.default_rng(seed)
    calm_days = vix.index[vix < CALM_VIX]
    pools = {p: port_returns[p].reindex(calm_days).dropna().to_numpy() for p in port_returns}
    names = list(pools)
    idx = pd.bdate_range("2000-01-03", periods=SIM_LENGTH)
    rows = []
    for run in range(runs):
        p = names[rng.integers(len(names))]
        r = block_bootstrap(pools[p], SIM_LENGTH, SIM_BLOCK, rng)
        jump = int(rng.integers(SIM_WARMUP + 150, SIM_LENGTH - 100))
        r[jump:] *= SIM_JUMP
        s = pd.Series(r, index=idx)
        for model, fn in (("historical", historical.forecast), ("ewma", ewma.forecast)):
            fc = fn(s)
            z = (s.reindex(fc.index) / fc["vol"]).to_numpy()
            b = (-s.reindex(fc.index) > fc["var95"]).to_numpy()
            j = jump - SIM_WARMUP  # jump position within the forecast index
            for det, a in (
                ("cusum", cusum(z)),
                ("page_hinkley", page_hinkley(z)),
                ("breach_count", breach_count(b)),
            ):
                pre = a[:j]
                post = np.flatnonzero(a[j : j + WINDOW_AFTER + 1])
                rows.append(
                    {
                        "run": run,
                        "portfolio": p,
                        "model": model,
                        "detector": det,
                        "detected": len(post) > 0,
                        "delay": int(post[0]) if len(post) else np.nan,
                        "false_alarms_per_year": pre.sum() / (j / 252),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- chart and main


def plot_alarms(fc: pd.DataFrame, vix: pd.Series, portfolio: str, model: str, path) -> None:
    plotting.style()
    starts = regime_starts(vix)
    g = fc[(fc["portfolio"] == portfolio) & (fc["model"] == model)]
    g = g[(g["date"] >= EVAL_START) & (g["date"] <= EVAL_END)].set_index("date")
    alarms = all_alarms(g, vix)
    v = vix.loc[g.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(v.index, v, color="#8a8984", linewidth=0.8, label="VIX")
    for s in starts[(starts >= EVAL_START) & (starts <= EVAL_END)]:
        ax.axvline(s, color=plotting.TEXT, linewidth=0.8, linestyle="--")
    ax.plot([], [], color=plotting.TEXT, linewidth=0.8, linestyle="--", label="regime start")
    top = float(v.max())
    for i, (det, a) in enumerate(alarms.items()):
        d = a.index[a.to_numpy()]
        ax.scatter(
            d,
            np.full(len(d), top * (1.08 + 0.07 * i)),
            marker="v",
            s=22,
            color=plotting.SERIES[i],
            label=det.replace("_", " "),
        )
    ax.set_ylabel("VIX")
    ax.set_title(
        f"VIX, regime starts, and detector alarms ({portfolio}, {model} model)", loc="left"
    )
    ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    from riskcheck.backtest import load_forecasts
    from riskcheck.data import load

    vix = load().market["vix"]
    fc = load_forecasts()
    detail, pooled = real_study(fc, vix)
    detail.to_csv(REPORTS_DIR / "drift_detail.csv", index=False)
    pooled.to_csv(REPORTS_DIR / "drift_results.csv", index=False)
    exploratory_share(fc, vix).to_csv(REPORTS_DIR / "drift_exploratory.csv", index=False)
    port = pd.read_parquet(REPORTS_DIR / "portfolio_returns.parquet")
    sim = simulate(port, vix)
    sim_sum = (
        sim.groupby(["model", "detector"])
        .agg(
            power=("detected", "mean"),
            median_delay_days=("delay", "median"),
            false_alarms_per_year=("false_alarms_per_year", "mean"),
        )
        .reset_index()
    )
    sim_sum.to_csv(REPORTS_DIR / "drift_simulation.csv", index=False)
    plot_alarms(fc, vix, "equal_sector", "historical", REPORTS_DIR / "drift_alarms.png")
    starts = regime_starts(vix)
    print("regime starts:", [str(d.date()) for d in starts])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(pooled.to_string(index=False))
        print("\nsimulation:")
        print(sim_sum.to_string(index=False))


if __name__ == "__main__":
    main()
