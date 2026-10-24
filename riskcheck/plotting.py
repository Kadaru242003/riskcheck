"""Shared chart style: one categorical order, recessive axes, thin marks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (fixed; never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BREACH = "#e34948"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
MODEL_COLORS = dict(zip(["historical", "ewma", "garch", "factor"], SERIES, strict=False))
ZONE_COLORS = {"green": "#1baf7a", "yellow": "#eda100", "red": "#e34948"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "font.size": 9,
        }
    )
