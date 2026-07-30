"""Figures for the N7 Re3450/Re8950 reconstructed x-t workflow."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

from nek_post.front_detection_io import preflight_output_paths
from nek_post.reconstructed_xt_comparison_io import (
    reconstructed_xt_output_paths,
)


LEGEND_LABELS = (
    "Nek5000 Re3450 N7 reconstructed",
    "Cantero Fig. 5a 3D Re3450",
    "Nek5000 Re8950 N7 reconstructed",
    "Cantero Fig. 5a 3D Re8950",
)

SIMULATION_STYLES = {
    "Re3450": {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.2,
        "marker": "None",
        "zorder": 2,
    },
    "Re8950": {
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.2,
        "marker": "None",
        "zorder": 2,
    },
}

PAPER_STYLES = {
    "Re3450": {
        "color": "tab:green",
        "marker": "o",
        "markersize": 2.0,
        "linestyle": "None",
        "zorder": 5,
    },
    "Re8950": {
        "color": "purple",
        "marker": "o",
        "markersize": 2.0,
        "linestyle": "None",
        "zorder": 4,
    },
}


def _save_figure(fig, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def _finite_xy(
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite time/value pairs without changing their values."""
    time_array = np.asarray(time, dtype=float)
    values_array = np.asarray(values, dtype=float)
    mask = np.isfinite(time_array) & np.isfinite(values_array)
    return time_array[mask], values_array[mask]


def _finite_paper_xy(
    reynolds_label: str,
    paper: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paper points or reject a legend-only paper artist."""
    time, values = _finite_xy(paper["time"], paper["paper_x"])
    if time.size == 0:
        raise ValueError(
            f"Cantero {reynolds_label} paper data has zero finite "
            "plotting points."
        )
    return time, values


def plot_pair_overlay(
    path: Path,
    reynolds_label: str,
    reconstructed: Mapping[str, np.ndarray],
    paper: Mapping[str, np.ndarray],
) -> None:
    """Plot one reconstructed simulation and its matching paper points."""
    if reynolds_label not in {"Re3450", "Re8950"}:
        raise ValueError("reynolds_label must be 'Re3450' or 'Re8950'.")
    simulation_label = (
        LEGEND_LABELS[0]
        if reynolds_label == "Re3450"
        else LEGEND_LABELS[2]
    )
    paper_label = (
        LEGEND_LABELS[1]
        if reynolds_label == "Re3450"
        else LEGEND_LABELS[3]
    )
    simulation_time, simulation_x = _finite_xy(
        reconstructed["time"],
        reconstructed["x_reconstructed_relative"],
    )
    paper_time, paper_x = _finite_paper_xy(reynolds_label, paper)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        simulation_time,
        simulation_x,
        label=simulation_label,
        **SIMULATION_STYLES[reynolds_label],
    )
    ax.plot(
        paper_time,
        paper_x,
        label=paper_label,
        **PAPER_STYLES[reynolds_label],
    )
    ax.set_title(
        f"N7 {reynolds_label} reconstructed front and Cantero Figure 5a"
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x_front - x0")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path)


def plot_fourway_overlay(
    path: Path,
    reconstructed_by_reynolds: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    papers_by_reynolds: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Plot both simulations first and both real paper datasets afterward."""
    re3450_time, re3450_x = _finite_xy(
        reconstructed_by_reynolds["Re3450"]["time"],
        reconstructed_by_reynolds["Re3450"][
            "x_reconstructed_relative"
        ],
    )
    re8950_time, re8950_x = _finite_xy(
        reconstructed_by_reynolds["Re8950"]["time"],
        reconstructed_by_reynolds["Re8950"][
            "x_reconstructed_relative"
        ],
    )
    re3450_paper_time, re3450_paper_x = _finite_paper_xy(
        "Re3450", papers_by_reynolds["Re3450"]
    )
    re8950_paper_time, re8950_paper_x = _finite_paper_xy(
        "Re8950", papers_by_reynolds["Re8950"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    re3450_simulation = ax.plot(
        re3450_time,
        re3450_x,
        label=LEGEND_LABELS[0],
        **SIMULATION_STYLES["Re3450"],
    )[0]
    re8950_simulation = ax.plot(
        re8950_time,
        re8950_x,
        label=LEGEND_LABELS[2],
        **SIMULATION_STYLES["Re8950"],
    )[0]
    re3450_paper = ax.plot(
        re3450_paper_time,
        re3450_paper_x,
        label=LEGEND_LABELS[1],
        **PAPER_STYLES["Re3450"],
    )[0]
    re8950_paper = ax.plot(
        re8950_paper_time,
        re8950_paper_x,
        label=LEGEND_LABELS[3],
        **PAPER_STYLES["Re8950"],
    )[0]

    ax.set_title("N7 reconstructed fronts and Cantero Figure 5a")
    ax.set_xlabel("t")
    ax.set_ylabel("x_front - x0")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=(
            re3450_simulation,
            re3450_paper,
            re8950_simulation,
            re8950_paper,
        )
    )
    _save_figure(fig, path)


def write_reconstructed_xt_plots(
    *,
    output_dir: Path,
    reconstructed_by_reynolds: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    papers_by_reynolds: Mapping[str, Mapping[str, np.ndarray]],
    slump_tmin: float,
    slump_tmax: float,
    overwrite: bool,
) -> list[Path]:
    """Write exactly the two pair plots and the combined four-way plot."""
    del slump_tmin, slump_tmax
    _finite_paper_xy("Re3450", papers_by_reynolds["Re3450"])
    _finite_paper_xy("Re8950", papers_by_reynolds["Re8950"])
    paths = list(
        reconstructed_xt_output_paths(
            output_dir, include_plots=True
        ).figures
    )
    preflight_output_paths(paths, overwrite)
    plot_pair_overlay(
        paths[0],
        "Re3450",
        reconstructed_by_reynolds["Re3450"],
        papers_by_reynolds["Re3450"],
    )
    plot_pair_overlay(
        paths[1],
        "Re8950",
        reconstructed_by_reynolds["Re8950"],
        papers_by_reynolds["Re8950"],
    )
    plot_fourway_overlay(
        paths[2],
        reconstructed_by_reynolds,
        papers_by_reynolds,
    )
    return paths
