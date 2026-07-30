"""Plots for the automatic spectral-front Figure 5a comparison."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

from nek_post.fig5a_detected_front import (
    detected_front_overlay_output_paths,
)
from nek_post.front_detection_io import preflight_output_paths


def _save_figure(fig, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def _plot_overlay(
    path: Path,
    case: str,
    automatic_front: Mapping[str, np.ndarray | float],
    paper: Mapping[str, np.ndarray],
    *,
    loglog: bool,
    slump_tmin: float | None = None,
    slump_tmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    paper_time = np.asarray(paper["time"], dtype=float)
    paper_x = np.asarray(paper["paper_x"], dtype=float)
    automatic_time = np.asarray(automatic_front["time"], dtype=float)
    automatic_x = np.asarray(
        automatic_front["x_relative"], dtype=float
    )

    paper_mask = np.isfinite(paper_time) & np.isfinite(paper_x)
    automatic_mask = np.isfinite(automatic_time) & np.isfinite(automatic_x)
    if loglog:
        paper_mask &= (paper_time > 0.0) & (paper_x > 0.0)
        automatic_mask &= (automatic_time > 0.0) & (automatic_x > 0.0)
    if slump_tmin is not None and slump_tmax is not None:
        paper_mask &= (paper_time >= slump_tmin) & (
            paper_time <= slump_tmax
        )
        automatic_mask &= (automatic_time >= slump_tmin) & (
            automatic_time <= slump_tmax
        )

    ax.plot(
        paper_time[paper_mask],
        paper_x[paper_mask],
        marker="o",
        linestyle="None",
        label="Cantero Figure 5a 3D Re8950",
    )
    ax.plot(
        automatic_time[automatic_mask],
        automatic_x[automatic_mask],
        linestyle="-",
        label=f"{case} automatic spectral front",
    )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    title = f"{case} automatic spectral front and Cantero Re8950"
    if loglog:
        title += " (log-log)"
    if slump_tmin is not None:
        title += " slumping interval"
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x_front - x0")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    _save_figure(fig, path)


def _plot_difference(
    path: Path,
    case: str,
    comparison: Mapping[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        comparison["time"],
        comparison["difference"],
        label="difference",
    )
    ax.axhline(0.0, linewidth=1.0, label="zero")
    ax.set_title(
        f"Automatic spectral front - Cantero Re8950: {case}"
    )
    ax.set_xlabel("t")
    ax.set_ylabel("automatic spectral front - Cantero Re8950")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path)


def write_detected_front_overlay_plots(
    *,
    output_dir: Path,
    case: str,
    automatic_front: Mapping[str, np.ndarray | float],
    paper: Mapping[str, np.ndarray],
    comparison: Mapping[str, np.ndarray],
    slump_tmin: float,
    slump_tmax: float,
    overwrite: bool,
) -> list[Path]:
    """Write all four Re8950 comparison figures in deterministic order."""
    paths = list(
        detected_front_overlay_output_paths(
            output_dir, case, include_plots=True
        ).figures
    )
    preflight_output_paths(paths, overwrite)
    _plot_overlay(
        paths[0],
        case,
        automatic_front,
        paper,
        loglog=False,
    )
    _plot_overlay(
        paths[1],
        case,
        automatic_front,
        paper,
        loglog=True,
    )
    _plot_difference(paths[2], case, comparison)
    _plot_overlay(
        paths[3],
        case,
        automatic_front,
        paper,
        loglog=False,
        slump_tmin=slump_tmin,
        slump_tmax=slump_tmax,
    )
    return paths
