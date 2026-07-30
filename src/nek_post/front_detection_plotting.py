"""Validation plots for automatic concentration-based front detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from nek_post.front_detection import FrontTrackingResult
from nek_post.front_detection_compare import FrontDetectionComparison
from nek_post.front_detection_io import (
    ensure_writable_output,
    front_detection_difference_path,
    front_detection_overlay_path,
    preflight_output_paths,
)


def save_figure(fig, path: Path, overwrite: bool) -> None:
    """Save with established layout and DPI, closing on success or failure."""
    try:
        ensure_writable_output(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def plot_front_detection_overlay(
    path: Path,
    case: str,
    tracking_result: FrontTrackingResult,
    reference_front: Mapping[str, NDArray[np.float64]],
    overwrite: bool,
) -> None:
    """Plot the full reference curve and finite automatic detections."""
    fig, ax = plt.subplots(figsize=(7, 4))
    reference_time = np.asarray(reference_front["time"], dtype=float)
    reference_x = np.abs(np.asarray(reference_front["x_front"], dtype=float))
    successful = np.isfinite(tracking_result.time) & np.isfinite(
        tracking_result.x_front
    )
    ax.plot(reference_time, reference_x, label="front_simple reference")
    ax.plot(
        tracking_result.time[successful],
        np.abs(tracking_result.x_front[successful]),
        linestyle="none",
        marker="o",
        markersize=4,
        label="automatic front",
    )
    ax.set_title(f"Automatic front and front_simple reference: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("front position")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def plot_front_detection_difference(
    path: Path,
    case: str,
    comparison: FrontDetectionComparison,
    overwrite: bool,
) -> None:
    """Plot the signed method-to-method difference on comparison times."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(comparison.time, comparison.difference, label="difference")
    ax.axhline(0.0, linewidth=1.0, label="zero")
    ax.set_title(f"Automatic front difference from front_simple: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("x_auto - x_front_simple")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def write_front_detection_plots(
    output_dir: str | Path,
    case: str,
    tracking_result: FrontTrackingResult,
    reference_front: Mapping[str, NDArray[np.float64]],
    comparison: FrontDetectionComparison,
    overwrite: bool,
) -> list[Path]:
    """Preflight and write overlay then difference figures."""
    if comparison.time.size == 0:
        return []
    paths = [
        front_detection_overlay_path(output_dir, case),
        front_detection_difference_path(output_dir, case),
    ]
    preflight_output_paths(paths, overwrite)
    plot_front_detection_overlay(
        paths[0],
        case,
        tracking_result,
        reference_front,
        overwrite=True,
    )
    plot_front_detection_difference(
        paths[1],
        case,
        comparison,
        overwrite=True,
    )
    return paths
