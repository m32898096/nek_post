"""Headless Figure-4-style plots of spanwise leading-edge evolution."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nek_post.front_detection_io import preflight_output_paths
from nek_post.leading_edge_io import (
    leading_edge_evolution_pdf_path,
    leading_edge_evolution_png_path,
)
from nek_post.leading_edge_workflow import (
    LeadingEdgeEvolution,
    LeadingEdgeTimeSelection,
)


def periodic_leading_edge_plot_arrays(
    y: ArrayLike,
    x_front: ArrayLike,
    y_max_periodic_endpoint: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return plotting copies closed at y_max without filling NaN values."""
    y_array = np.asarray(y, dtype=np.float64)
    x_array = np.asarray(x_front, dtype=np.float64)
    if y_array.ndim != 1 or x_array.ndim != 1 or y_array.shape != x_array.shape:
        raise ValueError("y and x_front must be matching one-dimensional arrays.")
    if y_array.size == 0 or not np.all(np.isfinite(y_array)):
        raise ValueError("y must be a nonempty finite array.")
    if np.any(np.diff(y_array) <= 0.0):
        raise ValueError("y must be strictly increasing.")
    y_max = float(y_max_periodic_endpoint)
    if not np.isfinite(y_max) or y_max <= y_array[-1]:
        raise ValueError(
            "y_max_periodic_endpoint must be finite and greater than the last "
            "computational y coordinate."
        )
    return (
        np.append(y_array, y_max),
        np.append(x_array, x_array[0]),
    )


def _physical_y_extent(
    evolution: LeadingEdgeEvolution,
) -> tuple[float, float]:
    metadata = evolution.horizontal_plan_metadata
    try:
        y_min = float(metadata["ymin"])
        y_max = float(metadata["ymax_periodic_endpoint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Horizontal-plan metadata must contain finite ymin and "
            "ymax_periodic_endpoint values."
        ) from exc
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        raise ValueError(
            "Horizontal-plan metadata must define a finite positive y period."
        )
    return y_min, y_max


def _selected_x_front(
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
) -> NDArray[np.float64]:
    if selection.evolution is not evolution:
        raise ValueError("Time selection must refer to the supplied evolution.")
    selected = np.asarray(selection.x_front, dtype=np.float64)
    expected_shape = (selection.actual_time.size, evolution.y.size)
    if selected.shape != expected_shape or selected.shape[0] == 0:
        raise ValueError(f"selection.x_front must have shape {expected_shape}.")
    if not np.any(np.isfinite(selected)):
        raise ValueError("Cannot plot a selection with no finite leading-edge point.")
    return selected


def _x_limits(values: NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot a selection with no finite leading-edge point.")
    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    span = x_max - x_min
    padding = 0.03 * span
    if padding == 0.0:
        padding = 0.03 * max(abs(x_min), 1.0)
    return x_min - padding, x_max + padding


def write_leading_edge_evolution_plots(
    output_dir: str | Path,
    case: str,
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
    overwrite: bool,
    *,
    reynolds_number: float | int | None = None,
) -> list[Path]:
    """Write one unsmoothed solid x_front(y) curve per selected actual time."""
    paths = [
        leading_edge_evolution_png_path(output_dir, case),
        leading_edge_evolution_pdf_path(output_dir, case),
    ]
    preflight_output_paths(paths, overwrite)

    selected = _selected_x_front(evolution, selection)
    y = np.asarray(evolution.y, dtype=np.float64)
    y_min, y_max = _physical_y_extent(evolution)
    if (
        y.ndim != 1
        or y.size == 0
        or not np.all(np.isfinite(y))
        or np.any(np.diff(y) <= 0.0)
    ):
        raise ValueError("Evolution y coordinates must be finite and increasing.")
    x_limits = _x_limits(selected)

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for x_front in selected:
            y_plot, x_plot = periodic_leading_edge_plot_arrays(
                y,
                x_front,
                y_max,
            )
            ax.plot(x_plot, y_plot, color="black", linewidth=0.9, linestyle="-")

        title_parameters = [
            case,
            f"C={evolution.threshold:.4g}",
            f"z={evolution.z_target:.4g}",
        ]
        if reynolds_number is not None:
            reynolds = float(reynolds_number)
            if not np.isfinite(reynolds):
                raise ValueError("reynolds_number must be finite when supplied.")
            title_parameters.insert(1, f"Re={reynolds:g}")
        ax.set_title("Leading-edge evolution: " + ", ".join(title_parameters))
        ax.set_xlabel("Streamwise position, x")
        ax.set_ylabel("Spanwise position, y")
        ax.set_xlim(*x_limits)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="0.9", linewidth=0.5)
        fig.tight_layout()

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths[0], dpi=200)
        fig.savefig(paths[1])
    finally:
        if fig is not None:
            plt.close(fig)
    return paths


__all__ = (
    "periodic_leading_edge_plot_arrays",
    "write_leading_edge_evolution_plots",
)
