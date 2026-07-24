"""Concentration-field diagnostics for already selected automatic fronts."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nek_post.front_detection_diagnostics import FrontFrameDiagnostic
from nek_post.front_detection_io import (
    ensure_writable_output,
    front_detection_diagnostic_path,
    preflight_output_paths,
)


def save_figure(fig, path: Path, overwrite: bool) -> None:
    """Save and close one diagnostic figure on success or failure."""
    try:
        ensure_writable_output(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def _draw_mask_boundary(
    ax: Any,
    Xi: np.ndarray,
    Zi: np.ndarray,
    mask: np.ndarray,
    **style: Any,
) -> None:
    mask_arr = np.asarray(mask, dtype=bool)
    if not np.any(mask_arr) or np.all(mask_arr):
        return
    ax.contour(
        Xi,
        Zi,
        mask_arr.astype(float),
        levels=[0.5],
        **style,
    )


def _format_value(value: float) -> str:
    return f"{value:.6g}" if np.isfinite(value) else "nan"


def _annotation_text(diagnostic: FrontFrameDiagnostic) -> str:
    tracking_difference = (
        diagnostic.x_front - diagnostic.predicted_x
        if np.isfinite(diagnostic.x_front)
        and np.isfinite(diagnostic.predicted_x)
        else float("nan")
    )
    return "\n".join(
        [
            f"status: {diagnostic.status}",
            f"selected label: {diagnostic.selected_component_label}",
            f"selected pixels: {diagnostic.selected_component_pixels}",
            f"component count: {diagnostic.component_count}",
            f"spatial candidate count: {diagnostic.spatial_candidate_count}",
            f"temporal candidate count: {diagnostic.temporal_candidate_count}",
            f"selected overlap pixels: {diagnostic.selected_overlap_pixels}",
            f"predicted x: {_format_value(diagnostic.predicted_x)}",
            f"automatic x: {_format_value(diagnostic.x_front)}",
            f"tracking difference: {_format_value(tracking_difference)}",
        ]
    )


def plot_front_frame_diagnostic(
    path: str | Path,
    case: str,
    sequence: Any,
    diagnostic: FrontFrameDiagnostic,
    *,
    overwrite: bool,
) -> None:
    """Plot one fixed-grid concentration frame and tracked-component overlays."""
    output_path = Path(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(
        sequence.Xi,
        sequence.Zi,
        diagnostic.concentration,
        shading="auto",
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("concentration C")

    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color="white",
            marker="s",
            markerfacecolor="none",
            markeredgecolor="black",
            label=f"C = {diagnostic.threshold:g} contour",
        )
    ]
    concentration = np.asarray(diagnostic.concentration, dtype=float)
    finite = np.isfinite(concentration)
    active = finite & (concentration > diagnostic.threshold)
    has_crossing = np.any(active) and np.any(finite & ~active)
    if has_crossing:
        ax.contour(
            sequence.Xi,
            sequence.Zi,
            concentration,
            levels=[diagnostic.threshold],
            colors="black",
            linewidths=0.8,
        )

    spatial_labels = {
        component.label for component in diagnostic.spatial_components
    }
    selected_label = (
        diagnostic.selected_component.label
        if diagnostic.selected_component is not None
        else None
    )
    rejected_components = tuple(
        component
        for component in diagnostic.components
        if component.label not in spatial_labels
    )
    other_spatial_components = tuple(
        component
        for component in diagnostic.spatial_components
        if component.label != selected_label
    )

    if rejected_components:
        for component in rejected_components:
            _draw_mask_boundary(
                ax,
                sequence.Xi,
                sequence.Zi,
                component.mask,
                colors="0.45",
                linewidths=0.7,
                linestyles="-",
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.45",
                linewidth=0.7,
                label="rejected threshold components",
            )
        )

    if other_spatial_components:
        for component in other_spatial_components:
            _draw_mask_boundary(
                ax,
                sequence.Xi,
                sequence.Zi,
                component.mask,
                colors="tab:orange",
                linewidths=1.3,
                linestyles="--",
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color="tab:orange",
                linewidth=1.3,
                linestyle="--",
                label="other spatially valid components",
            )
        )

    if diagnostic.selected_component is not None:
        _draw_mask_boundary(
            ax,
            sequence.Xi,
            sequence.Zi,
            diagnostic.selected_component.mask,
            colors="lime",
            linewidths=2.5,
            linestyles="-",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="lime",
                linewidth=2.5,
                label="selected component",
            )
        )

    if np.isfinite(diagnostic.x_front):
        ax.axvline(
            diagnostic.x_front,
            color="red",
            linewidth=2.5,
            linestyle="-",
            label="automatic front",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=2.5,
                label="automatic front",
            )
        )
    if np.isfinite(diagnostic.predicted_x):
        ax.axvline(
            diagnostic.predicted_x,
            color="cyan",
            linewidth=1.3,
            linestyle="--",
            label="predicted front",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="cyan",
                linewidth=1.3,
                linestyle="--",
                label="predicted front",
            )
        )
    if np.isfinite(diagnostic.reference_x):
        ax.axvline(
            diagnostic.reference_x,
            color="magenta",
            linewidth=1.3,
            linestyle=":",
            label="front_simple reference",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="magenta",
                linewidth=1.3,
                linestyle=":",
                label="front_simple reference",
            )
        )

    finite_x = np.asarray(sequence.Xi)[np.isfinite(sequence.Xi)]
    finite_z = np.asarray(sequence.Zi)[np.isfinite(sequence.Zi)]
    ax.set_xlim(float(np.min(finite_x)), float(np.max(finite_x)))
    ax.set_ylim(float(np.min(finite_z)), float(np.max(finite_z)))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        f"{case} front diagnostic — f{diagnostic.file_index:05d}, "
        f"t={diagnostic.time:g}"
    )
    ax.text(
        0.01,
        0.99,
        _annotation_text(diagnostic),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.85,
        },
        zorder=20,
    )
    fig.subplots_adjust(bottom=0.28)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    save_figure(fig, output_path, overwrite)


def write_front_frame_diagnostic_plots(
    output_dir: str | Path,
    case: str,
    sequence: Any,
    diagnostics: Iterable[FrontFrameDiagnostic],
    *,
    overwrite: bool,
) -> list[Path]:
    """Preflight and write diagnostic figures in requested frame order."""
    requested = tuple(diagnostics)
    paths = [
        front_detection_diagnostic_path(
            output_dir,
            case,
            diagnostic.file_index,
        )
        for diagnostic in requested
    ]
    preflight_output_paths(paths, overwrite)
    for path, diagnostic in zip(paths, requested, strict=True):
        plot_front_frame_diagnostic(
            path,
            case,
            sequence,
            diagnostic,
            overwrite=True,
        )
    return paths
