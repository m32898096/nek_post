"""Automatic spectral-front comparison with Cantero Figure 5a data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from nek_post.front_compare import (
    compare_automatic_front_to_paper,
    finite_mean,
    max_abs,
    mean_abs,
    rms,
    slumping_velocity_metrics,
)
from nek_post.front_detection_io import (
    format_csv_value,
    preflight_output_paths,
)
from nek_post.front_io import (
    read_detected_front_timeseries_csv,
    read_digitized_paper_csv,
)
from nek_post.paths import ProjectPaths


PAPER_DATASET = "Cantero_Fig5a_3D_Re8950"
PAPER_ROLE = "external_published_comparison"
FRONT_METHOD = "automatic_spectral_element"

COMPARISON_COLUMNS = (
    "time",
    "paper_x",
    "automatic_x_interp",
    "difference",
    "absolute_difference",
    "relative_difference",
    "log_difference",
)

SUMMARY_COLUMNS = (
    "case",
    "paper_dataset",
    "paper_role",
    "front_source",
    "front_method",
    "automatic_x0",
    "n_automatic_points",
    "n_paper_points",
    "n_comparison_points",
    "time_min_compared",
    "time_max_compared",
    "mean_signed_difference",
    "mean_absolute_difference",
    "rms_difference",
    "max_absolute_difference",
    "mean_signed_relative_difference",
    "mean_absolute_relative_difference",
    "max_absolute_relative_difference",
    "rms_log_difference",
    "slump_tmin",
    "slump_tmax",
    "n_slumping_points",
    "paper_slumping_velocity",
    "automatic_slumping_velocity",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
)


@dataclass(frozen=True)
class DetectedFrontOverlayOutputs:
    """Deterministic paths produced by one detected-front overlay run."""

    comparison_csv: Path
    summary_csv: Path
    figures: tuple[Path, ...]


def default_detected_front_csv(paths: ProjectPaths, case: str) -> Path:
    """Return the standard automatic spectral-front CSV for a case."""
    return (
        paths.front_detection_dir
        / f"{case}_spectral"
        / f"{case}_detected_front_timeseries.csv"
    )


def default_detected_front_overlay_dir(paths: ProjectPaths, case: str) -> Path:
    """Return the standard Re8950 comparison output directory."""
    return paths.fig5a_paper_overlay_dir / f"{case}_Re8950"


def detected_front_overlay_output_paths(
    output_dir: str | Path,
    case: str,
    *,
    include_plots: bool = True,
) -> DetectedFrontOverlayOutputs:
    """Build every deterministic CSV and optional figure path."""
    directory = Path(output_dir)
    stem = f"{case}_Re8950_fig5a"
    figures = (
        directory / f"{stem}_overlay_linear.png",
        directory / f"{stem}_overlay_loglog.png",
        directory / f"{stem}_difference.png",
        directory / f"{stem}_slumping_overlay.png",
    )
    return DetectedFrontOverlayOutputs(
        comparison_csv=directory / f"{stem}_comparison.csv",
        summary_csv=directory / f"{stem}_summary.csv",
        figures=figures if include_plots else (),
    )


def automatic_front_relative_to_initial(
    automatic_front: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Express an automatic front as displacement from its earliest point."""
    time = np.asarray(automatic_front["time"], dtype=np.float64)
    x_front_auto = np.asarray(
        automatic_front["x_front_auto"], dtype=np.float64
    )
    status = np.asarray(automatic_front["status"], dtype=str)
    automatic_x0 = float(x_front_auto[0])
    return {
        "time": time,
        "x_front_auto": x_front_auto,
        "status": status,
        "x_relative": x_front_auto - automatic_x0,
        "automatic_x0": automatic_x0,
    }


def build_detected_front_summary(
    *,
    case: str,
    front_source: str | Path,
    automatic_front: Mapping[str, np.ndarray | float],
    paper: Mapping[str, np.ndarray],
    comparison: Mapping[str, np.ndarray],
    slump_tmin: float,
    slump_tmax: float,
) -> dict[str, str | int | float]:
    """Build the one-row Re8950 detected-front comparison summary."""
    difference = np.asarray(comparison["difference"], dtype=float)
    relative_difference_values = np.asarray(
        comparison["relative_difference"], dtype=float
    )
    log_difference = np.asarray(comparison["log_difference"], dtype=float)
    time = np.asarray(comparison["time"], dtype=float)
    slumping = slumping_velocity_metrics(
        dict(comparison),
        compared_x_key="automatic_x_interp",
        compared_label="automatic",
        tmin=slump_tmin,
        tmax=slump_tmax,
    )
    return {
        "case": case,
        "paper_dataset": PAPER_DATASET,
        "paper_role": PAPER_ROLE,
        "front_source": str(front_source),
        "front_method": FRONT_METHOD,
        "automatic_x0": float(automatic_front["automatic_x0"]),
        "n_automatic_points": int(
            np.asarray(automatic_front["time"]).size
        ),
        "n_paper_points": int(np.asarray(paper["time"]).size),
        "n_comparison_points": int(time.size),
        "time_min_compared": float(time[0]),
        "time_max_compared": float(time[-1]),
        "mean_signed_difference": finite_mean(difference),
        "mean_absolute_difference": mean_abs(
            difference, finite_only=True
        ),
        "rms_difference": rms(difference, finite_only=True),
        "max_absolute_difference": max_abs(
            difference, finite_only=True
        ),
        "mean_signed_relative_difference": finite_mean(
            relative_difference_values
        ),
        "mean_absolute_relative_difference": mean_abs(
            relative_difference_values, finite_only=True
        ),
        "max_absolute_relative_difference": max_abs(
            relative_difference_values, finite_only=True
        ),
        "rms_log_difference": rms(log_difference, finite_only=True),
        "slump_tmin": slump_tmin,
        "slump_tmax": slump_tmax,
        **slumping,
    }


def _write_comparison_csv(
    path: Path,
    comparison: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for index in range(np.asarray(comparison["time"]).size):
            writer.writerow(
                {
                    column: format_csv_value(comparison[column][index])
                    for column in COMPARISON_COLUMNS
                }
            )


def _write_summary_csv(
    path: Path,
    summary: Mapping[str, str | int | float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                column: format_csv_value(summary[column])
                for column in SUMMARY_COLUMNS
            }
        )


def run_detected_front_overlay(
    *,
    case: str,
    front_csv: str | Path,
    paper_csv: str | Path,
    output_dir: str | Path,
    slump_tmin: float = 3.0,
    slump_tmax: float = 12.0,
    overwrite: bool = False,
    no_plots: bool = False,
) -> DetectedFrontOverlayOutputs:
    """Run the CSV comparison and optional plotting workflow."""
    if (
        not np.isfinite(slump_tmin)
        or not np.isfinite(slump_tmax)
        or slump_tmin > slump_tmax
    ):
        raise ValueError(
            "Slumping bounds must be finite with slump_tmin <= slump_tmax."
        )

    front_path = Path(front_csv)
    paper_path = Path(paper_csv)
    automatic_absolute = read_detected_front_timeseries_csv(front_path)
    automatic_relative = automatic_front_relative_to_initial(
        automatic_absolute
    )
    paper = read_digitized_paper_csv(paper_path, require_positive=False)
    comparison = compare_automatic_front_to_paper(
        automatic_relative, paper
    )
    summary = build_detected_front_summary(
        case=case,
        front_source=front_path,
        automatic_front=automatic_relative,
        paper=paper,
        comparison=comparison,
        slump_tmin=slump_tmin,
        slump_tmax=slump_tmax,
    )
    outputs = detected_front_overlay_output_paths(
        output_dir, case, include_plots=not no_plots
    )
    preflight_output_paths(
        (
            outputs.comparison_csv,
            outputs.summary_csv,
            *outputs.figures,
        ),
        overwrite,
    )
    _write_comparison_csv(outputs.comparison_csv, comparison)
    _write_summary_csv(outputs.summary_csv, summary)

    if not no_plots:
        from nek_post.fig5a_detected_front_plotting import (
            write_detected_front_overlay_plots,
        )

        written_figures = write_detected_front_overlay_plots(
            output_dir=Path(output_dir),
            case=case,
            automatic_front=automatic_relative,
            paper=paper,
            comparison=comparison,
            slump_tmin=slump_tmin,
            slump_tmax=slump_tmax,
            overwrite=overwrite,
        )
        if tuple(written_figures) != outputs.figures:
            raise RuntimeError("Detected-front plot paths did not match preflight.")

    return outputs
