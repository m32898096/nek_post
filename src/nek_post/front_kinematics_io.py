"""Output paths and CSV writing for front-kinematics analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from numbers import Integral
from pathlib import Path

import numpy as np


TIMESERIES_COLUMNS = (
    "time",
    "x_front",
    "v_raw",
    "v_smooth",
    "x_reconstructed",
    "x_reconstruction_error",
)
SUMMARY_COLUMNS = (
    "case",
    "n_points",
    "time_start",
    "time_end",
    "x_start",
    "x_end",
    "v_raw_min",
    "v_raw_max",
    "v_smooth_min",
    "v_smooth_max",
    "mean_abs_reconstruction_error",
    "max_abs_reconstruction_error",
    "rms_reconstruction_error",
    "final_reconstruction_error",
    "slumping_velocity_raw_position_fit",
    "slumping_velocity_reconstructed_position_fit",
)


def front_kinematics_timeseries_path(output_dir: Path, case: str) -> Path:
    return output_dir / f"{case}_front_kinematics_timeseries.csv"


def front_kinematics_summary_path(output_dir: Path) -> Path:
    return output_dir / "front_kinematics_summary.csv"


def front_position_figure_path(output_dir: Path) -> Path:
    return output_dir / "front_position_xt_vs_time.png"


def raw_velocity_figure_path(output_dir: Path) -> Path:
    return output_dir / "front_velocity_raw_vs_time.png"


def smoothed_velocity_figure_path(output_dir: Path) -> Path:
    return output_dir / "front_velocity_smoothed_vs_time.png"


def reconstruction_error_figure_path(output_dir: Path) -> Path:
    return output_dir / "front_reconstruction_error_vs_time.png"


def reconstruction_figure_path(output_dir: Path, case: str) -> Path:
    return output_dir / f"front_position_reconstructed_vs_original_{case}.png"


def ensure_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def format_numeric_value(value: object) -> str:
    """Format numbers using the established output precision."""
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(value)
    return f"{float(value):.16g}"


def write_front_kinematics_timeseries_csv(
    path: Path,
    kinematics: Mapping[str, np.ndarray],
    overwrite: bool,
) -> None:
    ensure_writable_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_COLUMNS)
        writer.writeheader()
        for index in range(kinematics["time"].size):
            writer.writerow(
                {
                    column: format_numeric_value(float(kinematics[column][index]))
                    for column in TIMESERIES_COLUMNS
                }
            )


def write_front_kinematics_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    overwrite: bool,
) -> None:
    ensure_writable_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_numeric_value(row[column]) for column in SUMMARY_COLUMNS})
