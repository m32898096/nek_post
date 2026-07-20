"""Summary metrics and deterministic reporting for front kinematics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from nek_post.front_compare import max_abs, mean_abs, rms, slumping_region_linear_fit
from nek_post.front_kinematics_io import format_numeric_value


SLUMPING_TMIN = 3.0
SLUMPING_TMAX = 12.0
SUMMARY_TABLE_COLUMNS = (
    "case",
    "n_points",
    "time_start",
    "time_end",
    "x_start",
    "x_end",
    "mean_abs_reconstruction_error",
    "max_abs_reconstruction_error",
    "rms_reconstruction_error",
    "final_reconstruction_error",
    "slumping_velocity_raw_position_fit",
    "slumping_velocity_reconstructed_position_fit",
)


def slumping_velocity(time: np.ndarray, values: np.ndarray) -> float:
    """Return the fitted position slope over the established slumping interval."""
    _n_points, slope = slumping_region_linear_fit(time, values, SLUMPING_TMIN, SLUMPING_TMAX)
    return slope


def build_front_kinematics_summary_row(
    case: str,
    kinematics: Mapping[str, np.ndarray],
) -> dict[str, str]:
    """Build one formatted summary row for a front-kinematics case."""
    time = kinematics["time"]
    x_front = kinematics["x_front"]
    v_raw = kinematics["v_raw"]
    v_smooth = kinematics["v_smooth"]
    x_reconstructed = kinematics["x_reconstructed"]
    error = kinematics["x_reconstruction_error"]
    values: dict[str, object] = {
        "case": case,
        "n_points": int(time.size),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "x_start": float(x_front[0]),
        "x_end": float(x_front[-1]),
        "v_raw_min": float(np.min(v_raw)),
        "v_raw_max": float(np.max(v_raw)),
        "v_smooth_min": float(np.min(v_smooth)),
        "v_smooth_max": float(np.max(v_smooth)),
        "mean_abs_reconstruction_error": mean_abs(error),
        "max_abs_reconstruction_error": max_abs(error),
        "rms_reconstruction_error": rms(error),
        "final_reconstruction_error": float(error[-1]),
        "slumping_velocity_raw_position_fit": slumping_velocity(time, x_front),
        "slumping_velocity_reconstructed_position_fit": slumping_velocity(time, x_reconstructed),
    }
    return {key: format_numeric_value(value) for key, value in values.items()}


def format_front_kinematics_summary_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Format the established terminal summary without printing it."""
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in SUMMARY_TABLE_COLUMNS
    }
    lines = [
        "Front kinematics summary:",
        "  ".join(column.ljust(widths[column]) for column in SUMMARY_TABLE_COLUMNS),
        "  ".join("-" * widths[column] for column in SUMMARY_TABLE_COLUMNS),
    ]
    lines.extend(
        "  ".join(row[column].ljust(widths[column]) for column in SUMMARY_TABLE_COLUMNS)
        for row in rows
    )
    return "\n".join(lines)
