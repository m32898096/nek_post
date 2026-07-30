"""Comparison helpers for front-position Figure 5a workflows."""

from __future__ import annotations

import numpy as np


def finite_mean(values: np.ndarray, *, finite_only: bool = True) -> float:
    """Return mean value, optionally ignoring non-finite entries."""
    values_arr = np.asarray(values, dtype=float)
    if finite_only:
        values_arr = values_arr[np.isfinite(values_arr)]
        if values_arr.size == 0:
            return float("nan")
    return float(np.mean(values_arr))


def mean_abs(values: np.ndarray, *, finite_only: bool = False) -> float:
    """Return mean absolute value."""
    values_arr = np.asarray(values, dtype=float)
    if finite_only:
        values_arr = values_arr[np.isfinite(values_arr)]
        if values_arr.size == 0:
            return float("nan")
    return float(np.mean(np.abs(values_arr)))


def max_abs(values: np.ndarray, *, finite_only: bool = False) -> float:
    """Return maximum absolute value."""
    values_arr = np.asarray(values, dtype=float)
    if finite_only:
        values_arr = values_arr[np.isfinite(values_arr)]
        if values_arr.size == 0:
            return float("nan")
    return float(np.max(np.abs(values_arr)))


def rms(values: np.ndarray, *, finite_only: bool = False) -> float:
    """Return root-mean-square value."""
    values_arr = np.asarray(values, dtype=float)
    if finite_only:
        values_arr = values_arr[np.isfinite(values_arr)]
        if values_arr.size == 0:
            return float("nan")
    return float(np.sqrt(np.mean(values_arr**2)))


def linear_fit_slope(time: np.ndarray, values: np.ndarray) -> float:
    """Return the slope from a linear fit."""
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if time_arr.size < 2:
        return float("nan")
    slope, _intercept = np.polyfit(time_arr, values_arr, deg=1)
    return float(slope)


def slumping_region_linear_fit(time: np.ndarray, values: np.ndarray, tmin: float = 3.0, tmax: float = 12.0) -> tuple[int, float]:
    """Fit a line over the slumping-region time window."""
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    mask = (time_arr >= tmin) & (time_arr <= tmax)
    n_points = int(np.count_nonzero(mask))
    if n_points < 2:
        return n_points, float("nan")
    return n_points, linear_fit_slope(time_arr[mask], values_arr[mask])


def relative_difference(value: float, reference: float) -> float:
    """Return relative difference against a reference value."""
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float((value - reference) / reference)


def interpolate_to_paper_times(
    source_time: np.ndarray,
    source_x: np.ndarray,
    paper_time: np.ndarray,
    paper_x: np.ndarray,
    *,
    interpolated_key: str,
    error_key: str,
) -> dict[str, np.ndarray]:
    """Interpolate source front data onto overlapping paper time points."""
    source_time_arr = np.asarray(source_time, dtype=float)
    source_x_arr = np.asarray(source_x, dtype=float)
    paper_time_arr = np.asarray(paper_time, dtype=float)
    paper_x_arr = np.asarray(paper_x, dtype=float)

    overlap = (paper_time_arr >= source_time_arr[0]) & (paper_time_arr <= source_time_arr[-1])
    if not np.any(overlap):
        raise ValueError("Simulation and paper time ranges do not overlap.")

    time = paper_time_arr[overlap]
    paper_x_overlap = paper_x_arr[overlap]
    source_interp = np.interp(time, source_time_arr, source_x_arr)
    error = source_interp - paper_x_overlap
    relative_error = np.full_like(error, np.nan, dtype=float)
    np.divide(
        error,
        paper_x_overlap,
        out=relative_error,
        where=paper_x_overlap != 0.0,
    )
    positive = (source_interp > 0.0) & (paper_x_overlap > 0.0)
    log_error = np.full_like(error, np.nan, dtype=float)
    log_error[positive] = np.log(source_interp[positive]) - np.log(paper_x_overlap[positive])
    return {
        "time": time,
        "paper_x": paper_x_overlap,
        interpolated_key: source_interp,
        error_key: error,
        "relative_error": relative_error,
        "log_error": log_error,
    }


def compare_front_to_paper(
    front: dict[str, np.ndarray],
    paper: dict[str, np.ndarray],
    *,
    front_x_key: str,
    interpolated_key: str,
    error_key: str,
    no_overlap_message: str = "Simulation and paper time ranges do not overlap.",
) -> dict[str, np.ndarray]:
    """Compare a front dictionary to digitized paper data."""
    try:
        return interpolate_to_paper_times(
            front["time"],
            front[front_x_key],
            paper["time"],
            paper["x"] if "x" in paper else paper["paper_x"],
            interpolated_key=interpolated_key,
            error_key=error_key,
        )
    except ValueError as exc:
        if str(exc) == "Simulation and paper time ranges do not overlap.":
            raise ValueError(no_overlap_message) from exc
        raise


def compare_automatic_front_to_paper(
    automatic_front: dict[str, np.ndarray],
    paper: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compare relative automatic-front positions at overlapping paper times."""
    comparison = compare_front_to_paper(
        automatic_front,
        paper,
        front_x_key="x_relative",
        interpolated_key="automatic_x_interp",
        error_key="difference",
        no_overlap_message="Automatic-front and paper time ranges do not overlap.",
    )
    difference = comparison["difference"]
    return {
        "time": comparison["time"],
        "paper_x": comparison["paper_x"],
        "automatic_x_interp": comparison["automatic_x_interp"],
        "difference": difference,
        "absolute_difference": np.abs(difference),
        "relative_difference": comparison["relative_error"],
        "log_difference": comparison["log_error"],
    }


def slumping_velocity_metrics(
    comparison: dict[str, np.ndarray],
    *,
    compared_x_key: str,
    compared_label: str,
    tmin: float = 3.0,
    tmax: float = 12.0,
    include_relative_error_stats: bool = False,
) -> dict[str, float | int]:
    """Return slumping-region velocity metrics for paper and compared fronts."""
    time = comparison["time"]
    paper_x = comparison["paper_x"]
    compared_x = comparison[compared_x_key]
    mask = (time >= tmin) & (time <= tmax)
    n_points = int(np.count_nonzero(mask))
    compared_velocity_key = f"{compared_label}_slumping_velocity"
    base: dict[str, float | int] = {
        "n_slumping_points": n_points,
        compared_velocity_key: float("nan"),
        "paper_slumping_velocity": float("nan"),
        "slumping_velocity_difference": float("nan"),
        "slumping_velocity_relative_difference": float("nan"),
    }
    if include_relative_error_stats:
        base["slumping_mean_abs_relative_error"] = float("nan")
        base["slumping_max_abs_relative_error"] = float("nan")
    if n_points < 2:
        return base

    paper_velocity = linear_fit_slope(time[mask], paper_x[mask])
    compared_velocity = linear_fit_slope(time[mask], compared_x[mask])
    base.update(
        {
            compared_velocity_key: compared_velocity,
            "paper_slumping_velocity": paper_velocity,
            "slumping_velocity_difference": compared_velocity - paper_velocity,
            "slumping_velocity_relative_difference": relative_difference(compared_velocity, paper_velocity),
        }
    )
    if include_relative_error_stats:
        relative_error = comparison["relative_error"]
        base["slumping_mean_abs_relative_error"] = mean_abs(relative_error[mask], finite_only=True)
        base["slumping_max_abs_relative_error"] = max_abs(relative_error[mask], finite_only=True)
    return base
