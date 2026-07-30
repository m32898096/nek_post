"""I/O helpers for front-position and Figure 5a workflows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nek_post.front_detection import (
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
)


SUCCESSFUL_AUTOMATIC_FRONT_STATUSES = frozenset(
    {STATUS_SELECTED_INITIAL, STATUS_SELECTED_TRACKED}
)


def parse_case_labels(raw: str) -> list[str]:
    """Parse comma-separated case labels such as ``N5,N7,N9``."""
    cases = [case.strip().upper() for case in raw.split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must include at least one case.")
    for case in cases:
        if not case.startswith("N") or not case[1:].isdigit():
            raise ValueError(f"Invalid case {case!r}; expected labels such as N5, N7, N9.")
    return cases


def front_simple_path(data_root: Path, case: str, filename: str = "front_simple.dat") -> Path:
    """Return the standard front-position path for a case."""
    return Path(data_root) / f"case_{case}" / filename


def processed_front_kinematics_path(processed_dir: Path, case: str) -> Path:
    """Return the processed front-kinematics timeseries CSV path for a case."""
    return Path(processed_dir) / f"{case}_front_kinematics_timeseries.csv"


def validate_finite_values(path: Path, **arrays: np.ndarray) -> None:
    """Raise a clear error if any named array contains non-finite values."""
    nonfinite = [name for name, values in arrays.items() if not np.all(np.isfinite(values))]
    if nonfinite:
        names = ", ".join(nonfinite)
        raise ValueError(f"{path} contains non-finite value(s) in: {names}.")


def sort_by_time(path: Path, time: np.ndarray, **values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sort arrays by time and reject duplicate time values after sorting."""
    time_arr = np.asarray(time, dtype=float)
    order = np.argsort(time_arr)
    sorted_time = time_arr[order]
    sorted_values = {name: np.asarray(value, dtype=float)[order] for name, value in values.items()}

    if np.any(np.diff(sorted_time) <= 0.0):
        duplicate_times = sorted_time[np.concatenate(([False], np.diff(sorted_time) == 0.0))]
        if duplicate_times.size:
            duplicates = ", ".join(f"{value:.16g}" for value in np.unique(duplicate_times))
            raise ValueError(f"{path} contains duplicate time value(s) after sorting: {duplicates}.")
        raise ValueError(f"{path} contains non-increasing time values after sorting.")
    return sorted_time, sorted_values


def read_front_simple_dat(path: Path) -> dict[str, np.ndarray]:
    """Read a two-column ``front_simple.dat`` file as absolute front position."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Front-position file not found: {path}")

    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: time and x_front.")

    time, values = sort_by_time(path, data[:, 0], x_front=data[:, 1])
    x_front = values["x_front"]
    if time.size < 2:
        raise ValueError(f"{path} must contain at least two time samples.")
    validate_finite_values(path, time=time, x_front=x_front)
    return {"time": time, "x_front": x_front}


def front_relative_to_initial(front: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return front data shifted by its initial front position."""
    x_front = np.asarray(front["x_front"], dtype=float)
    x0 = float(x_front[0])
    return {
        "time": np.asarray(front["time"], dtype=float),
        "x_front_minus_x0": x_front - x0,
        "x0": np.array([x0], dtype=float),
    }


def read_digitized_paper_csv(path: Path, *, require_positive: bool = True) -> dict[str, np.ndarray]:
    """Read digitized Cantero Figure 5a CSV data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Paper CSV not found: {path}")

    rows: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            try:
                time = float(row[0])
                x_value = float(row[1])
            except ValueError:
                if line_number == 1:
                    continue
                raise ValueError(f"{path}:{line_number} has non-numeric first or second column: {row}") from None
            if np.isfinite(time) and np.isfinite(x_value):
                rows.append((time, x_value))

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two finite paper data rows.")

    data = np.asarray(rows, dtype=float)
    time, values = sort_by_time(path, data[:, 0], x=data[:, 1])
    x_value = values["x"]
    positive = (time > 0.0) & (x_value > 0.0)
    if require_positive and not np.any(positive):
        raise ValueError(f"{path} has no finite positive rows for log-log plotting.")
    return {"time": time, "x": x_value, "paper_x": x_value, "positive_mask": positive}


def read_detected_front_timeseries_csv(path: Path) -> dict[str, np.ndarray]:
    """Read successful automatic detections from a front timeseries CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detected-front CSV not found: {path}")

    retained: list[tuple[float, float, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "x_front_auto", "status"}
        available = set(reader.fieldnames or [])
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            status = (row["status"] or "").strip()
            if status not in SUCCESSFUL_AUTOMATIC_FRONT_STATUSES:
                continue
            try:
                time = float(row["time"])
                x_front_auto = float(row["x_front_auto"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}:{line_number} has non-numeric time or x_front_auto."
                ) from None

            if not np.isfinite(time) or not np.isfinite(x_front_auto):
                raise ValueError(
                    f"{path}:{line_number} has non-finite time or x_front_auto "
                    f"for successful status {status!r}."
                )
            retained.append((time, x_front_auto, status))

    if len(retained) < 2:
        raise ValueError(
            f"{path} must contain at least two successful automatic-front points."
        )

    time = np.asarray([row[0] for row in retained], dtype=np.float64)
    x_front_auto = np.asarray([row[1] for row in retained], dtype=np.float64)
    status = np.asarray([row[2] for row in retained], dtype=str)
    order = np.argsort(time, kind="stable")
    time = time[order]
    x_front_auto = x_front_auto[order]
    status = status[order]
    if np.any(np.diff(time) <= 0.0):
        duplicate_times = time[
            np.concatenate(([False], np.diff(time) == 0.0))
        ]
        if duplicate_times.size:
            duplicates = ", ".join(
                f"{value:.16g}" for value in np.unique(duplicate_times)
            )
            raise ValueError(
                f"{path} contains duplicate retained time value(s): {duplicates}."
            )
        raise ValueError(f"{path} contains non-increasing retained time values.")

    return {
        "time": time,
        "x_front_auto": x_front_auto,
        "status": status,
    }


def read_processed_front_kinematics_csv(path: Path) -> dict[str, np.ndarray]:
    """Read processed front kinematics CSV data produced by script 13."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed front kinematics CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "x_reconstructed"}
        available = set(reader.fieldnames or [])
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        rows = [(float(row["time"]), float(row["x_reconstructed"])) for row in reader]

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two processed samples.")
    data = np.asarray(rows, dtype=float)
    time, values = sort_by_time(path, data[:, 0], x_reconstructed=data[:, 1])
    x_reconstructed = values["x_reconstructed"]
    validate_finite_values(path, time=time, x_reconstructed=x_reconstructed)

    x0 = float(x_reconstructed[0])
    return {
        "time": time,
        "x_reconstructed": x_reconstructed,
        "x_processed": x_reconstructed - x0,
        "processed_x0_shift": np.array([x0], dtype=float),
    }
