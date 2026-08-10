"""Cantero mean-front extraction from span-averaged equivalent height."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.cantero_equivalent_height import (
    CanteroEquivalentHeightPlan,
    apply_cantero_equivalent_height_plan,
    build_cantero_equivalent_height_plan,
)
from nek_post.io_nek import get_nek_time, read_nek_file


DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD = 0.01
DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X = 0.0
STATUS_SUCCESS = "success"
STATUS_NO_DOWNWARD_CROSSING = "no_downward_crossing"
STATUS_REFERENCE_BELOW_THRESHOLD = "reference_below_threshold"

CANTERO_MEAN_FRONT_COLUMNS = (
    "case",
    "file_index",
    "source_file",
    "time",
    "x_front",
    "x_front_minus_initial",
    "threshold",
    "reference_x",
    "left_index",
    "right_index",
    "x_left",
    "x_right",
    "h_left",
    "h_right",
    "crossing_count_in_search_region",
    "status",
)


class CanteroMeanFrontDetectionError(ValueError):
    """Raised when a profile cannot provide the specified Cantero front."""

    def __init__(self, message: str, *, status: str) -> None:
        super().__init__(message)
        self.status = status


def _readonly(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    return parsed


def _positive_threshold(value: object) -> float:
    threshold = _finite_float(value, "threshold")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    return threshold


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _validate_profile(
    x_coordinates: object,
    span_averaged_height: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if np.iscomplexobj(x_coordinates) or np.iscomplexobj(span_averaged_height):
        raise ValueError("x_coordinates and span_averaged_height must be real-valued.")
    try:
        x = np.asarray(x_coordinates, dtype=np.float64)
        height = np.asarray(span_averaged_height, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x_coordinates and span_averaged_height must be numeric.") from exc
    if x.ndim != 1 or height.ndim != 1:
        raise ValueError("x_coordinates and span_averaged_height must be one-dimensional.")
    if x.size != height.size:
        raise ValueError("x_coordinates and span_averaged_height must have identical lengths.")
    if x.size < 2:
        raise ValueError("x_coordinates and span_averaged_height need at least two points.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(height)):
        raise ValueError("x_coordinates and span_averaged_height must be finite.")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x_coordinates must be strictly increasing.")
    return x, height


@dataclass(frozen=True)
class CanteroMeanFrontDetection:
    """Auditable first downstream threshold crossing of one h_bar(x) profile."""

    x_front: float
    threshold: float
    reference_x: float
    left_index: int
    right_index: int
    x_left: float
    x_right: float
    h_left: float
    h_right: float
    crossing_count_in_search_region: int


def detect_cantero_mean_front(
    x_coordinates: object,
    span_averaged_height: object,
    *,
    threshold: object = DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD,
    reference_x: object = DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X,
) -> CanteroMeanFrontDetection:
    """Find the first positive-x downward h_bar threshold crossing.

    The search starts at the physical GLL node nearest the explicit interior
    reference, then proceeds monotonically toward increasing physical x.  An
    ``np.argmin`` tie resolves to the first (lower-x) node because x is
    increasing.  The selected node itself must satisfy ``h_bar >= threshold``;
    h_bar is not interpolated at ``reference_x``.  This prevents a disconnected
    downstream overshoot from replacing the advancing front.
    """
    x, height = _validate_profile(x_coordinates, span_averaged_height)
    threshold_value = _positive_threshold(threshold)
    reference_value = _finite_float(reference_x, "reference_x")
    if reference_value < x[0] or reference_value > x[-1]:
        raise CanteroMeanFrontDetectionError(
            "reference_x is outside the physical x domain "
            f"[{x[0]:.16g}, {x[-1]:.16g}].",
            status="invalid_reference_x",
        )
    start = int(np.argmin(np.abs(x - reference_value)))
    if height[start] < threshold_value:
        raise CanteroMeanFrontDetectionError(
            "span_averaged_height at the reference search node is below "
            f"threshold: h_bar={height[start]:.16g}, threshold={threshold_value:.16g}.",
            status=STATUS_REFERENCE_BELOW_THRESHOLD,
        )

    crossings = [
        index
        for index in range(start, x.size - 1)
        if height[index] >= threshold_value and height[index + 1] < threshold_value
    ]
    if not crossings:
        raise CanteroMeanFrontDetectionError(
            "No downward Cantero mean-front threshold crossing was found "
            "toward increasing x from the reference location.",
            status=STATUS_NO_DOWNWARD_CROSSING,
        )
    left_index = crossings[0]
    right_index = left_index + 1
    x_left = float(x[left_index])
    x_right = float(x[right_index])
    h_left = float(height[left_index])
    h_right = float(height[right_index])
    if h_left == threshold_value:
        x_front = x_left
    else:
        fraction = (threshold_value - h_left) / (h_right - h_left)
        x_front = x_left + fraction * (x_right - x_left)
    if not x_left <= x_front <= x_right:
        raise RuntimeError("Cantero mean-front interpolation left its threshold bracket.")
    return CanteroMeanFrontDetection(
        x_front=float(x_front),
        threshold=threshold_value,
        reference_x=reference_value,
        left_index=left_index,
        right_index=right_index,
        x_left=x_left,
        x_right=x_right,
        h_left=h_left,
        h_right=h_right,
        crossing_count_in_search_region=len(crossings),
    )


@dataclass(frozen=True)
class CanteroMeanFrontTimeseries:
    """Per-frame absolute Cantero front positions and crossing diagnostics."""

    case: str
    file_index: NDArray[np.int64]
    source_file: tuple[str, ...]
    time: NDArray[np.float64]
    x_front: NDArray[np.float64]
    x_front_minus_initial: NDArray[np.float64]
    threshold: float
    reference_x: float
    left_index: NDArray[np.int64]
    right_index: NDArray[np.int64]
    x_left: NDArray[np.float64]
    x_right: NDArray[np.float64]
    h_left: NDArray[np.float64]
    h_right: NDArray[np.float64]
    crossing_count_in_search_region: NDArray[np.int64]
    status: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, str) or not self.case.strip():
            raise ValueError("case must be a non-empty string.")
        threshold = _positive_threshold(self.threshold)
        reference_x = _finite_float(self.reference_x, "reference_x")
        file_index = _readonly(self.file_index, np.int64)
        time = _readonly(self.time, np.float64)
        x_front = _readonly(self.x_front, np.float64)
        relative = _readonly(self.x_front_minus_initial, np.float64)
        left_index = _readonly(self.left_index, np.int64)
        right_index = _readonly(self.right_index, np.int64)
        x_left = _readonly(self.x_left, np.float64)
        x_right = _readonly(self.x_right, np.float64)
        h_left = _readonly(self.h_left, np.float64)
        h_right = _readonly(self.h_right, np.float64)
        crossing_count = _readonly(self.crossing_count_in_search_region, np.int64)
        count = time.size
        arrays = (
            file_index,
            x_front,
            relative,
            left_index,
            right_index,
            x_left,
            x_right,
            h_left,
            h_right,
            crossing_count,
        )
        if count == 0 or any(values.shape != (count,) for values in arrays):
            raise ValueError("Cantero mean-front timeseries arrays must be nonempty vectors of one length.")
        if len(self.source_file) != count or len(self.status) != count:
            raise ValueError("source_file and status must match timeseries length.")
        if np.any(file_index < 0) or np.any(np.diff(file_index) <= 0):
            raise ValueError("file_index values must be strictly increasing non-negative integers.")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
            raise ValueError("time values must be finite and strictly increasing.")
        successful = np.asarray(
            [value == STATUS_SUCCESS for value in self.status], dtype=bool
        )
        if not np.any(successful):
            raise ValueError("Cantero mean-front timeseries has no successful frames.")
        if (
            not np.all(np.isfinite(x_front[successful]))
            or not np.all(np.isfinite(relative[successful]))
            or not np.all(np.isfinite(x_left[successful]))
            or not np.all(np.isfinite(x_right[successful]))
            or not np.all(np.isfinite(h_left[successful]))
            or not np.all(np.isfinite(h_right[successful]))
            or np.any(left_index[successful] < 0)
            or np.any(right_index[successful] != left_index[successful] + 1)
            or np.any(crossing_count[successful] < 1)
        ):
            raise ValueError("Successful Cantero mean-front rows must have finite crossing diagnostics.")
        if (
            np.any(np.isfinite(x_front[~successful]))
            or np.any(np.isfinite(relative[~successful]))
            or np.any(left_index[~successful] != -1)
            or np.any(right_index[~successful] != -1)
        ):
            raise ValueError("Failed Cantero mean-front rows must not contain a front location.")
        initial_front = float(x_front[np.flatnonzero(successful)[0]])
        if not np.allclose(relative[successful], x_front[successful] - initial_front):
            raise ValueError("x_front_minus_initial must use the first successful x_front.")
        object.__setattr__(self, "case", self.case.strip())
        object.__setattr__(self, "file_index", file_index)
        object.__setattr__(self, "source_file", tuple(str(value) for value in self.source_file))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "x_front", x_front)
        object.__setattr__(self, "x_front_minus_initial", relative)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "reference_x", reference_x)
        object.__setattr__(self, "left_index", left_index)
        object.__setattr__(self, "right_index", right_index)
        object.__setattr__(self, "x_left", x_left)
        object.__setattr__(self, "x_right", x_right)
        object.__setattr__(self, "h_left", h_left)
        object.__setattr__(self, "h_right", h_right)
        object.__setattr__(self, "crossing_count_in_search_region", crossing_count)
        object.__setattr__(self, "status", tuple(str(value) for value in self.status))

    @property
    def successful_mask(self) -> NDArray[np.bool_]:
        result = np.asarray(
            [value == STATUS_SUCCESS for value in self.status], dtype=np.bool_
        )
        result.setflags(write=False)
        return result


def _frame_values(frame: object) -> tuple[int, Path]:
    try:
        index = _nonnegative_integer(getattr(frame, "index"), "frame index")
        path = Path(getattr(frame, "path"))
    except AttributeError as exc:
        raise ValueError("Each frame must provide index and path attributes.") from exc
    return index, path


def compute_cantero_mean_front_timeseries(
    frames: Sequence[object],
    *,
    case: str,
    threshold: object = DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD,
    reference_x: object = DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X,
    reader: Callable[[str | Path], Any] = read_nek_file,
) -> CanteroMeanFrontTimeseries:
    """Process ordered snapshots with one reusable equivalent-height plan."""
    if not isinstance(case, str) or not case.strip():
        raise ValueError("case must be a non-empty string.")
    if not callable(reader):
        raise ValueError("reader must be callable.")
    threshold_value = _positive_threshold(threshold)
    reference_value = _finite_float(reference_x, "reference_x")
    resolved_frames = tuple(_frame_values(frame) for frame in frames)
    if not resolved_frames:
        raise ValueError("At least one Nek frame is required.")
    file_indices = np.asarray([frame[0] for frame in resolved_frames], dtype=np.int64)
    if np.any(np.diff(file_indices) <= 0):
        raise ValueError("Frame indices must be unique and strictly increasing.")

    first_index, first_path = resolved_frames[0]
    first_data = reader(first_path)
    plan = build_cantero_equivalent_height_plan(first_data)
    records: list[dict[str, object]] = []
    previous_time: float | None = None
    for frame_position, (file_index, source_path) in enumerate(resolved_frames):
        data = first_data if frame_position == 0 else reader(source_path)
        time = _finite_float(get_nek_time(data), f"Nek time for frame {file_index}")
        if previous_time is not None and time <= previous_time:
            raise ValueError("Nek frame times must be strictly increasing in file-index order.")
        previous_time = time
        equivalent_height = apply_cantero_equivalent_height_plan(
            plan, data, source_file=source_path
        )
        try:
            detection = detect_cantero_mean_front(
                equivalent_height.x_coordinates,
                equivalent_height.span_averaged_height,
                threshold=threshold_value,
                reference_x=reference_value,
            )
        except CanteroMeanFrontDetectionError as exc:
            records.append(
                {
                    "file_index": file_index,
                    "source_file": str(source_path),
                    "time": time,
                    "x_front": np.nan,
                    "left_index": -1,
                    "right_index": -1,
                    "x_left": np.nan,
                    "x_right": np.nan,
                    "h_left": np.nan,
                    "h_right": np.nan,
                    "crossing_count": 0,
                    "status": exc.status,
                }
            )
            continue
        records.append(
            {
                "file_index": file_index,
                "source_file": str(source_path),
                "time": time,
                "x_front": detection.x_front,
                "left_index": detection.left_index,
                "right_index": detection.right_index,
                "x_left": detection.x_left,
                "x_right": detection.x_right,
                "h_left": detection.h_left,
                "h_right": detection.h_right,
                "crossing_count": detection.crossing_count_in_search_region,
                "status": STATUS_SUCCESS,
            }
        )
    successful_positions = [
        position
        for position, record in enumerate(records)
        if record["status"] == STATUS_SUCCESS
    ]
    if not successful_positions:
        raise ValueError("No Cantero mean-front threshold crossing succeeded in any frame.")
    initial_front = float(records[successful_positions[0]]["x_front"])
    relative = np.asarray(
        [
            float(record["x_front"]) - initial_front
            if record["status"] == STATUS_SUCCESS
            else np.nan
            for record in records
        ],
        dtype=np.float64,
    )
    return CanteroMeanFrontTimeseries(
        case=case,
        file_index=np.asarray([record["file_index"] for record in records], dtype=np.int64),
        source_file=tuple(str(record["source_file"]) for record in records),
        time=np.asarray([record["time"] for record in records], dtype=np.float64),
        x_front=np.asarray([record["x_front"] for record in records], dtype=np.float64),
        x_front_minus_initial=relative,
        threshold=threshold_value,
        reference_x=reference_value,
        left_index=np.asarray([record["left_index"] for record in records], dtype=np.int64),
        right_index=np.asarray([record["right_index"] for record in records], dtype=np.int64),
        x_left=np.asarray([record["x_left"] for record in records], dtype=np.float64),
        x_right=np.asarray([record["x_right"] for record in records], dtype=np.float64),
        h_left=np.asarray([record["h_left"] for record in records], dtype=np.float64),
        h_right=np.asarray([record["h_right"] for record in records], dtype=np.float64),
        crossing_count_in_search_region=np.asarray(
            [record["crossing_count"] for record in records], dtype=np.int64
        ),
        status=tuple(str(record["status"]) for record in records),
    )


def cantero_mean_front_timeseries_path(
    output_dir: str | Path,
    case: str,
) -> Path:
    """Return the deterministic Cantero mean-front CSV path."""
    if not isinstance(case, str) or not case.strip():
        raise ValueError("case must be a non-empty string.")
    return Path(output_dir) / case.strip() / f"{case.strip()}_cantero_mean_front_timeseries.csv"


def _csv_value(value: object) -> str:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return f"{float(value):.16g}"
    return str(value)


def write_cantero_mean_front_timeseries_csv(
    path: str | Path,
    series: CanteroMeanFrontTimeseries,
    *,
    overwrite: bool,
) -> Path:
    """Write all frames, retaining failed crossings as explicit blank rows."""
    if not isinstance(series, CanteroMeanFrontTimeseries):
        raise ValueError("series must be a CanteroMeanFrontTimeseries.")
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANTERO_MEAN_FRONT_COLUMNS)
        writer.writeheader()
        for position in range(series.time.size):
            row = {
                "case": series.case,
                "file_index": series.file_index[position],
                "source_file": series.source_file[position],
                "time": series.time[position],
                "x_front": series.x_front[position],
                "x_front_minus_initial": series.x_front_minus_initial[position],
                "threshold": series.threshold,
                "reference_x": series.reference_x,
                "left_index": series.left_index[position],
                "right_index": series.right_index[position],
                "x_left": series.x_left[position],
                "x_right": series.x_right[position],
                "h_left": series.h_left[position],
                "h_right": series.h_right[position],
                "crossing_count_in_search_region": series.crossing_count_in_search_region[position],
                "status": series.status[position],
            }
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return output_path


def _csv_float(row: Mapping[str, str], name: str, path: Path, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line} has malformed {name}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{path}:{line} has non-finite {name}.")
    return value


def _csv_integer(row: Mapping[str, str], name: str, path: Path, line: int) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line} has malformed integer {name}.") from exc
    if value < 0:
        raise ValueError(f"{path}:{line} has negative {name}.")
    return value


def read_cantero_mean_front_timeseries_csv(path: str | Path) -> dict[str, object]:
    """Load successful Cantero mean-front rows without old-detector semantics."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cantero mean-front CSV not found: {csv_path}")
    successful: list[dict[str, object]] = []
    previous_time: float | None = None
    previous_index: int | None = None
    threshold: float | None = None
    reference_x: float | None = None
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(CANTERO_MEAN_FRONT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): {', '.join(sorted(missing))}."
            )
        for line, row in enumerate(reader, start=2):
            index = _csv_integer(row, "file_index", csv_path, line)
            time = _csv_float(row, "time", csv_path, line)
            row_threshold = _positive_threshold(_csv_float(row, "threshold", csv_path, line))
            row_reference = _csv_float(row, "reference_x", csv_path, line)
            if previous_index is not None and index <= previous_index:
                raise ValueError(f"{csv_path}:{line} file_index is not strictly increasing.")
            if previous_time is not None and time <= previous_time:
                raise ValueError(f"{csv_path}:{line} time is not strictly increasing.")
            previous_index, previous_time = index, time
            if threshold is None:
                threshold, reference_x = row_threshold, row_reference
            elif row_threshold != threshold or row_reference != reference_x:
                raise ValueError(f"{csv_path}:{line} has inconsistent threshold or reference_x.")
            if (row.get("status") or "").strip() != STATUS_SUCCESS:
                continue
            left_index = _csv_integer(row, "left_index", csv_path, line)
            right_index = _csv_integer(row, "right_index", csv_path, line)
            if right_index != left_index + 1:
                raise ValueError(f"{csv_path}:{line} has non-adjacent threshold bracket indices.")
            successful.append(
                {
                    "file_index": index,
                    "time": time,
                    "x_front": _csv_float(row, "x_front", csv_path, line),
                    "x_front_minus_initial": _csv_float(
                        row, "x_front_minus_initial", csv_path, line
                    ),
                    "left_index": left_index,
                    "right_index": right_index,
                    "x_left": _csv_float(row, "x_left", csv_path, line),
                    "x_right": _csv_float(row, "x_right", csv_path, line),
                    "h_left": _csv_float(row, "h_left", csv_path, line),
                    "h_right": _csv_float(row, "h_right", csv_path, line),
                }
            )
    if len(successful) < 2:
        raise ValueError(f"{csv_path} must contain at least two successful Cantero mean-front rows.")
    return {
        "time": _readonly([row["time"] for row in successful], np.float64),
        "file_index": _readonly([row["file_index"] for row in successful], np.int64),
        "x_front": _readonly([row["x_front"] for row in successful], np.float64),
        "x_front_minus_initial": _readonly(
            [row["x_front_minus_initial"] for row in successful], np.float64
        ),
        "threshold": threshold,
        "reference_x": reference_x,
    }


__all__ = (
    "CANTERO_MEAN_FRONT_COLUMNS",
    "DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X",
    "DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD",
    "STATUS_NO_DOWNWARD_CROSSING",
    "STATUS_REFERENCE_BELOW_THRESHOLD",
    "STATUS_SUCCESS",
    "CanteroMeanFrontDetection",
    "CanteroMeanFrontDetectionError",
    "CanteroMeanFrontTimeseries",
    "cantero_mean_front_timeseries_path",
    "compute_cantero_mean_front_timeseries",
    "detect_cantero_mean_front",
    "read_cantero_mean_front_timeseries_csv",
    "write_cantero_mean_front_timeseries_csv",
)
