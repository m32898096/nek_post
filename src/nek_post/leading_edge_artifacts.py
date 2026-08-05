"""Validated reconstruction of plot-ready leading-edge CSV artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nek_post.leading_edge_io import (
    LEADING_EDGE_METADATA_COLUMNS,
    LEADING_EDGE_TIMESERIES_COLUMNS,
)
from nek_post.leading_edge_methods import (
    DEFAULT_LEADING_EDGE_METHOD,
    normalize_leading_edge_method,
)


@dataclass(frozen=True)
class LeadingEdgePlotData:
    """Immutable selected leading-edge curves reconstructed from CSV files."""

    case: str
    file_indices: NDArray[np.int64]
    target_time: NDArray[np.float64]
    actual_time: NDArray[np.float64]
    time_error: NDArray[np.float64]
    y: NDArray[np.float64]
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    threshold: float
    z_target: float
    nx: int
    native_ny: int
    dense_ny: int
    y_upsample_factor: int
    y_min: float
    y_max_periodic_endpoint: float
    periodic_endpoint_included: bool
    target_time_spacing: float | None
    extraction_method: str = DEFAULT_LEADING_EDGE_METHOD


@dataclass(frozen=True)
class _TimeseriesRow:
    case: str
    file_index: int
    source_file: str
    target_time: float
    actual_time: float
    time_error: float
    y: float
    x_front: float
    success: bool
    crossing_count: int
    threshold: float
    z_target: float
    nx: int
    native_ny: int
    dense_ny: int
    y_upsample_factor: int


def _readonly_copy(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _read_csv_rows(
    path: str | Path,
    expected_columns: tuple[str, ...],
    artifact_name: str,
) -> tuple[Path, list[dict[str, str]]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Leading-edge {artifact_name} CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Leading-edge {artifact_name} CSV is empty: {csv_path}")
        actual_columns = tuple(reader.fieldnames)
        if actual_columns != expected_columns:
            missing = [name for name in expected_columns if name not in actual_columns]
            unexpected = [name for name in actual_columns if name not in expected_columns]
            details: list[str] = []
            if missing:
                details.append("missing columns: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected columns: " + ", ".join(unexpected))
            if not details:
                details.append("column order does not match the required schema")
            raise ValueError(
                f"Invalid leading-edge {artifact_name} CSV schema in {csv_path}: "
                + "; ".join(details)
                + "."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(
            f"Leading-edge {artifact_name} CSV contains no data rows: {csv_path}"
        )
    return csv_path, rows


def _text(row: dict[str, str], name: str, context: str) -> str:
    value = row.get(name)
    if value is None or value == "":
        raise ValueError(f"{context} has an empty {name!r} value.")
    return value


def _integer(
    row: dict[str, str],
    name: str,
    context: str,
    *,
    minimum: int | None = None,
) -> int:
    text = _text(row, name, context)
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{context} has malformed integer {name}={text!r}.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} requires {name} >= {minimum}; got {value}.")
    return value


def _float(
    row: dict[str, str],
    name: str,
    context: str,
    *,
    allow_nan: bool = False,
) -> float:
    text = _text(row, name, context)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{context} has malformed numeric {name}={text!r}.") from exc
    if np.isinf(value) or (np.isnan(value) and not allow_nan):
        raise ValueError(f"{context} has invalid numeric {name}={text!r}.")
    return value


def _boolean(row: dict[str, str], name: str, context: str) -> bool:
    text = _text(row, name, context)
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(
        f"{context} has malformed boolean {name}={text!r}; expected True or False."
    )


def _metadata_values(row: dict[str, str], path: Path) -> dict[str, object]:
    context = f"metadata row in {path}"
    case = _text(row, "case", context)
    values: dict[str, object] = {
        "case": case,
        "extraction_method": normalize_leading_edge_method(
            _text(row, "extraction_method", context)
        ),
        "n_input_frames": _integer(
            row, "n_input_frames", context, minimum=1
        ),
        "n_selected_frames": _integer(
            row, "n_selected_frames", context, minimum=1
        ),
        "actual_time_start": _float(row, "actual_time_start", context),
        "actual_time_end": _float(row, "actual_time_end", context),
        "threshold": _float(row, "threshold", context),
        "z_target": _float(row, "z_target", context),
        "nx": _integer(row, "nx", context, minimum=2),
        "native_ny": _integer(row, "native_ny", context, minimum=1),
        "dense_ny": _integer(row, "dense_ny", context, minimum=1),
        "y_upsample_factor": _integer(
            row, "y_upsample_factor", context, minimum=1
        ),
        "y_min": _float(row, "y_min", context),
        "y_max_periodic_endpoint": _float(
            row, "y_max_periodic_endpoint", context
        ),
        "periodic_endpoint_included": _boolean(
            row, "periodic_endpoint_included", context
        ),
        "algorithm_version": _integer(
            row, "algorithm_version", context, minimum=0
        ),
        "inverse_mapping_target_count": _integer(
            row, "inverse_mapping_target_count", context, minimum=0
        ),
        "inverse_mapping_success_count": _integer(
            row, "inverse_mapping_success_count", context, minimum=0
        ),
        "inverse_mapping_failure_count": _integer(
            row, "inverse_mapping_failure_count", context, minimum=0
        ),
        "ambiguous_boundary_point_count": _integer(
            row, "ambiguous_boundary_point_count", context, minimum=0
        ),
        "maximum_successful_residual": _float(
            row, "maximum_successful_residual", context
        ),
        "maximum_iteration_count": _integer(
            row, "maximum_iteration_count", context, minimum=0
        ),
    }
    raw_spacing = _float(
        row,
        "target_time_spacing",
        context,
        allow_nan=True,
    )
    if np.isnan(raw_spacing):
        values["target_time_spacing"] = None
    elif raw_spacing <= 0.0:
        raise ValueError(
            f"{context} requires target_time_spacing to be positive or nan."
        )
    else:
        values["target_time_spacing"] = raw_spacing

    if values["n_selected_frames"] > values["n_input_frames"]:  # type: ignore[operator]
        raise ValueError("Metadata n_selected_frames exceeds n_input_frames.")
    if values["actual_time_end"] < values["actual_time_start"]:  # type: ignore[operator]
        raise ValueError("Metadata actual_time_end precedes actual_time_start.")
    if values["y_max_periodic_endpoint"] <= values["y_min"]:  # type: ignore[operator]
        raise ValueError("Metadata periodic y extent must have y_max greater than y_min.")
    if values["periodic_endpoint_included"] is not False:
        raise ValueError("Metadata periodic_endpoint_included must be False.")
    if values["dense_ny"] != (  # type: ignore[comparison-overlap]
        values["y_upsample_factor"] * values["native_ny"]  # type: ignore[operator]
    ):
        raise ValueError(
            "Metadata requires dense_ny == y_upsample_factor * native_ny."
        )
    return values


def _timeseries_row(
    row: dict[str, str],
    row_number: int,
    path: Path,
) -> _TimeseriesRow:
    context = f"timeseries row {row_number} in {path}"
    source_file = _text(row, "source_file", context)
    return _TimeseriesRow(
        case=_text(row, "case", context),
        file_index=_integer(row, "file_index", context, minimum=0),
        source_file=source_file,
        target_time=_float(row, "target_time", context),
        actual_time=_float(row, "actual_time", context),
        time_error=_float(row, "time_error", context),
        y=_float(row, "y", context),
        x_front=_float(row, "x_front", context, allow_nan=True),
        success=_boolean(row, "success", context),
        crossing_count=_integer(row, "crossing_count", context, minimum=0),
        threshold=_float(row, "threshold", context),
        z_target=_float(row, "z_target", context),
        nx=_integer(row, "nx", context, minimum=2),
        native_ny=_integer(row, "native_ny", context, minimum=1),
        dense_ny=_integer(row, "dense_ny", context, minimum=1),
        y_upsample_factor=_integer(
            row, "y_upsample_factor", context, minimum=1
        ),
    )


def _require_row_metadata_consistency(
    row: _TimeseriesRow,
    metadata: dict[str, object],
) -> None:
    if row.case != metadata["case"]:
        raise ValueError(
            f"Inconsistent case values: timeseries has {row.case!r}, "
            f"metadata has {metadata['case']!r}."
        )
    for name in (
        "threshold",
        "z_target",
        "nx",
        "native_ny",
        "dense_ny",
        "y_upsample_factor",
    ):
        if getattr(row, name) != metadata[name]:
            raise ValueError(
                f"Inconsistent {name}: timeseries has {getattr(row, name)!r}, "
                f"metadata has {metadata[name]!r}."
            )


def _require_frame_constants(
    rows: list[_TimeseriesRow],
    file_index: int,
) -> None:
    first = rows[0]
    for row in rows[1:]:
        for name in (
            "source_file",
            "target_time",
            "actual_time",
            "time_error",
        ):
            if getattr(row, name) != getattr(first, name):
                raise ValueError(
                    f"File index {file_index} has inconsistent {name} values."
                )


def read_leading_edge_artifacts(
    timeseries_csv: str | Path,
    metadata_csv: str | Path,
) -> LeadingEdgePlotData:
    """Read and validate CSV artifacts without accessing Nek5000 field files."""
    timeseries_path, raw_timeseries_rows = _read_csv_rows(
        timeseries_csv,
        LEADING_EDGE_TIMESERIES_COLUMNS,
        "timeseries",
    )
    metadata_path, raw_metadata_rows = _read_csv_rows(
        metadata_csv,
        LEADING_EDGE_METADATA_COLUMNS,
        "metadata",
    )
    if len(raw_metadata_rows) != 1:
        raise ValueError(
            f"Leading-edge metadata CSV must contain exactly one data row: "
            f"{metadata_path}"
        )
    metadata = _metadata_values(raw_metadata_rows[0], metadata_path)
    rows = [
        _timeseries_row(row, row_number, timeseries_path)
        for row_number, row in enumerate(raw_timeseries_rows, start=2)
    ]
    grouped: dict[int, list[_TimeseriesRow]] = {}
    seen_file_y: set[tuple[int, float]] = set()
    for row in rows:
        _require_row_metadata_consistency(row, metadata)
        row_key = (row.file_index, row.y)
        if row_key in seen_file_y:
            raise ValueError(
                "Duplicate leading-edge timeseries row for "
                f"file_index={row.file_index}, y={row.y:.16g}."
            )
        seen_file_y.add(row_key)
        grouped.setdefault(row.file_index, []).append(row)

    ordered_indices = sorted(grouped)
    if len(ordered_indices) != metadata["n_selected_frames"]:
        raise ValueError(
            "Metadata n_selected_frames does not match the timeseries frame count."
        )
    dense_ny = int(metadata["dense_ny"])
    shared_y: np.ndarray | None = None
    target_times: list[float] = []
    actual_times: list[float] = []
    time_errors: list[float] = []
    x_front_rows: list[list[float]] = []
    success_rows: list[list[bool]] = []
    crossing_rows: list[list[int]] = []

    for file_index in ordered_indices:
        frame_rows = sorted(grouped[file_index], key=lambda row: row.y)
        _require_frame_constants(frame_rows, file_index)
        if len(frame_rows) != dense_ny:
            raise ValueError(
                f"File index {file_index} has {len(frame_rows)} y rows; "
                f"metadata dense_ny is {dense_ny}."
            )
        frame_y = np.asarray([row.y for row in frame_rows], dtype=np.float64)
        if np.any(np.diff(frame_y) <= 0.0):
            raise ValueError(
                f"File index {file_index} y coordinates must be strictly increasing."
            )
        if shared_y is None:
            shared_y = frame_y
        elif not np.array_equal(frame_y, shared_y):
            raise ValueError("Selected frames have inconsistent y grids.")
        first = frame_rows[0]
        target_times.append(first.target_time)
        actual_times.append(first.actual_time)
        time_errors.append(first.time_error)
        x_front_rows.append([row.x_front for row in frame_rows])
        success_rows.append([row.success for row in frame_rows])
        crossing_rows.append([row.crossing_count for row in frame_rows])

    assert shared_y is not None
    actual_time_array = np.asarray(actual_times, dtype=np.float64)
    if np.any(np.diff(actual_time_array) <= 0.0):
        raise ValueError(
            "Selected actual times must be strictly increasing in file-index order."
        )
    target_time_array = np.asarray(target_times, dtype=np.float64)
    if np.any(np.diff(target_time_array) <= 0.0):
        raise ValueError("Selected target times must be strictly increasing.")
    y_min = float(metadata["y_min"])
    y_max = float(metadata["y_max_periodic_endpoint"])
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        abs(y_min), abs(y_max), 1.0
    )
    if abs(float(shared_y[0]) - y_min) > tolerance:
        raise ValueError("Timeseries y grid does not begin at metadata y_min.")
    if shared_y[-1] >= y_max:
        raise ValueError(
            "Timeseries computational y grid must exclude y_max_periodic_endpoint."
        )

    return LeadingEdgePlotData(
        case=str(metadata["case"]),
        extraction_method=str(metadata["extraction_method"]),
        file_indices=_readonly_copy(ordered_indices, np.int64),
        target_time=_readonly_copy(target_time_array, np.float64),
        actual_time=_readonly_copy(actual_time_array, np.float64),
        time_error=_readonly_copy(time_errors, np.float64),
        y=_readonly_copy(shared_y, np.float64),
        x_front=_readonly_copy(x_front_rows, np.float64),
        success_mask=_readonly_copy(success_rows, np.bool_),
        crossing_count=_readonly_copy(crossing_rows, np.int64),
        threshold=float(metadata["threshold"]),
        z_target=float(metadata["z_target"]),
        nx=int(metadata["nx"]),
        native_ny=int(metadata["native_ny"]),
        dense_ny=dense_ny,
        y_upsample_factor=int(metadata["y_upsample_factor"]),
        y_min=y_min,
        y_max_periodic_endpoint=y_max,
        periodic_endpoint_included=False,
        target_time_spacing=metadata["target_time_spacing"],  # type: ignore[arg-type]
    )


__all__ = ("LeadingEdgePlotData", "read_leading_edge_artifacts")
