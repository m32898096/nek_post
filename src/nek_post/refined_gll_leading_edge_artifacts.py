"""Validation and same-case method diagnostics for refined-GLL artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from nek_post.leading_edge_io import LEADING_EDGE_TIMESERIES_COLUMNS
from nek_post.leading_edge_methods import EXTRACTION_X_CONDITION


REFINED_GLL_METHODS = ("rightmost-crossing", "moore-boundary")


def _finite_increasing(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array))
            or np.any(np.diff(array) <= 0.0)):
        raise ValueError(f"{name} must be a finite strictly increasing vector.")
    return array


def _bool(value: str, *, path: Path, line: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{path}:{line}: invalid Boolean value {value!r}.")


def read_refined_gll_run(
    directory: str | Path,
    *,
    case: str,
    method: str,
    expected_frame_count: int | None = None,
    source_node_count: int = 8,
    target_node_count: int = 10,
) -> dict[str, Any]:
    """Read one saved route without imposing another case's physical grid.

    The CSV and JSON are treated as one artifact. The actual non-uniform x/y
    coordinates recorded by the refined-GLL plan are retained exactly. This
    reader deliberately has no cross-case pointwise comparison operation.
    """
    directory = Path(directory)
    csv_path = directory / f"{case}_leading_edge_timeseries.csv"
    metadata_path = directory / f"{case}_leading_edge_sampling_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "metadata_format_version": 1,
        "case": case,
        "sampling_mode": "refined-gll",
        "source_node_count": source_node_count,
        "source_polynomial_order": source_node_count - 1,
        "target_node_count": target_node_count,
        "extraction_method": method,
        "extraction_x_min": 0.0,
        "extraction_x_condition": EXTRACTION_X_CONDITION,
        "threshold": 0.1,
        "z_target": 0.04,
        "periodic_endpoint_included": False,
        "y_upsample_factor": None,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{metadata_path}: expected {key}={value!r}.")
    if "periodic_y" in metadata and metadata["periodic_y"] is not True:
        raise ValueError(f"{metadata_path}: refined-GLL y must be periodic.")
    if ("plane_values_finite_required" in metadata
            and metadata["plane_values_finite_required"] is not True):
        raise ValueError(f"{metadata_path}: refined-GLL planes must require finite values.")
    if metadata.get("spatial_extrapolation", False) is not False:
        raise ValueError(f"{metadata_path}: spatial extrapolation is not allowed.")

    x = _finite_increasing(metadata.get("x"), "refined-GLL x")
    y = _finite_increasing(metadata.get("y"), "refined-GLL y")
    period = float(metadata.get("y_period"))
    if not np.isfinite(period) or period <= 0.0 or not y[-1] < y[0] + period:
        raise ValueError("Refined-GLL y must exclude its finite periodic upper endpoint.")
    if int(metadata.get("nx", -1)) != x.size or int(metadata.get("output_ny", -1)) != y.size:
        raise ValueError("Recorded refined-GLL coordinate counts are inconsistent.")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEADING_EDGE_TIMESERIES_COLUMNS:
            raise ValueError(f"Unexpected timeseries schema: {csv_path}.")
        rows = list(reader)
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = {}
    for line, row in enumerate(rows, start=2):
        if row["case"] != case or row["y_upsample_factor"] != "":
            raise ValueError(f"{csv_path}:{line}: incompatible case or y-grid metadata.")
        if (float(row["threshold"]) != 0.1 or float(row["z_target"]) != 0.04
                or int(row["nx"]) != x.size or int(row["dense_ny"]) != y.size
                or int(row["native_ny"]) != int(metadata.get("native_ny", -1))):
            raise ValueError(f"{csv_path}:{line}: CSV/JSON definition mismatch.")
        grouped.setdefault(int(row["file_index"]), []).append((line, row))

    ordered = sorted(grouped.items(), key=lambda item: float(item[1][0][1]["actual_time"]))
    if expected_frame_count is not None and len(ordered) != expected_frame_count:
        raise ValueError(f"Expected {expected_frame_count} frames; found {len(ordered)} in {csv_path}.")
    if len(ordered) != int(metadata.get("n_selected_frames", -1)):
        raise ValueError("CSV frame count differs from refined-GLL metadata.")
    if int(metadata.get("n_input_frames", -1)) < len(ordered):
        raise ValueError("Input-frame count cannot be smaller than selected-frame count.")
    curves, successes, crossings, times, indices, sources = [], [], [], [], [], []
    for index, entries in ordered:
        entries.sort(key=lambda item: float(item[1]["y"]))
        frame_y = np.asarray([float(row["y"]) for _, row in entries])
        if frame_y.shape != y.shape or not np.allclose(frame_y, y, rtol=0.0, atol=1.0e-14):
            raise ValueError(f"Frame {index} does not preserve the recorded physical y grid.")
        for key in ("actual_time", "target_time", "time_error", "source_file"):
            if len({row[key] for _, row in entries}) != 1:
                raise ValueError(f"Frame {index} has inconsistent {key} values.")
        curve = np.asarray([float(row["x_front"]) for _, row in entries])
        success = np.asarray([_bool(row["success"], path=csv_path, line=line)
                              for line, row in entries])
        crossing = np.asarray([int(row["crossing_count"]) for _, row in entries], dtype=np.int64)
        if (np.any(np.isinf(curve)) or np.any(crossing < 0)
                or not np.array_equal(success, np.isfinite(curve))):
            raise ValueError(f"Frame {index} has invalid front, success, or crossing values.")
        curves.append(curve)
        successes.append(success)
        crossings.append(crossing)
        times.append(float(entries[0][1]["actual_time"]))
        indices.append(index)
        sources.append(entries[0][1]["source_file"])
    times_array = np.asarray(times)
    if times_array.size == 0 or np.any(np.diff(times_array) <= 0.0):
        raise ValueError("Stored refined-GLL physical times must be strictly increasing.")
    if (times_array[0] != float(metadata.get("actual_time_start"))
            or times_array[-1] != float(metadata.get("actual_time_end"))):
        raise ValueError("CSV time range differs from refined-GLL metadata.")
    for key, actual in (
        ("source_file_indices", indices),
        ("source_files", sources),
        ("stored_physical_times", times),
    ):
        if key in metadata and metadata[key] != actual:
            raise ValueError(f"CSV {key} differ from refined-GLL metadata.")
    return {
        "metadata": metadata,
        "x": x,
        "y": y,
        "x_front": np.asarray(curves),
        "success": np.asarray(successes),
        "crossing_count": np.asarray(crossings),
        "times": times_array,
        "indices": np.asarray(indices, dtype=np.int64),
        "source_files": tuple(sources),
        "inputs": (str(csv_path), str(metadata_path)),
        "x_nonuniform": bool(not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-12, atol=1e-14)),
        "y_nonuniform": bool(not np.allclose(np.diff(y), np.diff(y)[0], rtol=1e-12, atol=1e-14)),
    }


def compare_refined_gll_methods(
    rightmost: dict[str, Any], moore: dict[str, Any],
) -> dict[str, Any]:
    """Compare extraction methods only when they share one case-native grid."""
    for key in ("x", "y", "times", "indices"):
        if not np.array_equal(rightmost[key], moore[key]):
            raise ValueError(f"Same-case refined-GLL method runs differ in {key}.")
    a, b = np.asarray(rightmost["x_front"]), np.asarray(moore["x_front"])
    sa, sb = np.asarray(rightmost["success"]), np.asarray(moore["success"])
    if a.shape != b.shape or sa.shape != a.shape or sb.shape != a.shape:
        raise ValueError("Same-case refined-GLL method arrays have incompatible shapes.")
    rms, maximum, differing = [], [], []
    identical = []
    for av, bv, am, bm in zip(a, b, sa, sb, strict=True):
        mask = am & bm & np.isfinite(av) & np.isfinite(bv)
        difference = av[mask] - bv[mask]
        rms.append(float(np.sqrt(np.mean(difference * difference))) if difference.size else None)
        maximum.append(float(np.max(np.abs(difference))) if difference.size else None)
        different = (am != bm) | (np.isfinite(av) != np.isfinite(bv))
        both = np.isfinite(av) & np.isfinite(bv)
        different |= both & (av != bv)
        differing.append(int(np.count_nonzero(different)))
        identical.append(bool(np.array_equal(av, bv, equal_nan=True) and np.array_equal(am, bm)))
    return {
        "frame_count": int(a.shape[0]),
        "rms_by_frame": rms,
        "maximum_absolute_difference_by_frame": maximum,
        "differing_y_count_by_frame": differing,
        "exactly_identical_frame_count": int(sum(identical)),
        "exactly_identical_frame_fraction": float(np.mean(identical)),
    }


__all__ = (
    "REFINED_GLL_METHODS",
    "compare_refined_gll_methods",
    "read_refined_gll_run",
)
