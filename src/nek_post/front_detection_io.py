"""Discovery, output paths, CSV writing, and reporting for front detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

import numpy as np

from nek_post.front_detection import FrontTrackingResult


TIMESERIES_COLUMNS = (
    "file_index",
    "source_file",
    "time",
    "x_front_auto",
    "predicted_x",
    "tracking_error",
    "status",
    "component_count",
    "spatial_candidate_count",
    "temporal_candidate_count",
    "selected_component_label",
    "selected_component_pixels",
    "selected_component_xmin",
    "selected_component_xmax",
    "selected_bottom_contact",
    "selected_overlap_pixels",
    "concentration_finite_fraction",
    "concentration_min",
    "concentration_max",
    "threshold",
    "min_component_pixels",
    "bottom_rows",
    "max_front_jump",
    "connectivity",
)

COMPARISON_COLUMNS = (
    "file_index",
    "time",
    "x_front_auto",
    "x_front_reference",
    "difference",
    "absolute_difference",
)

SUMMARY_COLUMNS = (
    "case",
    "reference_file",
    "reference_role",
    "comparison_status",
    "n_input_frames",
    "n_successful_detections",
    "success_fraction",
    "n_selected_initial",
    "n_selected_tracked",
    "n_no_threshold_component",
    "n_no_valid_spatial_candidate",
    "n_no_valid_temporal_candidate",
    "time_start",
    "time_end",
    "index_start",
    "index_end",
    "threshold",
    "min_component_pixels",
    "bottom_rows",
    "max_front_jump",
    "connectivity",
    "nx",
    "nz",
    "interpolation_method",
    "interpolation_engine",
    "spectral_element_shape",
    "spectral_polynomial_order",
    "spectral_slice_y",
    "n_comparison_points",
    "mean_signed_difference",
    "mean_absolute_difference",
    "rms_difference",
    "max_absolute_difference",
    "n_slumping_points",
    "auto_slumping_velocity",
    "reference_slumping_velocity",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
)

SUMMARY_TABLE_COLUMNS = (
    "case",
    "n_input_frames",
    "n_successful_detections",
    "success_fraction",
    "n_comparison_points",
    "mean_absolute_difference",
    "rms_difference",
    "max_absolute_difference",
)


@dataclass(frozen=True)
class NekFramePath:
    """One discovered Nek5000 field file and its parsed five-digit index."""

    index: int
    path: Path


def _validate_index_filter(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return parsed


def discover_nek_frame_paths(
    case_dir: str | Path,
    *,
    file_prefix: str,
    start_index: int | None = None,
    end_index: int | None = None,
) -> tuple[NekFramePath, ...]:
    """Discover exact ``PREFIX.fNNNNN`` files, sorted by parsed index."""
    directory = Path(case_dir)
    start = _validate_index_filter(start_index, "start_index")
    end = _validate_index_filter(end_index, "end_index")
    if start is not None and end is not None and start > end:
        raise ValueError("start_index must be less than or equal to end_index.")

    filename_pattern = re.compile(rf"^{re.escape(file_prefix)}\.f([0-9]{{5}})$")
    frames_by_index: dict[int, Path] = {}
    if directory.is_dir():
        for path in directory.iterdir():
            match = filename_pattern.fullmatch(path.name)
            if match is None or not path.is_file():
                continue
            index = int(match.group(1))
            if index in frames_by_index:
                raise ValueError(
                    f"Duplicate Nek5000 frame index {index} in {directory}: "
                    f"{frames_by_index[index]} and {path}."
                )
            frames_by_index[index] = path

    selected = tuple(
        NekFramePath(index=index, path=frames_by_index[index])
        for index in sorted(frames_by_index)
        if (start is None or index >= start) and (end is None or index <= end)
    )
    if not selected:
        raise FileNotFoundError(
            f"No Nek5000 frame files matching {filename_pattern.pattern!r} "
            f"remain in {directory}."
        )
    return selected


def detected_front_timeseries_path(output_dir: str | Path, case: str) -> Path:
    return Path(output_dir) / f"{case}_detected_front_timeseries.csv"


def front_detection_comparison_path(output_dir: str | Path, case: str) -> Path:
    return Path(output_dir) / f"{case}_front_detection_comparison.csv"


def front_detection_summary_path(output_dir: str | Path, case: str) -> Path:
    return Path(output_dir) / f"{case}_front_detection_summary.csv"


def front_detection_overlay_path(output_dir: str | Path, case: str) -> Path:
    return Path(output_dir) / f"{case}_front_detection_overlay.png"


def front_detection_difference_path(output_dir: str | Path, case: str) -> Path:
    return Path(output_dir) / f"{case}_front_detection_difference.png"


def front_detection_diagnostics_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "diagnostics"


def front_detection_diagnostic_path(
    output_dir: str | Path,
    case: str,
    file_index: int,
) -> Path:
    if not isinstance(file_index, Integral) or isinstance(
        file_index, (bool, np.bool_)
    ):
        raise ValueError("file_index must be a non-negative integer.")
    parsed_index = int(file_index)
    if parsed_index < 0:
        raise ValueError("file_index must be greater than or equal to 0.")
    return (
        front_detection_diagnostics_dir(output_dir)
        / f"{case}_front_diagnostic_f{parsed_index:05d}.png"
    )


def ensure_writable_output(path: str | Path, overwrite: bool) -> None:
    """Reject an existing output unless replacement was explicitly enabled."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )


def preflight_output_paths(
    paths: Iterable[str | Path],
    overwrite: bool,
) -> tuple[Path, ...]:
    """Check every requested output before any writer changes the filesystem."""
    output_paths = tuple(Path(path) for path in paths)
    for path in output_paths:
        ensure_writable_output(path, overwrite)
    return output_paths


def format_csv_value(value: Any) -> str:
    """Format CSV values without coercing integers, booleans, or strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return f"{float(value):.16g}"
    return str(value)


def _write_rows(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    overwrite: bool,
) -> None:
    ensure_writable_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: format_csv_value(row[column]) for column in columns}
            )


def write_detected_front_timeseries_csv(
    path: str | Path,
    sequence: Any,
    tracking_result: FrontTrackingResult,
    overwrite: bool,
) -> None:
    """Write one row per input frame in exact sequence order."""
    output_path = Path(path)
    n_frames = len(sequence.time)
    if len(tracking_result.time) != n_frames:
        raise ValueError("Sequence and tracking result lengths must match.")

    def rows() -> Iterable[dict[str, Any]]:
        for index in range(n_frames):
            yield {
                "file_index": sequence.file_indices[index],
                "source_file": str(sequence.source_files[index]),
                "time": sequence.time[index],
                "x_front_auto": tracking_result.x_front[index],
                "predicted_x": tracking_result.predicted_x[index],
                "tracking_error": tracking_result.tracking_error[index],
                "status": tracking_result.status[index],
                "component_count": tracking_result.component_count[index],
                "spatial_candidate_count": tracking_result.spatial_candidate_count[
                    index
                ],
                "temporal_candidate_count": tracking_result.temporal_candidate_count[
                    index
                ],
                "selected_component_label": tracking_result.selected_component_label[
                    index
                ],
                "selected_component_pixels": tracking_result.selected_component_pixels[
                    index
                ],
                "selected_component_xmin": tracking_result.selected_component_xmin[
                    index
                ],
                "selected_component_xmax": tracking_result.selected_component_xmax[
                    index
                ],
                "selected_bottom_contact": tracking_result.selected_bottom_contact[
                    index
                ],
                "selected_overlap_pixels": tracking_result.selected_overlap_pixels[
                    index
                ],
                "concentration_finite_fraction": sequence.finite_fraction[index],
                "concentration_min": sequence.concentration_min[index],
                "concentration_max": sequence.concentration_max[index],
                "threshold": tracking_result.threshold,
                "min_component_pixels": tracking_result.min_component_pixels,
                "bottom_rows": tracking_result.bottom_rows,
                "max_front_jump": tracking_result.max_front_jump,
                "connectivity": tracking_result.connectivity,
            }

    _write_rows(output_path, TIMESERIES_COLUMNS, rows(), overwrite)


def write_front_detection_comparison_csv(
    path: str | Path,
    comparison: Any,
    overwrite: bool,
) -> None:
    """Write successful, overlapping automatic/reference comparison rows."""

    def rows() -> Iterable[dict[str, Any]]:
        for index in range(len(comparison.time)):
            yield {
                "file_index": comparison.file_indices[index],
                "time": comparison.time[index],
                "x_front_auto": comparison.x_front_auto[index],
                "x_front_reference": comparison.x_front_reference[index],
                "difference": comparison.difference[index],
                "absolute_difference": comparison.absolute_difference[index],
            }

    _write_rows(Path(path), COMPARISON_COLUMNS, rows(), overwrite)


def write_front_detection_summary_csv(
    path: str | Path,
    summary: Mapping[str, Any],
    overwrite: bool,
) -> None:
    """Write one deterministic front-detection summary row."""
    _write_rows(Path(path), SUMMARY_COLUMNS, [summary], overwrite)


def write_front_detection_csvs(
    output_dir: str | Path,
    case: str,
    sequence: Any,
    tracking_result: FrontTrackingResult,
    comparison: Any,
    summary: Mapping[str, Any],
    overwrite: bool,
) -> list[Path]:
    """Preflight and write all CSV outputs in established order."""
    paths = [
        detected_front_timeseries_path(output_dir, case),
        front_detection_comparison_path(output_dir, case),
        front_detection_summary_path(output_dir, case),
    ]
    preflight_output_paths(paths, overwrite)
    write_detected_front_timeseries_csv(
        paths[0], sequence, tracking_result, overwrite=True
    )
    write_front_detection_comparison_csv(paths[1], comparison, overwrite=True)
    write_front_detection_summary_csv(paths[2], summary, overwrite=True)
    return paths


def format_front_detection_summary_table(summary: Mapping[str, Any]) -> str:
    """Format a compact deterministic terminal summary."""
    formatted = {
        column: format_csv_value(summary[column])
        for column in SUMMARY_TABLE_COLUMNS
    }
    widths = {
        column: max(len(column), len(formatted[column]))
        for column in SUMMARY_TABLE_COLUMNS
    }
    return "\n".join(
        [
            "Front detection summary:",
            "  ".join(
                column.ljust(widths[column]) for column in SUMMARY_TABLE_COLUMNS
            ),
            "  ".join("-" * widths[column] for column in SUMMARY_TABLE_COLUMNS),
            "  ".join(
                formatted[column].ljust(widths[column])
                for column in SUMMARY_TABLE_COLUMNS
            ),
        ]
    )
