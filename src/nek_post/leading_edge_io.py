"""Artifact paths and CSV serialization for leading-edge evolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nek_post.front_detection_io import (
    ensure_writable_output,
    format_csv_value,
    preflight_output_paths,
)
from nek_post.leading_edge_methods import normalize_leading_edge_method
from nek_post.leading_edge_methods import EXTRACTION_X_CONDITION
if TYPE_CHECKING:
    from nek_post.leading_edge_workflow import (
        LeadingEdgeEvolution,
        LeadingEdgeTimeSelection,
    )


LEADING_EDGE_TIMESERIES_COLUMNS = (
    "case",
    "file_index",
    "source_file",
    "target_time",
    "actual_time",
    "time_error",
    "y",
    "x_front",
    "success",
    "crossing_count",
    "threshold",
    "z_target",
    "nx",
    "native_ny",
    "dense_ny",
    "y_upsample_factor",
)

LEADING_EDGE_METADATA_COLUMNS = (
    "case",
    "extraction_method",
    "extraction_x_min",
    "extraction_x_condition",
    "n_input_frames",
    "n_selected_frames",
    "actual_time_start",
    "actual_time_end",
    "target_time_spacing",
    "threshold",
    "z_target",
    "nx",
    "native_ny",
    "dense_ny",
    "y_upsample_factor",
    "y_min",
    "y_max_periodic_endpoint",
    "periodic_endpoint_included",
    "algorithm_version",
    "inverse_mapping_target_count",
    "inverse_mapping_success_count",
    "inverse_mapping_failure_count",
    "ambiguous_boundary_point_count",
    "maximum_successful_residual",
    "maximum_iteration_count",
)


def leading_edge_timeseries_path(output_dir: str | Path, case: str) -> Path:
    """Return the tidy selected-frame leading-edge CSV path."""
    return Path(output_dir) / f"{case}_leading_edge_timeseries.csv"


def leading_edge_metadata_path(output_dir: str | Path, case: str) -> Path:
    """Return the one-row leading-edge metadata CSV path."""
    return Path(output_dir) / f"{case}_leading_edge_metadata.csv"


def leading_edge_sampling_metadata_path(output_dir: str | Path, case: str) -> Path:
    """Return explicitly tagged refined-GLL metadata, separate from the legacy CSV."""
    return Path(output_dir) / f"{case}_leading_edge_sampling_metadata.json"


def leading_edge_evolution_png_path(output_dir: str | Path, case: str) -> Path:
    """Return the leading-edge evolution PNG path."""
    return Path(output_dir) / f"{case}_leading_edge_evolution.png"


def leading_edge_evolution_pdf_path(output_dir: str | Path, case: str) -> Path:
    """Return the leading-edge evolution PDF path."""
    return Path(output_dir) / f"{case}_leading_edge_evolution.pdf"


def _validate_evolution_selection(
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
) -> tuple[int, int]:
    n_input = int(np.asarray(evolution.time).size)
    n_selected = int(np.asarray(selection.actual_time).size)
    dense_ny = int(np.asarray(evolution.y).size)
    if selection.evolution is not evolution:
        raise ValueError("Time selection must refer to the supplied evolution.")
    expected = (n_selected, dense_ny)
    for name in ("x_front", "success_mask", "crossing_count"):
        if np.asarray(getattr(selection, name)).shape != expected:
            raise ValueError(f"selection.{name} must have shape {expected}.")
    for name in (
        "selected_positions",
        "file_indices",
        "target_time",
        "actual_time",
        "time_error",
        "finite_leading_edge_fraction",
        "successful_y_count",
    ):
        if np.asarray(getattr(selection, name)).shape != (n_selected,):
            raise ValueError(f"selection.{name} must have shape ({n_selected},).")
    if len(selection.source_files) != n_selected:
        raise ValueError("selection.source_files length must match selected frames.")
    if n_input == 0 or n_selected == 0 or dense_ny == 0:
        raise ValueError("Evolution and selection must be nonempty.")
    return n_input, n_selected


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


def write_leading_edge_timeseries_csv(
    path: str | Path,
    case: str,
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
    overwrite: bool,
) -> None:
    """Write selected leading edges in frame-major, increasing-y row order."""
    _validate_evolution_selection(evolution, selection)

    def rows() -> Iterable[dict[str, Any]]:
        for frame_position in range(selection.actual_time.size):
            for y_position in range(evolution.y.size):
                yield {
                    "case": case,
                    "file_index": selection.file_indices[frame_position],
                    "source_file": str(selection.source_files[frame_position]),
                    "target_time": selection.target_time[frame_position],
                    "actual_time": selection.actual_time[frame_position],
                    "time_error": selection.time_error[frame_position],
                    "y": evolution.y[y_position],
                    "x_front": selection.x_front[frame_position, y_position],
                    "success": selection.success_mask[frame_position, y_position],
                    "crossing_count": selection.crossing_count[
                        frame_position, y_position
                    ],
                    "threshold": evolution.threshold,
                    "z_target": evolution.z_target,
                    "nx": evolution.nx,
                    "native_ny": evolution.native_ny,
                    "dense_ny": evolution.dense_ny,
                    "y_upsample_factor": (
                        "" if evolution.y_upsample_factor is None else evolution.y_upsample_factor
                    ),
                }

    _write_rows(
        Path(path),
        LEADING_EDGE_TIMESERIES_COLUMNS,
        rows(),
        overwrite,
    )


def leading_edge_metadata(
    case: str,
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
) -> Mapping[str, Any]:
    """Return the deterministic one-row metadata mapping used by the CSV."""
    n_input, n_selected = _validate_evolution_selection(evolution, selection)
    plan_metadata = evolution.horizontal_plan_metadata
    if evolution.sampling_mode == "refined-gll":
        if evolution.source_node_count is None or evolution.target_node_count is None:
            raise ValueError("Refined-GLL metadata requires source and target node counts.")
        return {
            "metadata_format_version": 1,
            "case": case, "sampling_mode": evolution.sampling_mode,
            "interpretation": "Original source polynomial evaluated on a target GLL nodal set; no new solution information.",
            "source_node_count": evolution.source_node_count,
            "source_polynomial_order": evolution.source_node_count - 1,
            "target_node_count": evolution.target_node_count,
            "extraction_method": evolution.extraction_method,
            "extraction_x_min": evolution.extraction_x_min,
            "extraction_x_condition": "unrestricted" if evolution.extraction_x_min is None else EXTRACTION_X_CONDITION,
            "threshold": evolution.threshold, "z_target": evolution.z_target,
            "nx": evolution.nx, "native_ny": evolution.native_ny,
            "output_ny": evolution.dense_ny, "dense_ny": evolution.dense_ny,
            "y_upsample_factor": None, "periodic_endpoint_included": False,
            "y_period": plan_metadata["ymax_periodic_endpoint"] - plan_metadata["ymin"],
            "x": evolution.x.tolist(), "y": evolution.y.tolist(),
            "n_input_frames": n_input, "n_selected_frames": n_selected,
            "actual_time_start": float(evolution.time[0]),
            "actual_time_end": float(evolution.time[-1]),
            "target_time_spacing": selection.target_time_spacing,
            "plan_build_seconds": evolution.plan_build_seconds,
            "processing_runtime_seconds": evolution.processing_runtime_seconds,
            "horizontal_plan_metadata": dict(plan_metadata),
        }
    required_plan_keys = (
        "ymin",
        "ymax_periodic_endpoint",
        "spectral_horizontal_algorithm_version",
        "inverse_mapping_target_count",
        "inverse_mapping_success_count",
        "inverse_mapping_failure_count",
        "ambiguous_boundary_point_count",
        "maximum_successful_residual",
        "maximum_iteration_count",
    )
    missing = [key for key in required_plan_keys if key not in plan_metadata]
    if missing:
        raise ValueError(
            "Horizontal-plan metadata is missing required values: "
            + ", ".join(missing)
            + "."
        )
    spacing = (
        np.nan
        if selection.target_time_spacing is None
        else selection.target_time_spacing
    )
    extraction_x_min = (
        np.nan
        if evolution.extraction_x_min is None
        else float(evolution.extraction_x_min)
    )
    return {
        "case": case,
        "extraction_method": normalize_leading_edge_method(
            evolution.extraction_method
        ),
        "extraction_x_min": extraction_x_min,
        "extraction_x_condition": (
            "unrestricted"
            if evolution.extraction_x_min is None
            else EXTRACTION_X_CONDITION
        ),
        "n_input_frames": n_input,
        "n_selected_frames": n_selected,
        "actual_time_start": evolution.time[0],
        "actual_time_end": evolution.time[-1],
        "target_time_spacing": spacing,
        "threshold": evolution.threshold,
        "z_target": evolution.z_target,
        "nx": evolution.nx,
        "native_ny": evolution.native_ny,
        "dense_ny": evolution.dense_ny,
        "y_upsample_factor": evolution.y_upsample_factor,
        "y_min": plan_metadata["ymin"],
        "y_max_periodic_endpoint": plan_metadata["ymax_periodic_endpoint"],
        "periodic_endpoint_included": False,
        "algorithm_version": plan_metadata[
            "spectral_horizontal_algorithm_version"
        ],
        "inverse_mapping_target_count": plan_metadata[
            "inverse_mapping_target_count"
        ],
        "inverse_mapping_success_count": plan_metadata[
            "inverse_mapping_success_count"
        ],
        "inverse_mapping_failure_count": plan_metadata[
            "inverse_mapping_failure_count"
        ],
        "ambiguous_boundary_point_count": plan_metadata[
            "ambiguous_boundary_point_count"
        ],
        "maximum_successful_residual": plan_metadata[
            "maximum_successful_residual"
        ],
        "maximum_iteration_count": plan_metadata["maximum_iteration_count"],
    }


def write_leading_edge_metadata_csv(
    path: str | Path,
    case: str,
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
    overwrite: bool,
) -> None:
    """Write one row describing processing, selection, grid, and inversion."""
    if evolution.sampling_mode != "uniform-spectral":
        raise ValueError("Refined-GLL metadata uses JSON; use write_leading_edge_csvs.")
    row = leading_edge_metadata(case, evolution, selection)
    _write_rows(
        Path(path),
        LEADING_EDGE_METADATA_COLUMNS,
        [row],
        overwrite,
    )


def write_leading_edge_csvs(
    output_dir: str | Path,
    case: str,
    evolution: LeadingEdgeEvolution,
    selection: LeadingEdgeTimeSelection,
    overwrite: bool,
) -> list[Path]:
    """Write timeseries and metadata after preflight of both paths.

    Uniform artifacts retain their original CSV schemas and bytes. Refined-GLL
    uses the same timeseries columns (blank uniform-only upsample factor) plus
    tagged JSON metadata; it is not a legacy uniform-spectral metadata CSV.
    """
    paths = [
        leading_edge_timeseries_path(output_dir, case),
        (leading_edge_sampling_metadata_path(output_dir, case)
         if evolution.sampling_mode == "refined-gll"
         else leading_edge_metadata_path(output_dir, case)),
    ]
    preflight_output_paths(paths, overwrite)
    write_leading_edge_timeseries_csv(
        paths[0], case, evolution, selection, overwrite=True
    )
    if evolution.sampling_mode == "refined-gll":
        paths[1].write_text(
            json.dumps(leading_edge_metadata(case, evolution, selection), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        write_leading_edge_metadata_csv(
            paths[1], case, evolution, selection, overwrite=True
        )
    return paths


__all__ = (
    "LEADING_EDGE_METADATA_COLUMNS",
    "LEADING_EDGE_TIMESERIES_COLUMNS",
    "leading_edge_evolution_pdf_path",
    "leading_edge_evolution_png_path",
    "leading_edge_metadata",
    "leading_edge_metadata_path",
    "leading_edge_sampling_metadata_path",
    "leading_edge_timeseries_path",
    "write_leading_edge_csvs",
    "write_leading_edge_metadata_csv",
    "write_leading_edge_timeseries_csv",
)
