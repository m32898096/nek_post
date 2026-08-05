from __future__ import annotations

import csv
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import nek_post.leading_edge_io as leading_edge_io
from nek_post.leading_edge_io import (
    LEADING_EDGE_METADATA_COLUMNS,
    LEADING_EDGE_TIMESERIES_COLUMNS,
    leading_edge_evolution_pdf_path,
    leading_edge_evolution_png_path,
    leading_edge_metadata_path,
    leading_edge_timeseries_path,
    write_leading_edge_csvs,
)
from nek_post.leading_edge_workflow import (
    LeadingEdgeEvolution,
    select_leading_edge_times,
)


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _outputs() -> tuple[LeadingEdgeEvolution, object]:
    x_front = np.array([[0.2, np.nan, 0.4], [0.5, 0.6, 0.7]])
    metadata = MappingProxyType(
        {
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": -0.5,
            "ymax_periodic_endpoint": 1.0,
            "nx": 4,
            "native_ny": 3,
            "dense_ny": 3,
            "y_upsample_factor": 1,
            "z_target": 0.04,
            "spectral_horizontal_algorithm_version": 7,
            "inverse_mapping_target_count": 12,
            "inverse_mapping_success_count": 11,
            "inverse_mapping_failure_count": 1,
            "ambiguous_boundary_point_count": 2,
            "maximum_successful_residual": 3.0e-12,
            "maximum_iteration_count": 6,
        }
    )
    evolution = LeadingEdgeEvolution(
        file_indices=_readonly([12, 18], np.int64),
        source_files=(Path("GC0.f00012"), Path("GC0.f00018")),
        time=_readonly([1.0, 1.4], np.float64),
        x=_readonly([0.0, 0.3, 0.6, 1.0], np.float64),
        y=_readonly([-0.5, 0.0, 0.5], np.float64),
        x_front=_readonly(x_front, np.float64),
        success_mask=_readonly(np.isfinite(x_front), np.bool_),
        crossing_count=_readonly([[1, 0, 2], [1, 1, 1]], np.int64),
        finite_leading_edge_fraction=_readonly([2.0 / 3.0, 1.0], np.float64),
        successful_y_count=_readonly([2, 3], np.int64),
        threshold=0.1,
        z_target=0.04,
        nx=4,
        native_ny=3,
        dense_ny=3,
        y_upsample_factor=1,
        horizontal_plan_metadata=metadata,
        periodic_endpoint_included=False,
    )
    return evolution, select_leading_edge_times(evolution, spacing=None)


def test_exact_output_file_names(tmp_path: Path) -> None:
    assert leading_edge_timeseries_path(tmp_path, "N7").name == (
        "N7_leading_edge_timeseries.csv"
    )
    assert leading_edge_metadata_path(tmp_path, "N7").name == (
        "N7_leading_edge_metadata.csv"
    )
    assert leading_edge_evolution_png_path(tmp_path, "N7").name == (
        "N7_leading_edge_evolution.png"
    )
    assert leading_edge_evolution_pdf_path(tmp_path, "N7").name == (
        "N7_leading_edge_evolution.pdf"
    )


def test_csv_headers_rows_nan_boolean_and_metadata_values(tmp_path: Path) -> None:
    evolution, selection = _outputs()

    paths = write_leading_edge_csvs(
        tmp_path,
        "N7",
        evolution,
        selection,  # type: ignore[arg-type]
        overwrite=False,
    )

    with paths[0].open(newline="", encoding="utf-8") as handle:
        timeseries_rows = list(csv.reader(handle))
    with paths[1].open(newline="", encoding="utf-8") as handle:
        metadata_rows = list(csv.reader(handle))
    assert timeseries_rows[0] == list(LEADING_EDGE_TIMESERIES_COLUMNS)
    assert metadata_rows[0] == list(LEADING_EDGE_METADATA_COLUMNS)
    assert len(timeseries_rows) == 1 + 2 * 3
    assert len(metadata_rows) == 2

    x_column = LEADING_EDGE_TIMESERIES_COLUMNS.index("x_front")
    success_column = LEADING_EDGE_TIMESERIES_COLUMNS.index("success")
    y_column = LEADING_EDGE_TIMESERIES_COLUMNS.index("y")
    assert timeseries_rows[2][x_column] == "nan"
    assert timeseries_rows[2][success_column] == "False"
    assert timeseries_rows[1][success_column] == "True"
    assert [row[y_column] for row in timeseries_rows[1:4]] == ["-0.5", "0", "0.5"]
    assert [row[1] for row in timeseries_rows[1:]] == [
        "12",
        "12",
        "12",
        "18",
        "18",
        "18",
    ]

    metadata_row = dict(zip(metadata_rows[0], metadata_rows[1], strict=True))
    assert metadata_row["case"] == "N7"
    assert metadata_row["extraction_method"] == "rightmost-crossing"
    assert metadata_row["n_input_frames"] == "2"
    assert metadata_row["n_selected_frames"] == "2"
    assert metadata_row["actual_time_start"] == "1"
    assert metadata_row["actual_time_end"] == "1.4"
    assert metadata_row["target_time_spacing"] == "nan"
    assert metadata_row["threshold"] == "0.1"
    assert metadata_row["z_target"] == "0.04"
    assert metadata_row["y_min"] == "-0.5"
    assert metadata_row["y_max_periodic_endpoint"] == "1"
    assert metadata_row["periodic_endpoint_included"] == "False"
    assert metadata_row["algorithm_version"] == "7"
    assert metadata_row["inverse_mapping_target_count"] == "12"
    assert metadata_row["inverse_mapping_success_count"] == "11"
    assert metadata_row["inverse_mapping_failure_count"] == "1"
    assert metadata_row["ambiguous_boundary_point_count"] == "2"
    assert metadata_row["maximum_successful_residual"] == "3e-12"
    assert metadata_row["maximum_iteration_count"] == "6"


def test_io_module_has_no_matplotlib_dependency() -> None:
    assert "matplotlib" not in leading_edge_io.__dict__
    assert "plt" not in leading_edge_io.__dict__


def test_preflight_conflict_creates_no_partial_output(tmp_path: Path) -> None:
    evolution, selection = _outputs()
    conflict = leading_edge_metadata_path(tmp_path, "N7")
    conflict.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        write_leading_edge_csvs(
            tmp_path,
            "N7",
            evolution,
            selection,  # type: ignore[arg-type]
            overwrite=False,
        )

    assert not leading_edge_timeseries_path(tmp_path, "N7").exists()
    assert conflict.read_text(encoding="utf-8") == "keep"


def test_success_creates_parent_only_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, selection = _outputs()
    output_dir = tmp_path / "new" / "nested"
    original_preflight = leading_edge_io.preflight_output_paths
    observed: list[bool] = []

    def checking_preflight(paths: object, overwrite: bool) -> tuple[Path, ...]:
        observed.append(output_dir.exists())
        return original_preflight(paths, overwrite)  # type: ignore[arg-type]

    monkeypatch.setattr(
        leading_edge_io,
        "preflight_output_paths",
        checking_preflight,
    )
    write_leading_edge_csvs(
        output_dir,
        "N7",
        evolution,
        selection,  # type: ignore[arg-type]
        overwrite=False,
    )

    assert observed == [False]
    assert output_dir.is_dir()


def test_overwrite_false_rejects_and_true_replaces_both_csvs(tmp_path: Path) -> None:
    evolution, selection = _outputs()
    paths = [
        leading_edge_timeseries_path(tmp_path, "N7"),
        leading_edge_metadata_path(tmp_path, "N7"),
    ]
    for path in paths:
        path.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_leading_edge_csvs(
            tmp_path,
            "N7",
            evolution,
            selection,  # type: ignore[arg-type]
            overwrite=False,
        )
    assert all(path.read_text(encoding="utf-8") == "old" for path in paths)

    written = write_leading_edge_csvs(
        tmp_path,
        "N7",
        evolution,
        selection,  # type: ignore[arg-type]
        overwrite=True,
    )
    assert written == paths
    assert all(path.read_text(encoding="utf-8") != "old" for path in paths)
