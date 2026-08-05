from __future__ import annotations

import csv
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.leading_edge_artifacts import read_leading_edge_artifacts
from nek_post.leading_edge_io import (
    LEADING_EDGE_METADATA_COLUMNS,
    LEADING_EDGE_TIMESERIES_COLUMNS,
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


def _evolution() -> LeadingEdgeEvolution:
    x_front = np.asarray(
        [[0.2, np.nan, 0.4, 0.3], [0.5, 0.6, 0.7, 0.8]],
        dtype=np.float64,
    )
    metadata = MappingProxyType(
        {
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax_periodic_endpoint": 1.0,
            "nx": 4,
            "native_ny": 2,
            "dense_ny": 4,
            "y_upsample_factor": 2,
            "z_target": 0.04,
            "spectral_horizontal_algorithm_version": 1,
            "inverse_mapping_target_count": 16,
            "inverse_mapping_success_count": 15,
            "inverse_mapping_failure_count": 1,
            "ambiguous_boundary_point_count": 2,
            "maximum_successful_residual": 3.0e-13,
            "maximum_iteration_count": 6,
        }
    )
    return LeadingEdgeEvolution(
        file_indices=_readonly([10, 20], np.int64),
        source_files=(Path("GC0.f00010"), Path("GC0.f00020")),
        time=_readonly([0.5, 0.75], np.float64),
        x=_readonly([0.0, 0.3, 0.6, 1.0], np.float64),
        y=_readonly([0.0, 0.25, 0.5, 0.75], np.float64),
        x_front=_readonly(x_front, np.float64),
        success_mask=_readonly(np.isfinite(x_front), np.bool_),
        crossing_count=_readonly([[1, 0, 2, 1], [1, 1, 1, 2]], np.int64),
        finite_leading_edge_fraction=_readonly([0.75, 1.0], np.float64),
        successful_y_count=_readonly([3, 4], np.int64),
        threshold=0.1,
        z_target=0.04,
        nx=4,
        native_ny=2,
        dense_ny=4,
        y_upsample_factor=2,
        horizontal_plan_metadata=metadata,
        periodic_endpoint_included=False,
    )


def _write_valid(tmp_path: Path) -> tuple[Path, Path]:
    evolution = _evolution()
    selection = select_leading_edge_times(evolution)
    paths = write_leading_edge_csvs(
        tmp_path,
        "N7",
        evolution,
        selection,
        overwrite=False,
    )
    return paths[0], paths[1]


def _read_dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_dict_rows(
    path: Path,
    columns: list[str] | tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_writer_round_trip_reconstructs_exact_selected_plot_data(
    tmp_path: Path,
) -> None:
    timeseries, metadata = _write_valid(tmp_path)

    result = read_leading_edge_artifacts(timeseries, metadata)

    assert result.case == "N7"
    assert result.extraction_method == "rightmost-crossing"
    assert_array_equal(result.file_indices, [10, 20])
    assert_allclose(result.target_time, [0.5, 0.75])
    assert_allclose(result.actual_time, [0.5, 0.75])
    assert_allclose(result.time_error, [0.0, 0.0])
    assert_array_equal(result.y, [0.0, 0.25, 0.5, 0.75])
    assert_allclose(result.x_front[0, [0, 2, 3]], [0.2, 0.4, 0.3])
    assert np.isnan(result.x_front[0, 1])
    assert_array_equal(
        result.success_mask,
        [[True, False, True, True], [True, True, True, True]],
    )
    assert_array_equal(
        result.crossing_count,
        [[1, 0, 2, 1], [1, 1, 1, 2]],
    )
    assert result.threshold == 0.1
    assert result.z_target == 0.04
    assert result.nx == 4
    assert result.native_ny == 2
    assert result.dense_ny == 4
    assert result.y_upsample_factor == 2
    assert result.y_min == 0.0
    assert result.y_max_periodic_endpoint == 1.0
    assert not result.periodic_endpoint_included
    assert result.target_time_spacing == 0.25


def test_all_plot_data_arrays_are_read_only_copies(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)

    result = read_leading_edge_artifacts(timeseries, metadata)

    expected_dtypes = {
        "file_indices": np.dtype(np.int64),
        "target_time": np.dtype(np.float64),
        "actual_time": np.dtype(np.float64),
        "time_error": np.dtype(np.float64),
        "y": np.dtype(np.float64),
        "x_front": np.dtype(np.float64),
        "success_mask": np.dtype(np.bool_),
        "crossing_count": np.dtype(np.int64),
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(result, name)
        assert array.dtype == dtype, name
        assert not array.flags.writeable, name


def test_scrambled_csv_rows_reconstruct_in_file_and_y_order(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(timeseries)
    _write_dict_rows(timeseries, columns, list(reversed(rows)))

    result = read_leading_edge_artifacts(timeseries, metadata)

    assert_array_equal(result.file_indices, [10, 20])
    assert_array_equal(result.y, [0.0, 0.25, 0.5, 0.75])
    assert result.x_front[0, 0] == 0.2
    assert result.x_front[1, -1] == 0.8


def test_missing_file_and_empty_artifact_are_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    with pytest.raises(FileNotFoundError, match="timeseries CSV not found"):
        read_leading_edge_artifacts(tmp_path / "missing.csv", metadata)

    timeseries.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="timeseries CSV is empty"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    _columns, rows = _read_dict_rows(timeseries)
    columns = [name for name in LEADING_EDGE_TIMESERIES_COLUMNS if name != "x_front"]
    for row in rows:
        row.pop("x_front")
    _write_dict_rows(timeseries, columns, rows)

    with pytest.raises(ValueError, match="missing columns: x_front"):
        read_leading_edge_artifacts(timeseries, metadata)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("case", "N9", "Inconsistent case"),
        ("threshold", "0.2", "Inconsistent threshold"),
        ("z_target", "0.05", "Inconsistent z_target"),
        ("nx", "8", "Inconsistent nx"),
    ),
)
def test_inconsistent_metadata_is_rejected(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(metadata)
    rows[0][column] = value
    _write_dict_rows(metadata, columns, rows)

    with pytest.raises(ValueError, match=message):
        read_leading_edge_artifacts(timeseries, metadata)


def test_duplicate_file_index_y_row_is_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(timeseries)
    rows.append(dict(rows[0]))
    _write_dict_rows(timeseries, columns, rows)

    with pytest.raises(ValueError, match="Duplicate.*file_index=10"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_inconsistent_y_grids_between_frames_are_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(timeseries)
    rows[5]["y"] = "0.3"
    _write_dict_rows(timeseries, columns, rows)

    with pytest.raises(ValueError, match="inconsistent y grids"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_dense_ny_must_equal_each_frame_y_row_count(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    timeseries_columns, timeseries_rows = _read_dict_rows(timeseries)
    metadata_columns, metadata_rows = _read_dict_rows(metadata)
    metadata_rows[0]["native_ny"] = "5"
    metadata_rows[0]["dense_ny"] = "5"
    metadata_rows[0]["y_upsample_factor"] = "1"
    for row in timeseries_rows:
        row["native_ny"] = "5"
        row["dense_ny"] = "5"
        row["y_upsample_factor"] = "1"
    _write_dict_rows(timeseries, timeseries_columns, timeseries_rows)
    _write_dict_rows(metadata, metadata_columns, metadata_rows)

    with pytest.raises(ValueError, match="has 4 y rows.*dense_ny is 5"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_dense_ny_factor_relation_is_validated(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(metadata)
    rows[0]["dense_ny"] = "5"
    _write_dict_rows(metadata, columns, rows)

    with pytest.raises(ValueError, match="dense_ny == y_upsample_factor"):
        read_leading_edge_artifacts(timeseries, metadata)


@pytest.mark.parametrize(
    ("artifact", "column"),
    (("timeseries", "success"), ("metadata", "periodic_endpoint_included")),
)
def test_malformed_booleans_are_rejected(
    tmp_path: Path,
    artifact: str,
    column: str,
) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    path = timeseries if artifact == "timeseries" else metadata
    columns, rows = _read_dict_rows(path)
    rows[0][column] = "not-a-boolean"
    _write_dict_rows(path, columns, rows)

    with pytest.raises(ValueError, match="malformed boolean"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_non_increasing_actual_times_are_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(timeseries)
    for row in rows[4:]:
        row["actual_time"] = "0.4"
    _write_dict_rows(timeseries, columns, rows)

    with pytest.raises(ValueError, match="actual times must be strictly increasing"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_malformed_numeric_value_is_rejected(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(timeseries)
    rows[0]["crossing_count"] = "one"
    _write_dict_rows(timeseries, columns, rows)

    with pytest.raises(ValueError, match="malformed integer crossing_count"):
        read_leading_edge_artifacts(timeseries, metadata)


def test_metadata_schema_remains_exact_writer_schema(tmp_path: Path) -> None:
    _timeseries, metadata = _write_valid(tmp_path)
    columns, _rows = _read_dict_rows(metadata)
    assert columns == list(LEADING_EDGE_METADATA_COLUMNS)


def test_metadata_rejects_unsupported_extraction_method(tmp_path: Path) -> None:
    timeseries, metadata = _write_valid(tmp_path)
    columns, rows = _read_dict_rows(metadata)
    rows[0]["extraction_method"] = "moore-boundary"
    _write_dict_rows(metadata, columns, rows)

    with pytest.raises(ValueError, match="method"):
        read_leading_edge_artifacts(timeseries, metadata)
