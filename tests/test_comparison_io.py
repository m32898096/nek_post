from pathlib import Path
import re

import numpy as np
import pytest

from nek_post.comparison_io import (
    CONCENTRATION_ERROR_COLUMNS,
    FRONT_POSITION_COLUMNS,
    PRESSURE_ERROR_COLUMNS,
    VELOCITY_ERROR_COLUMNS,
    append_comparison_log,
    check_slice_files,
    comparison_log_path,
    comparison_metadata_path,
    concentration_error_table_path,
    concentration_interpolated_path,
    front_position_table_path,
    load_slice_file,
    pressure_error_table_path,
    pressure_interpolated_path,
    save_concentration_interpolated,
    save_pressure_interpolated,
    save_velocity_interpolated,
    slice_path,
    velocity_error_table_path,
    velocity_interpolated_path,
    write_comparison_set_metadata,
    write_concentration_error_table,
    write_front_position_table,
    write_pressure_error_table,
    write_velocity_error_table,
)
from nek_post.paths import ProjectPaths


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N5": tmp_path / "data" / "case_N5"},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])


def test_exact_comparison_artifact_paths(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    assert slice_path(paths, "N5", 79) == tmp_path / "postproc/slices/N5/slice_N5_f00079.npz"
    assert concentration_interpolated_path(paths, "N5", 79) == tmp_path / "postproc/interpolated/C/interp_C_N5_f00079.npz"
    assert velocity_interpolated_path(paths, "N5", 79) == tmp_path / "postproc/interpolated/velocity/interp_velocity_N5_f00079.npz"
    assert pressure_interpolated_path(paths, "N5", 79) == tmp_path / "postproc/interpolated/pressure/interp_pressure_N5_f00079.npz"
    assert concentration_error_table_path(paths, "t19p5") == tmp_path / "results/tables/concentration_error_t19p5.csv"
    assert front_position_table_path(paths, "t19p5") == tmp_path / "results/tables/front_position_t19p5.csv"
    assert velocity_error_table_path(paths, "t19p5") == tmp_path / "results/tables/velocity_error_t19p5.csv"
    assert pressure_error_table_path(paths, "t19p5") == tmp_path / "results/tables/pressure_error_t19p5.csv"
    assert comparison_metadata_path(paths, "t19p5") == tmp_path / "results/tables/comparison_set_t19p5_metadata.txt"
    assert comparison_log_path(paths) == tmp_path / "postproc/logs/compare_poly_orders.log"


def test_slice_npz_loading_closes_archive(tmp_path: Path) -> None:
    path = tmp_path / "slice.npz"
    np.savez(path, x=np.array([1.0, 2.0]), time=19.5)

    loaded = load_slice_file(path)

    assert set(loaded) == {"x", "time"}
    assert np.array_equal(loaded["x"], np.array([1.0, 2.0]))
    path.unlink()
    assert not path.exists()


def test_missing_slice_validation_includes_regeneration_command(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        check_slice_files(paths, {"N5": 79})

    message = str(exc_info.value)
    assert "Missing required slice files:" in message
    assert str(slice_path(paths, "N5", 79)) in message
    assert "python scripts/02_extract_midspan_slice.py --case N5 --index 79" in message


def test_concentration_npz_schema(tmp_path: Path) -> None:
    Xi, Zi = _arrays()
    path = tmp_path / "nested/concentration.npz"
    source = tmp_path / "slice.npz"
    save_concentration_interpolated(path, Xi, Zi, Xi + Zi, "N5", 79, source, "linear")

    with np.load(path) as data:
        assert set(data.files) == {
            "Xi",
            "Zi",
            "C_grid",
            "case",
            "index",
            "source_slice_file",
            "interpolation_method",
        }
        assert data["case"].item() == "N5"
        assert data["index"].item() == 79
        assert data["source_slice_file"].item() == str(source)


def test_velocity_npz_schema_and_empty_comparison_set(tmp_path: Path) -> None:
    Xi, Zi = _arrays()
    path = tmp_path / "velocity.npz"
    save_velocity_interpolated(path, Xi, Zi, Xi, Zi, Xi + Zi, Xi - Zi, "N5", 79, None, tmp_path / "slice.npz", "linear")

    with np.load(path) as data:
        assert set(data.files) == {
            "Xi",
            "Zi",
            "u_grid",
            "v_grid",
            "w_grid",
            "speed_grid",
            "case",
            "index",
            "comparison_set",
            "source_slice_file",
            "interpolation_method",
        }
        assert data["comparison_set"].item() == ""


def test_pressure_npz_schema_and_empty_comparison_set(tmp_path: Path) -> None:
    Xi, Zi = _arrays()
    path = tmp_path / "pressure.npz"
    save_pressure_interpolated(path, Xi, Zi, Xi + Zi, Xi - Zi, "N5", 79, None, tmp_path / "slice.npz", "linear")

    with np.load(path) as data:
        assert set(data.files) == {
            "Xi",
            "Zi",
            "p_grid",
            "p_prime_grid",
            "case",
            "index",
            "comparison_set",
            "source_slice_file",
            "interpolation_method",
        }
        assert data["comparison_set"].item() == ""


@pytest.mark.parametrize(
    ("writer", "columns", "filename"),
    [
        (write_concentration_error_table, CONCENTRATION_ERROR_COLUMNS, "concentration.csv"),
        (write_front_position_table, FRONT_POSITION_COLUMNS, "front.csv"),
        (write_velocity_error_table, VELOCITY_ERROR_COLUMNS, "velocity.csv"),
        (write_pressure_error_table, PRESSURE_ERROR_COLUMNS, "pressure.csv"),
    ],
)
def test_csv_writers_preserve_exact_column_order(writer, columns: tuple[str, ...], filename: str, tmp_path: Path) -> None:
    path = tmp_path / "tables" / filename
    writer(path, [{column: column for column in columns}])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(columns)
    assert lines[1] == ",".join(columns)


def test_metadata_text_structure(tmp_path: Path) -> None:
    path = tmp_path / "metadata.txt"
    write_comparison_set_metadata(path, "t19p5", 19.5, "N11", {"N5": 79, "N11": 40})

    assert path.read_text(encoding="utf-8") == (
        "comparison_set: t19p5\n"
        "target_time: 19.5\n"
        "reference_case: N11\n"
        "case_indices:\n"
        "  N5: 79\n"
        "  N11: 40\n"
    )


def test_append_log_timestamp_and_block_structure(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    append_comparison_log(paths, "first block")
    append_comparison_log(paths, "second block")

    text = comparison_log_path(paths).read_text(encoding="utf-8")
    timestamp = r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]"
    assert re.fullmatch(f"{timestamp}\nfirst block\n\n{timestamp}\nsecond block\n\n", text)
