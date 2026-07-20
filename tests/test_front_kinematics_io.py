import csv
from pathlib import Path

import numpy as np
import pytest

from nek_post.front_kinematics_io import (
    SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
    format_numeric_value,
    front_kinematics_summary_path,
    front_kinematics_timeseries_path,
    write_front_kinematics_summary_csv,
    write_front_kinematics_timeseries_csv,
)


def _kinematics() -> dict[str, np.ndarray]:
    return {
        "time": np.array([0.0, 1.0]),
        "x_front": np.array([1.0 / 3.0, 2.0]),
        "v_raw": np.array([1.0, 2.0]),
        "v_smooth": np.array([1.25, 1.75]),
        "x_reconstructed": np.array([1.0 / 3.0, 1.5]),
        "x_reconstruction_error": np.array([0.0, -0.5]),
    }


def _summary_row(case: str, n_points: int) -> dict[str, object]:
    return {
        "case": case,
        "n_points": n_points,
        "time_start": 0.0,
        "time_end": 15.0,
        "x_start": 1.0 / 3.0,
        "x_end": 31.0,
        "v_raw_min": 1.0,
        "v_raw_max": 2.0,
        "v_smooth_min": 1.25,
        "v_smooth_max": 1.75,
        "mean_abs_reconstruction_error": 0.25,
        "max_abs_reconstruction_error": 0.5,
        "rms_reconstruction_error": np.sqrt(0.125),
        "final_reconstruction_error": -0.5,
        "slumping_velocity_raw_position_fit": 2.0,
        "slumping_velocity_reconstructed_position_fit": 1.75,
    }


def test_exact_output_filenames(tmp_path: Path) -> None:
    assert front_kinematics_timeseries_path(tmp_path, "N7").name == "N7_front_kinematics_timeseries.csv"
    assert front_kinematics_summary_path(tmp_path).name == "front_kinematics_summary.csv"


def test_timeseries_writer_creates_parents_and_preserves_header_and_formatting(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "parents" / "timeseries.csv"

    write_front_kinematics_timeseries_csv(path, _kinematics(), overwrite=False)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(TIMESERIES_COLUMNS)
    assert rows[1][TIMESERIES_COLUMNS.index("x_front")] == "0.3333333333333333"


def test_summary_writer_preserves_header_integer_format_and_row_order(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "summary.csv"

    write_front_kinematics_summary_csv(
        path,
        [_summary_row("N7", 16), _summary_row("N5", 9)],
        overwrite=False,
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(SUMMARY_COLUMNS)
    assert rows[1][SUMMARY_COLUMNS.index("case")] == "N7"
    assert rows[1][SUMMARY_COLUMNS.index("n_points")] == "16"
    assert rows[2][SUMMARY_COLUMNS.index("case")] == "N5"
    assert rows[2][SUMMARY_COLUMNS.index("n_points")] == "9"


def test_numeric_formatting_uses_16_significant_digits_and_integer_strings() -> None:
    assert format_numeric_value(1.0 / 3.0) == "0.3333333333333333"
    assert format_numeric_value(17) == "17"


@pytest.mark.parametrize("kind", ["timeseries", "summary"])
def test_csv_overwrite_protection(kind: str, tmp_path: Path) -> None:
    path = tmp_path / "existing.csv"
    path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"Output exists: .* Pass --overwrite to replace it\."):
        if kind == "timeseries":
            write_front_kinematics_timeseries_csv(path, _kinematics(), overwrite=False)
        else:
            write_front_kinematics_summary_csv(path, [_summary_row("N5", 2)], overwrite=False)
    assert path.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("kind", ["timeseries", "summary"])
def test_csv_successful_overwrite(kind: str, tmp_path: Path) -> None:
    path = tmp_path / "existing.csv"
    path.write_text("old contents", encoding="utf-8")

    if kind == "timeseries":
        write_front_kinematics_timeseries_csv(path, _kinematics(), overwrite=True)
        expected_header = ",".join(TIMESERIES_COLUMNS)
    else:
        write_front_kinematics_summary_csv(path, [_summary_row("N5", 2)], overwrite=True)
        expected_header = ",".join(SUMMARY_COLUMNS)

    assert path.read_text(encoding="utf-8").splitlines()[0] == expected_header
    assert "old contents" not in path.read_text(encoding="utf-8")
