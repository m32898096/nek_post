import numpy as np
import pytest

from nek_post.front_kinematics_io import SUMMARY_COLUMNS
from nek_post.front_kinematics_reporting import (
    build_front_kinematics_summary_row,
    format_front_kinematics_summary_table,
)


DISPLAY_COLUMNS = (
    "case",
    "n_points",
    "time_start",
    "time_end",
    "x_start",
    "x_end",
    "mean_abs_reconstruction_error",
    "max_abs_reconstruction_error",
    "rms_reconstruction_error",
    "final_reconstruction_error",
    "slumping_velocity_raw_position_fit",
    "slumping_velocity_reconstructed_position_fit",
)


def _kinematics() -> dict[str, np.ndarray]:
    time = np.arange(16, dtype=float)
    x_front = 2.0 * time + 1.0
    x_reconstructed = 3.0 * time - 2.0
    return {
        "time": time,
        "x_front": x_front,
        "v_raw": np.linspace(-2.0, 4.0, time.size),
        "v_smooth": np.linspace(-1.0, 3.0, time.size),
        "x_reconstructed": x_reconstructed,
        "x_reconstruction_error": x_reconstructed - x_front,
    }


def test_summary_row_has_exact_keys_extrema_error_metrics_and_slumping_fits() -> None:
    row = build_front_kinematics_summary_row("N7", _kinematics())

    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["case"] == "N7"
    assert row["n_points"] == "16"
    assert row["time_start"] == "0"
    assert row["time_end"] == "15"
    assert row["x_start"] == "1"
    assert row["x_end"] == "31"
    assert row["v_raw_min"] == "-2"
    assert row["v_raw_max"] == "4"
    assert row["v_smooth_min"] == "-1"
    assert row["v_smooth_max"] == "3"
    assert float(row["mean_abs_reconstruction_error"]) == pytest.approx(5.25)
    assert float(row["max_abs_reconstruction_error"]) == pytest.approx(12.0)
    assert float(row["rms_reconstruction_error"]) == pytest.approx(np.sqrt(41.5))
    assert float(row["final_reconstruction_error"]) == pytest.approx(12.0)
    assert float(row["slumping_velocity_raw_position_fit"]) == pytest.approx(2.0)
    assert float(row["slumping_velocity_reconstructed_position_fit"]) == pytest.approx(3.0)


def _table_row(case: str, n_points: str, value: str) -> dict[str, str]:
    return {
        "case": case,
        "n_points": n_points,
        "time_start": value,
        "time_end": value,
        "x_start": value,
        "x_end": value,
        "mean_abs_reconstruction_error": value,
        "max_abs_reconstruction_error": value,
        "rms_reconstruction_error": value,
        "final_reconstruction_error": value,
        "slumping_velocity_raw_position_fit": value,
        "slumping_velocity_reconstructed_position_fit": value,
    }


def test_complete_terminal_summary_is_deterministic_and_preserves_case_order() -> None:
    first = _table_row("N7", "4", "1")
    second = _table_row("N5", "12", "2")

    actual = format_front_kinematics_summary_table([first, second])
    widths = {column: len(column) for column in DISPLAY_COLUMNS}
    expected = "\n".join(
        [
            "Front kinematics summary:",
            "  ".join(DISPLAY_COLUMNS),
            "  ".join("-" * widths[column] for column in DISPLAY_COLUMNS),
            "  ".join(first[column].ljust(widths[column]) for column in DISPLAY_COLUMNS),
            "  ".join(second[column].ljust(widths[column]) for column in DISPLAY_COLUMNS),
        ]
    )

    assert actual == expected
    assert actual.splitlines()[3].startswith("N7")
    assert actual.splitlines()[4].startswith("N5")
