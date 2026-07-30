from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from nek_post.fig5a_detected_front import (
    COMPARISON_COLUMNS,
    FRONT_METHOD,
    PAPER_DATASET,
    PAPER_ROLE,
    SUMMARY_COLUMNS,
    automatic_front_relative_to_initial,
    build_detected_front_summary,
    detected_front_overlay_output_paths,
    run_detected_front_overlay,
)
from nek_post.front_compare import compare_automatic_front_to_paper
from nek_post.front_io import read_detected_front_timeseries_csv


def _write_detected_csv(path: Path, rows: list[tuple[object, object, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "x_front_auto", "status"))
        writer.writerows(rows)


def _write_paper_csv(path: Path) -> None:
    path.write_text(
        "time,x\n"
        "0,0\n"
        "1,1\n"
        "3,3\n"
        "4,4\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "header",
    [
        "x_front_auto,status\n",
        "time,status\n",
        "time,x_front_auto\n",
    ],
)
def test_detected_front_reader_requires_columns(
    tmp_path: Path, header: str
) -> None:
    path = tmp_path / "front.csv"
    path.write_text(header, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        read_detected_front_timeseries_csv(path)


def test_reader_retains_established_success_statuses_sorts_and_excludes_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "front.csv"
    _write_detected_csv(
        path,
        [
            (3, 13, "selected_tracked"),
            ("not-a-time", "not-a-front", "no_valid_temporal_candidate"),
            (1, 11, "selected_initial"),
            (4, 14, "selected_tracked"),
        ],
    )

    front = read_detected_front_timeseries_csv(path)

    assert front["time"].dtype == np.float64
    assert front["x_front_auto"].dtype == np.float64
    np.testing.assert_array_equal(front["time"], [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(front["x_front_auto"], [11.0, 13.0, 14.0])
    np.testing.assert_array_equal(
        front["status"],
        ["selected_initial", "selected_tracked", "selected_tracked"],
    )


@pytest.mark.parametrize(("time", "x_front"), [("nan", 1), (1, "inf")])
def test_reader_rejects_nonfinite_successful_rows(
    tmp_path: Path, time: object, x_front: object
) -> None:
    path = tmp_path / "front.csv"
    _write_detected_csv(
        path,
        [
            (time, x_front, "selected_initial"),
            (2, 2, "selected_tracked"),
        ],
    )

    with pytest.raises(ValueError, match="non-finite"):
        read_detected_front_timeseries_csv(path)


def test_reader_rejects_duplicate_retained_times(tmp_path: Path) -> None:
    path = tmp_path / "front.csv"
    _write_detected_csv(
        path,
        [
            (1, 10, "selected_initial"),
            (1, 11, "selected_tracked"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate retained time"):
        read_detected_front_timeseries_csv(path)


def test_reader_requires_two_successful_points(tmp_path: Path) -> None:
    path = tmp_path / "front.csv"
    _write_detected_csv(
        path,
        [
            (1, 10, "selected_initial"),
            (2, "nan", "no_threshold_component"),
        ],
    )

    with pytest.raises(ValueError, match="at least two successful"):
        read_detected_front_timeseries_csv(path)


def test_relative_front_uses_first_success_without_abs_or_time_shift() -> None:
    front = {
        "time": np.array([2.0, 4.0, 8.0]),
        "x_front_auto": np.array([-2.0, -3.0, 1.0]),
        "status": np.array(
            ["selected_initial", "selected_tracked", "selected_tracked"]
        ),
    }

    relative = automatic_front_relative_to_initial(front)

    assert relative["automatic_x0"] == -2.0
    np.testing.assert_array_equal(relative["time"], [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(relative["x_relative"], [0.0, -1.0, 3.0])


def test_comparison_uses_overlap_interpolation_and_difference_definitions() -> None:
    automatic = {
        "time": np.array([1.0, 3.0, 5.0]),
        "x_relative": np.array([0.0, 4.0, 8.0]),
    }
    paper = {
        "time": np.array([0.0, 2.0, 4.0, 6.0]),
        "paper_x": np.array([9.0, 0.0, 7.0, 9.0]),
    }

    comparison = compare_automatic_front_to_paper(automatic, paper)

    np.testing.assert_array_equal(comparison["time"], [2.0, 4.0])
    np.testing.assert_array_equal(
        comparison["automatic_x_interp"], [2.0, 6.0]
    )
    np.testing.assert_array_equal(comparison["difference"], [2.0, -1.0])
    np.testing.assert_array_equal(
        comparison["absolute_difference"], [2.0, 1.0]
    )
    assert np.isnan(comparison["relative_difference"][0])
    assert comparison["relative_difference"][1] == pytest.approx(-1.0 / 7.0)
    assert np.isnan(comparison["log_difference"][0])
    assert comparison["log_difference"][1] == pytest.approx(
        np.log(6.0) - np.log(7.0)
    )


def test_log_difference_is_nan_for_nonpositive_automatic_value() -> None:
    comparison = compare_automatic_front_to_paper(
        {
            "time": np.array([1.0, 2.0]),
            "x_relative": np.array([-1.0, 1.0]),
        },
        {
            "time": np.array([1.0, 2.0]),
            "paper_x": np.array([1.0, 1.0]),
        },
    )

    assert np.isnan(comparison["log_difference"][0])
    assert comparison["log_difference"][1] == 0.0


def test_comparison_rejects_nonoverlapping_ranges_without_extrapolation() -> None:
    with pytest.raises(ValueError, match="do not overlap"):
        compare_automatic_front_to_paper(
            {
                "time": np.array([3.0, 4.0]),
                "x_relative": np.array([0.0, 1.0]),
            },
            {
                "time": np.array([1.0, 2.0]),
                "paper_x": np.array([1.0, 2.0]),
            },
        )


def test_summary_statistics_and_slumping_velocities() -> None:
    time = np.array([3.0, 6.0, 9.0, 12.0])
    paper_x = 2.0 * time
    automatic_x = 3.0 * time
    difference = automatic_x - paper_x
    comparison = {
        "time": time,
        "paper_x": paper_x,
        "automatic_x_interp": automatic_x,
        "difference": difference,
        "absolute_difference": np.abs(difference),
        "relative_difference": difference / paper_x,
        "log_difference": np.log(automatic_x) - np.log(paper_x),
    }
    summary = build_detected_front_summary(
        case="GC8950_N7",
        front_source=Path("automatic.csv"),
        automatic_front={
            "time": np.array([0.0, 12.0]),
            "automatic_x0": -2.5,
        },
        paper={"time": time},
        comparison=comparison,
        slump_tmin=3.0,
        slump_tmax=12.0,
    )

    assert summary["paper_dataset"] == PAPER_DATASET
    assert summary["paper_role"] == PAPER_ROLE
    assert summary["front_method"] == FRONT_METHOD
    assert summary["automatic_x0"] == -2.5
    assert summary["n_automatic_points"] == 2
    assert summary["n_paper_points"] == 4
    assert summary["n_comparison_points"] == 4
    assert summary["mean_signed_difference"] == pytest.approx(7.5)
    assert summary["mean_absolute_difference"] == pytest.approx(7.5)
    assert summary["rms_difference"] == pytest.approx(
        np.sqrt(np.mean(difference**2))
    )
    assert summary["max_absolute_difference"] == 12.0
    assert summary["mean_signed_relative_difference"] == 0.5
    assert summary["rms_log_difference"] == pytest.approx(np.log(1.5))
    assert summary["n_slumping_points"] == 4
    assert summary["paper_slumping_velocity"] == pytest.approx(2.0)
    assert summary["automatic_slumping_velocity"] == pytest.approx(3.0)
    assert summary["slumping_velocity_difference"] == pytest.approx(1.0)
    assert summary["slumping_velocity_relative_difference"] == pytest.approx(
        0.5
    )


def test_summary_writes_nan_metrics_without_runtime_warning() -> None:
    comparison = {
        "time": np.array([4.0]),
        "paper_x": np.array([0.0]),
        "automatic_x_interp": np.array([0.0]),
        "difference": np.array([0.0]),
        "absolute_difference": np.array([0.0]),
        "relative_difference": np.array([np.nan]),
        "log_difference": np.array([np.nan]),
    }
    with np.errstate(all="raise"):
        summary = build_detected_front_summary(
            case="GC8950_N7",
            front_source="automatic.csv",
            automatic_front={
                "time": np.array([1.0, 2.0]),
                "automatic_x0": 4.0,
            },
            paper={"time": np.array([4.0])},
            comparison=comparison,
            slump_tmin=3.0,
            slump_tmax=12.0,
        )

    assert np.isnan(summary["mean_signed_relative_difference"])
    assert np.isnan(summary["rms_log_difference"])
    assert np.isnan(summary["automatic_slumping_velocity"])


def test_deterministic_output_paths() -> None:
    outputs = detected_front_overlay_output_paths(
        Path("/tmp/results/GC8950_N7_Re8950"),
        "GC8950_N7",
    )

    assert outputs.comparison_csv.name == (
        "GC8950_N7_Re8950_fig5a_comparison.csv"
    )
    assert outputs.summary_csv.name == "GC8950_N7_Re8950_fig5a_summary.csv"
    assert [path.name for path in outputs.figures] == [
        "GC8950_N7_Re8950_fig5a_overlay_linear.png",
        "GC8950_N7_Re8950_fig5a_overlay_loglog.png",
        "GC8950_N7_Re8950_fig5a_difference.png",
        "GC8950_N7_Re8950_fig5a_slumping_overlay.png",
    ]


def test_no_plots_writes_csvs_and_overwrite_protection(
    tmp_path: Path,
) -> None:
    front_csv = tmp_path / "detected.csv"
    paper_csv = tmp_path / "paper.csv"
    output_dir = tmp_path / "outputs"
    _write_detected_csv(
        front_csv,
        [
            (0, 10, "selected_initial"),
            (2, 12, "selected_tracked"),
            (4, 14, "selected_tracked"),
        ],
    )
    _write_paper_csv(paper_csv)

    outputs = run_detected_front_overlay(
        case="GC8950_N7",
        front_csv=front_csv,
        paper_csv=paper_csv,
        output_dir=output_dir,
        no_plots=True,
    )

    assert outputs.figures == ()
    assert outputs.comparison_csv.is_file()
    assert outputs.summary_csv.is_file()
    assert list(csv.reader(outputs.comparison_csv.open(encoding="utf-8")))[
        0
    ] == list(COMPARISON_COLUMNS)
    summary_rows = list(
        csv.DictReader(outputs.summary_csv.open(encoding="utf-8"))
    )
    assert list(summary_rows[0]) == list(SUMMARY_COLUMNS)
    assert summary_rows[0]["automatic_x0"] == "10"
    assert not list(output_dir.glob("*.png"))

    comparison_before = outputs.comparison_csv.read_bytes()
    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        run_detected_front_overlay(
            case="GC8950_N7",
            front_csv=front_csv,
            paper_csv=paper_csv,
            output_dir=output_dir,
            no_plots=True,
        )
    assert outputs.comparison_csv.read_bytes() == comparison_before

    rerun = run_detected_front_overlay(
        case="GC8950_N7",
        front_csv=front_csv,
        paper_csv=paper_csv,
        output_dir=output_dir,
        overwrite=True,
        no_plots=True,
    )
    assert rerun == outputs
