from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from nek_post import reconstructed_xt_comparison as comparison_module
from nek_post.front_kinematics import (
    front_velocity,
    integrate_velocity,
    moving_average,
)
from nek_post.reconstructed_xt_comparison import (
    FRONT_METHOD,
    PAPER_ROLE,
    ReconstructionInput,
    build_reconstructed_xt_summary_row,
    compare_reconstructed_to_paper,
    effective_smoothing_configuration,
    reconstruct_detected_front,
    run_n7_reconstructed_xt_fourway,
)
from nek_post.reconstructed_xt_comparison_io import (
    PAPER_COMPARISON_COLUMNS,
    RECONSTRUCTED_TIMESERIES_COLUMNS,
    SUMMARY_COLUMNS,
    reconstructed_xt_output_paths,
)


def _write_detected_csv(
    path: Path,
    rows: list[tuple[object, object, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "x_front_auto", "status"))
        writer.writerows(rows)


def _write_paper_csv(
    path: Path,
    rows: list[tuple[float, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "x"))
        writer.writerows(rows)


def test_reconstruction_maps_automatic_front_into_compute_kinematics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, np.ndarray], str, int, int]] = []

    def fake_compute(front, method, window, polyorder):
        calls.append((front, method, window, polyorder))
        return {
            "time": front["time"],
            "x_front": front["x_front"],
            "v_raw": np.array([1.0, 2.0, 3.0]),
            "v_smooth": np.array([1.5, 2.0, 2.5]),
            "x_reconstructed": np.array([-4.0, -2.5, 0.0]),
            "x_reconstruction_error": np.array([0.0, 0.5, 1.0]),
        }

    monkeypatch.setattr(
        comparison_module, "compute_kinematics", fake_compute
    )
    detected = {
        "time": np.array([2.0, 4.0, 7.0]),
        "x_front_auto": np.array([-4.0, -3.0, -1.0]),
    }

    reconstructed = reconstruct_detected_front(
        detected,
        smooth_method="moving_average",
        smooth_window=3,
        savgol_polyorder=2,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0]["time"], detected["time"])
    np.testing.assert_array_equal(
        calls[0][0]["x_front"], detected["x_front_auto"]
    )
    assert calls[0][1:] == ("moving_average", 3, 2)
    np.testing.assert_array_equal(
        reconstructed["x_front_auto"], [-4.0, -3.0, -1.0]
    )
    np.testing.assert_array_equal(
        reconstructed["x_detected_relative"], [0.0, 1.0, 3.0]
    )
    np.testing.assert_array_equal(
        reconstructed["x_reconstructed_relative"], [0.0, 1.5, 4.0]
    )
    np.testing.assert_array_equal(
        reconstructed["x_reconstruction_difference"], [0.0, 0.5, 1.0]
    )


def test_actual_reconstruction_reuses_gradient_moving_average_and_trapezoid() -> None:
    time = np.arange(5, dtype=np.float64)
    x_front = np.array([10.0, 11.0, 14.0, 15.0, 20.0])

    reconstructed = reconstruct_detected_front(
        {"time": time, "x_front_auto": x_front},
        smooth_method="moving_average",
        smooth_window=3,
        savgol_polyorder=3,
    )

    expected_raw = front_velocity(time, x_front)
    expected_smooth = moving_average(expected_raw, 3)
    expected_reconstructed = integrate_velocity(
        time, expected_smooth, x_front[0]
    )
    np.testing.assert_allclose(reconstructed["v_raw"], expected_raw)
    np.testing.assert_allclose(
        reconstructed["v_smooth"], expected_smooth
    )
    np.testing.assert_allclose(
        reconstructed["x_reconstructed"], expected_reconstructed
    )
    np.testing.assert_allclose(
        reconstructed["x_reconstruction_difference"],
        expected_reconstructed - x_front,
    )
    assert reconstructed["x_detected_relative"][0] == 0.0
    assert reconstructed["x_reconstructed_relative"][0] == 0.0
    assert (
        reconstructed["x_reconstructed"][-1]
        != reconstructed["x_front_auto"][-1]
    )


def test_savgol_velocity_smoothing_is_established_implementation() -> None:
    time = np.arange(7, dtype=float)
    x_front = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0])

    reconstructed = reconstruct_detected_front(
        {"time": time, "x_front_auto": x_front},
        smooth_method="savgol",
        smooth_window=5,
        savgol_polyorder=2,
    )

    expected_raw = front_velocity(time, x_front)
    np.testing.assert_allclose(
        reconstructed["v_smooth"],
        savgol_filter(
            expected_raw,
            window_length=5,
            polyorder=2,
            mode="interp",
        ),
    )


@pytest.mark.parametrize(
    ("method", "window", "polyorder", "n_points", "expected"),
    [
        ("moving_average", 4, 3, 8, (5, np.nan)),
        ("savgol", 12, 9, 6, (5, 4)),
        ("savgol", 1, 3, 6, (1, np.nan)),
    ],
)
def test_effective_smoothing_configuration(
    method: str,
    window: int,
    polyorder: int,
    n_points: int,
    expected: tuple[int, float],
) -> None:
    actual = effective_smoothing_configuration(
        method=method,
        window=window,
        polyorder=polyorder,
        n_points=n_points,
    )

    assert actual[0] == expected[0]
    if np.isnan(expected[1]):
        assert np.isnan(actual[1])
    else:
        assert actual[1] == expected[1]


def test_pairwise_comparison_is_overlap_only_and_uses_required_differences() -> None:
    reconstructed = {
        "time": np.array([1.0, 3.0, 5.0]),
        "x_reconstructed_relative": np.array([0.0, 4.0, 8.0]),
    }
    paper = {
        "time": np.array([0.0, 2.0, 4.0, 6.0]),
        "paper_x": np.array([8.0, 0.0, 7.0, 10.0]),
    }

    result = compare_reconstructed_to_paper(reconstructed, paper)

    np.testing.assert_array_equal(result["time"], [2.0, 4.0])
    np.testing.assert_array_equal(
        result["reconstructed_x_interp"], [2.0, 6.0]
    )
    np.testing.assert_array_equal(result["difference"], [2.0, -1.0])
    np.testing.assert_array_equal(
        result["absolute_difference"], [2.0, 1.0]
    )
    assert np.isnan(result["relative_difference"][0])
    assert result["relative_difference"][1] == pytest.approx(-1.0 / 7.0)
    assert np.isnan(result["log_difference"][0])
    assert result["log_difference"][1] == pytest.approx(
        np.log(6.0) - np.log(7.0)
    )


def test_pairwise_comparison_rejects_no_overlap_without_extrapolating() -> None:
    with pytest.raises(ValueError, match="do not overlap"):
        compare_reconstructed_to_paper(
            {
                "time": np.array([3.0, 4.0]),
                "x_reconstructed_relative": np.array([0.0, 1.0]),
            },
            {
                "time": np.array([0.0, 1.0]),
                "paper_x": np.array([0.0, 1.0]),
            },
        )


def test_nonpositive_reconstructed_value_gives_nan_log_difference() -> None:
    result = compare_reconstructed_to_paper(
        {
            "time": np.array([1.0, 2.0]),
            "x_reconstructed_relative": np.array([-1.0, 1.0]),
        },
        {
            "time": np.array([1.0, 2.0]),
            "paper_x": np.array([1.0, 1.0]),
        },
    )

    assert np.isnan(result["log_difference"][0])
    assert result["log_difference"][1] == 0.0


def test_summary_fields_metrics_and_slumping_velocities() -> None:
    time = np.array([3.0, 6.0, 9.0, 12.0])
    reconstruction_difference = np.array([0.0, 1.0, -1.0, 2.0])
    reconstructed = {
        "time": time,
        "x_front_auto": 2.0 * time + 10.0,
        "x_reconstructed": (
            2.0 * time + 10.0 + reconstruction_difference
        ),
        "x_reconstruction_difference": reconstruction_difference,
    }
    paper_x = 2.0 * time
    reconstructed_interp = 3.0 * time
    paper_difference = reconstructed_interp - paper_x
    comparison = {
        "time": time,
        "paper_x": paper_x,
        "reconstructed_x_interp": reconstructed_interp,
        "difference": paper_difference,
    }
    source = ReconstructionInput(
        "Re3450",
        "N7",
        3450,
        Path("re3450-front.csv"),
        Path("re3450-paper.csv"),
    )

    summary = build_reconstructed_xt_summary_row(
        reconstruction_input=source,
        reconstructed=reconstructed,
        paper={"time": time},
        comparison=comparison,
        smooth_method="savgol",
        smooth_window=5,
        savgol_polyorder=2,
        slump_tmin=3.0,
        slump_tmax=12.0,
    )

    assert summary["case"] == "N7"
    assert summary["reynolds_number"] == 3450
    assert summary["front_method"] == FRONT_METHOD
    assert summary["paper_role"] == PAPER_ROLE
    assert summary["paper_dataset"] == "Cantero_Fig5a_3D_Re3450"
    assert summary["smooth_window_effective"] == 3
    assert summary["savgol_polyorder_effective"] == 2
    assert summary["mean_signed_reconstruction_difference"] == 0.5
    assert summary["mean_absolute_reconstruction_difference"] == 1.0
    assert summary["rms_reconstruction_difference"] == pytest.approx(
        np.sqrt(1.5)
    )
    assert summary["max_absolute_reconstruction_difference"] == 2.0
    assert summary["final_reconstruction_difference"] == 2.0
    assert summary["mean_signed_paper_difference"] == 7.5
    assert summary["paper_slumping_velocity"] == pytest.approx(2.0)
    assert summary["reconstructed_slumping_velocity"] == pytest.approx(3.0)
    assert summary["slumping_velocity_difference"] == pytest.approx(1.0)
    assert summary["slumping_velocity_relative_difference"] == pytest.approx(
        0.5
    )


def test_unavailable_summary_metrics_are_nan_without_runtime_warnings() -> None:
    source = ReconstructionInput(
        "Re8950",
        "GC8950_N7",
        8950,
        Path("re8950-front.csv"),
        Path("re8950-paper.csv"),
    )
    with np.errstate(all="raise"):
        summary = build_reconstructed_xt_summary_row(
            reconstruction_input=source,
            reconstructed={
                "time": np.array([1.0, 2.0]),
                "x_front_auto": np.array([5.0, 6.0]),
                "x_reconstructed": np.array([5.0, 6.0]),
                "x_reconstruction_difference": np.array([0.0, 0.0]),
            },
            paper={"time": np.array([1.0, 2.0])},
            comparison={
                "time": np.array([1.0]),
                "paper_x": np.array([0.0]),
                "reconstructed_x_interp": np.array([0.0]),
                "difference": np.array([0.0]),
            },
            smooth_method="moving_average",
            smooth_window=11,
            savgol_polyorder=3,
            slump_tmin=3.0,
            slump_tmax=12.0,
        )

    assert np.isnan(summary["savgol_polyorder_effective"])
    assert np.isnan(summary["paper_slumping_velocity"])
    assert np.isnan(summary["reconstructed_slumping_velocity"])


def test_exact_output_filenames() -> None:
    outputs = reconstructed_xt_output_paths(Path("/tmp/output"))

    assert outputs.re3450_timeseries_csv.name == (
        "Re3450_N7_detected_front_reconstructed_xt.csv"
    )
    assert outputs.re8950_timeseries_csv.name == (
        "Re8950_N7_detected_front_reconstructed_xt.csv"
    )
    assert outputs.re3450_comparison_csv.name == (
        "Re3450_N7_reconstructed_vs_Cantero_Re3450.csv"
    )
    assert outputs.re8950_comparison_csv.name == (
        "Re8950_N7_reconstructed_vs_Cantero_Re8950.csv"
    )
    assert outputs.summary_csv.name == (
        "N7_Re3450_Re8950_reconstructed_xt_summary.csv"
    )
    assert [path.name for path in outputs.figures] == [
        "Re3450_N7_reconstructed_vs_Cantero_Re3450.png",
        "Re8950_N7_reconstructed_vs_Cantero_Re8950.png",
        "N7_Re3450_Re8950_reconstructed_fourway_overlay.png",
    ]


def test_end_to_end_loads_independent_fronts_pairs_papers_and_writes_csvs_only(
    tmp_path: Path,
) -> None:
    re3450_front = tmp_path / "re3450-detected.csv"
    re8950_front = tmp_path / "re8950-detected.csv"
    re3450_paper = tmp_path / "re3450-paper.csv"
    re8950_paper = tmp_path / "re8950-paper.csv"
    output_dir = tmp_path / "outputs"
    _write_detected_csv(
        re3450_front,
        [
            (0, 10, "selected_initial"),
            (1, "unused", "no_threshold_component"),
            (2, 12, "selected_tracked"),
            (4, 14, "selected_tracked"),
        ],
    )
    _write_detected_csv(
        re8950_front,
        [
            (1, -5, "selected_initial"),
            (3, -1, "selected_tracked"),
            (5, 3, "selected_tracked"),
            (7, "unused", "no_valid_temporal_candidate"),
        ],
    )
    _write_paper_csv(
        re3450_paper, [(0, 0), (1, 1), (2, 2), (4, 4)]
    )
    _write_paper_csv(
        re8950_paper, [(0, 100), (1, 0), (2, 2), (4, 6), (6, 10)]
    )

    outputs = run_n7_reconstructed_xt_fourway(
        re3450_front_csv=re3450_front,
        re8950_front_csv=re8950_front,
        re3450_paper_csv=re3450_paper,
        re8950_paper_csv=re8950_paper,
        output_dir=output_dir,
        smooth_method="moving_average",
        smooth_window=1,
        no_plots=True,
    )

    assert outputs.figures == ()
    assert all(path.is_file() for path in outputs.all_paths())
    assert not list(output_dir.glob("*.png"))
    re3450_rows = list(
        csv.DictReader(
            outputs.re3450_timeseries_csv.open(encoding="utf-8")
        )
    )
    re8950_rows = list(
        csv.DictReader(
            outputs.re8950_timeseries_csv.open(encoding="utf-8")
        )
    )
    assert list(re3450_rows[0]) == list(RECONSTRUCTED_TIMESERIES_COLUMNS)
    assert [float(row["time"]) for row in re3450_rows] == [0.0, 2.0, 4.0]
    assert [float(row["time"]) for row in re8950_rows] == [1.0, 3.0, 5.0]
    assert [float(row["x_front_auto"]) for row in re3450_rows] == [
        10.0,
        12.0,
        14.0,
    ]
    assert [float(row["x_front_auto"]) for row in re8950_rows] == [
        -5.0,
        -1.0,
        3.0,
    ]

    re3450_comparison = list(
        csv.DictReader(
            outputs.re3450_comparison_csv.open(encoding="utf-8")
        )
    )
    re8950_comparison = list(
        csv.DictReader(
            outputs.re8950_comparison_csv.open(encoding="utf-8")
        )
    )
    assert list(re3450_comparison[0]) == list(PAPER_COMPARISON_COLUMNS)
    assert [float(row["difference"]) for row in re3450_comparison] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [float(row["time"]) for row in re8950_comparison] == [
        1.0,
        2.0,
        4.0,
    ]
    assert [float(row["difference"]) for row in re8950_comparison] == [
        0.0,
        0.0,
        0.0,
    ]

    summary = list(
        csv.DictReader(outputs.summary_csv.open(encoding="utf-8"))
    )
    assert list(summary[0]) == list(SUMMARY_COLUMNS)
    assert [row["reynolds_number"] for row in summary] == ["3450", "8950"]
    assert [row["case"] for row in summary] == ["N7", "GC8950_N7"]
    assert [row["paper_dataset"] for row in summary] == [
        "Cantero_Fig5a_3D_Re3450",
        "Cantero_Fig5a_3D_Re8950",
    ]
    assert [row["front_source"] for row in summary] == [
        str(re3450_front),
        str(re8950_front),
    ]

    kept = outputs.summary_csv.read_bytes()
    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        run_n7_reconstructed_xt_fourway(
            re3450_front_csv=re3450_front,
            re8950_front_csv=re8950_front,
            re3450_paper_csv=re3450_paper,
            re8950_paper_csv=re8950_paper,
            output_dir=output_dir,
            smooth_window=1,
            no_plots=True,
        )
    assert outputs.summary_csv.read_bytes() == kept

    rerun = run_n7_reconstructed_xt_fourway(
        re3450_front_csv=re3450_front,
        re8950_front_csv=re8950_front,
        re3450_paper_csv=re3450_paper,
        re8950_paper_csv=re8950_paper,
        output_dir=output_dir,
        smooth_window=1,
        overwrite=True,
        no_plots=True,
    )
    assert rerun == outputs
