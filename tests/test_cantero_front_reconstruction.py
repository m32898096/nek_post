from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import nek_post.cantero_front_reconstruction as reconstruction_module
from nek_post.cantero_front_reconstruction import (
    CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS,
    CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS,
    CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS,
    CanteroFrontReconstruction,
    build_cantero_front_reconstruction_summary,
    cantero_front_reconstruction_output_paths,
    compare_cantero_reconstructed_to_paper,
    reconstruct_cantero_mean_front,
    run_cantero_front_reconstruction,
)
from nek_post.cantero_mean_front import (
    STATUS_REFERENCE_BELOW_THRESHOLD,
    STATUS_SUCCESS,
    CanteroMeanFrontTimeseries,
    write_cantero_mean_front_timeseries_csv,
)


def _mean_front(
    *,
    time: np.ndarray | None = None,
    x_front: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    resolved_time = np.array([0.0, 1.5, 4.0]) if time is None else time
    resolved_x_front = np.array([10.0, 13.0, 18.0]) if x_front is None else x_front
    return {
        "time": resolved_time,
        "file_index": np.arange(1, resolved_time.size + 1),
        "x_front": resolved_x_front,
        "x_front_minus_initial": resolved_x_front - resolved_x_front[0],
    }


def _write_paper(path: Path, rows: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "x"))
        writer.writerows(rows)


def _phase_two_series_with_failure() -> CanteroMeanFrontTimeseries:
    return CanteroMeanFrontTimeseries(
        case="N7",
        file_index=np.array([1, 2, 3]),
        source_file=("f1", "f2", "f3"),
        time=np.array([0.0, 1.0, 2.0]),
        x_front=np.array([8.0, np.nan, 10.0]),
        x_front_minus_initial=np.array([0.0, np.nan, 2.0]),
        threshold=0.01,
        reference_x=0.0,
        left_index=np.array([1, -1, 1]),
        right_index=np.array([2, -1, 2]),
        x_left=np.array([7.0, np.nan, 9.0]),
        x_right=np.array([8.0, np.nan, 10.0]),
        h_left=np.array([0.02, np.nan, 0.02]),
        h_right=np.array([0.0, np.nan, 0.0]),
        crossing_count_in_search_region=np.array([1, 0, 1]),
        status=(STATUS_SUCCESS, STATUS_REFERENCE_BELOW_THRESHOLD, STATUS_SUCCESS),
    )


@pytest.mark.parametrize(
    ("method", "window", "polyorder"),
    (("moving_average", 7, 3), ("savgol", 5, 2)),
)
def test_reconstruction_delegates_absolute_phase_two_front_to_compute_kinematics(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    window: int,
    polyorder: int,
) -> None:
    calls: list[tuple[dict[str, np.ndarray], str, int, int]] = []

    def fake_compute(front: dict[str, np.ndarray], method: str, window: int, polyorder: int) -> dict[str, np.ndarray]:
        calls.append((front, method, window, polyorder))
        return {
            "v_raw": np.array([2.0, 2.0, 2.0]),
            "v_smooth": np.array([2.0, 2.0, 2.0]),
            "x_reconstructed": np.array([10.0, 13.0, 18.0]),
            "x_reconstruction_error": np.array([0.0, 0.0, 0.0]),
        }

    monkeypatch.setattr(reconstruction_module, "compute_kinematics", fake_compute)
    mean_front = _mean_front()
    result = reconstruct_cantero_mean_front(
        mean_front,
        smooth_method=method,
        smooth_window=window,
        savgol_polyorder=polyorder,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0]["time"], mean_front["time"])
    np.testing.assert_array_equal(calls[0][0]["x_front"], mean_front["x_front"])
    assert calls[0][1:] == (method, window, polyorder)
    np.testing.assert_array_equal(result.x_front_relative, [0.0, 3.0, 8.0])
    np.testing.assert_array_equal(result.x_reconstructed_relative, [0.0, 3.0, 8.0])


def test_constant_velocity_nonuniform_time_reconstructs_exactly() -> None:
    time = np.array([0.0, 0.5, 2.0, 4.5])
    x_front = 7.0 + 3.25 * time

    result = reconstruct_cantero_mean_front(
        _mean_front(time=time, x_front=x_front), smooth_window=1
    )

    np.testing.assert_allclose(result.v_raw, 3.25, atol=2.0e-14)
    np.testing.assert_allclose(result.v_smooth, 3.25, atol=2.0e-14)
    np.testing.assert_allclose(result.x_reconstructed, x_front, atol=2.0e-14)
    assert result.x_front_relative[0] == 0.0
    assert result.x_reconstructed_relative[0] == 0.0
    np.testing.assert_allclose(
        result.x_reconstruction_difference,
        result.x_reconstructed - result.x_front,
        atol=2.0e-14,
    )


def test_comparison_uses_reconstructed_relative_with_numerical_minus_paper_sign() -> None:
    reconstruction = CanteroFrontReconstruction(
        time=np.array([0.0, 2.0, 4.0]),
        file_index=np.array([1, 2, 3]),
        x_front=np.array([10.0, 30.0, 50.0]),
        x_front_relative=np.array([0.0, 20.0, 40.0]),
        v_raw=np.array([1.0, 1.0, 1.0]),
        v_smooth=np.array([1.0, 1.0, 1.0]),
        x_reconstructed=np.array([10.0, 12.0, 14.0]),
        x_reconstructed_relative=np.array([0.0, 2.0, 4.0]),
        x_reconstruction_difference=np.array([0.0, -18.0, -36.0]),
    )

    comparison = compare_cantero_reconstructed_to_paper(
        reconstruction,
        {"time": np.array([1.0, 3.0]), "x": np.array([0.5, 2.5])},
    )

    np.testing.assert_allclose(comparison.reconstructed_x_interp, [1.0, 3.0])
    np.testing.assert_allclose(comparison.difference, [0.5, 0.5])
    np.testing.assert_allclose(comparison.absolute_difference, [0.5, 0.5])


def test_summary_and_slumping_metrics_use_reconstructed_not_raw_trajectory() -> None:
    time = np.array([3.0, 6.0, 9.0, 12.0])
    raw_relative = 2.0 * (time - time[0])
    reconstructed_relative = time - time[0]
    paper_x = 0.5 * (time - time[0])
    reconstruction = CanteroFrontReconstruction(
        time=time,
        file_index=np.array([1, 2, 3, 4]),
        x_front=100.0 + raw_relative,
        x_front_relative=raw_relative,
        v_raw=np.full(time.size, 2.0),
        v_smooth=np.full(time.size, 1.0),
        x_reconstructed=10.0 + reconstructed_relative,
        x_reconstructed_relative=reconstructed_relative,
        x_reconstruction_difference=(10.0 + reconstructed_relative)
        - (100.0 + raw_relative),
    )
    comparison = compare_cantero_reconstructed_to_paper(
        reconstruction, {"time": time, "x": paper_x}
    )

    summary = build_cantero_front_reconstruction_summary(
        case="N7",
        front_csv="mean-front.csv",
        paper_csv="paper.csv",
        reconstruction=reconstruction,
        paper={"time": time},
        comparison=comparison,
        smooth_method="moving_average",
        smooth_window=1,
        savgol_polyorder=3,
        slump_tmin=3.0,
        slump_tmax=12.0,
    )

    expected_difference = reconstructed_relative - paper_x
    raw_difference = raw_relative - paper_x
    assert summary["mean_signed_paper_difference"] == pytest.approx(
        float(np.mean(expected_difference))
    )
    assert summary["mean_absolute_paper_difference"] == pytest.approx(
        float(np.mean(np.abs(expected_difference)))
    )
    assert summary["rms_paper_difference"] == pytest.approx(
        float(np.sqrt(np.mean(expected_difference**2)))
    )
    assert summary["max_absolute_paper_difference"] == pytest.approx(
        float(np.max(np.abs(expected_difference)))
    )
    assert summary["mean_signed_paper_difference"] != pytest.approx(
        float(np.mean(raw_difference))
    )
    assert summary["reconstructed_slumping_velocity"] == pytest.approx(1.0)
    assert summary["paper_slumping_velocity"] == pytest.approx(0.5)
    assert summary["slumping_velocity_difference"] == pytest.approx(0.5)
    assert summary["slumping_velocity_relative_difference"] == pytest.approx(1.0)


def test_outputs_contain_only_reconstructed_paper_comparison_artifacts() -> None:
    outputs = cantero_front_reconstruction_output_paths("/tmp/outputs", "N7")

    assert outputs.timeseries_csv.name == "N7_cantero_front_reconstruction_timeseries.csv"
    assert outputs.comparison_csv.name == "N7_cantero_front_reconstruction_comparison.csv"
    assert outputs.overlay_figure is not None
    assert "raw" not in " ".join(path.name for path in outputs.all_paths()).lower()
    assert "x_front_relative" not in CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS


def test_run_uses_public_phase_two_loader_and_writes_reconstructed_only_csvs(
    tmp_path: Path,
) -> None:
    front_csv = tmp_path / "mean-front.csv"
    paper_csv = tmp_path / "paper.csv"
    write_cantero_mean_front_timeseries_csv(
        front_csv, _phase_two_series_with_failure(), overwrite=False
    )
    _write_paper(paper_csv, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])

    run = run_cantero_front_reconstruction(
        case="N7",
        front_csv=front_csv,
        paper_csv=paper_csv,
        output_dir=tmp_path / "outputs",
        smooth_window=1,
        no_plots=True,
    )

    np.testing.assert_array_equal(run.reconstruction.file_index, [1, 3])
    np.testing.assert_allclose(run.reconstruction.x_front, [8.0, 10.0])
    assert run.outputs.overlay_figure is None
    with run.outputs.timeseries_csv.open(newline="", encoding="utf-8") as handle:
        timeseries_rows = list(csv.DictReader(handle))
    with run.outputs.comparison_csv.open(newline="", encoding="utf-8") as handle:
        comparison_rows = list(csv.DictReader(handle))
    with run.outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert tuple(timeseries_rows[0]) == CANTERO_RECONSTRUCTION_TIMESERIES_COLUMNS
    assert tuple(comparison_rows[0]) == CANTERO_RECONSTRUCTION_COMPARISON_COLUMNS
    assert tuple(summary_rows[0]) == CANTERO_RECONSTRUCTION_SUMMARY_COLUMNS
    assert "x_front_relative" not in comparison_rows[0]
    assert run.summary["n_front_points"] == 2
    assert run.summary["n_paper_points"] == 3
    assert run.summary["n_comparison_points"] == 3
    assert run.summary["max_absolute_reconstruction_difference"] == 0.0
    assert run.summary["max_absolute_paper_difference"] == 0.0


def test_overwrite_replaces_preflighted_phase_three_outputs(tmp_path: Path) -> None:
    front_csv = tmp_path / "mean-front.csv"
    paper_csv = tmp_path / "paper.csv"
    write_cantero_mean_front_timeseries_csv(
        front_csv, _phase_two_series_with_failure(), overwrite=False
    )
    _write_paper(paper_csv, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    first = run_cantero_front_reconstruction(
        case="N7",
        front_csv=front_csv,
        paper_csv=paper_csv,
        output_dir=tmp_path / "outputs",
        smooth_window=1,
        no_plots=True,
    )
    first.outputs.timeseries_csv.write_text("old contents", encoding="utf-8")

    second = run_cantero_front_reconstruction(
        case="N7",
        front_csv=front_csv,
        paper_csv=paper_csv,
        output_dir=tmp_path / "outputs",
        smooth_window=1,
        overwrite=True,
        no_plots=True,
    )

    assert second.outputs.timeseries_csv.read_text(encoding="utf-8").startswith("time,")


def test_partial_output_preflight_prevents_all_expensive_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = cantero_front_reconstruction_output_paths(
        tmp_path / "outputs", "N7", include_plots=True
    )
    original = "keep existing artifact"
    outputs.timeseries_csv.parent.mkdir(parents=True)
    outputs.timeseries_csv.write_text(original, encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("partial-output preflight must run before expensive work")

    monkeypatch.setattr(
        reconstruction_module, "read_cantero_mean_front_timeseries_csv", fail_if_called
    )
    monkeypatch.setattr(reconstruction_module, "read_digitized_paper_csv", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "compute_kinematics", fail_if_called)
    monkeypatch.setattr(
        reconstruction_module, "compare_cantero_reconstructed_to_paper", fail_if_called
    )
    monkeypatch.setattr(
        reconstruction_module, "write_cantero_front_reconstruction_overlay", fail_if_called
    )

    with pytest.raises(FileExistsError, match="Output exists"):
        run_cantero_front_reconstruction(
            case="N7",
            front_csv=tmp_path / "mean-front.csv",
            paper_csv=tmp_path / "paper.csv",
            output_dir=tmp_path / "outputs",
        )

    assert outputs.timeseries_csv.read_text(encoding="utf-8") == original
    assert not outputs.comparison_csv.exists()
    assert not outputs.summary_csv.exists()
    assert outputs.overlay_figure is not None
    assert not outputs.overlay_figure.exists()


def test_overlay_plots_only_paper_and_reconstructed_relative_displacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Axis:
        def plot(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def legend(self) -> None:
            return None

        def grid(self, *_args: object, **_kwargs: object) -> None:
            return None

    class Figure:
        def tight_layout(self) -> None:
            return None

        def savefig(self, _path: Path, **_kwargs: object) -> None:
            return None

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (Figure(), Axis()))
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    reconstruction = CanteroFrontReconstruction(
        time=np.array([0.0, 1.0]),
        file_index=np.array([1, 2]),
        x_front=np.array([9.0, 99.0]),
        x_front_relative=np.array([0.0, 90.0]),
        v_raw=np.array([1.0, 1.0]),
        v_smooth=np.array([1.0, 1.0]),
        x_reconstructed=np.array([9.0, 10.0]),
        x_reconstructed_relative=np.array([0.0, 1.0]),
        x_reconstruction_difference=np.array([0.0, -89.0]),
    )

    reconstruction_module.write_cantero_front_reconstruction_overlay(
        tmp_path / "overlay.png",
        paper={"time": np.array([0.0, 1.0]), "x": np.array([0.0, 1.0])},
        reconstruction=reconstruction,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0][1], reconstruction.x_reconstructed_relative)
    assert not np.array_equal(calls[1][0][1], reconstruction.x_front_relative)
