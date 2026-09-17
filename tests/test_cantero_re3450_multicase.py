from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import nek_post.cantero_re3450_multicase as multicase_module
from nek_post.cantero_front_reconstruction import (
    CanteroFrontReconstruction,
    compare_cantero_reconstructed_to_paper,
)
from nek_post.cantero_re3450_multicase import (
    FORMAL_RE3450_CASES,
    MULTICASE_SUMMARY_COLUMNS,
    build_cantero_re3450_overlay_series,
    cantero_re3450_front_csvs,
    cantero_re3450_multicase_output_paths,
    compare_cantero_reconstructions_to_reference,
    run_cantero_re3450_multicase,
    write_cantero_re3450_multicase_overlays,
)


TIME = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
PAPER = {"time": TIME, "x": 0.5 * TIME}
SLOPES = {"N5": 1.0, "N7": 2.0, "N9": 3.0}


def _reconstruction(case: str) -> CanteroFrontReconstruction:
    slope = SLOPES[case]
    raw_slope = 10.0 + slope
    x0 = {"N5": 5.0, "N7": 7.0, "N9": 9.0}[case]
    x_front = x0 + raw_slope * TIME
    x_reconstructed = x0 + slope * TIME
    return CanteroFrontReconstruction(
        time=TIME,
        file_index=np.arange(1, TIME.size + 1),
        x_front=x_front,
        x_front_relative=x_front - x_front[0],
        v_raw=np.full(TIME.size, raw_slope),
        v_smooth=np.full(TIME.size, slope),
        x_reconstructed=x_reconstructed,
        x_reconstructed_relative=x_reconstructed - x_reconstructed[0],
        x_reconstruction_difference=x_reconstructed - x_front,
    )


def _front_csvs(root: Path) -> dict[str, Path]:
    return {case: root / f"{case}.csv" for case in FORMAL_RE3450_CASES}


def test_formal_cases_and_independent_phase_two_paths() -> None:
    paths = cantero_re3450_front_csvs("/results/cantero_mean_front")

    assert FORMAL_RE3450_CASES == ("N5", "N7", "N9")
    assert tuple(paths) == FORMAL_RE3450_CASES
    for case in FORMAL_RE3450_CASES:
        assert paths[case] == Path(
            f"/results/cantero_mean_front/{case}/{case}_cantero_mean_front_timeseries.csv"
        )
    assert len(set(paths.values())) == 3
    assert tuple(cantero_re3450_front_csvs(
        "/results/cantero_mean_front", ("N9", "N5", "N7"))) == FORMAL_RE3450_CASES


def test_run_reconstructs_each_absolute_front_once_and_writes_ordered_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    front_csvs = _front_csvs(tmp_path)
    case_by_path = {path: case for case, path in front_csvs.items()}
    input_fronts = {
        case: _reconstruction(case).x_front.copy() for case in FORMAL_RE3450_CASES
    }
    read_calls: list[Path] = []
    reconstruction_calls: list[np.ndarray] = []
    comparison_inputs: list[np.ndarray] = []

    def read_front(path: Path) -> dict[str, np.ndarray]:
        resolved = Path(path)
        read_calls.append(resolved)
        case = case_by_path[resolved]
        return {
            "time": TIME,
            "file_index": np.arange(1, TIME.size + 1),
            "x_front": input_fronts[case],
        }

    def reconstruct(mean_front: dict[str, np.ndarray], **_kwargs: object) -> CanteroFrontReconstruction:
        absolute_front = np.asarray(mean_front["x_front"])
        reconstruction_calls.append(absolute_front.copy())
        case = next(
            case
            for case, expected in input_fronts.items()
            if np.array_equal(absolute_front, expected)
        )
        return _reconstruction(case)

    original_compare = compare_cantero_reconstructed_to_paper

    def compare(reconstruction: CanteroFrontReconstruction, paper: object):
        comparison_inputs.append(reconstruction.x_reconstructed_relative.copy())
        return original_compare(reconstruction, paper)

    monkeypatch.setattr(multicase_module, "read_digitized_paper_csv", lambda *_args, **_kwargs: PAPER)
    monkeypatch.setattr(multicase_module, "read_cantero_mean_front_timeseries_csv", read_front)
    monkeypatch.setattr(multicase_module, "reconstruct_cantero_mean_front", reconstruct)
    monkeypatch.setattr(multicase_module, "compare_cantero_reconstructed_to_paper", compare)

    run = run_cantero_re3450_multicase(
        front_csvs=front_csvs,
        paper_csv=tmp_path / "paper.csv",
        output_dir=tmp_path / "outputs",
        no_plots=True,
    )

    assert run.cases == FORMAL_RE3450_CASES
    assert read_calls == [front_csvs[case] for case in FORMAL_RE3450_CASES]
    assert len(reconstruction_calls) == 3
    for position, case in enumerate(FORMAL_RE3450_CASES):
        np.testing.assert_array_equal(reconstruction_calls[position], input_fronts[case])
        np.testing.assert_array_equal(
            comparison_inputs[position], SLOPES[case] * TIME
        )
    assert run.outputs.linear_overlay is None
    assert run.outputs.loglog_overlay is None
    assert all(path.is_file() for path in run.outputs.all_paths())

    with run.outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == MULTICASE_SUMMARY_COLUMNS
    assert [row["case"] for row in rows] == ["N5", "N7", "N9"]
    for row, case in zip(rows, FORMAL_RE3450_CASES, strict=True):
        error_slope = SLOPES[case] - 0.5
        expected_difference = error_slope * TIME
        assert float(row["mean_signed_paper_difference"]) == pytest.approx(
            float(np.mean(expected_difference))
        )
        assert float(row["mean_absolute_paper_difference"]) == pytest.approx(
            float(np.mean(np.abs(expected_difference)))
        )
        assert float(row["rms_paper_difference"]) == pytest.approx(
            float(np.sqrt(np.mean(expected_difference**2)))
        )
        assert float(row["max_absolute_paper_difference"]) == pytest.approx(
            float(np.max(np.abs(expected_difference)))
        )
        assert float(row["slumping_velocity_reconstructed"]) == pytest.approx(
            SLOPES[case]
        )
        assert float(row["slumping_velocity_paper"]) == pytest.approx(0.5)
        assert float(row["slumping_velocity_relative_difference"]) == pytest.approx(
            error_slope / 0.5
        )


def test_linear_and_loglog_series_are_same_four_reconstructed_datasets() -> None:
    reconstructions = {
        case: _reconstruction(case) for case in FORMAL_RE3450_CASES
    }

    linear = build_cantero_re3450_overlay_series(
        PAPER, reconstructions, loglog=False
    )
    loglog = build_cantero_re3450_overlay_series(
        PAPER, reconstructions, loglog=True
    )

    assert [series.key for series in linear] == ["paper", "N5", "N7", "N9"]
    assert [series.key for series in loglog] == ["paper", "N5", "N7", "N9"]
    for position, case in enumerate(FORMAL_RE3450_CASES, start=1):
        np.testing.assert_array_equal(
            linear[position].displacement,
            reconstructions[case].x_reconstructed_relative,
        )
        positive = (
            (linear[position].time > 0.0)
            & (linear[position].displacement > 0.0)
        )
        np.testing.assert_array_equal(
            loglog[position].time, linear[position].time[positive]
        )
        np.testing.assert_array_equal(
            loglog[position].displacement,
            linear[position].displacement[positive],
        )
        assert not np.array_equal(
            linear[position].displacement, reconstructions[case].x_front_relative
        )


def test_loglog_mask_omits_zero_and_nonpositive_without_shifting() -> None:
    reconstructions = {
        case: _reconstruction(case) for case in FORMAL_RE3450_CASES
    }
    original = reconstructions["N5"]
    reconstructed_relative = np.array([0.0, -1.0, 2.0, 3.0, 4.0])
    x_reconstructed = original.x_reconstructed[0] + reconstructed_relative
    reconstructions["N5"] = CanteroFrontReconstruction(
        time=original.time,
        file_index=original.file_index,
        x_front=original.x_front,
        x_front_relative=original.x_front_relative,
        v_raw=original.v_raw,
        v_smooth=original.v_smooth,
        x_reconstructed=x_reconstructed,
        x_reconstructed_relative=reconstructed_relative,
        x_reconstruction_difference=x_reconstructed - original.x_front,
    )

    loglog = build_cantero_re3450_overlay_series(
        PAPER, reconstructions, loglog=True
    )
    n5 = loglog[1]

    np.testing.assert_array_equal(n5.time, [6.0, 9.0, 12.0])
    np.testing.assert_array_equal(n5.displacement, [2.0, 3.0, 4.0])
    assert np.all(n5.time > 0.0)
    assert np.all(n5.displacement > 0.0)


def test_overlay_writer_sends_four_series_to_each_combined_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = cantero_re3450_multicase_output_paths(tmp_path)
    reconstructions = {
        case: _reconstruction(case) for case in FORMAL_RE3450_CASES
    }
    calls: list[tuple[Path, tuple[str, ...], bool]] = []

    def plot(path: Path, series: object, *, loglog: bool) -> Path:
        keys = tuple(value.key for value in series)
        calls.append((path, keys, loglog))
        return path

    monkeypatch.setattr(multicase_module, "_plot_series", plot)

    written = write_cantero_re3450_multicase_overlays(
        outputs, paper=PAPER, reconstructions=reconstructions
    )

    assert written == (outputs.linear_overlay, outputs.loglog_overlay)
    assert calls == [
        (outputs.linear_overlay, ("paper", "N5", "N7", "N9"), False),
        (outputs.loglog_overlay, ("paper", "N5", "N7", "N9"), True),
    ]


def test_outputs_have_no_raw_comparison_artifacts() -> None:
    outputs = cantero_re3450_multicase_output_paths("/tmp/outputs")

    assert len(outputs.all_paths()) == 9
    assert all("raw" not in path.name.lower() for path in outputs.all_paths())
    assert len(outputs.timeseries_csvs) == 3
    assert len(outputs.comparison_csvs) == 3


def test_arbitrary_case_paths_preserve_order_and_add_reference_tables() -> None:
    cases = ("N7_H", "N7_VH", "N7_VVH")
    fronts = cantero_re3450_front_csvs("/results/h_refinement/mean_front", cases)
    outputs = cantero_re3450_multicase_output_paths(
        "/results/h_refinement/reconstruction", cases=cases,
        include_input_summary=True, reference_case="N7_VVH")

    assert tuple(fronts) == cases
    assert outputs.cases == cases
    assert tuple(outputs.timeseries_csvs) == cases
    assert outputs.input_summary_csv is not None
    assert outputs.reference_timeseries_csv is not None
    assert "N7_VVH" in outputs.reference_timeseries_csv.name
    assert len(outputs.all_paths()) == 12
    # Historical defaults remain byte-for-byte path compatible.
    assert len(cantero_re3450_multicase_output_paths("/tmp/p").all_paths()) == 9


def test_reference_comparison_uses_actual_case_times_without_extrapolation() -> None:
    def reconstruction(time: np.ndarray, slope: float) -> CanteroFrontReconstruction:
        absolute = 10.0 + slope * time
        relative = absolute - absolute[0]
        return CanteroFrontReconstruction(
            time=time, file_index=np.arange(1, time.size + 1), x_front=absolute,
            x_front_relative=relative, v_raw=np.full(time.size, slope),
            v_smooth=np.full(time.size, slope), x_reconstructed=absolute,
            x_reconstructed_relative=relative,
            x_reconstruction_difference=np.zeros(time.size),
        )

    cases = ("N7_H", "N7_VH", "N7_VVH")
    values = {
        "N7_H": reconstruction(np.array([-1.0, 0.5, 1.5, 3.0]), 2.0),
        "N7_VH": reconstruction(np.array([0.25, 1.25]), 1.5),
        "N7_VVH": reconstruction(np.array([0.0, 1.0, 2.0]), 1.0),
    }
    rows, summaries = compare_cantero_reconstructions_to_reference(
        values, cases=cases, reference_case="N7_VVH")

    h_rows = [row for row in rows if row["case"] == "N7_H"]
    assert [row["time"] for row in h_rows] == [0.5, 1.5]
    assert [row["file_index"] for row in h_rows] == [2, 3]
    assert all(0.0 <= row["time"] <= 2.0 for row in rows)
    ref_rows = [row for row in rows if row["case"] == "N7_VVH"]
    assert all(row["front_position_difference"] == 0.0 for row in ref_rows)
    assert all(row["raw_velocity_difference"] == 0.0 for row in ref_rows)
    ref_summary = next(row for row in summaries if row["case"] == "N7_VVH")
    assert ref_summary["is_reference"] is True
    assert ref_summary["max_absolute_reconstructed_front_difference"] == 0.0
    assert "no extrapolation" in ref_summary["time_alignment"]


def test_generic_run_writes_input_and_reference_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = ("N7_H", "N7_VH", "N7_VVH")
    front_csvs = {case: tmp_path / f"{case}.csv" for case in cases}
    time = np.array([0.0, 1.0, 2.0])
    slopes = dict(zip(cases, (1.2, 1.1, 1.0), strict=True))

    def read_front(path: Path) -> dict[str, object]:
        case = next(case for case, candidate in front_csvs.items() if candidate == Path(path))
        x = 8.0 + slopes[case] * time
        return {
            "time": time, "file_index": np.array([1, 2, 3]), "x_front": x,
            "x_front_minus_initial": x - x[0], "threshold": .01,
            "reference_x": 0.0, "n_input_frames": 4,
            "n_successful_frames": 3, "n_failed_frames": 1,
            "input_time_start": 0.0, "input_time_end": 3.0,
            "failed_file_indices": (4,), "failed_statuses": ("no_downward_crossing",),
        }

    monkeypatch.setattr(multicase_module, "read_cantero_mean_front_timeseries_csv", read_front)
    monkeypatch.setattr(multicase_module, "read_digitized_paper_csv",
                        lambda *_args, **_kwargs: {"time": time, "x": time})
    run = run_cantero_re3450_multicase(
        front_csvs=front_csvs, paper_csv=tmp_path / "paper.csv",
        output_dir=tmp_path / "out", cases=cases, smooth_window=1,
        no_plots=True, reference_case="N7_VVH", include_input_summary=True)

    assert run.outputs.input_summary_csv is not None
    assert run.outputs.reference_timeseries_csv is not None
    assert run.outputs.reference_summary_csv is not None
    assert all(path.is_file() for path in run.outputs.all_paths())
    with run.outputs.input_summary_csv.open(newline="", encoding="utf-8") as handle:
        inputs = list(csv.DictReader(handle))
    assert [row["case"] for row in inputs] == list(cases)
    assert all(row["threshold"] == "0.01" and row["reference_x"] == "0" for row in inputs)
    assert all(row["n_failed_frames"] == "1" and row["failed_file_indices"] == "4" for row in inputs)
    with run.outputs.reference_summary_csv.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert summary[-1]["case"] == "N7_VVH"
    assert summary[-1]["max_absolute_front_position_difference"] == "0"


def test_partial_output_preflight_stops_before_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = cantero_re3450_multicase_output_paths(tmp_path / "outputs")
    existing = outputs.timeseries_csvs["N5"]
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("preflight must run before reads, reconstruction, or plotting")

    monkeypatch.setattr(multicase_module, "read_digitized_paper_csv", fail_if_called)
    monkeypatch.setattr(multicase_module, "read_cantero_mean_front_timeseries_csv", fail_if_called)
    monkeypatch.setattr(multicase_module, "reconstruct_cantero_mean_front", fail_if_called)
    monkeypatch.setattr(multicase_module, "write_cantero_re3450_multicase_overlays", fail_if_called)

    with pytest.raises(FileExistsError, match="Output exists"):
        run_cantero_re3450_multicase(
            front_csvs=_front_csvs(tmp_path),
            paper_csv=tmp_path / "paper.csv",
            output_dir=tmp_path / "outputs",
        )

    assert existing.read_text(encoding="utf-8") == "keep"
    assert not any(path.exists() for path in outputs.all_paths() if path != existing)
