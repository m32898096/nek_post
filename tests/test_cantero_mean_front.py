from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nek_post.cantero_mean_front as mean_front
from nek_post.cantero_mean_front import (
    STATUS_NO_DOWNWARD_CROSSING,
    STATUS_REFERENCE_BELOW_THRESHOLD,
    STATUS_SUCCESS,
    CanteroMeanFrontDetectionError,
    cantero_mean_front_timeseries_path,
    compute_cantero_mean_front_timeseries,
    detect_cantero_mean_front,
    read_cantero_mean_front_timeseries_csv,
    write_cantero_mean_front_timeseries_csv,
)


def test_linear_downward_crossing_uses_physical_linear_interpolation() -> None:
    result = detect_cantero_mean_front(
        [0.0, 1.0, 2.0, 3.0], [1.0, 0.5, 0.02, 0.0]
    )

    assert result.left_index == 2
    assert result.right_index == 3
    assert result.x_front == pytest.approx(2.5)
    assert result.h_left >= result.threshold > result.h_right


def test_nonuniform_gll_like_spacing_uses_x_not_sample_index() -> None:
    result = detect_cantero_mean_front(
        [-2.0, 0.0, 0.1, 5.0], [1.0, 0.2, 0.02, 0.0]
    )

    assert result.x_front == pytest.approx(2.55)
    assert result.x_left <= result.x_front <= result.x_right


def test_exact_threshold_node_returns_its_exact_coordinate() -> None:
    result = detect_cantero_mean_front([0.0, 1.0, 3.0], [1.0, 0.01, 0.0])

    assert result.left_index == 1
    assert result.x_front == 1.0


def test_exact_threshold_plateau_uses_its_final_node_before_below_threshold() -> None:
    result = detect_cantero_mean_front(
        [0.0, 1.0, 3.0, 6.0], [1.0, 0.01, 0.01, 0.0]
    )

    assert result.left_index == 2
    assert result.right_index == 3
    assert result.x_front == 3.0


def test_reference_tie_uses_first_lower_x_gll_node() -> None:
    result = detect_cantero_mean_front(
        [0.0, 2.0, 4.0], [0.02, 0.0, 0.0], reference_x=1.0
    )

    assert result.left_index == 0
    assert result.x_front == pytest.approx(1.0)


def test_reference_uses_selected_node_without_interpolating_at_reference() -> None:
    with pytest.raises(
        CanteroMeanFrontDetectionError,
        match="reference search node is below threshold",
    ) as error:
        detect_cantero_mean_front(
            [0.0, 2.0, 4.0], [1.0, 0.0, 0.0], reference_x=1.01
        )

    assert error.value.status == STATUS_REFERENCE_BELOW_THRESHOLD


def test_first_outward_crossing_wins_over_downstream_false_island() -> None:
    result = detect_cantero_mean_front(
        [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.02, 0.0, 0.03, 0.0]
    )

    assert result.left_index == 1
    assert result.x_front == pytest.approx(1.5)
    assert result.crossing_count_in_search_region == 2


def test_negative_downstream_spectral_value_is_not_clipped() -> None:
    result = detect_cantero_mean_front([0.0, 2.0, 5.0], [1.0, 0.02, -0.005])

    assert result.x_front == pytest.approx(3.2)
    assert result.h_right == -0.005


def test_invalid_reference_and_threshold_conditions_fail_explicitly() -> None:
    with pytest.raises(CanteroMeanFrontDetectionError, match="outside"):
        detect_cantero_mean_front([0.0, 1.0], [1.0, 0.0], reference_x=2.0)
    with pytest.raises(
        CanteroMeanFrontDetectionError,
        match="below threshold",
    ) as error:
        detect_cantero_mean_front([0.0, 1.0], [0.0, 1.0])
    assert error.value.status == STATUS_REFERENCE_BELOW_THRESHOLD
    with pytest.raises(CanteroMeanFrontDetectionError, match="No downward") as error:
        detect_cantero_mean_front([0.0, 1.0], [1.0, 0.5])
    assert error.value.status == STATUS_NO_DOWNWARD_CROSSING


@pytest.mark.parametrize(
    ("x", "height", "message"),
    (
        ([0.0, 0.0], [1.0, 0.0], "strictly increasing"),
        ([0.0, 1.0], [1.0, np.nan], "must be finite"),
        ([0.0, 1.0], [1.0], "identical lengths"),
        ([0.0], [1.0], "at least two"),
    ),
)
def test_invalid_profile_inputs_are_rejected(
    x: list[float], height: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_cantero_mean_front(x, height)


def _compute_timeseries_for_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profiles: list[np.ndarray],
    times: list[float],
) -> object:
    frames = tuple(
        SimpleNamespace(index=index, path=tmp_path / f"GC0.f{index:05d}")
        for index in range(1, len(profiles) + 1)
    )
    data_by_path = {
        frame.path: SimpleNamespace(time=time)
        for frame, time in zip(frames, times, strict=True)
    }
    profiles_by_path = dict(zip((frame.path for frame in frames), profiles, strict=True))
    plan = object()
    monkeypatch.setattr(mean_front, "build_cantero_equivalent_height_plan", lambda _: plan)

    def apply(supplied_plan: object, _data: object, *, source_file: Path) -> object:
        if supplied_plan is not plan:
            pytest.fail("timeseries must reuse its one Phase-1 plan")
        return SimpleNamespace(
            x_coordinates=np.array([0.0, 1.0, 2.0]),
            span_averaged_height=profiles_by_path[source_file],
        )

    monkeypatch.setattr(
        mean_front,
        "apply_cantero_equivalent_height_plan",
        apply,
    )
    return compute_cantero_mean_front_timeseries(
        frames, case="N7", reader=data_by_path.__getitem__
    )


def test_timeseries_first_failure_uses_first_later_success_as_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    series = _compute_timeseries_for_profiles(
        monkeypatch,
        tmp_path,
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.02, 0.0]),
            np.array([1.0, 0.04, 0.0]),
        ],
        [0.0, 1.0, 2.0],
    )

    assert series.status == (
        STATUS_REFERENCE_BELOW_THRESHOLD,
        STATUS_SUCCESS,
        STATUS_SUCCESS,
    )
    assert np.isnan(series.x_front[0])
    assert np.isnan(series.x_front_minus_initial[0])
    np.testing.assert_allclose(series.x_front[1:], [1.5, 1.75])
    np.testing.assert_allclose(series.x_front_minus_initial[1:], [0.0, 0.25])


def test_timeseries_failure_between_successes_is_not_forward_filled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    series = _compute_timeseries_for_profiles(
        monkeypatch,
        tmp_path,
        [
            np.array([1.0, 0.02, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.04, 0.0]),
        ],
        [0.0, 1.0, 2.0],
    )

    assert series.status == (
        STATUS_SUCCESS,
        STATUS_REFERENCE_BELOW_THRESHOLD,
        STATUS_SUCCESS,
    )
    np.testing.assert_allclose(series.x_front[[0, 2]], [1.5, 1.75])
    np.testing.assert_allclose(series.x_front_minus_initial[[0, 2]], [0.0, 0.25])
    assert np.isnan(series.x_front[1])
    assert np.isnan(series.x_front_minus_initial[1])


def test_timeseries_reuses_one_phase_one_plan_and_keeps_failed_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = tuple(
        SimpleNamespace(index=index, path=tmp_path / f"GC0.f{index:05d}")
        for index in (1, 2, 3)
    )
    data_by_path = {
        frame.path: SimpleNamespace(time=0.5 * position)
        for position, frame in enumerate(frames, start=1)
    }
    profiles = {
        frames[0].path: np.array([1.0, 0.02, 0.0]),
        frames[1].path: np.array([1.0, 0.0, 0.0]),
        frames[2].path: np.array([0.0, 0.0, 0.0]),
    }
    calls: dict[str, list[object]] = {"build": [], "apply": []}
    plan = object()

    def build(data: object) -> object:
        calls["build"].append(data)
        return plan

    def apply(supplied_plan: object, data: object, *, source_file: Path) -> object:
        assert supplied_plan is plan
        calls["apply"].append((data, source_file))
        return SimpleNamespace(
            x_coordinates=np.array([0.0, 1.0, 2.0]),
            span_averaged_height=profiles[source_file],
        )

    monkeypatch.setattr(mean_front, "build_cantero_equivalent_height_plan", build)
    monkeypatch.setattr(mean_front, "apply_cantero_equivalent_height_plan", apply)
    series = compute_cantero_mean_front_timeseries(
        frames,
        case="N7",
        reader=data_by_path.__getitem__,
    )

    assert len(calls["build"]) == 1
    assert len(calls["apply"]) == 3
    np.testing.assert_array_equal(series.file_index, [1, 2, 3])
    np.testing.assert_allclose(series.time, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(series.x_front[:2], [1.5, 0.99])
    assert np.isnan(series.x_front[2])
    np.testing.assert_allclose(series.x_front_minus_initial[:2], [0.0, -0.51])
    assert np.isnan(series.x_front_minus_initial[2])
    assert series.status == (
        STATUS_SUCCESS,
        STATUS_SUCCESS,
        STATUS_REFERENCE_BELOW_THRESHOLD,
    )

    path = cantero_mean_front_timeseries_path(tmp_path, "N7")
    write_cantero_mean_front_timeseries_csv(path, series, overwrite=False)
    loaded = read_cantero_mean_front_timeseries_csv(path)
    np.testing.assert_array_equal(loaded["file_index"], [1, 2])
    np.testing.assert_allclose(loaded["x_front"], [1.5, 0.99])
    np.testing.assert_allclose(loaded["x_front_minus_initial"], [0.0, -0.51])


def test_timeseries_rejects_nonincreasing_frame_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = tuple(
        SimpleNamespace(index=index, path=tmp_path / f"GC0.f{index:05d}")
        for index in (1, 2)
    )
    data_by_path = {
        frames[0].path: SimpleNamespace(time=1.0),
        frames[1].path: SimpleNamespace(time=1.0),
    }
    monkeypatch.setattr(mean_front, "build_cantero_equivalent_height_plan", lambda _: object())
    monkeypatch.setattr(
        mean_front,
        "apply_cantero_equivalent_height_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            x_coordinates=np.array([0.0, 1.0]),
            span_averaged_height=np.array([1.0, 0.0]),
        ),
    )

    with pytest.raises(ValueError, match="times must be strictly increasing"):
        compute_cantero_mean_front_timeseries(
            frames, case="N7", reader=data_by_path.__getitem__
        )


def test_timeseries_rejects_decreasing_frame_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = tuple(
        SimpleNamespace(index=index, path=tmp_path / f"GC0.f{index:05d}")
        for index in (1, 2, 3)
    )
    data_by_path = {
        frame.path: SimpleNamespace(time=time)
        for frame, time in zip(frames, [0.0, 2.0, 1.0], strict=True)
    }
    monkeypatch.setattr(mean_front, "build_cantero_equivalent_height_plan", lambda _: object())
    monkeypatch.setattr(
        mean_front,
        "apply_cantero_equivalent_height_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            x_coordinates=np.array([0.0, 1.0]),
            span_averaged_height=np.array([1.0, 0.0]),
        ),
    )

    with pytest.raises(ValueError, match="times must be strictly increasing"):
        compute_cantero_mean_front_timeseries(
            frames, case="N7", reader=data_by_path.__getitem__
        )
