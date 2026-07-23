from dataclasses import fields

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.front_detection import (
    FrontComponent,
    FrontTrackingResult,
    STATUS_NO_THRESHOLD_COMPONENT,
    STATUS_NO_VALID_SPATIAL_CANDIDATE,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
    detect_front_components,
    filter_spatial_components,
    predict_front_position,
    select_initial_component,
    select_temporal_component,
    track_concentration_front,
)


def _grid(nz: int = 4, nx: int = 10) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    z = np.arange(nz, dtype=float)
    return np.meshgrid(x, z)


def _component(
    label: int,
    mask: np.ndarray,
    *,
    x_max: float,
    pixel_count: int | None = None,
    bottom_contact: bool = True,
) -> FrontComponent:
    count = int(np.count_nonzero(mask)) if pixel_count is None else pixel_count
    return FrontComponent(
        label=label,
        pixel_count=count,
        x_min=0.0,
        x_max=x_max,
        z_min=0.0,
        z_max=1.0,
        bottom_contact=bottom_contact,
        mask=mask,
    )


def _track(
    time: np.ndarray,
    Xi: np.ndarray,
    Zi: np.ndarray,
    frames: np.ndarray,
    **overrides,
) -> FrontTrackingResult:
    settings = {
        "threshold": 0.5,
        "min_component_pixels": 2,
        "bottom_rows": 1,
        "max_front_jump": 2.0,
        "connectivity": 8,
    }
    settings.update(overrides)
    return track_concentration_front(time, Xi, Zi, frames, **settings)


def test_threshold_is_strict_and_excludes_nonfinite_concentrations() -> None:
    Xi, Zi = _grid(nz=2, nx=4)
    concentration = np.array(
        [
            [0.5, 0.6, np.nan, np.inf],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    components = detect_front_components(
        Xi,
        Zi,
        concentration,
        threshold=0.5,
        bottom_rows=1,
    )

    assert len(components) == 1
    assert components[0].pixel_count == 1
    expected_mask = np.zeros_like(concentration, dtype=bool)
    expected_mask[0, 1] = True
    assert_array_equal(components[0].mask, expected_mask)


def test_connectivity_controls_diagonal_labeling() -> None:
    Xi, Zi = _grid(nz=2, nx=2)
    concentration = np.array([[1.0, 0.0], [0.0, 1.0]])

    diagonal = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=1, connectivity=8
    )
    orthogonal = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=1, connectivity=4
    )

    assert [(item.label, item.pixel_count) for item in diagonal] == [(1, 2)]
    assert [(item.label, item.pixel_count) for item in orthogonal] == [(1, 1), (2, 1)]


def test_component_statistics_and_masks_match_source_pixels() -> None:
    Xi, Zi = _grid(nz=4, nx=5)
    concentration = np.zeros_like(Xi)
    concentration[0, 0:2] = 1.0
    concentration[2:4, 3:5] = 1.0

    components = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=1
    )

    assert [component.label for component in components] == [1, 2]
    assert [component.pixel_count for component in components] == [2, 4]
    assert (
        components[0].x_min,
        components[0].x_max,
        components[0].z_min,
        components[0].z_max,
    ) == (0.0, 1.0, 0.0, 0.0)
    assert (
        components[1].x_min,
        components[1].x_max,
        components[1].z_min,
        components[1].z_max,
    ) == (3.0, 4.0, 2.0, 3.0)
    assert_array_equal(
        components[0].mask, (concentration == 1.0) & (Zi == 0.0)
    )
    assert_array_equal(components[1].mask, (Zi >= 2.0) & (Xi >= 3.0))


def test_bottom_contact_uses_z_levels_not_array_row_orientation() -> None:
    Xi, Zi = _grid(nz=4, nx=4)
    Zi = Zi[::-1].copy()
    concentration = np.zeros_like(Xi)
    concentration[3, 0] = 1.0
    concentration[0, 3] = 1.0

    components = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=1, connectivity=4
    )

    assert components[0].bottom_contact is False
    assert components[1].bottom_contact is True


def test_bottom_rows_accepts_either_lowest_level_and_clips_oversized_value() -> None:
    Xi, Zi = _grid(nz=3, nx=4)
    concentration = np.zeros_like(Xi)
    concentration[0, 0] = 1.0
    concentration[1, 2] = 1.0
    concentration[2, 3] = 1.0

    two_rows = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=2, connectivity=4
    )
    all_rows = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=20, connectivity=4
    )

    assert [component.bottom_contact for component in two_rows] == [True, True, False]
    assert all(component.bottom_contact for component in all_rows)


def test_spatial_filter_rejects_small_and_floating_components() -> None:
    Xi, Zi = _grid(nz=5, nx=8)
    concentration = np.zeros_like(Xi)
    concentration[0, 0:3] = 1.0
    concentration[0, 7] = 1.0
    concentration[3:5, 3:7] = 1.0

    components = detect_front_components(
        Xi, Zi, concentration, threshold=0.5, bottom_rows=1
    )
    selected = filter_spatial_components(components, min_component_pixels=3)

    assert len(components) == 3
    assert len(selected) == 1
    assert selected[0].pixel_count == 3
    assert selected[0].bottom_contact is True
    assert selected[0].x_max == 2.0


def test_no_spatial_candidate_reports_spatial_failure() -> None:
    Xi, Zi = _grid(nz=4, nx=5)
    frame = np.zeros_like(Xi)
    frame[3, 1:4] = 1.0

    result = _track(np.array([0.0]), Xi, Zi, frame[None, ...])

    assert result.status == (STATUS_NO_VALID_SPATIAL_CANDIDATE,)
    assert result.component_count[0] == 1
    assert result.spatial_candidate_count[0] == 0
    assert np.isnan(result.x_front[0])


def test_initial_selection_uses_area_then_rightmost_then_smallest_label() -> None:
    shape = (2, 4)
    mask = np.zeros(shape, dtype=bool)
    largest = _component(3, mask, x_max=1.0, pixel_count=5)
    smaller = _component(1, mask, x_max=3.0, pixel_count=4)
    assert select_initial_component([smaller, largest]) is largest

    left = _component(1, mask, x_max=2.0, pixel_count=4)
    right = _component(2, mask, x_max=3.0, pixel_count=4)
    assert select_initial_component([left, right]) is right

    higher_label = _component(2, mask, x_max=3.0, pixel_count=4)
    lower_label = _component(1, mask, x_max=3.0, pixel_count=4)
    assert select_initial_component([higher_label, lower_label]) is lower_label


def test_larger_floating_component_is_ignored_during_initial_selection() -> None:
    shape = (2, 4)
    mask = np.zeros(shape, dtype=bool)
    valid = _component(1, mask, x_max=2.0, pixel_count=3)
    floating = _component(
        2, mask, x_max=3.0, pixel_count=20, bottom_contact=False
    )

    spatial = filter_spatial_components(
        [valid, floating], min_component_pixels=2
    )

    assert select_initial_component(spatial) is valid


def test_prediction_uses_last_front_then_nonuniform_constant_velocity() -> None:
    assert predict_front_position([1.0], [2.5], 7.0) == 2.5
    predicted = predict_front_position([1.0, 3.0], [2.0, 6.0], 6.0)
    assert predicted == 12.0


def test_temporal_distance_rejection_and_infinite_tolerance() -> None:
    mask = np.ones((2, 2), dtype=bool)
    candidate = _component(1, mask, x_max=10.0)

    rejected, overlap = select_temporal_component(
        [candidate],
        predicted_x=2.0,
        max_front_jump=1.0,
        previous_selected_mask=mask,
    )
    accepted, accepted_overlap = select_temporal_component(
        [candidate],
        predicted_x=2.0,
        max_front_jump=np.inf,
        previous_selected_mask=mask,
    )

    assert rejected is None
    assert overlap == 0
    assert accepted is candidate
    assert accepted_overlap == 4


def test_temporal_ranking_prioritizes_overlap_over_area_and_proximity() -> None:
    previous = np.zeros((3, 5), dtype=bool)
    previous[0, 0:2] = True
    overlap_mask = previous.copy()
    large_mask = np.zeros_like(previous)
    large_mask[1:, 0:4] = True
    overlap_candidate = _component(1, overlap_mask, x_max=4.0, pixel_count=2)
    larger_closer = _component(2, large_mask, x_max=3.1, pixel_count=8)

    selected, overlap = select_temporal_component(
        [larger_closer, overlap_candidate],
        predicted_x=3.0,
        max_front_jump=2.0,
        previous_selected_mask=previous,
    )

    assert selected is overlap_candidate
    assert overlap == 2


def test_temporal_ranking_uses_area_then_proximity_for_overlap_ties() -> None:
    previous = np.zeros((2, 5), dtype=bool)
    previous[0, 0] = True
    mask = previous.copy()
    small = _component(1, mask, x_max=3.0, pixel_count=2)
    large = _component(2, mask, x_max=4.0, pixel_count=3)

    selected, _ = select_temporal_component(
        [small, large],
        predicted_x=2.0,
        max_front_jump=3.0,
        previous_selected_mask=previous,
    )
    assert selected is large

    equally_large_far = _component(1, mask, x_max=4.0, pixel_count=3)
    equally_large_near = _component(2, mask, x_max=2.5, pixel_count=3)
    selected, _ = select_temporal_component(
        [equally_large_far, equally_large_near],
        predicted_x=2.0,
        max_front_jump=3.0,
        previous_selected_mask=previous,
    )
    assert selected is equally_large_near


def test_advancing_main_body_ignores_small_isolated_fragment() -> None:
    Xi, Zi = _grid(nz=4, nx=10)
    frames = np.zeros((4, *Xi.shape))
    for frame_index in range(4):
        frames[frame_index, 0, frame_index : frame_index + 3] = 1.0
        frames[frame_index, 2, 8] = 1.0

    result = _track(np.arange(4.0), Xi, Zi, frames, min_component_pixels=2)

    assert_array_equal(result.x_front, [2.0, 3.0, 4.0, 5.0])
    assert result.status == (
        STATUS_SELECTED_INITIAL,
        STATUS_SELECTED_TRACKED,
        STATUS_SELECTED_TRACKED,
        STATUS_SELECTED_TRACKED,
    )


def test_detached_larger_object_is_not_selected() -> None:
    Xi, Zi = _grid(nz=5, nx=8)
    frame = np.zeros_like(Xi)
    frame[0, 0:3] = 1.0
    frame[3:5, 3:7] = 1.0

    result = _track(np.array([0.0]), Xi, Zi, frame[None, ...])

    assert result.x_front[0] == 2.0
    assert result.selected_component_pixels[0] == 3
    assert result.selected_bottom_contact[0]


def test_temporal_jump_is_rejected_and_next_frame_recovers() -> None:
    Xi, Zi = _grid(nz=3, nx=10)
    frames = np.zeros((4, *Xi.shape))
    frames[0, 0, 0:3] = 1.0
    frames[1, 0, 1:4] = 1.0
    frames[2, 0, 8:10] = 1.0
    frames[3, 0, 3:6] = 1.0

    result = _track(np.arange(4.0), Xi, Zi, frames, max_front_jump=1.0)

    assert_allclose(result.x_front, [2.0, 3.0, np.nan, 5.0], equal_nan=True)
    assert_allclose(result.predicted_x, [np.nan, 2.0, 4.0, 5.0], equal_nan=True)
    assert result.status[2] == STATUS_NO_VALID_TEMPORAL_CANDIDATE
    assert result.status[3] == STATUS_SELECTED_TRACKED
    assert result.selected_component_label[2] == -1
    assert result.tracking_error[3] == 0.0


def test_initial_missing_frames_are_skipped_until_first_detection() -> None:
    Xi, Zi = _grid(nz=3, nx=6)
    frames = np.zeros((4, *Xi.shape))
    frames[2, 0, 0:2] = 1.0
    frames[3, 0, 1:3] = 1.0

    result = _track(np.arange(4.0), Xi, Zi, frames)

    assert_allclose(result.x_front, [np.nan, np.nan, 1.0, 2.0], equal_nan=True)
    assert result.status == (
        STATUS_NO_THRESHOLD_COMPONENT,
        STATUS_NO_THRESHOLD_COMPONENT,
        STATUS_SELECTED_INITIAL,
        STATUS_SELECTED_TRACKED,
    )
    assert np.isnan(result.predicted_x[2])
    assert result.predicted_x[3] == 1.0
    assert result.tracking_error[3] == 1.0


def test_result_schema_dtypes_failure_sentinels_and_metadata() -> None:
    Xi, Zi = _grid(nz=2, nx=3)
    frames = np.zeros((1, *Xi.shape))

    result = _track(
        np.array([0.0]),
        Xi,
        Zi,
        frames,
        threshold=0.25,
        min_component_pixels=3,
        bottom_rows=2,
        max_front_jump=np.inf,
        connectivity=4,
    )

    assert {item.name for item in fields(result)} == {
        "time",
        "x_front",
        "predicted_x",
        "tracking_error",
        "selected_component_label",
        "selected_component_pixels",
        "selected_component_xmin",
        "selected_component_xmax",
        "selected_bottom_contact",
        "component_count",
        "spatial_candidate_count",
        "temporal_candidate_count",
        "selected_overlap_pixels",
        "status",
        "threshold",
        "min_component_pixels",
        "bottom_rows",
        "max_front_jump",
        "connectivity",
    }
    per_frame_arrays = [
        result.time,
        result.x_front,
        result.predicted_x,
        result.tracking_error,
        result.selected_component_label,
        result.selected_component_pixels,
        result.selected_component_xmin,
        result.selected_component_xmax,
        result.selected_bottom_contact,
        result.component_count,
        result.spatial_candidate_count,
        result.temporal_candidate_count,
        result.selected_overlap_pixels,
    ]
    assert all(array.shape == (1,) for array in per_frame_arrays)
    assert result.x_front.dtype.kind == "f"
    assert result.tracking_error.dtype.kind == "f"
    assert result.selected_component_label.dtype.kind == "i"
    assert result.component_count.dtype.kind == "i"
    assert result.selected_bottom_contact.dtype.kind == "b"
    assert result.status == (STATUS_NO_THRESHOLD_COMPONENT,)
    assert np.isnan(result.x_front[0])
    assert np.isnan(result.tracking_error[0])
    assert result.selected_component_label[0] == -1
    assert result.selected_component_pixels[0] == 0
    assert np.isnan(result.selected_component_xmin[0])
    assert np.isnan(result.selected_component_xmax[0])
    assert not result.selected_bottom_contact[0]
    assert result.selected_overlap_pixels[0] == 0
    assert result.threshold == 0.25
    assert result.min_component_pixels == 3
    assert result.bottom_rows == 2
    assert np.isinf(result.max_front_jump)
    assert result.connectivity == 4


def _valid_inputs() -> dict[str, object]:
    Xi, Zi = _grid(nz=2, nx=3)
    return {
        "time": np.array([0.0, 1.0]),
        "Xi": Xi,
        "Zi": Zi,
        "C_frames": np.zeros((2, *Xi.shape)),
        "threshold": 0.5,
        "min_component_pixels": 1,
        "bottom_rows": 1,
        "max_front_jump": 1.0,
        "connectivity": 8,
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"time": np.array([[0.0, 1.0]])}, "one-dimensional"),
        ({"time": np.array([]), "C_frames": np.zeros((0, 2, 3))}, "at least one"),
        ({"time": np.array([0.0, np.nan])}, "finite"),
        ({"time": np.array([0.0, 0.0])}, "strictly increasing"),
        ({"Xi": np.zeros((2, 2))}, "matching shapes"),
        ({"C_frames": np.zeros((1, 2, 3))}, "frame count"),
        ({"C_frames": np.zeros((2, 3, 3))}, "grid shape"),
        ({"threshold": np.nan}, "threshold must be finite"),
        ({"min_component_pixels": 0}, "min_component_pixels"),
        ({"bottom_rows": 0}, "bottom_rows"),
        ({"max_front_jump": 0.0}, "max_front_jump"),
        ({"connectivity": 6}, "exactly 4 or 8"),
    ],
)
def test_sequence_input_validation(
    update: dict[str, object], message: str
) -> None:
    arguments = _valid_inputs()
    arguments.update(update)

    with pytest.raises(ValueError, match=message):
        track_concentration_front(**arguments)


def test_validation_requires_finite_coordinate_pair() -> None:
    arguments = _valid_inputs()
    arguments["Xi"] = np.full((2, 3), np.nan)

    with pytest.raises(ValueError, match="finite coordinate pair"):
        track_concentration_front(**arguments)


def test_validation_requires_a_finite_z_level() -> None:
    arguments = _valid_inputs()
    arguments["Zi"] = np.full((2, 3), np.nan)

    with pytest.raises(ValueError, match="finite z level"):
        track_concentration_front(**arguments)
