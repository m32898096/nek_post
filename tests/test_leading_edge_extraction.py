from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.leading_edge_extraction import (
    LeadingEdgeCurve,
    extract_spanwise_leading_edge,
)


def _extract_one(
    concentration: object,
    *,
    x: object = (0.0, 1.0, 2.0),
    threshold: float = 0.0,
) -> LeadingEdgeCurve:
    return extract_spanwise_leading_edge(
        x,
        [0.25],
        [concentration],
        threshold=threshold,
    )


def test_increasing_crossing_uses_exact_linear_interpolation() -> None:
    result = _extract_one([-1.0, 1.0], x=[0.0, 2.0])

    assert result.x_front[0] == pytest.approx(1.0)
    assert result.crossing_count[0] == 1
    assert result.success_mask[0]


def test_decreasing_crossing_uses_exact_linear_interpolation() -> None:
    result = _extract_one([2.0, -1.0], x=[0.0, 3.0])

    assert result.x_front[0] == pytest.approx(2.0)
    assert result.crossing_count[0] == 1


def test_several_crossings_select_rightmost_and_report_count() -> None:
    result = _extract_one(
        [-1.0, 1.0, -1.0, 1.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    assert result.x_front[0] == pytest.approx(3.5)
    assert result.crossing_count[0] == 4


def test_exact_threshold_at_one_sample_is_not_counted_twice() -> None:
    result = _extract_one([-1.0, 0.0, 1.0])

    assert result.x_front[0] == 1.0
    assert result.crossing_count[0] == 1


@pytest.mark.parametrize(
    "concentration",
    (
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 1.0],
    ),
)
def test_contiguous_plateau_counts_once_at_its_rightmost_x(
    concentration: list[float],
) -> None:
    result = _extract_one(concentration, x=[0.0, 1.0, 2.0, 3.0])

    assert result.x_front[0] == 2.0
    assert result.crossing_count[0] == 1


def test_complete_row_threshold_plateau_returns_rightmost_x() -> None:
    result = _extract_one([0.0, 0.0, 0.0])

    assert result.x_front[0] == 2.0
    assert result.crossing_count[0] == 1
    assert result.success_mask[0]


def test_exact_threshold_at_first_x_sample() -> None:
    result = _extract_one([0.0, 2.0, 3.0])

    assert result.x_front[0] == 0.0
    assert result.crossing_count[0] == 1


def test_exact_threshold_at_final_x_sample() -> None:
    result = _extract_one([-2.0, -1.0, 0.0])

    assert result.x_front[0] == 2.0
    assert result.crossing_count[0] == 1


def test_multiple_isolated_exact_threshold_contacts_are_distinct() -> None:
    result = _extract_one(
        [-1.0, 0.0, -1.0, 0.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    assert result.x_front[0] == 3.0
    assert result.crossing_count[0] == 2


@pytest.mark.parametrize("concentration", ([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]))
def test_row_entirely_on_one_side_has_no_crossing(
    concentration: list[float],
) -> None:
    result = _extract_one(concentration)

    assert np.isnan(result.x_front[0])
    assert result.crossing_count[0] == 0
    assert not result.success_mask[0]


def test_nan_region_is_not_bridged() -> None:
    result = _extract_one([-1.0, np.nan, 1.0])

    assert np.isnan(result.x_front[0])
    assert result.crossing_count[0] == 0


def test_finite_crossings_on_both_sides_of_nan_are_detected_independently() -> None:
    result = _extract_one(
        [-1.0, 1.0, np.nan, 1.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    assert result.x_front[0] == pytest.approx(3.5)
    assert result.crossing_count[0] == 2


@pytest.mark.parametrize(
    ("concentration", "expected_x", "expected_count"),
    (
        ([np.nan, -1.0, 1.0], 1.5, 1),
        ([-1.0, 1.0, np.nan], 0.5, 1),
        ([np.nan, 0.0, np.nan], 1.0, 1),
    ),
)
def test_nan_at_left_or_right_does_not_hide_finite_intersections(
    concentration: list[float],
    expected_x: float,
    expected_count: int,
) -> None:
    result = _extract_one(concentration)

    assert result.x_front[0] == pytest.approx(expected_x)
    assert result.crossing_count[0] == expected_count


def test_multiple_disconnected_concentration_bodies_choose_rightmost() -> None:
    result = _extract_one(
        [-1.0, 1.0, -1.0, np.nan, -1.0, 1.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    assert result.x_front[0] == pytest.approx(5.5)
    assert result.crossing_count[0] == 4


def test_multiple_y_rows_are_extracted_independently_in_original_order() -> None:
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([0.1, 0.4, 0.9])
    concentration = np.asarray(
        [
            [-1.0, 1.0, 2.0],
            [2.0, 1.0, -1.0],
            [-1.0, -2.0, -3.0],
        ]
    )

    result = extract_spanwise_leading_edge(
        x,
        y,
        concentration,
        threshold=0.0,
    )

    assert_array_equal(result.y, y)
    assert_allclose(result.x_front[:2], [0.5, 1.5])
    assert np.isnan(result.x_front[2])
    assert_array_equal(result.success_mask, [True, True, False])
    assert_array_equal(result.crossing_count, [1, 1, 0])


def test_nonuniform_x_uses_physical_linear_interpolation() -> None:
    result = _extract_one(
        [-1.0, -0.5, 1.0],
        x=[0.0, 0.2, 2.0],
    )

    assert result.x_front[0] == pytest.approx(0.8)


def test_default_threshold_is_point_one() -> None:
    result = extract_spanwise_leading_edge(
        [0.0, 2.0],
        [0.0],
        [[0.0, 0.4]],
    )

    assert result.threshold == 0.1
    assert result.x_front[0] == pytest.approx(0.5)


def test_concentration_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape mismatch.*expected \(2, 3\)"):
        extract_spanwise_leading_edge(
            [0.0, 1.0, 2.0],
            [0.0, 1.0],
            np.zeros((3, 3)),
        )


@pytest.mark.parametrize(
    "x",
    (
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0],
    ),
)
def test_nonmonotonic_or_duplicate_x_is_rejected(x: list[float]) -> None:
    with pytest.raises(ValueError, match="x must be strictly increasing and unique"):
        extract_spanwise_leading_edge(x, [0.0], [np.zeros(len(x))])


@pytest.mark.parametrize(
    "y",
    (
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0],
    ),
)
def test_nonmonotonic_or_duplicate_y_is_rejected(y: list[float]) -> None:
    with pytest.raises(ValueError, match="y must be strictly increasing and unique"):
        extract_spanwise_leading_edge(
            [0.0, 1.0],
            y,
            np.zeros((len(y), 2)),
        )


@pytest.mark.parametrize(
    ("name", "x", "y"),
    (
        ("x", [0.0, np.nan], [0.0]),
        ("x", [0.0, np.inf], [0.0]),
        ("y", [0.0, 1.0], [0.0, np.nan]),
        ("y", [0.0, 1.0], [0.0, np.inf]),
    ),
)
def test_nonfinite_coordinate_is_rejected(
    name: str,
    x: list[float],
    y: list[float],
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must contain only finite"):
        extract_spanwise_leading_edge(x, y, np.zeros((len(y), len(x))))


@pytest.mark.parametrize("threshold", (np.nan, np.inf, -np.inf, "not-a-number"))
def test_nonfinite_or_nonnumeric_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        extract_spanwise_leading_edge(
            [0.0, 1.0],
            [0.0],
            [[0.0, 1.0]],
            threshold=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("concentration", (np.zeros(3), np.zeros((1, 1, 3))))
def test_non_two_dimensional_concentration_is_rejected(
    concentration: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="concentration must be a two-dimensional"):
        extract_spanwise_leading_edge(
            [0.0, 1.0, 2.0],
            [0.0],
            concentration,
        )


def test_result_arrays_have_expected_shapes_dtypes_and_are_immutable() -> None:
    result = extract_spanwise_leading_edge(
        [0.0, 1.0],
        [0.0, 0.5],
        [[-1.0, 1.0], [1.0, -1.0]],
        threshold=0.0,
    )

    assert result.y.shape == (2,)
    assert result.x_front.shape == (2,)
    assert result.success_mask.shape == (2,)
    assert result.crossing_count.shape == (2,)
    assert result.y.dtype == np.float64
    assert result.x_front.dtype == np.float64
    assert result.success_mask.dtype == np.bool_
    assert result.crossing_count.dtype == np.int64
    for name in ("y", "x_front", "success_mask", "crossing_count"):
        assert not getattr(result, name).flags.writeable
    with pytest.raises(ValueError):
        result.x_front[0] = 3.0
    with pytest.raises(FrozenInstanceError):
        result.threshold = 0.5  # type: ignore[misc]


def test_input_arrays_and_writeability_are_not_modified() -> None:
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([0.0, 0.5])
    concentration = np.asarray(
        [[-1.0, 1.0, 2.0], [2.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    x_before = x.copy()
    y_before = y.copy()
    concentration_before = concentration.copy()

    result = extract_spanwise_leading_edge(
        x,
        y,
        concentration,
        threshold=0.0,
    )

    assert_array_equal(x, x_before)
    assert_array_equal(y, y_before)
    assert_array_equal(concentration, concentration_before)
    assert x.flags.writeable
    assert y.flags.writeable
    assert concentration.flags.writeable
    assert not np.shares_memory(result.y, y)
