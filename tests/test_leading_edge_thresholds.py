from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose
import pytest

from nek_post.leading_edge_thresholds import (
    ThresholdIntersection,
    row_threshold_intersections,
)


def _intersections(
    concentration: object,
    *,
    x: object,
    threshold: float = 0.0,
) -> tuple[ThresholdIntersection, ...]:
    return row_threshold_intersections(x, concentration, threshold)


def test_strict_crossing_uses_linear_physical_x_interpolation() -> None:
    intersections = _intersections([-1.0, 1.0], x=[0.0, 2.0])

    assert len(intersections) == 1
    assert intersections[0].x == pytest.approx(1.0)
    assert (intersections[0].left_ix, intersections[0].right_ix) == (0, 1)


def test_reversed_strict_crossing_preserves_support_order() -> None:
    intersections = _intersections([2.0, -1.0], x=[0.0, 3.0])

    assert len(intersections) == 1
    assert intersections[0].x == pytest.approx(2.0)
    assert (intersections[0].left_ix, intersections[0].right_ix) == (0, 1)


def test_exact_sample_is_one_intersection() -> None:
    intersections = _intersections([-1.0, 0.0, 1.0], x=[0.0, 1.0, 2.0])

    assert len(intersections) == 1
    assert intersections[0].x == 1.0


def test_exact_plateau_is_once_at_rightmost_x_with_entry_support() -> None:
    intersections = _intersections(
        [1.0, 0.0, 0.0, -1.0],
        x=[0.0, 0.4, 1.1, 2.0],
    )

    assert len(intersections) == 1
    assert intersections[0].x == 1.1
    assert (intersections[0].left_ix, intersections[0].right_ix) == (0, 1)


def test_multiple_crossings_are_all_returned() -> None:
    intersections = _intersections(
        [-1.0, 1.0, -1.0, 1.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    assert_allclose([item.x for item in intersections], [0.5, 1.5, 2.5, 3.5])


def test_nan_separates_but_does_not_hide_finite_crossings() -> None:
    intersections = _intersections(
        [-1.0, 1.0, np.nan, 1.0, -1.0],
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    assert_allclose([item.x for item in intersections], [0.5, 3.5])


def test_intersection_record_is_immutable() -> None:
    intersection = _intersections([-1.0, 1.0], x=[0.0, 1.0])[0]

    with pytest.raises(FrozenInstanceError):
        intersection.x = 2.0  # type: ignore[misc]
