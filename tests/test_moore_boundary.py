from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import nek_post.leading_edge_methods.moore_boundary as moore
from nek_post.leading_edge_methods.moore_boundary import (
    MOORE_INITIAL_INDICATOR,
    MOORE_NEIGHBOR_OFFSETS,
    MooreBoundaryTrace,
    build_low_side_boundary_candidate_mask,
    extract_moore_boundary,
    moore_neighbor_offset,
    select_moore_boundary_start,
    trace_moore_boundary,
    wrap_moore_indicator,
)


def test_fortran_indicator_offsets_and_wrapping_are_exact() -> None:
    assert MOORE_INITIAL_INDICATOR == 4
    assert MOORE_NEIGHBOR_OFFSETS == (
        (+1, -1),
        (+1, 0),
        (+1, +1),
        (0, +1),
        (-1, +1),
        (-1, 0),
        (-1, -1),
        (0, -1),
    )
    for indicator, expected in enumerate(MOORE_NEIGHBOR_OFFSETS, start=1):
        assert moore_neighbor_offset(indicator) == expected
        assert moore_neighbor_offset(indicator + 8) == expected
    assert wrap_moore_indicator(0) == 8
    assert wrap_moore_indicator(9) == 1
    assert wrap_moore_indicator(17) == 1


def test_local_search_applies_plus_five_turn_then_advances_on_rejection() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    current = (2, 2)
    # From initial indicator 4, +5 wraps to 1. Indicators 1 and 2 are
    # rejected; indicator 3 is the first acceptable candidate.
    mask[3, 3] = True

    found = moore._next_moore_candidate(
        mask, current, MOORE_INITIAL_INDICATOR
    )

    assert found == ((3, 3), 3)


def test_candidate_mask_threshold_finite_and_adjacency_semantics() -> None:
    concentration = np.full((4, 5), np.nan)
    concentration[0, 1] = 1.0  # high side
    concentration[0, 2] = 0.5  # exact threshold: low candidate
    concentration[1, 1] = 0.2  # below threshold: low candidate
    concentration[3, 1] = 0.1  # periodic-y adjacent low candidate
    concentration[0, 0] = np.inf
    concentration[1, 0] = -np.inf
    concentration[2, 4] = 0.0  # isolated low side

    candidate = build_low_side_boundary_candidate_mask(
        concentration, threshold=0.5
    )

    expected = np.zeros_like(candidate)
    expected[0, 2] = True
    expected[1, 1] = True
    expected[3, 1] = True
    assert_array_equal(candidate, expected)
    assert not candidate.flags.writeable


def test_candidate_mask_does_not_wrap_x_adjacency() -> None:
    concentration = np.full((3, 4), np.nan)
    concentration[1, 0] = 1.0
    concentration[1, 3] = 0.0

    candidate = build_low_side_boundary_candidate_mask(
        concentration, threshold=0.5
    )

    assert not candidate[1, 3]


def test_start_prefers_upper_seam_then_greatest_x() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 5] = True
    mask[3, 2] = True
    mask[3, 4] = True

    assert select_moore_boundary_start(mask) == (4, 3)


def test_start_falls_back_to_global_x_then_greatest_y_tie() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[1, 5] = True
    mask[3, 5] = True
    mask[2, 4] = True

    assert select_moore_boundary_start(mask) == (5, 3)
    assert select_moore_boundary_start(np.zeros_like(mask)) is None


def test_simple_closed_boundary_has_deterministic_order_and_closure() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1:4] = True
    mask[2, (1, 3)] = True
    mask[3, 1:4] = True

    trace = trace_moore_boundary(mask)

    assert trace.start_point == (3, 3)
    assert trace.closed
    assert trace.step_count == 8
    assert trace.path_length == 8
    assert list(zip(trace.ix.tolist(), trace.iy.tolist(), strict=True)) == [
        (3, 3),
        (2, 3),
        (1, 3),
        (1, 2),
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
    ]
    assert np.count_nonzero((trace.ix == 3) & (trace.iy == 3)) == 1
    assert not trace.ix.flags.writeable
    assert not trace.iy.flags.writeable


def test_diagonal_boundary_path_is_deterministic() -> None:
    mask = np.eye(4, dtype=bool)

    trace = trace_moore_boundary(mask)

    assert trace.closed
    assert list(zip(trace.ix.tolist(), trace.iy.tolist(), strict=True)) == [
        (3, 3),
        (2, 2),
        (1, 1),
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_periodic_y_seam_is_crossed_without_duplicate_closure_points() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, 2] = True

    trace = trace_moore_boundary(mask)

    assert trace.closed
    assert_array_equal(trace.ix, [2, 2, 2, 2])
    assert_array_equal(trace.iy, [3, 0, 1, 2])
    assert len(set(zip(trace.ix.tolist(), trace.iy.tolist()))) == 4


def test_trace_raises_when_candidate_has_no_neighbour() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    with pytest.raises(RuntimeError, match="no acceptable next"):
        trace_moore_boundary(mask)


def test_trace_detects_premature_directed_edge_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.ones((3, 3), dtype=bool)
    sequence = iter(
        [
            ((1, 2), 1),
            ((0, 2), 1),
            ((1, 2), 2),
            ((0, 2), 1),
        ]
    )
    monkeypatch.setattr(
        moore,
        "_next_moore_candidate",
        lambda *_args, **_kwargs: next(sequence),
    )

    with pytest.raises(RuntimeError, match="premature directed-edge cycle"):
        trace_moore_boundary(mask)


def test_trace_enforces_maximum_step_protection() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, 2] = True

    with pytest.raises(RuntimeError, match="maximum-step bound of 2"):
        trace_moore_boundary(mask, maximum_steps=2)


def test_empty_trace_is_valid_and_has_no_start() -> None:
    trace = trace_moore_boundary(np.zeros((3, 4), dtype=bool))

    assert trace.start_point is None
    assert not trace.closed
    assert trace.step_count == 0
    assert trace.path_length == 0


def test_trace_conversion_counts_unique_x_and_preserves_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_trace = MooreBoundaryTrace(
        ix=np.array([1, 1, 2, 3]),
        iy=np.array([0, 0, 0, 2]),
        start_point=(1, 0),
        closed=True,
        step_count=4,
    )
    monkeypatch.setattr(moore, "trace_moore_boundary", lambda _mask: fake_trace)
    concentration = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    result = extract_moore_boundary(
        np.array([0.0, 0.5, 1.0, 1.5]),
        np.array([0.0, 1.0, 2.0]),
        concentration,
        threshold=0.5,
    )

    assert result.method == "moore-boundary"
    assert result.threshold == 0.5
    assert_array_equal(result.x_front[[0, 2]], [1.0, 1.5])
    assert np.isnan(result.x_front[1])
    assert_array_equal(result.success_mask, [True, False, True])
    assert_array_equal(result.crossing_count, [2, 0, 1])
    for values in (
        result.y,
        result.x_front,
        result.success_mask,
        result.crossing_count,
    ):
        assert not values.flags.writeable


def test_no_boundary_candidates_return_common_empty_result() -> None:
    result = extract_moore_boundary(
        np.arange(4, dtype=float),
        np.arange(3, dtype=float),
        np.ones((3, 4)),
        threshold=0.5,
    )

    assert result.method == "moore-boundary"
    assert np.all(np.isnan(result.x_front))
    assert_array_equal(result.success_mask, np.zeros(3, dtype=bool))
    assert_array_equal(result.crossing_count, np.zeros(3, dtype=np.int64))


def test_analytic_periodic_front_returns_first_low_side_grid_column() -> None:
    x = np.arange(6, dtype=float)
    y = np.arange(4, dtype=float) / 4.0
    concentration = np.where(x[None, :] < 3.0, 1.0, 0.0)
    concentration = np.broadcast_to(concentration, (y.size, x.size)).copy()

    result = extract_moore_boundary(
        x,
        y,
        concentration,
        threshold=0.5,
    )

    assert_array_equal(result.x_front, np.full(y.size, 3.0))
    assert_array_equal(result.success_mask, np.ones(y.size, dtype=bool))
    assert_array_equal(result.crossing_count, np.ones(y.size, dtype=np.int64))
