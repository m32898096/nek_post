from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import nek_post.leading_edge_methods.moore_boundary as moore
from nek_post.leading_edge_methods.moore_boundary import (
    MOORE_INITIAL_INDICATOR,
    MOORE_NEIGHBOR_OFFSETS,
    MooreBoundaryComponent,
    MooreBoundaryTrace,
    build_low_side_boundary_candidate_mask,
    extract_moore_boundary,
    label_periodic_boundary_components,
    moore_neighbor_offset,
    moore_trace_winding_number,
    primary_moore_component_mask,
    select_moore_boundary_start,
    select_primary_moore_component,
    select_periodic_moore_trace,
    trace_moore_boundary,
    wrap_moore_indicator,
)


def _component(
    points: list[tuple[int, int]],
    *,
    shape: tuple[int, int] = (8, 12),
) -> MooreBoundaryComponent:
    ny, nx = shape
    return MooreBoundaryComponent(
        flat_indices=np.asarray([iy * nx + ix for iy, ix in points]),
        mask_shape=(ny, nx),
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


def test_component_labeling_connects_periodic_y_seam() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[0, 2] = True
    mask[-1, 2] = True

    components = label_periodic_boundary_components(mask)

    assert len(components) == 1
    assert components[0].pixel_count == 2
    assert components[0].unique_y_count == 2


def test_component_labeling_connects_diagonal_periodic_seam() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[0, 2] = True
    mask[-1, 3] = True

    components = label_periodic_boundary_components(mask)

    assert len(components) == 1
    assert_array_equal(components[0].flat_indices, [2, 27])


def test_component_labeling_does_not_wrap_x_edges() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 0] = True
    mask[2, -1] = True

    components = label_periodic_boundary_components(mask)

    assert len(components) == 2
    assert [component.minimum_flat_index for component in components] == [12, 17]


def test_component_measurements_and_mask_are_exact_and_immutable() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 3] = True
    mask[1, 4] = True
    mask[1, 5] = True

    component = label_periodic_boundary_components(mask)[0]
    reconstructed = component.to_mask()

    assert component.pixel_count == 3
    assert component.unique_y_count == 2
    assert component.max_ix == 5
    assert component.minimum_flat_index == 3
    assert_array_equal(component.flat_indices, [3, 11, 12])
    assert_array_equal(reconstructed, mask)
    assert not component.flat_indices.flags.writeable
    assert not reconstructed.flags.writeable


def test_primary_selection_prioritizes_unique_y_coverage_over_front_x() -> None:
    broad = _component([(0, 1), (1, 1), (2, 1)])
    frontmost = _component([(4, 10), (5, 10)])

    assert select_primary_moore_component((frontmost, broad)) is broad


def test_primary_selection_uses_pixel_count_after_equal_y_coverage() -> None:
    larger = _component([(0, 2), (0, 3), (1, 2), (1, 3)])
    smaller = _component([(4, 10), (5, 10)])

    assert select_primary_moore_component((smaller, larger)) is larger


def test_primary_selection_uses_max_x_after_equal_coverage_and_size() -> None:
    rear = _component([(0, 2), (1, 2)])
    front = _component([(4, 8), (5, 8)])

    assert select_primary_moore_component((rear, front)) is front


def test_primary_selection_final_tie_uses_smallest_flat_index() -> None:
    first = _component([(0, 5), (1, 5)])
    later = _component([(2, 5), (3, 5)])

    assert select_primary_moore_component((later, first)) is first
    assert select_primary_moore_component(()) is None


def test_primary_mask_prefers_all_y_component_over_two_row_seam_component() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[:, 2] = True
    mask[0, 7] = True
    mask[-1, 7] = True

    primary = primary_moore_component_mask(mask)

    expected = np.zeros_like(mask)
    expected[:, 2] = True
    assert_array_equal(primary, expected)
    assert not primary.flags.writeable


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


def test_explicit_trace_start_and_indicator_are_supported_and_validated() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1:4] = True
    mask[2, (1, 3)] = True
    mask[3, 1:4] = True

    trace = trace_moore_boundary(
        mask,
        start_point=(1, 1),
        initial_indicator=1,
    )

    assert trace.start_point == (1, 1)
    with pytest.raises(ValueError, match="start_point.*belong"):
        trace_moore_boundary(mask, start_point=(0, 0))
    with pytest.raises(ValueError, match="initial_indicator"):
        trace_moore_boundary(mask, initial_indicator=9)


def _closed_trace(points: list[tuple[int, int]]) -> MooreBoundaryTrace:
    return MooreBoundaryTrace(
        ix=np.asarray([point[0] for point in points]),
        iy=np.asarray([point[1] for point in points]),
        start_point=points[0],
        closed=True,
        step_count=len(points),
    )


def test_winding_number_local_triangle_is_zero_and_includes_closing_edge() -> None:
    trace = _closed_trace([(2, 5), (3, 0), (2, 0)])

    assert moore_trace_winding_number(trace, 6) == 0


def test_winding_number_positive_periodic_circuit_is_one() -> None:
    trace = _closed_trace(
        [(2, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    )

    assert moore_trace_winding_number(trace, 6) == 1


def test_winding_number_negative_periodic_circuit_is_minus_one() -> None:
    trace = _closed_trace(
        [(2, 0), (2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]
    )

    assert moore_trace_winding_number(trace, 6) == -1


def test_winding_number_rejects_non_neighbour_edge() -> None:
    trace = _closed_trace([(0, 0), (2, 0)])

    with pytest.raises(ValueError, match="not a valid periodic Moore-neighbour"):
        moore_trace_winding_number(trace, 6)


def test_periodic_selector_prefers_winding_trace_over_frontward_triangle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.zeros((6, 10), dtype=bool)
    mask[:, 2] = True
    mask[5, 8] = True
    mask[4, 8:10] = True
    mask[4, 3:8] = True
    triangle = _closed_trace([(8, 5), (9, 4), (8, 4)])
    winding = _closed_trace(
        [(2, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    )
    calls: list[tuple[tuple[int, int], int]] = []

    def fake_trace(
        _mask: object,
        *,
        start_point: tuple[int, int],
        initial_indicator: int,
        **_kwargs: object,
    ) -> MooreBoundaryTrace:
        calls.append((start_point, initial_indicator))
        if start_point == (2, 5) and initial_indicator == 3:
            return winding
        if start_point[0] >= 8:
            return triangle
        raise RuntimeError("synthetic failed walk")

    monkeypatch.setattr(moore, "trace_moore_boundary", fake_trace)

    selection = select_periodic_moore_trace(mask)

    assert selection.selected_candidate.winding_number == 1
    assert selection.selected_candidate.maximum_ix == 2
    assert any(
        candidate.winding_number == 0 and candidate.maximum_ix == 9
        for candidate in selection.candidates
    )
    assert any(
        candidate.failure_reason == "synthetic failed walk"
        for candidate in selection.candidates
    )
    assert calls[:8] == [((8, 5), indicator) for indicator in range(1, 9)]
    assert calls.index(((2, 5), 3)) > calls.index(((8, 5), 8))


def test_equivalent_trace_candidates_are_deduplicated_deterministically() -> None:
    mask = np.zeros((6, 5), dtype=bool)
    mask[:, 2] = True

    first = select_periodic_moore_trace(mask)
    second = select_periodic_moore_trace(mask)

    assert first.selected_candidate == second.selected_candidate
    assert first.selected_candidate.start_point == (2, 5)
    assert first.selected_candidate.initial_indicator == 1
    assert sum(candidate.selected for candidate in first.candidates) == 1
    assert any(
        candidate.duplicate_of == ((2, 5), 1)
        for candidate in first.candidates
    )


def test_full_y_component_without_one_winding_trace_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.zeros((6, 10), dtype=bool)
    mask[:, 2] = True
    mask[5, 8] = True
    mask[4, 3:10] = True
    triangle = _closed_trace([(8, 5), (9, 4), (8, 4)])
    monkeypatch.setattr(
        moore,
        "trace_moore_boundary",
        lambda *_args, **_kwargs: triangle,
    )

    with pytest.raises(RuntimeError, match="spans every y row.*winding 1"):
        select_periodic_moore_trace(mask)


def test_partial_y_component_allows_zero_winding_fallback() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 2:4] = True
    mask[3, 2] = True

    selection = select_periodic_moore_trace(mask)

    assert selection.selected_candidate.winding_number == 0
    assert selection.selected_candidate.unique_y_count == 2


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
    monkeypatch.setattr(
        moore,
        "_select_extraction_trace",
        lambda _mask, _x: fake_trace,
    )
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


def test_extraction_traces_primary_all_y_component_not_small_seam_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_mask = np.zeros((6, 8), dtype=bool)
    candidate_mask[:, 2] = True
    candidate_mask[0, 7] = True
    candidate_mask[-1, 7] = True
    monkeypatch.setattr(
        moore,
        "build_low_side_boundary_candidate_mask",
        lambda *_args, **_kwargs: candidate_mask,
    )
    x = np.arange(8, dtype=float)
    y = np.arange(6, dtype=float) / 6.0

    result = extract_moore_boundary(
        x,
        y,
        np.zeros((y.size, x.size)),
        threshold=0.5,
    )

    assert_array_equal(result.x_front, np.full(y.size, 2.0))
    assert_array_equal(result.success_mask, np.ones(y.size, dtype=bool))
    assert_array_equal(result.crossing_count, np.ones(y.size, dtype=np.int64))


def test_full_span_component_selection_prefers_rowwise_physical_x_over_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_mask = np.zeros((6, 11), dtype=bool)
    candidate_mask[:, 2] = True
    candidate_mask[2:5, 3:6] = True  # Make the rear component much larger.
    candidate_mask[:, 9] = True
    monkeypatch.setattr(
        moore,
        "build_low_side_boundary_candidate_mask",
        lambda *_args, **_kwargs: candidate_mask,
    )
    x = np.linspace(0.1, 4.0, candidate_mask.shape[1]) ** 2
    y = np.arange(candidate_mask.shape[0], dtype=float) / candidate_mask.shape[0]

    result = extract_moore_boundary(
        x,
        y,
        np.zeros(candidate_mask.shape),
        threshold=0.5,
    )

    assert_array_equal(result.x_front, np.full(y.size, x[9]))


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
