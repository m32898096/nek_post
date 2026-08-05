"""Fortran-derived pure-NumPy Moore boundary tracing."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from nek_post.leading_edge_methods.common import LeadingEdgeExtractionResult


MOORE_BOUNDARY_METHOD = "moore-boundary"
MOORE_INITIAL_INDICATOR = 4
MOORE_NEIGHBOR_OFFSETS = (
    (+1, -1),
    (+1, 0),
    (+1, +1),
    (0, +1),
    (-1, +1),
    (-1, 0),
    (-1, -1),
    (0, -1),
)


def wrap_moore_indicator(indicator: int) -> int:
    """Wrap a one-based Moore indicator cyclically into ``1...8``."""
    if not isinstance(indicator, Integral) or isinstance(
        indicator, (bool, np.bool_)
    ):
        raise ValueError("Moore indicator must be an integer.")
    return (int(indicator) - 1) % 8 + 1


def moore_neighbor_offset(indicator: int) -> tuple[int, int]:
    """Return the exact Fortran-derived ``(delta_ix, delta_iy)`` offset."""
    return MOORE_NEIGHBOR_OFFSETS[wrap_moore_indicator(indicator) - 1]


def build_low_side_boundary_candidate_mask(
    concentration: object,
    *,
    threshold: float,
) -> NDArray[np.bool_]:
    """Return low-side pixels adjacent to a high-side eight-neighbour.

    The first array dimension is periodic y. The second is non-periodic x.
    Non-finite values belong to neither threshold side.
    """
    values = np.asarray(concentration, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("concentration must be a nonempty two-dimensional array.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite.") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")

    finite = np.isfinite(values)
    high_side = finite & (values > threshold_value)
    low_side = finite & (values <= threshold_value)
    adjacent_high = np.zeros(values.shape, dtype=np.bool_)
    for delta_ix, delta_iy in MOORE_NEIGHBOR_OFFSETS:
        shifted_y = np.roll(high_side, -delta_iy, axis=0)
        if delta_ix == 0:
            adjacent_high |= shifted_y
        elif delta_ix > 0:
            adjacent_high[:, :-1] |= shifted_y[:, 1:]
        else:
            adjacent_high[:, 1:] |= shifted_y[:, :-1]
    candidate = low_side & adjacent_high
    candidate.setflags(write=False)
    return candidate


def select_moore_boundary_start(
    candidate_mask: object,
) -> tuple[int, int] | None:
    """Choose the deterministic front-biased ``(ix, iy)`` start point."""
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("candidate_mask must be a nonempty two-dimensional array.")
    seam_ix = np.flatnonzero(mask[-1])
    if seam_ix.size:
        return int(seam_ix[-1]), int(mask.shape[0] - 1)
    iy_values, ix_values = np.nonzero(mask)
    if ix_values.size == 0:
        return None
    greatest_ix = int(np.max(ix_values))
    greatest_iy = int(np.max(iy_values[ix_values == greatest_ix]))
    return greatest_ix, greatest_iy


def _readonly_integer_copy(values: object) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MooreBoundaryTrace:
    """Immutable traced Moore path in integer grid-index coordinates."""

    ix: NDArray[np.int64]
    iy: NDArray[np.int64]
    start_point: tuple[int, int] | None
    closed: bool
    step_count: int

    def __post_init__(self) -> None:
        ix = _readonly_integer_copy(self.ix)
        iy = _readonly_integer_copy(self.iy)
        if ix.ndim != 1 or iy.shape != ix.shape:
            raise ValueError("Moore trace ix and iy must be matching vectors.")
        if not isinstance(self.closed, (bool, np.bool_)):
            raise ValueError("Moore trace closed must be a boolean.")
        if not isinstance(self.step_count, Integral) or isinstance(
            self.step_count, (bool, np.bool_)
        ):
            raise ValueError("Moore trace step_count must be an integer.")
        steps = int(self.step_count)
        if steps < 0:
            raise ValueError("Moore trace step_count must be non-negative.")
        start = self.start_point
        if start is not None:
            if len(start) != 2:
                raise ValueError("Moore trace start_point must contain ix and iy.")
            start = (int(start[0]), int(start[1]))
        object.__setattr__(self, "ix", ix)
        object.__setattr__(self, "iy", iy)
        object.__setattr__(self, "start_point", start)
        object.__setattr__(self, "closed", bool(self.closed))
        object.__setattr__(self, "step_count", steps)

    @property
    def path_length(self) -> int:
        """Number of retained, non-duplicated path points."""
        return int(self.ix.size)


def _next_moore_candidate(
    mask: NDArray[np.bool_],
    current: tuple[int, int],
    indicator: int,
) -> tuple[tuple[int, int], int] | None:
    """Apply the +5 turn and test no more than eight neighbours."""
    ny, nx = mask.shape
    search_indicator = wrap_moore_indicator(indicator + 5)
    ix, iy = current
    for _ in range(8):
        delta_ix, delta_iy = moore_neighbor_offset(search_indicator)
        next_ix = ix + delta_ix
        next_iy = (iy + delta_iy) % ny
        if 0 <= next_ix < nx and mask[next_iy, next_ix]:
            return (next_ix, next_iy), search_indicator
        search_indicator = wrap_moore_indicator(search_indicator + 1)
    return None


def trace_moore_boundary(
    candidate_mask: object,
    *,
    maximum_steps: int | None = None,
) -> MooreBoundaryTrace:
    """Trace one Moore boundary using directed-edge closure and cycle checks."""
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("candidate_mask must be a nonempty two-dimensional array.")
    start = select_moore_boundary_start(mask)
    if start is None:
        return MooreBoundaryTrace(
            ix=np.empty(0, dtype=np.int64),
            iy=np.empty(0, dtype=np.int64),
            start_point=None,
            closed=False,
            step_count=0,
        )
    candidate_count = int(np.count_nonzero(mask))
    default_bound = 8 * candidate_count + 8
    if maximum_steps is None:
        step_bound = default_bound
    else:
        if not isinstance(maximum_steps, Integral) or isinstance(
            maximum_steps, (bool, np.bool_)
        ):
            raise ValueError("maximum_steps must be a positive integer.")
        step_bound = int(maximum_steps)
        if step_bound < 1:
            raise ValueError("maximum_steps must be a positive integer.")

    current = start
    indicator = MOORE_INITIAL_INDICATOR
    ix_path = [start[0]]
    iy_path = [start[1]]
    first_edge: tuple[tuple[int, int], tuple[int, int]] | None = None
    traversed_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while len(traversed_edges) < step_bound:
        found = _next_moore_candidate(mask, current, indicator)
        if found is None:
            raise RuntimeError(
                "Moore boundary trace has no acceptable next boundary neighbour "
                f"at point {current}."
            )
        next_point, next_indicator = found
        edge = (current, next_point)
        if first_edge is None:
            first_edge = edge
        elif edge == first_edge:
            if len(ix_path) > 1 and (ix_path[-1], iy_path[-1]) == start:
                ix_path.pop()
                iy_path.pop()
            return MooreBoundaryTrace(
                ix=ix_path,
                iy=iy_path,
                start_point=start,
                closed=True,
                step_count=len(traversed_edges),
            )
        if edge in traversed_edges:
            raise RuntimeError(
                "Moore boundary trace encountered a premature directed-edge "
                f"cycle at edge {edge}."
            )
        traversed_edges.add(edge)
        ix_path.append(next_point[0])
        iy_path.append(next_point[1])
        current = next_point
        indicator = next_indicator

    raise RuntimeError(
        "Moore boundary trace exceeded its deterministic maximum-step bound "
        f"of {step_bound}."
    )


def extract_moore_boundary(
    x: object,
    y: object,
    concentration: object,
    *,
    threshold: float,
) -> LeadingEdgeExtractionResult:
    """Trace low-side boundary pixels and reduce them to common ``x_front(y)``."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    field = np.asarray(concentration, dtype=np.float64)
    if (
        x_values.ndim != 1
        or x_values.size < 2
        or not np.all(np.isfinite(x_values))
        or np.any(np.diff(x_values) <= 0.0)
    ):
        raise ValueError("x must be a finite, strictly increasing vector.")
    if (
        y_values.ndim != 1
        or y_values.size == 0
        or not np.all(np.isfinite(y_values))
        or np.any(np.diff(y_values) <= 0.0)
    ):
        raise ValueError("y must be a finite, strictly increasing vector.")
    if field.shape != (y_values.size, x_values.size):
        raise ValueError("concentration must have shape (y.size, x.size).")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite.") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")

    candidate_mask = build_low_side_boundary_candidate_mask(
        field,
        threshold=threshold_value,
    )
    if not np.any(candidate_mask):
        return LeadingEdgeExtractionResult(
            method=MOORE_BOUNDARY_METHOD,
            y=y_values,
            x_front=np.full(y_values.size, np.nan),
            success_mask=np.zeros(y_values.size, dtype=np.bool_),
            crossing_count=np.zeros(y_values.size, dtype=np.int64),
            threshold=threshold_value,
        )

    trace = trace_moore_boundary(candidate_mask)
    x_front = np.full(y_values.size, np.nan, dtype=np.float64)
    success_mask = np.zeros(y_values.size, dtype=np.bool_)
    crossing_count = np.zeros(y_values.size, dtype=np.int64)
    for iy in range(y_values.size):
        visited_ix = np.unique(trace.ix[trace.iy == iy])
        if visited_ix.size:
            x_front[iy] = float(np.max(x_values[visited_ix]))
            success_mask[iy] = True
            crossing_count[iy] = int(visited_ix.size)
    return LeadingEdgeExtractionResult(
        method=MOORE_BOUNDARY_METHOD,
        y=y_values,
        x_front=x_front,
        success_mask=success_mask,
        crossing_count=crossing_count,
        threshold=threshold_value,
    )


__all__ = (
    "MOORE_BOUNDARY_METHOD",
    "MOORE_INITIAL_INDICATOR",
    "MOORE_NEIGHBOR_OFFSETS",
    "MooreBoundaryTrace",
    "build_low_side_boundary_candidate_mask",
    "extract_moore_boundary",
    "moore_neighbor_offset",
    "select_moore_boundary_start",
    "trace_moore_boundary",
    "wrap_moore_indicator",
)
