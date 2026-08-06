"""Fortran-derived pure-Python/NumPy Moore boundary tracing."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field as dataclass_field, replace
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


@dataclass(frozen=True)
class MooreBoundaryComponent:
    """Immutable diagnostics for one periodic-y boundary component."""

    flat_indices: NDArray[np.int64]
    mask_shape: tuple[int, int]
    pixel_count: int = dataclass_field(init=False)
    unique_y_count: int = dataclass_field(init=False)
    max_ix: int = dataclass_field(init=False)
    minimum_flat_index: int = dataclass_field(init=False)

    def __post_init__(self) -> None:
        try:
            ny, nx = self.mask_shape
        except (TypeError, ValueError) as exc:
            raise ValueError("component mask_shape must contain ny and nx.") from exc
        if (
            not isinstance(ny, Integral)
            or isinstance(ny, (bool, np.bool_))
            or not isinstance(nx, Integral)
            or isinstance(nx, (bool, np.bool_))
            or int(ny) < 1
            or int(nx) < 1
        ):
            raise ValueError("component mask_shape values must be positive integers.")
        shape = (int(ny), int(nx))
        indices = _readonly_integer_copy(self.flat_indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("component flat_indices must be a nonempty vector.")
        indices = np.unique(indices)
        if indices[0] < 0 or indices[-1] >= shape[0] * shape[1]:
            raise ValueError("component flat_indices are outside mask_shape.")
        indices.setflags(write=False)
        iy_values = indices // shape[1]
        ix_values = indices % shape[1]
        object.__setattr__(self, "flat_indices", indices)
        object.__setattr__(self, "mask_shape", shape)
        object.__setattr__(self, "pixel_count", int(indices.size))
        object.__setattr__(
            self, "unique_y_count", int(np.unique(iy_values).size)
        )
        object.__setattr__(self, "max_ix", int(np.max(ix_values)))
        object.__setattr__(self, "minimum_flat_index", int(indices[0]))

    def to_mask(self) -> NDArray[np.bool_]:
        """Return an immutable boolean mask containing only this component."""
        mask = np.zeros(self.mask_shape, dtype=np.bool_)
        mask.flat[self.flat_indices] = True
        mask.setflags(write=False)
        return mask


def _validated_candidate_mask(candidate_mask: object) -> NDArray[np.bool_]:
    mask = np.asarray(candidate_mask, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("candidate_mask must be a nonempty two-dimensional array.")
    return mask


def label_periodic_boundary_components(
    candidate_mask: object,
) -> tuple[MooreBoundaryComponent, ...]:
    """Label deterministic 8-components with periodic y and non-periodic x."""
    mask = _validated_candidate_mask(candidate_mask)
    ny, nx = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[MooreBoundaryComponent] = []
    for start_flat in np.flatnonzero(mask):
        start_flat_value = int(start_flat)
        start_iy, start_ix = divmod(start_flat_value, nx)
        if visited[start_iy, start_ix]:
            continue
        visited[start_iy, start_ix] = True
        queue: deque[tuple[int, int]] = deque([(start_ix, start_iy)])
        component_indices: list[int] = []
        while queue:
            ix, iy = queue.popleft()
            component_indices.append(iy * nx + ix)
            for delta_ix, delta_iy in MOORE_NEIGHBOR_OFFSETS:
                next_ix = ix + delta_ix
                if not 0 <= next_ix < nx:
                    continue
                next_iy = (iy + delta_iy) % ny
                if mask[next_iy, next_ix] and not visited[next_iy, next_ix]:
                    visited[next_iy, next_ix] = True
                    queue.append((next_ix, next_iy))
        components.append(
            MooreBoundaryComponent(
                flat_indices=np.asarray(component_indices, dtype=np.int64),
                mask_shape=mask.shape,
            )
        )
    return tuple(components)


def select_primary_moore_component(
    components: Sequence[MooreBoundaryComponent],
) -> MooreBoundaryComponent | None:
    """Select by y coverage, size, front position, then smallest identifier."""
    supplied = tuple(components)
    if not supplied:
        return None
    if not all(isinstance(item, MooreBoundaryComponent) for item in supplied):
        raise ValueError("components must contain MooreBoundaryComponent values.")
    return max(
        supplied,
        key=lambda item: (
            item.unique_y_count,
            item.pixel_count,
            item.max_ix,
            -item.minimum_flat_index,
        ),
    )


def primary_moore_component_mask(
    candidate_mask: object,
) -> NDArray[np.bool_]:
    """Return an immutable mask containing only the selected primary component."""
    mask = _validated_candidate_mask(candidate_mask)
    primary = select_primary_moore_component(
        label_periodic_boundary_components(mask)
    )
    if primary is None:
        result = np.zeros(mask.shape, dtype=np.bool_)
        result.setflags(write=False)
        return result
    return primary.to_mask()


def select_moore_boundary_start(
    candidate_mask: object,
) -> tuple[int, int] | None:
    """Choose the deterministic front-biased ``(ix, iy)`` start point."""
    mask = _validated_candidate_mask(candidate_mask)
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
    start_point: tuple[int, int] | None = None,
    initial_indicator: int = MOORE_INITIAL_INDICATOR,
    maximum_steps: int | None = None,
) -> MooreBoundaryTrace:
    """Trace one Moore boundary using directed-edge closure and cycle checks."""
    mask = _validated_candidate_mask(candidate_mask)
    if not isinstance(initial_indicator, Integral) or isinstance(
        initial_indicator, (bool, np.bool_)
    ):
        raise ValueError("initial_indicator must be an integer from 1 through 8.")
    indicator_value = int(initial_indicator)
    if not 1 <= indicator_value <= 8:
        raise ValueError("initial_indicator must be an integer from 1 through 8.")
    if start_point is None:
        start = select_moore_boundary_start(mask)
    else:
        try:
            raw_ix, raw_iy = start_point
        except (TypeError, ValueError) as exc:
            raise ValueError("start_point must contain integer ix and iy.") from exc
        if (
            not isinstance(raw_ix, Integral)
            or isinstance(raw_ix, (bool, np.bool_))
            or not isinstance(raw_iy, Integral)
            or isinstance(raw_iy, (bool, np.bool_))
        ):
            raise ValueError("start_point must contain integer ix and iy.")
        start = (int(raw_ix), int(raw_iy))
        if (
            not 0 <= start[0] < mask.shape[1]
            or not 0 <= start[1] < mask.shape[0]
            or not mask[start[1], start[0]]
        ):
            raise ValueError("start_point must belong to the supplied candidate mask.")
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
    indicator = indicator_value
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


def moore_trace_winding_number(
    trace: MooreBoundaryTrace,
    ny: int,
) -> int:
    """Return the integer periodic-y winding, including the closing edge."""
    if not isinstance(ny, Integral) or isinstance(ny, (bool, np.bool_)):
        raise ValueError("ny must be an integer greater than one.")
    ny_value = int(ny)
    if ny_value < 2:
        raise ValueError("ny must be an integer greater than one.")
    if not trace.closed:
        raise ValueError("Moore trace winding requires a closed trace.")
    if trace.path_length < 2:
        raise ValueError("Moore trace winding requires at least two path points.")
    if np.any(trace.iy < 0) or np.any(trace.iy >= ny_value):
        raise ValueError("Moore trace y indices are outside the supplied period.")

    winding_sum = 0
    for position in range(trace.path_length):
        next_position = (position + 1) % trace.path_length
        current_ix = int(trace.ix[position])
        current_iy = int(trace.iy[position])
        next_ix = int(trace.ix[next_position])
        next_iy = int(trace.iy[next_position])
        delta_ix = next_ix - current_ix
        if current_iy == ny_value - 1 and next_iy == 0:
            delta_iy = 1
        elif current_iy == 0 and next_iy == ny_value - 1:
            delta_iy = -1
        else:
            delta_iy = next_iy - current_iy
        if (
            delta_ix not in (-1, 0, 1)
            or delta_iy not in (-1, 0, 1)
            or (delta_ix == 0 and delta_iy == 0)
        ):
            raise ValueError(
                "Moore trace contains an edge that is not a valid periodic "
                f"Moore-neighbour step: {((current_ix, current_iy), (next_ix, next_iy))}."
            )
        winding_sum += delta_iy
    if winding_sum % ny_value != 0:
        raise ValueError(
            "Closed Moore trace has a non-integer periodic-y winding: "
            f"unwrapped y sum {winding_sum} for ny={ny_value}."
        )
    return winding_sum // ny_value


@dataclass(frozen=True)
class MooreTraceCandidateDiagnostic:
    """Immutable outcome and topology metrics for one deterministic walk."""

    start_point: tuple[int, int]
    initial_indicator: int
    closed: bool
    winding_number: int | None
    unique_y_count: int
    path_length: int
    maximum_ix: int | None
    median_rowwise_x: float | None
    mean_rowwise_x: float | None
    selected: bool = False
    failure_reason: str | None = None
    duplicate_of: tuple[tuple[int, int], int] | None = None


@dataclass(frozen=True)
class MooreTraceSelection:
    """Selected trace and diagnostics for all attempted deterministic walks."""

    trace: MooreBoundaryTrace
    selected_candidate: MooreTraceCandidateDiagnostic
    candidates: tuple[MooreTraceCandidateDiagnostic, ...]


def _canonical_trace_edges(
    trace: MooreBoundaryTrace,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    points = tuple(
        (int(ix), int(iy))
        for ix, iy in zip(trace.ix, trace.iy, strict=True)
    )
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for position, point in enumerate(points):
        next_point = points[(position + 1) % len(points)]
        edges.append(
            (point, next_point) if point <= next_point else (next_point, point)
        )
    return tuple(sorted(edges))


def _periodic_seam_starts(mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    ny, nx = mask.shape
    points = {
        (int(ix), iy)
        for iy in (ny - 1, 0)
        for ix in np.flatnonzero(mask[iy])
    }
    if not points:
        fallback = select_moore_boundary_start(mask)
        if fallback is not None:
            points.add(fallback)
    return tuple(
        sorted(
            points,
            key=lambda point: (
                -point[0],
                -point[1],
                point[1] * nx + point[0],
            ),
        )
    )


def select_periodic_moore_trace(
    component_mask: object,
    *,
    x: object | None = None,
) -> MooreTraceSelection:
    """Select a one-winding trace, preferring greatest rowwise physical x."""
    mask = _validated_candidate_mask(component_mask)
    if x is None:
        x_values = np.arange(mask.shape[1], dtype=np.float64)
    else:
        x_values = np.asarray(x, dtype=np.float64)
        if (
            x_values.ndim != 1
            or x_values.size != mask.shape[1]
            or not np.all(np.isfinite(x_values))
            or np.any(np.diff(x_values) <= 0.0)
        ):
            raise ValueError(
                "x must be a finite, strictly increasing vector matching the "
                "component-mask columns."
            )
    component_y_count = int(np.count_nonzero(np.any(mask, axis=1)))
    if component_y_count == 0:
        raise ValueError("component_mask must contain at least one boundary pixel.")

    diagnostics: list[MooreTraceCandidateDiagnostic] = []
    successful: list[tuple[MooreBoundaryTrace, int]] = []
    canonical_owners: dict[
        tuple[tuple[tuple[int, int], tuple[int, int]], ...],
        tuple[tuple[int, int], int],
    ] = {}
    for start in _periodic_seam_starts(mask):
        for indicator in range(1, 9):
            try:
                trace = trace_moore_boundary(
                    mask,
                    start_point=start,
                    initial_indicator=indicator,
                )
                if not trace.closed:
                    raise RuntimeError("candidate walk did not produce a closed trace")
                winding = moore_trace_winding_number(trace, mask.shape[0])
            except (RuntimeError, ValueError) as exc:
                diagnostics.append(
                    MooreTraceCandidateDiagnostic(
                        start_point=start,
                        initial_indicator=indicator,
                        closed=False,
                        winding_number=None,
                        unique_y_count=0,
                        path_length=0,
                        maximum_ix=None,
                        median_rowwise_x=None,
                        mean_rowwise_x=None,
                        failure_reason=str(exc),
                    )
                )
                continue

            rowwise_x = np.asarray(
                [
                    np.max(x_values[trace.ix[trace.iy == iy]])
                    for iy in np.unique(trace.iy)
                ],
                dtype=np.float64,
            )
            diagnostic = MooreTraceCandidateDiagnostic(
                start_point=start,
                initial_indicator=indicator,
                closed=True,
                winding_number=winding,
                unique_y_count=int(np.unique(trace.iy).size),
                path_length=trace.path_length,
                maximum_ix=int(np.max(trace.ix)),
                median_rowwise_x=float(np.median(rowwise_x)),
                mean_rowwise_x=float(np.mean(rowwise_x)),
            )
            canonical = _canonical_trace_edges(trace)
            owner = canonical_owners.get(canonical)
            if owner is not None:
                diagnostics.append(replace(diagnostic, duplicate_of=owner))
                continue
            canonical_owners[canonical] = (start, indicator)
            diagnostic_index = len(diagnostics)
            diagnostics.append(diagnostic)
            successful.append((trace, diagnostic_index))

    one_winding = [
        item
        for item in successful
        if abs(int(diagnostics[item[1]].winding_number)) == 1
    ]
    if one_winding:
        eligible = one_winding
    else:
        if component_y_count == mask.shape[0]:
            raise RuntimeError(
                "Selected Moore boundary component spans every y row, but no "
                "closed trace with absolute periodic-y winding 1 was found."
            )
        eligible = [
            item
            for item in successful
            if diagnostics[item[1]].winding_number == 0
        ]
        if not eligible:
            raise RuntimeError(
                "Selected partial-y Moore boundary component produced no "
                "closed zero-winding trace."
            )

    def selection_key(
        item: tuple[MooreBoundaryTrace, int],
    ) -> tuple[int, float, float, int, int]:
        diagnostic = diagnostics[item[1]]
        start_flat = (
            diagnostic.start_point[1] * mask.shape[1]
            + diagnostic.start_point[0]
        )
        return (
            diagnostic.unique_y_count,
            float(diagnostic.median_rowwise_x),
            float(diagnostic.mean_rowwise_x),
            -start_flat,
            -diagnostic.initial_indicator,
        )

    selected_trace, selected_index = max(eligible, key=selection_key)
    diagnostics[selected_index] = replace(
        diagnostics[selected_index], selected=True
    )
    return MooreTraceSelection(
        trace=selected_trace,
        selected_candidate=diagnostics[selected_index],
        candidates=tuple(diagnostics),
    )


def _select_extraction_trace(
    candidate_mask: NDArray[np.bool_],
    x_values: NDArray[np.float64],
) -> MooreBoundaryTrace:
    """Select across full-y components without pixel count outranking x."""
    components = label_periodic_boundary_components(candidate_mask)
    full_span: list[tuple[MooreBoundaryComponent, MooreTraceSelection]] = []
    for component in components:
        if component.unique_y_count != candidate_mask.shape[0]:
            continue
        try:
            selection = select_periodic_moore_trace(
                component.to_mask(),
                x=x_values,
            )
        except RuntimeError:
            continue
        if abs(int(selection.selected_candidate.winding_number)) == 1:
            full_span.append((component, selection))

    if full_span:
        def selection_key(
            item: tuple[MooreBoundaryComponent, MooreTraceSelection],
        ) -> tuple[int, float, float, int, int, int]:
            component, selection = item
            diagnostic = selection.selected_candidate
            start_flat = (
                diagnostic.start_point[1] * candidate_mask.shape[1]
                + diagnostic.start_point[0]
            )
            return (
                diagnostic.unique_y_count,
                float(diagnostic.median_rowwise_x),
                float(diagnostic.mean_rowwise_x),
                -component.minimum_flat_index,
                -start_flat,
                -diagnostic.initial_indicator,
            )

        return max(full_span, key=selection_key)[1].trace

    primary = select_primary_moore_component(components)
    if primary is None:
        raise RuntimeError("No Moore boundary component is available for tracing.")
    return select_periodic_moore_trace(
        primary.to_mask(),
        x=x_values,
    ).trace


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

    trace = _select_extraction_trace(candidate_mask, x_values)
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
    "MooreBoundaryComponent",
    "MooreBoundaryTrace",
    "MooreTraceCandidateDiagnostic",
    "MooreTraceSelection",
    "build_low_side_boundary_candidate_mask",
    "extract_moore_boundary",
    "moore_neighbor_offset",
    "moore_trace_winding_number",
    "label_periodic_boundary_components",
    "primary_moore_component_mask",
    "select_primary_moore_component",
    "select_periodic_moore_trace",
    "select_moore_boundary_start",
    "trace_moore_boundary",
    "wrap_moore_indicator",
)
