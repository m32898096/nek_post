"""Directional GLL quadrature on structured axis-aligned Nek5000 meshes.

Geometry is accepted as affine when coordinate samples agree with the
endpoint affine map within an element-local tolerance.  That tolerance is the
larger of a two-float32-ULP storage floor at the element's coordinate magnitude
and float64 roundoff terms for its coordinate magnitude and physical span. An element
whose span is not at least 1024 such tolerances is rejected: its storage noise
would exceed 0.1 percent of the span, so an affine determination would not be
scientifically meaningful.  This admits four-byte Nek coordinates while
rejecting materially curved or poorly resolved elements.  Transverse replicas
of an assembled value must agree to ``rtol=1e-10`` and ``atol=1e-12`` before
they are averaged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_coordinates
from nek_post.gll import gll_nodes, gll_quadrature_weights
from nek_post.spectral_interpolation import element_coordinate_signature


PHYSICAL_COORDINATE_NAMES = ("x", "y", "z")
SUPPORTED_INTEGRATION_DIRECTIONS = PHYSICAL_COORDINATE_NAMES
# A float32-rounded sample can differ from the exact value by half an ULP.
# Comparing it with an affine profile reconstructed from two independently
# rounded endpoints has a one-ULP error budget; two ULPs retain a guard margin.
FLOAT32_STORAGE_ULP_FACTOR = 2.0
# The profile is formed and checked in float64.  This allows the accumulated
# arithmetic error (both at the coordinate magnitude and relative to the span)
# while remaining negligible compared with float32 storage.
FLOAT64_AFFINE_ROUNDOFF_FACTOR = 256.0
# Certify affine geometry only when the local uncertainty is at most 1 / 1024
# of the physical span (about 0.1 percent).
MIN_AFFINE_SPAN_TO_TOLERANCE_RATIO = 1024.0
TRANSVERSE_VALUE_RTOL = 1.0e-10
TRANSVERSE_VALUE_ATOL = 1.0e-12

_OUTPUT_COORDINATE_INDICES = {
    "x": (1, 2),  # horizontal y, vertical z
    "y": (0, 2),  # horizontal x, vertical z
    "z": (0, 1),  # horizontal x, vertical y
}


def _readonly_copy(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def normalize_integration_direction(direction: object) -> str:
    """Return one of the three supported physical integration directions."""
    if (
        not isinstance(direction, str)
        or direction not in SUPPORTED_INTEGRATION_DIRECTIONS
    ):
        raise ValueError("direction must be exactly one of: x, y, z.")
    return direction


@dataclass(frozen=True)
class GLLDirectionalIntegrationPlan:
    """Reusable geometry and assembly plan for one physical direction.

    Reuse requires identical stationary geometry and the same element ordering
    as the snapshot from which the plan was built.
    """

    direction: str
    physical_to_reference_axes: tuple[int, int, int]
    integration_reference_axis: int
    horizontal_coordinate_name: str
    vertical_coordinate_name: str
    horizontal_reference_axis: int
    vertical_reference_axis: int
    element_shape: tuple[int, int, int]
    element_count: int
    element_interval_counts: tuple[int, int, int]
    element_coordinate_signatures: tuple[str, ...]
    element_cell_indices: NDArray[np.int64]
    element_half_spans: NDArray[np.float64]
    element_physical_axis_increasing: NDArray[np.bool_]
    column_element_indices: NDArray[np.int64]
    column_horizontal_cell_indices: NDArray[np.int64]
    column_vertical_cell_indices: NDArray[np.int64]
    local_output_transpose: tuple[int, int]
    quadrature_weights: NDArray[np.float64]
    horizontal_coordinates: NDArray[np.float64]
    vertical_coordinates: NDArray[np.float64]
    horizontal_local_to_global: NDArray[np.int64]
    vertical_local_to_global: NDArray[np.int64]
    element_geometry_tolerances: NDArray[np.float64]
    topology_tolerances: tuple[float, float, float]


@dataclass(frozen=True)
class GLLDirectionalIntegrationResult:
    """One globally assembled physical-coordinate integration plane."""

    direction: str
    horizontal_coordinate_name: str
    vertical_coordinate_name: str
    horizontal_coordinates: NDArray[np.float64]
    vertical_coordinates: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        direction = normalize_integration_direction(self.direction)
        horizontal = _readonly_copy(self.horizontal_coordinates, np.float64)
        vertical = _readonly_copy(self.vertical_coordinates, np.float64)
        values = _readonly_copy(self.values, np.float64)
        if (
            horizontal.ndim != 1
            or vertical.ndim != 1
            or horizontal.size == 0
            or vertical.size == 0
            or not np.all(np.isfinite(horizontal))
            or not np.all(np.isfinite(vertical))
            or np.any(np.diff(horizontal) <= 0.0)
            or np.any(np.diff(vertical) <= 0.0)
        ):
            raise ValueError("Output coordinates must be finite increasing vectors.")
        if values.shape != (vertical.size, horizontal.size):
            raise ValueError(
                "values must have shape "
                "(vertical_coordinates.size, horizontal_coordinates.size)."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Integrated values must be finite.")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "horizontal_coordinates", horizontal)
        object.__setattr__(self, "vertical_coordinates", vertical)
        object.__setattr__(self, "values", values)


def _elements(data: object) -> tuple[Any, ...]:
    try:
        elements = tuple(data.elem)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "Nek data must provide an iterable 'elem' collection."
        ) from exc
    if not elements:
        raise ValueError("Nek data contains no elements.")
    return elements


def _validated_geometry(
    data: object,
) -> tuple[
    tuple[Any, ...],
    tuple[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        ...,
    ],
    tuple[int, int, int],
    tuple[str, ...],
]:
    elements = _elements(data)
    geometry = []
    element_shape: tuple[int, int, int] | None = None
    signatures = []
    for element_index, element in enumerate(elements):
        try:
            raw_coordinates = get_coordinates(element)
        except Exception as exc:
            raise ValueError(
                f"Element {element_index} does not provide complete coordinates: {exc}"
            ) from exc
        if any(np.iscomplexobj(component) for component in raw_coordinates):
            raise ValueError(
                f"Element {element_index} coordinates must be real numeric arrays."
            )
        coordinates = tuple(
            np.asarray(component, dtype=np.float64)
            for component in raw_coordinates
        )
        shapes = tuple(component.shape for component in coordinates)
        if any(component.ndim != 3 for component in coordinates):
            raise ValueError(
                f"Element {element_index} coordinates must be three-dimensional; "
                f"got {shapes}."
            )
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Element {element_index} coordinate shapes must match; got {shapes}."
            )
        shape = tuple(int(size) for size in shapes[0])
        if any(size < 2 for size in shape):
            raise ValueError(
                f"Element {element_index} must have at least two nodes per axis."
            )
        if element_shape is None:
            element_shape = shape
        elif shape != element_shape:
            raise ValueError(
                "All elements must use one consistent shape: "
                f"expected {element_shape}, element {element_index} has {shape}."
            )
        for physical_name, coordinate in zip(
            PHYSICAL_COORDINATE_NAMES, coordinates, strict=True
        ):
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(
                    f"Element {element_index} has a non-finite {physical_name} "
                    "coordinate."
                )
        geometry.append(coordinates)
        signatures.append(element_coordinate_signature(*coordinates))
    assert element_shape is not None
    return (
        elements,
        tuple(geometry),
        element_shape,
        tuple(signatures),
    )


def _element_geometry_tolerance(profile: NDArray[np.float64]) -> float:
    """Return a storage-aware affine tolerance for one coordinate profile.

    The float32 term uses the actual IEEE-754 spacing at the element's largest
    coordinate magnitude rather than a whole-domain scale.  The independent
    float64 term covers profile construction and affine reconstruction.
    """
    lower = float(profile[0])
    upper = float(profile[-1])
    span = abs(upper - lower)
    magnitude = float(np.max(np.abs(profile)))
    float32_ulp = abs(float(np.spacing(np.float32(magnitude))))
    storage_floor = FLOAT32_STORAGE_ULP_FACTOR * float32_ulp
    float64_roundoff = (
        FLOAT64_AFFINE_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * (magnitude + span + np.finfo(np.float64).tiny)
    )
    tolerance = max(storage_floor, float64_roundoff)
    if span <= MIN_AFFINE_SPAN_TO_TOLERANCE_RATIO * tolerance:
        raise ValueError(
            "Physical element span is too small relative to coordinate storage "
            "precision for a reliable affine validation: "
            f"span {span:.6g}, tolerance {tolerance:.6g}."
        )
    return tolerance


def _axis_profile(
    coordinate: NDArray[np.float64],
    axis: int,
) -> tuple[NDArray[np.float64], float]:
    axis_size = coordinate.shape[axis]
    lines = np.moveaxis(coordinate, axis, 0).reshape(axis_size, -1)
    profile = np.mean(lines, axis=1, dtype=np.float64)
    transverse_spread = float(np.max(np.ptp(lines, axis=1)))
    return profile, transverse_spread


def _affine_coordinate_values(
    profile: NDArray[np.float64],
    nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    midpoint = 0.5 * (profile[0] + profile[-1])
    half_span = 0.5 * (profile[-1] - profile[0])
    return midpoint + half_span * nodes


def _detect_geometry_mapping(
    geometry: tuple[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        ...,
    ],
    element_shape: tuple[int, int, int],
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[NDArray[np.float64], ...], ...],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    node_sets = tuple(gll_nodes(size) for size in element_shape)
    mapping: tuple[int, int, int] | None = None
    all_profiles = []
    all_increasing = np.empty((len(geometry), 3), dtype=np.bool_)
    bounds = np.empty((len(geometry), 3, 2), dtype=np.float64)
    element_tolerances = np.empty((len(geometry), 3), dtype=np.float64)

    for element_index, coordinates in enumerate(geometry):
        element_axes = []
        element_profiles = []
        for physical_index, coordinate in enumerate(coordinates):
            bounds[element_index, physical_index] = (
                float(np.min(coordinate)),
                float(np.max(coordinate)),
            )
            candidates = []
            for reference_axis in range(3):
                profile, transverse_spread = _axis_profile(
                    coordinate, reference_axis
                )
                differences = np.diff(profile)
                monotonic = bool(
                    np.all(differences > 0.0)
                    or np.all(differences < 0.0)
                )
                if not monotonic:
                    continue
                tolerance = _element_geometry_tolerance(profile)
                if transverse_spread <= tolerance and monotonic:
                    candidates.append((reference_axis, profile, tolerance))
            if len(candidates) != 1:
                raise ValueError(
                    "Ambiguous physical/reference-axis mapping for element "
                    f"{element_index}, coordinate "
                    f"{PHYSICAL_COORDINATE_NAMES[physical_index]}: expected one "
                    f"monotonic transverse-constant axis, found {len(candidates)}."
                )
            reference_axis, profile, tolerance = candidates[0]
            expected_profile = _affine_coordinate_values(
                profile, node_sets[reference_axis]
            )
            expected_shape = [1, 1, 1]
            expected_shape[reference_axis] = element_shape[reference_axis]
            expected_values = expected_profile.reshape(expected_shape)
            residual = float(np.max(np.abs(coordinate - expected_values)))
            if residual > tolerance:
                raise ValueError(
                    f"Element {element_index} physical "
                    f"{PHYSICAL_COORDINATE_NAMES[physical_index]} coordinate is "
                    "materially non-affine: maximum residual "
                    f"{residual:.6g} exceeds tolerance {tolerance:.6g}."
                )
            element_axes.append(reference_axis)
            element_profiles.append(profile)
            element_tolerances[element_index, physical_index] = tolerance
            all_increasing[element_index, physical_index] = bool(
                profile[-1] > profile[0]
            )
        element_mapping = tuple(element_axes)
        if len(set(element_mapping)) != 3:
            raise ValueError(
                "Physical/reference-axis mapping must be bijective for element "
                f"{element_index}; got {element_mapping}."
            )
        if mapping is None:
            mapping = element_mapping  # type: ignore[assignment]
        elif element_mapping != mapping:
            raise ValueError(
                "Physical/reference-axis mapping is inconsistent between "
                f"elements: expected {mapping}, element {element_index} has "
                f"{element_mapping}."
            )
        all_profiles.append(tuple(element_profiles))
    assert mapping is not None
    return (
        mapping,
        tuple(all_profiles),
        all_increasing,
        bounds,
        element_tolerances,
    )


def _cluster_intervals(
    bounds: NDArray[np.float64],
    tolerance: float,
    physical_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    order = np.lexsort((bounds[:, 1], bounds[:, 0]))
    groups: list[list[int]] = []
    representatives: list[NDArray[np.float64]] = []
    for raw_index in order:
        element_index = int(raw_index)
        pair = bounds[element_index]
        if (
            not representatives
            or np.max(np.abs(pair - representatives[-1])) > tolerance
        ):
            groups.append([element_index])
            representatives.append(pair.copy())
        else:
            groups[-1].append(element_index)
            representatives[-1] = np.mean(
                bounds[groups[-1]], axis=0, dtype=np.float64
            )
    intervals = np.asarray(representatives, dtype=np.float64)
    identifiers = np.empty(bounds.shape[0], dtype=np.int64)
    for interval_index, members in enumerate(groups):
        identifiers[members] = interval_index
    if np.any(intervals[:, 1] - intervals[:, 0] <= tolerance):
        raise ValueError(f"Physical {physical_name} element intervals are degenerate.")
    for interval_index in range(intervals.shape[0] - 1):
        gap = float(
            intervals[interval_index + 1, 0]
            - intervals[interval_index, 1]
        )
        if abs(gap) > tolerance:
            relation = "gap" if gap > 0.0 else "overlap"
            raise ValueError(
                f"Physical {physical_name} intervals contain a {relation} of "
                f"{abs(gap):.6g}."
            )
    return intervals, identifiers


def _validate_complete_topology(
    cell_indices: NDArray[np.int64],
    interval_counts: tuple[int, int, int],
) -> dict[tuple[int, int, int], int]:
    expected_count = int(np.prod(interval_counts, dtype=np.int64))
    lookup: dict[tuple[int, int, int], int] = {}
    for element_index, indices in enumerate(cell_indices):
        key = tuple(int(value) for value in indices)
        if key in lookup:
            raise ValueError(
                "Structured topology contains duplicate elements for physical "
                f"cell {key}: elements {lookup[key]} and {element_index}."
            )
        lookup[key] = element_index
    if len(lookup) != expected_count:
        missing = next(
            (
                (ix, iy, iz)
                for ix in range(interval_counts[0])
                for iy in range(interval_counts[1])
                for iz in range(interval_counts[2])
                if (ix, iy, iz) not in lookup
            ),
            None,
        )
        raise ValueError(
            "Structured topology is incomplete: expected "
            f"{expected_count} cells, found {len(lookup)}; first missing cell "
            f"is {missing}."
        )
    return lookup


def _physical_cell_profiles(
    profiles: tuple[tuple[NDArray[np.float64], ...], ...],
    increasing: NDArray[np.bool_],
    cell_indices: NDArray[np.int64],
    interval_counts: tuple[int, int, int],
    element_tolerances: NDArray[np.float64],
) -> tuple[tuple[NDArray[np.float64], ...], ...]:
    result = []
    for physical_index, physical_name in enumerate(PHYSICAL_COORDINATE_NAMES):
        coordinate_profiles = []
        for cell_index in range(interval_counts[physical_index]):
            members = np.flatnonzero(
                cell_indices[:, physical_index] == cell_index
            )
            tolerance = float(
                np.max(element_tolerances[members, physical_index])
            )
            normalized = np.stack(
                [
                    profiles[int(element_index)][physical_index]
                    if increasing[int(element_index), physical_index]
                    else profiles[int(element_index)][physical_index][::-1]
                    for element_index in members
                ]
            )
            reference = normalized[0]
            mismatch = float(np.max(np.abs(normalized - reference)))
            if mismatch > tolerance:
                raise ValueError(
                    f"Elements sharing physical {physical_name} cell "
                    f"{cell_index} have incompatible GLL coordinates: maximum "
                    f"difference {mismatch:.6g}."
                )
            coordinate_profiles.append(
                np.mean(normalized, axis=0, dtype=np.float64)
            )
        for cell_index in range(len(coordinate_profiles) - 1):
            tolerance = max(
                float(
                    np.max(
                        element_tolerances[
                            cell_indices[:, physical_index] == cell_index,
                            physical_index,
                        ]
                    )
                ),
                float(
                    np.max(
                        element_tolerances[
                            cell_indices[:, physical_index] == cell_index + 1,
                            physical_index,
                        ]
                    )
                ),
            )
            mismatch = abs(
                float(
                    coordinate_profiles[cell_index][-1]
                    - coordinate_profiles[cell_index + 1][0]
                )
            )
            if mismatch > tolerance:
                raise ValueError(
                    f"Adjacent physical {physical_name} cells have incompatible "
                    f"interface coordinates: difference {mismatch:.6g}."
                )
        result.append(tuple(coordinate_profiles))
    return tuple(result)


def _assemble_global_coordinate(
    cell_profiles: tuple[NDArray[np.float64], ...],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    node_count = cell_profiles[0].size
    local_to_global = np.empty(
        (len(cell_profiles), node_count), dtype=np.int64
    )
    coordinates = np.empty(
        len(cell_profiles) * (node_count - 1) + 1,
        dtype=np.float64,
    )
    for cell_index, profile in enumerate(cell_profiles):
        start = cell_index * (node_count - 1)
        stop = start + node_count
        local_to_global[cell_index] = np.arange(start, stop, dtype=np.int64)
        if cell_index == 0:
            coordinates[:node_count] = profile
        else:
            coordinates[start] = 0.5 * (coordinates[start] + profile[0])
            coordinates[start + 1 : stop] = profile[1:]
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("Assembled physical GLL coordinates are not increasing.")
    return coordinates, local_to_global


def build_gll_directional_integration_plan(
    data: object,
    *,
    direction: object,
) -> GLLDirectionalIntegrationPlan:
    """Build a reusable plan for one selectable physical direction."""
    normalized_direction = normalize_integration_direction(direction)
    elements, geometry, element_shape, signatures = _validated_geometry(data)
    mapping, profiles, increasing, bounds, element_tolerances = (
        _detect_geometry_mapping(
            geometry,
            element_shape,
        )
    )
    topology_tolerances = tuple(
        float(np.max(element_tolerances[:, physical]))
        for physical in range(3)
    )

    intervals = []
    cell_indices = np.empty((len(elements), 3), dtype=np.int64)
    for physical_index, physical_name in enumerate(PHYSICAL_COORDINATE_NAMES):
        physical_intervals, identifiers = _cluster_intervals(
            bounds[:, physical_index],
            topology_tolerances[physical_index],
            physical_name,
        )
        intervals.append(physical_intervals)
        cell_indices[:, physical_index] = identifiers
    interval_counts = tuple(int(value.shape[0]) for value in intervals)
    cell_lookup = _validate_complete_topology(cell_indices, interval_counts)
    cell_profiles = _physical_cell_profiles(
        profiles,
        increasing,
        cell_indices,
        interval_counts,
        element_tolerances,
    )

    integration_physical_axis = PHYSICAL_COORDINATE_NAMES.index(
        normalized_direction
    )
    horizontal_physical_axis, vertical_physical_axis = (
        _OUTPUT_COORDINATE_INDICES[normalized_direction]
    )
    integration_reference_axis = mapping[integration_physical_axis]
    horizontal_reference_axis = mapping[horizontal_physical_axis]
    vertical_reference_axis = mapping[vertical_physical_axis]
    remaining_reference_axes = tuple(
        axis for axis in range(3) if axis != integration_reference_axis
    )
    local_output_transpose = (
        remaining_reference_axes.index(vertical_reference_axis),
        remaining_reference_axes.index(horizontal_reference_axis),
    )

    horizontal_coordinates, horizontal_local_to_global = (
        _assemble_global_coordinate(cell_profiles[horizontal_physical_axis])
    )
    vertical_coordinates, vertical_local_to_global = (
        _assemble_global_coordinate(cell_profiles[vertical_physical_axis])
    )

    column_elements = []
    column_horizontal_cells = []
    column_vertical_cells = []
    for vertical_cell in range(interval_counts[vertical_physical_axis]):
        for horizontal_cell in range(interval_counts[horizontal_physical_axis]):
            elements_in_column = []
            for integration_cell in range(
                interval_counts[integration_physical_axis]
            ):
                key_values = [0, 0, 0]
                key_values[integration_physical_axis] = integration_cell
                key_values[horizontal_physical_axis] = horizontal_cell
                key_values[vertical_physical_axis] = vertical_cell
                key = tuple(key_values)
                try:
                    elements_in_column.append(cell_lookup[key])
                except KeyError as exc:
                    raise ValueError(
                        f"Directional column is missing physical cell {key}."
                    ) from exc
            column_elements.append(elements_in_column)
            column_horizontal_cells.append(horizontal_cell)
            column_vertical_cells.append(vertical_cell)

    half_spans = 0.5 * (
        bounds[:, integration_physical_axis, 1]
        - bounds[:, integration_physical_axis, 0]
    )
    if np.any(half_spans <= 0.0):
        raise ValueError("Integration-direction element spans must be positive.")

    return GLLDirectionalIntegrationPlan(
        direction=normalized_direction,
        physical_to_reference_axes=mapping,
        integration_reference_axis=integration_reference_axis,
        horizontal_coordinate_name=PHYSICAL_COORDINATE_NAMES[
            horizontal_physical_axis
        ],
        vertical_coordinate_name=PHYSICAL_COORDINATE_NAMES[
            vertical_physical_axis
        ],
        horizontal_reference_axis=horizontal_reference_axis,
        vertical_reference_axis=vertical_reference_axis,
        element_shape=element_shape,
        element_count=len(elements),
        element_interval_counts=interval_counts,
        element_coordinate_signatures=signatures,
        element_cell_indices=_readonly_copy(cell_indices, np.int64),
        element_half_spans=_readonly_copy(half_spans, np.float64),
        element_physical_axis_increasing=_readonly_copy(increasing, np.bool_),
        column_element_indices=_readonly_copy(column_elements, np.int64),
        column_horizontal_cell_indices=_readonly_copy(
            column_horizontal_cells, np.int64
        ),
        column_vertical_cell_indices=_readonly_copy(
            column_vertical_cells, np.int64
        ),
        local_output_transpose=local_output_transpose,
        quadrature_weights=gll_quadrature_weights(
            element_shape[integration_reference_axis]
        ),
        horizontal_coordinates=_readonly_copy(
            horizontal_coordinates, np.float64
        ),
        vertical_coordinates=_readonly_copy(vertical_coordinates, np.float64),
        horizontal_local_to_global=_readonly_copy(
            horizontal_local_to_global, np.int64
        ),
        vertical_local_to_global=_readonly_copy(
            vertical_local_to_global, np.int64
        ),
        element_geometry_tolerances=_readonly_copy(
            element_tolerances, np.float64
        ),
        topology_tolerances=topology_tolerances,
    )


def _validate_plan_geometry(
    plan: GLLDirectionalIntegrationPlan,
    data: object,
    source_file: object | None,
) -> tuple[Any, ...]:
    elements, _geometry, shape, signatures = _validated_geometry(data)
    context = "" if source_file is None else f" for {source_file}"
    if len(elements) != plan.element_count:
        raise ValueError(
            f"Incompatible geometry{context}: expected {plan.element_count} "
            f"elements, found {len(elements)}."
        )
    if shape != plan.element_shape:
        raise ValueError(
            f"Incompatible geometry{context}: expected element shape "
            f"{plan.element_shape}, found {shape}."
        )
    for element_index, (expected, actual) in enumerate(
        zip(plan.element_coordinate_signatures, signatures, strict=True)
    ):
        if actual != expected:
            raise ValueError(
                f"Incompatible geometry{context}: element {element_index} "
                "coordinate signature changed."
            )
    return elements


def _validated_scalar_fields(
    elements: tuple[Any, ...],
    field_getter: Callable[[Any], object],
    expected_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float64], ...]:
    fields = []
    for element_index, element in enumerate(elements):
        try:
            raw_field = field_getter(element)
            if np.iscomplexobj(raw_field):
                raise ValueError("field must be real")
            field = np.asarray(raw_field, dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"Could not obtain scalar field for element {element_index}: {exc}"
            ) from exc
        if field.shape != expected_shape:
            raise ValueError(
                f"Scalar field shape mismatch for element {element_index}: "
                f"expected {expected_shape}, got {field.shape}."
            )
        if not np.all(np.isfinite(field)):
            raise ValueError(
                f"Scalar field for element {element_index} contains non-finite values."
            )
        fields.append(field)
    return tuple(fields)


def apply_gll_directional_integration_plan(
    plan: GLLDirectionalIntegrationPlan,
    data: object,
    *,
    field_getter: Callable[[Any], object],
    source_file: object | None = None,
) -> GLLDirectionalIntegrationResult:
    """Apply a stationary-geometry directional integration plan.

    A plan is reusable only for identical stationary geometry in the same
    element ordering used to build it.  A reordered snapshot is intentionally
    rejected by coordinate-signature validation rather than remapped silently.
    """
    if not isinstance(plan, GLLDirectionalIntegrationPlan):
        raise ValueError("plan must be a GLLDirectionalIntegrationPlan.")
    if not callable(field_getter):
        raise ValueError("field_getter must be callable.")
    elements = _validate_plan_geometry(plan, data, source_file)
    fields = _validated_scalar_fields(
        elements, field_getter, plan.element_shape
    )

    output_sum = np.zeros(
        (
            plan.vertical_coordinates.size,
            plan.horizontal_coordinates.size,
        ),
        dtype=np.float64,
    )
    output_count = np.zeros(output_sum.shape, dtype=np.int64)
    horizontal_physical_axis = PHYSICAL_COORDINATE_NAMES.index(
        plan.horizontal_coordinate_name
    )
    vertical_physical_axis = PHYSICAL_COORDINATE_NAMES.index(
        plan.vertical_coordinate_name
    )

    for column_index, element_indices in enumerate(plan.column_element_indices):
        column_value = None
        for raw_element_index in element_indices:
            element_index = int(raw_element_index)
            local_value = np.tensordot(
                plan.quadrature_weights,
                fields[element_index],
                axes=(0, plan.integration_reference_axis),
            )
            local_value = np.transpose(
                local_value, axes=plan.local_output_transpose
            )
            if not plan.element_physical_axis_increasing[
                element_index, vertical_physical_axis
            ]:
                local_value = local_value[::-1, :]
            if not plan.element_physical_axis_increasing[
                element_index, horizontal_physical_axis
            ]:
                local_value = local_value[:, ::-1]
            local_value = (
                plan.element_half_spans[element_index] * local_value
            )
            if column_value is None:
                column_value = np.array(local_value, dtype=np.float64, copy=True)
            else:
                column_value += local_value
        assert column_value is not None

        horizontal_cell = int(
            plan.column_horizontal_cell_indices[column_index]
        )
        vertical_cell = int(plan.column_vertical_cell_indices[column_index])
        horizontal_indices = plan.horizontal_local_to_global[horizontal_cell]
        vertical_indices = plan.vertical_local_to_global[vertical_cell]
        output_index = np.ix_(vertical_indices, horizontal_indices)
        existing_count = output_count[output_index]
        overlap = existing_count > 0
        if np.any(overlap):
            existing_average = np.zeros_like(column_value)
            existing_average[overlap] = (
                output_sum[output_index][overlap] / existing_count[overlap]
            )
            agrees = np.isclose(
                existing_average[overlap],
                column_value[overlap],
                rtol=TRANSVERSE_VALUE_RTOL,
                atol=TRANSVERSE_VALUE_ATOL,
            )
            if not np.all(agrees):
                maximum_difference = float(
                    np.max(
                        np.abs(
                            existing_average[overlap]
                            - column_value[overlap]
                        )
                    )
                )
                raise ValueError(
                    "Integrated values from transverse element replicas are "
                    "incompatible: maximum difference "
                    f"{maximum_difference:.6g}."
                )
        output_sum[output_index] += column_value
        output_count[output_index] += 1

    if np.any(output_count == 0):
        raise RuntimeError("Directional integration left unassembled output nodes.")
    values = output_sum / output_count
    return GLLDirectionalIntegrationResult(
        direction=plan.direction,
        horizontal_coordinate_name=plan.horizontal_coordinate_name,
        vertical_coordinate_name=plan.vertical_coordinate_name,
        horizontal_coordinates=plan.horizontal_coordinates,
        vertical_coordinates=plan.vertical_coordinates,
        values=values,
    )


__all__ = (
    "FLOAT32_STORAGE_ULP_FACTOR",
    "FLOAT64_AFFINE_ROUNDOFF_FACTOR",
    "GLLDirectionalIntegrationPlan",
    "GLLDirectionalIntegrationResult",
    "MIN_AFFINE_SPAN_TO_TOLERANCE_RATIO",
    "PHYSICAL_COORDINATE_NAMES",
    "SUPPORTED_INTEGRATION_DIRECTIONS",
    "TRANSVERSE_VALUE_ATOL",
    "TRANSVERSE_VALUE_RTOL",
    "apply_gll_directional_integration_plan",
    "build_gll_directional_integration_plan",
    "normalize_integration_direction",
)
