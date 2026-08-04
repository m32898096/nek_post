"""Element-aware interpolation on a fixed physical horizontal plane.

The target grid is uniform in physical ``x`` and periodic physical ``y``.
Every target has coordinates ``(Xi[row, column], Yi[row, column], z_target)``.
Rows therefore vary in ``y`` and columns vary in ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_concentration, get_coordinates
from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_nodes,
)
from nek_post.spectral_interpolation import (
    InverseMapResult,
    SpectralGeometryMismatchError,
    SpectralInverseMappingDiagnostics,
    element_coordinate_signature,
    invert_map_newton_3d,
)


SPECTRAL_HORIZONTAL_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class SpectralHorizontalSlicePlan:
    """Immutable stationary-geometry plan for one physical ``z`` plane.

    ``target_shape`` is ``(dense_ny, nx)``. Consequently, arrays returned by
    :func:`apply_spectral_horizontal_slice_plan` are indexed as ``C[iy, ix]``.
    """

    target_shape: tuple[int, int]
    Xi: NDArray[np.float64]
    Yi: NDArray[np.float64]
    z_target: float
    y_min: float
    y_max: float
    native_ny: int
    dense_ny: int
    y_upsample_factor: int
    element_count: int
    element_shape: tuple[int, int, int]
    polynomial_order: tuple[int, int, int]
    element_coordinate_signatures: tuple[str, ...]
    target_valid_mask: NDArray[np.bool_]
    owner_element_index: NDArray[np.int64]
    valid_target_flat_indices: NDArray[np.int64]
    q0: NDArray[np.float64]
    q1: NDArray[np.float64]
    q2: NDArray[np.float64]
    basis_q0: NDArray[np.float64]
    basis_q1: NDArray[np.float64]
    basis_q2: NDArray[np.float64]
    inverse_mapping_diagnostics: SpectralInverseMappingDiagnostics
    algorithm_version: int = SPECTRAL_HORIZONTAL_ALGORITHM_VERSION


def _readonly(
    array: object,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _positive_grid_size(value: int, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return parsed


def _elements(data: object) -> tuple[Any, ...]:
    try:
        elements = tuple(data.elem)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise ValueError("Nek data must provide an iterable 'elem' collection.") from exc
    if not elements:
        raise ValueError("Nek data contains no spectral elements.")
    return elements


def _element_geometry(
    element: object,
    element_index: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    try:
        coordinates = tuple(
            np.asarray(component, dtype=np.float64)
            for component in get_coordinates(element)
        )
    except Exception as exc:
        raise ValueError(
            f"Element {element_index} does not provide complete coordinates: {exc}"
        ) from exc
    shapes = tuple(component.shape for component in coordinates)
    if any(component.ndim != 3 for component in coordinates):
        raise ValueError(
            f"Element {element_index} coordinate arrays must be three-dimensional; "
            f"got shapes {shapes}."
        )
    if len(set(shapes)) != 1:
        raise ValueError(
            f"Element {element_index} x, y, and z shapes must match; got {shapes}."
        )
    if any(size < 2 for size in shapes[0]):
        raise ValueError(
            f"Element {element_index} must have at least two GLL nodes per axis; "
            f"got shape {shapes[0]}."
        )
    for name, component in zip(("x", "y", "z"), coordinates, strict=True):
        if not np.all(np.isfinite(component)):
            bad = tuple(int(value) for value in np.argwhere(~np.isfinite(component))[0])
            raise ValueError(
                f"Element {element_index} has non-finite {name} coordinate at {bad}."
            )
    return coordinates  # type: ignore[return-value]


def _element_concentration(
    element: object,
    element_index: int,
    expected_shape: tuple[int, int, int],
) -> NDArray[np.float64]:
    try:
        concentration = np.asarray(get_concentration(element), dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Element {element_index} does not provide concentration: {exc}"
        ) from exc
    if concentration.ndim != 3 or concentration.shape != expected_shape:
        raise ValueError(
            f"Element {element_index} concentration shape mismatch: expected "
            f"{expected_shape}, got {concentration.shape}."
        )
    return concentration


def _validated_snapshot(
    data: object,
    *,
    require_concentration: bool,
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
    geometry = tuple(
        _element_geometry(element, index) for index, element in enumerate(elements)
    )
    element_shape = tuple(int(size) for size in geometry[0][0].shape)
    for index, coordinates in enumerate(geometry[1:], start=1):
        actual_shape = tuple(int(size) for size in coordinates[0].shape)
        if actual_shape != element_shape:
            raise ValueError(
                "All spectral elements in one snapshot must use the same shape: "
                f"element 0 has {element_shape}, element {index} has {actual_shape}."
            )
    if require_concentration:
        for index, element in enumerate(elements):
            _element_concentration(element, index, element_shape)
    signatures = tuple(
        element_coordinate_signature(*coordinates) for coordinates in geometry
    )
    return elements, geometry, element_shape, signatures


def _coordinate_tolerance(
    lower: float,
    upper: float,
    tolerance_factor: float,
) -> float:
    span = upper - lower
    coordinate_scale = max(abs(lower), abs(upper), span, 1.0)
    return max(
        64.0 * np.finfo(np.float64).eps * coordinate_scale,
        tolerance_factor * max(span, np.finfo(np.float64).tiny),
    )


def _deduplicate_sorted_coordinates(
    values: object,
    tolerance: float,
) -> NDArray[np.float64]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if sorted_values.size < 2:
        raise ValueError("The spanwise mesh must provide at least two y coordinates.")
    groups: list[list[float]] = [[float(sorted_values[0])]]
    for value in sorted_values[1:]:
        value_float = float(value)
        if value_float - groups[-1][-1] <= tolerance:
            groups[-1].append(value_float)
        else:
            groups.append([value_float])
    representatives = np.asarray(
        [float(np.mean(group, dtype=np.float64)) for group in groups],
        dtype=np.float64,
    )
    if representatives.size < 2 or np.any(np.diff(representatives) <= tolerance):
        raise ValueError(
            "Could not establish distinct ordered global spanwise coordinates."
        )
    return representatives


def _coordinate_level_index(
    levels: NDArray[np.float64],
    value: float,
    tolerance: float,
) -> int:
    insertion = int(np.searchsorted(levels, value))
    candidates = tuple(
        index for index in (insertion - 1, insertion) if 0 <= index < levels.size
    )
    if not candidates:
        raise ValueError("Could not match a spanwise coordinate to a global level.")
    best = min(candidates, key=lambda index: abs(float(levels[index]) - value))
    if abs(float(levels[best]) - value) > tolerance:
        raise ValueError("Could not match a spanwise coordinate to a global level.")
    return best


def _element_spanwise_profile(
    y: NDArray[np.float64],
    element_index: int,
    tolerance: float,
) -> tuple[int, NDArray[np.float64]]:
    candidates: list[tuple[int, NDArray[np.float64]]] = []
    for axis, axis_size in enumerate(y.shape):
        rows = np.moveaxis(y, axis, 0).reshape(axis_size, -1)
        profile = np.mean(rows, axis=1, dtype=np.float64)
        transverse_spread = np.max(np.ptp(rows, axis=1))
        differences = np.diff(profile)
        monotone = bool(
            np.all(differences > tolerance) or np.all(differences < -tolerance)
        )
        if transverse_spread <= tolerance and monotone:
            candidates.append((axis, np.sort(profile)))
    if len(candidates) != 1:
        raise ValueError(
            "Structured spanwise geometry could not be established for element "
            f"{element_index}: expected y to vary monotonically along exactly one "
            f"reference axis, found {len(candidates)} candidates."
        )
    return candidates[0]


def _native_periodic_y_resolution(
    geometry: tuple[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        ...,
    ],
    *,
    tolerance_factor: float,
) -> tuple[int, float, float, float]:
    y_min = min(float(np.min(coordinates[1])) for coordinates in geometry)
    y_max = max(float(np.max(coordinates[1])) for coordinates in geometry)
    if not y_max > y_min:
        raise ValueError("The spectral mesh does not define a non-zero y span.")
    tolerance = _coordinate_tolerance(y_min, y_max, tolerance_factor)

    profiles: list[NDArray[np.float64]] = []
    spanwise_axes: list[int] = []
    for element_index, coordinates in enumerate(geometry):
        axis, profile = _element_spanwise_profile(
            coordinates[1], element_index, tolerance
        )
        spanwise_axes.append(axis)
        profiles.append(profile)
    if len(set(spanwise_axes)) != 1:
        raise ValueError(
            "Structured spanwise geometry requires one consistent reference-axis "
            "orientation across all elements."
        )

    levels = _deduplicate_sorted_coordinates(
        np.concatenate(profiles), tolerance
    )
    unique_spans: dict[tuple[int, int], tuple[int, ...]] = {}
    for element_index, profile in enumerate(profiles):
        indices = tuple(
            _coordinate_level_index(levels, float(value), tolerance)
            for value in profile
        )
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError(
                "Element spanwise GLL coordinates do not map to distinct ordered "
                f"global levels for element {element_index}."
            )
        key = (indices[0], indices[-1])
        previous_profile = unique_spans.get(key)
        if previous_profile is not None and previous_profile != indices:
            raise ValueError(
                "Elements sharing a physical y interval have incompatible GLL "
                f"coordinates near y=[{levels[key[0]]:.16g}, "
                f"{levels[key[1]]:.16g}]."
            )
        unique_spans[key] = indices

    ordered_spans = sorted(unique_spans.items())
    if ordered_spans[0][0][0] != 0 or ordered_spans[-1][0][1] != levels.size - 1:
        raise ValueError("Spanwise element intervals do not cover the complete y domain.")
    previous_upper = ordered_spans[0][0][1]
    for (lower, upper), _indices in ordered_spans[1:]:
        if lower != previous_upper:
            raise ValueError(
                "Spanwise element intervals must form one contiguous, non-overlapping "
                "structured partition."
            )
        previous_upper = upper

    interval_count = sum(len(indices) - 1 for _key, indices in ordered_spans)
    native_ny = int(levels.size - 1)
    if interval_count != native_ny:
        raise ValueError(
            "Deduplicated spanwise GLL levels are incompatible with the element "
            f"partition: found {native_ny} global intervals but {interval_count} "
            "element-local intervals."
        )
    # y_min and y_max are distinct physical coordinates but the same periodic seam.
    # Thus N+1 global levels represent N independent periodic intervals.
    return native_ny, float(levels[0]), float(levels[-1]), tolerance


def _bin_shape(
    element_count: int,
    x_span: float,
    y_span: float,
) -> tuple[int, int]:
    target_bin_count = max(1, 4 * element_count)
    if x_span <= 0.0 or y_span <= 0.0:
        return 1, 1
    aspect = x_span / y_span
    x_bins = max(1, min(512, int(np.ceil(np.sqrt(target_bin_count * aspect)))))
    y_bins = max(1, min(512, int(np.ceil(target_bin_count / x_bins))))
    return x_bins, y_bins


def _bin_coordinate(
    value: float,
    lower: float,
    upper: float,
    count: int,
) -> int:
    if count == 1 or upper <= lower:
        return 0
    scaled = (value - lower) / (upper - lower)
    return int(np.clip(np.floor(scaled * count), 0, count - 1))


def build_spectral_horizontal_slice_plan(
    data: object,
    *,
    nx: int,
    z_target: float,
    y_upsample_factor: int = 2,
    physical_tolerance_factor: float = 1.0e-11,
    reference_tolerance: float = 1.0e-8,
    max_iterations: int = 30,
) -> SpectralHorizontalSlicePlan:
    """Build an element-aware reusable plan for a uniform physical x-y grid."""
    nx_value = _positive_grid_size(nx, "nx", 2)
    upsample_factor = _positive_grid_size(
        y_upsample_factor, "y_upsample_factor", 1
    )
    try:
        resolved_z = float(z_target)
    except (TypeError, ValueError) as exc:
        raise ValueError("z_target must be finite.") from exc
    if not np.isfinite(resolved_z):
        raise ValueError("z_target must be finite.")
    if physical_tolerance_factor <= 0.0:
        raise ValueError("physical_tolerance_factor must be positive.")
    if reference_tolerance < 0.0:
        raise ValueError("reference_tolerance must be non-negative.")
    _positive_grid_size(max_iterations, "max_iterations", 1)

    elements, geometry, element_shape, signatures = _validated_snapshot(
        data, require_concentration=True
    )
    polynomial_order = tuple(size - 1 for size in element_shape)
    native_ny, y_min, y_max, spanwise_tolerance = _native_periodic_y_resolution(
        geometry,
        tolerance_factor=physical_tolerance_factor,
    )
    dense_ny = upsample_factor * native_ny

    aabbs = np.asarray(
        [
            [
                float(np.min(coordinates[0])),
                float(np.max(coordinates[0])),
                float(np.min(coordinates[1])),
                float(np.max(coordinates[1])),
                float(np.min(coordinates[2])),
                float(np.max(coordinates[2])),
            ]
            for coordinates in geometry
        ],
        dtype=np.float64,
    )
    global_spans = (
        float(np.max(aabbs[:, 1]) - np.min(aabbs[:, 0])),
        float(np.max(aabbs[:, 3]) - np.min(aabbs[:, 2])),
        float(np.max(aabbs[:, 5]) - np.min(aabbs[:, 4])),
    )
    global_scale = float(np.linalg.norm(global_spans))
    aabb_tolerance = max(
        32.0 * np.finfo(np.float64).eps * max(global_scale, 1.0),
        physical_tolerance_factor
        * max(global_scale, np.finfo(np.float64).tiny),
        spanwise_tolerance,
    )
    eligible = np.flatnonzero(
        (aabbs[:, 4] - aabb_tolerance <= resolved_z)
        & (resolved_z <= aabbs[:, 5] + aabb_tolerance)
    )
    if eligible.size == 0:
        z_min = float(np.min(aabbs[:, 4]))
        z_max = float(np.max(aabbs[:, 5]))
        raise ValueError(
            f"No spectral element z AABB intersects z_target={resolved_z:.16g}; "
            f"mesh z range is [{z_min:.16g}, {z_max:.16g}]."
        )

    x_min = float(np.min(aabbs[eligible, 0]))
    x_max = float(np.max(aabbs[eligible, 1]))
    if not x_max > x_min:
        raise ValueError(
            "Intersecting spectral elements do not define a non-zero x span."
        )
    x_coordinates = np.linspace(x_min, x_max, nx_value, dtype=np.float64)
    y_coordinates = np.linspace(
        y_min,
        y_max,
        dense_ny,
        endpoint=False,
        dtype=np.float64,
    )
    Xi, Yi = np.meshgrid(x_coordinates, y_coordinates)
    target_shape = tuple(int(value) for value in Xi.shape)

    x_bins, y_bins = _bin_shape(
        int(eligible.size), x_max - x_min, y_max - y_min
    )
    spatial_bins: list[list[int]] = [
        [] for _ in range(x_bins * y_bins)
    ]
    for element_index in eligible:
        bounds = aabbs[element_index]
        ix0 = _bin_coordinate(bounds[0] - aabb_tolerance, x_min, x_max, x_bins)
        ix1 = _bin_coordinate(bounds[1] + aabb_tolerance, x_min, x_max, x_bins)
        iy0 = _bin_coordinate(bounds[2] - aabb_tolerance, y_min, y_max, y_bins)
        iy1 = _bin_coordinate(bounds[3] + aabb_tolerance, y_min, y_max, y_bins)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                spatial_bins[iy * x_bins + ix].append(int(element_index))
    bin_candidates = tuple(
        tuple(sorted(set(values))) for values in spatial_bins
    )

    owner = np.full(Xi.size, -1, dtype=np.int64)
    q_rows: list[NDArray[np.float64]] = []
    valid_indices: list[int] = []
    candidate_pairs = 0
    inverse_attempts = 0
    inverse_successes = 0
    inverse_failures = 0
    ambiguous = 0
    maximum_residual = 0.0
    maximum_iterations_used = 0

    for flat_index, (x_value, y_value) in enumerate(
        zip(Xi.ravel(), Yi.ravel(), strict=True)
    ):
        ix = _bin_coordinate(float(x_value), x_min, x_max, x_bins)
        iy = _bin_coordinate(float(y_value), y_min, y_max, y_bins)
        candidates = bin_candidates[iy * x_bins + ix]
        candidate_pairs += len(candidates)
        accepted: list[tuple[float, int, InverseMapResult]] = []
        target = (float(x_value), float(y_value), resolved_z)
        for element_index in candidates:
            bounds = aabbs[element_index]
            if not (
                bounds[0] - aabb_tolerance <= x_value <= bounds[1] + aabb_tolerance
                and bounds[2] - aabb_tolerance
                <= y_value
                <= bounds[3] + aabb_tolerance
                and bounds[4] - aabb_tolerance
                <= resolved_z
                <= bounds[5] + aabb_tolerance
            ):
                continue
            inverse_attempts += 1
            result = invert_map_newton_3d(
                *geometry[element_index],
                target,
                max_iterations=max_iterations,
                physical_tolerance_factor=physical_tolerance_factor,
                reference_tolerance=reference_tolerance,
            )
            maximum_iterations_used = max(
                maximum_iterations_used, result.iterations
            )
            if result.converged:
                inverse_successes += 1
                maximum_residual = max(maximum_residual, result.residual)
                accepted.append((result.residual, element_index, result))
            else:
                inverse_failures += 1
        if not accepted:
            continue
        if len(accepted) > 1:
            ambiguous += 1
        _residual, element_index, best = min(
            accepted, key=lambda item: (item[0], item[1])
        )
        owner[flat_index] = element_index
        valid_indices.append(flat_index)
        q_rows.append(best.q)

    q_values = (
        np.vstack(q_rows).astype(np.float64, copy=False)
        if q_rows
        else np.empty((0, 3), dtype=np.float64)
    )
    node_sets = tuple(gll_nodes(size) for size in element_shape)
    weight_sets = tuple(barycentric_weights(nodes) for nodes in node_sets)
    basis_rows = [
        np.empty((q_values.shape[0], size), dtype=np.float64)
        for size in element_shape
    ]
    for row_index, q in enumerate(q_values):
        for axis in range(3):
            basis_rows[axis][row_index] = barycentric_basis_and_derivative(
                node_sets[axis], weight_sets[axis], q[axis]
            )[0]

    valid_mask = owner >= 0
    diagnostics = SpectralInverseMappingDiagnostics(
        target_point_count=int(Xi.size),
        candidate_pair_count=candidate_pairs,
        inverse_attempt_count=inverse_attempts,
        inverse_success_count=inverse_successes,
        inverse_failure_count=inverse_failures,
        ambiguous_boundary_point_count=ambiguous,
        maximum_successful_residual=maximum_residual,
        maximum_iteration_count=maximum_iterations_used,
        spatial_bin_shape=(x_bins, y_bins),
    )
    return SpectralHorizontalSlicePlan(
        target_shape=target_shape,
        Xi=_readonly(Xi, np.float64),
        Yi=_readonly(Yi, np.float64),
        z_target=resolved_z,
        y_min=y_min,
        y_max=y_max,
        native_ny=native_ny,
        dense_ny=dense_ny,
        y_upsample_factor=upsample_factor,
        element_count=len(elements),
        element_shape=element_shape,
        polynomial_order=polynomial_order,
        element_coordinate_signatures=signatures,
        target_valid_mask=_readonly(valid_mask.reshape(target_shape), np.bool_),
        owner_element_index=_readonly(owner.reshape(target_shape), np.int64),
        valid_target_flat_indices=_readonly(valid_indices, np.int64),
        q0=_readonly(q_values[:, 0], np.float64),
        q1=_readonly(q_values[:, 1], np.float64),
        q2=_readonly(q_values[:, 2], np.float64),
        basis_q0=_readonly(basis_rows[0], np.float64),
        basis_q1=_readonly(basis_rows[1], np.float64),
        basis_q2=_readonly(basis_rows[2], np.float64),
        inverse_mapping_diagnostics=diagnostics,
    )


def validate_spectral_horizontal_geometry(
    plan: SpectralHorizontalSlicePlan,
    data: object,
    *,
    source_file: object | None = None,
) -> None:
    """Verify exact element shapes and coordinate signatures for plan reuse."""
    source = "<unknown source>" if source_file is None else str(source_file)
    elements = _elements(data)
    if len(elements) != plan.element_count:
        raise SpectralGeometryMismatchError(
            f"Spectral horizontal geometry mismatch for {source}: expected "
            f"{plan.element_count} elements, got {len(elements)}."
        )
    for index, element in enumerate(elements):
        coordinates = _element_geometry(element, index)
        actual_shape = tuple(int(size) for size in coordinates[0].shape)
        if actual_shape != plan.element_shape:
            raise SpectralGeometryMismatchError(
                f"Spectral horizontal geometry mismatch for {source}, element "
                f"{index}: expected shape {plan.element_shape}, got {actual_shape}."
            )
        actual_signature = element_coordinate_signature(*coordinates)
        expected_signature = plan.element_coordinate_signatures[index]
        if actual_signature != expected_signature:
            raise SpectralGeometryMismatchError(
                f"Spectral horizontal geometry mismatch for {source}, element "
                f"{index}: expected coordinate signature {expected_signature}, got "
                f"{actual_signature}."
            )


def apply_spectral_horizontal_slice_plan(
    plan: SpectralHorizontalSlicePlan,
    data: object,
    *,
    source_file: object | None = None,
) -> NDArray[np.float64]:
    """Evaluate concentration as ``C[iy, ix]`` on the plan's x-y grid.

    The returned shape is ``(dense_ny, nx)``. Rows follow increasing periodic
    physical y and columns follow increasing physical x. Unmapped targets stay
    NaN; concentration is neither clipped nor otherwise modified.
    """
    validate_spectral_horizontal_geometry(plan, data, source_file=source_file)
    elements = _elements(data)
    concentrations = tuple(
        _element_concentration(element, index, plan.element_shape)
        for index, element in enumerate(elements)
    )
    result = np.full(np.prod(plan.target_shape), np.nan, dtype=np.float64)
    valid_owner = plan.owner_element_index.ravel()[plan.valid_target_flat_indices]
    for element_index in np.unique(valid_owner):
        owned_rows = np.flatnonzero(valid_owner == element_index)
        target_indices = plan.valid_target_flat_indices[owned_rows]
        result[target_indices] = np.einsum(
            "pa,pb,pc,abc->p",
            plan.basis_q0[owned_rows],
            plan.basis_q1[owned_rows],
            plan.basis_q2[owned_rows],
            concentrations[int(element_index)],
            optimize=True,
        )
    return result.reshape(plan.target_shape)


def spectral_horizontal_plan_metadata(
    plan: SpectralHorizontalSlicePlan,
) -> Mapping[str, float | int]:
    """Return immutable numeric metadata for a horizontal interpolation plan."""
    return MappingProxyType(
        {
            "xmin": float(np.min(plan.Xi)),
            "xmax": float(np.max(plan.Xi)),
            "ymin": float(plan.y_min),
            "ymax_periodic_endpoint": float(plan.y_max),
            "nx": int(plan.target_shape[1]),
            "native_ny": int(plan.native_ny),
            "dense_ny": int(plan.dense_ny),
            "y_upsample_factor": int(plan.y_upsample_factor),
            "z_target": float(plan.z_target),
            "spectral_element_n0": int(plan.element_shape[0]),
            "spectral_element_n1": int(plan.element_shape[1]),
            "spectral_element_n2": int(plan.element_shape[2]),
            "spectral_polynomial_order_q0": int(plan.polynomial_order[0]),
            "spectral_polynomial_order_q1": int(plan.polynomial_order[1]),
            "spectral_polynomial_order_q2": int(plan.polynomial_order[2]),
            "spectral_horizontal_algorithm_version": int(plan.algorithm_version),
        }
    )


__all__ = (
    "SPECTRAL_HORIZONTAL_ALGORITHM_VERSION",
    "SpectralHorizontalSlicePlan",
    "apply_spectral_horizontal_slice_plan",
    "build_spectral_horizontal_slice_plan",
    "spectral_horizontal_plan_metadata",
    "validate_spectral_horizontal_geometry",
)
