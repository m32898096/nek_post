"""Element-aware tensor-product interpolation of Nek5000 spectral fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_concentration, get_coordinates
from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_nodes,
)


SPECTRAL_ALGORITHM_VERSION = 1


class SpectralGeometryMismatchError(ValueError):
    """Raised when a snapshot cannot reuse a stationary spectral plan."""


@dataclass(frozen=True)
class InverseMapResult:
    """Result and diagnostics from one physical-to-reference inversion."""

    q: NDArray[np.float64]
    converged: bool
    residual: float
    iterations: int
    reason: str


@dataclass(frozen=True)
class SpectralInverseMappingDiagnostics:
    """Aggregate diagnostics recorded while constructing a slice plan."""

    target_point_count: int
    candidate_pair_count: int
    inverse_attempt_count: int
    inverse_success_count: int
    inverse_failure_count: int
    ambiguous_boundary_point_count: int
    maximum_successful_residual: float
    maximum_iteration_count: int
    spatial_bin_shape: tuple[int, int]


@dataclass(frozen=True)
class SpectralSliceInterpolationPlan:
    """Reusable stationary-geometry plan with factorized tensor basis rows."""

    target_shape: tuple[int, int]
    Xi: NDArray[np.float64]
    Zi: NDArray[np.float64]
    y_target: float
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
    algorithm_version: int = SPECTRAL_ALGORITHM_VERSION


def _readonly(
    array: object,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


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
        raw_coordinates = get_coordinates(element)
    except Exception as exc:
        raise ValueError(
            f"Element {element_index} does not provide complete coordinates: {exc}"
        ) from exc
    coordinates = tuple(
        np.asarray(component, dtype=np.float64) for component in raw_coordinates
    )
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
    shape = shapes[0]
    if any(size < 2 for size in shape):
        raise ValueError(
            f"Element {element_index} must have at least two GLL nodes per axis; "
            f"got shape {shape}."
        )
    for name, component in zip(("x", "y", "z"), coordinates, strict=True):
        if not np.all(np.isfinite(component)):
            bad = tuple(int(value) for value in np.argwhere(~np.isfinite(component))[0])
            raise ValueError(
                f"Element {element_index} has non-finite {name} coordinate at {bad}."
            )
    return coordinates


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


def element_coordinate_signature(
    x: object,
    y: object,
    z: object,
) -> str:
    """Return a stable BLAKE2b signature for one element's geometry."""
    coordinates = tuple(np.asarray(value, dtype=np.float64) for value in (x, y, z))
    if any(value.ndim != 3 for value in coordinates):
        raise ValueError("Coordinate signatures require three-dimensional arrays.")
    if len({value.shape for value in coordinates}) != 1:
        raise ValueError("Coordinate signatures require matching x, y, and z shapes.")
    digest = blake2b(digest_size=20, person=b"nek-spectral-v1")
    shape = np.asarray(coordinates[0].shape, dtype="<i8")
    digest.update(shape.tobytes(order="C"))
    for coordinate in coordinates:
        canonical = np.ascontiguousarray(coordinate, dtype="<f8")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


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
        _element_geometry(element, index)
        for index, element in enumerate(elements)
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


def _basis_triplet(
    node_sets: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    weight_sets: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    q: Sequence[float],
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    return tuple(
        barycentric_basis_and_derivative(nodes, weights, coordinate)
        for nodes, weights, coordinate in zip(
            node_sets, weight_sets, q, strict=True
        )
    )  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _reference_data(
    element_shape: tuple[int, int, int],
) -> tuple[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
]:
    node_sets = tuple(gll_nodes(size) for size in element_shape)
    weight_sets = tuple(barycentric_weights(nodes) for nodes in node_sets)
    return node_sets, weight_sets  # type: ignore[return-value]


def evaluate_tensor_bary3(
    values: object,
    basis_q0: object,
    basis_q1: object,
    basis_q2: object,
) -> float:
    """Evaluate one tensor-product field from three factorized basis rows."""
    values_arr = np.asarray(values, dtype=np.float64)
    basis = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (basis_q0, basis_q1, basis_q2)
    )
    if values_arr.ndim != 3:
        raise ValueError("values must be a three-dimensional element array.")
    expected = tuple(row.size for row in basis)
    if any(row.ndim != 1 for row in basis) or values_arr.shape != expected:
        raise ValueError(
            f"Basis sizes must match values shape {values_arr.shape}; got {expected}."
        )
    return float(
        np.einsum(
            "a,b,c,abc->",
            basis[0],
            basis[1],
            basis[2],
            values_arr,
            optimize=False,
        )
    )


def eval_tensor_bary3(
    values: object,
    basis_q0: object,
    basis_q1: object,
    basis_q2: object,
) -> float:
    """Compatibility spelling matching the original MATLAB utility name."""
    return evaluate_tensor_bary3(values, basis_q0, basis_q1, basis_q2)


def _mapping_only(
    coordinates: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    basis_data: tuple[
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
) -> NDArray[np.float64]:
    basis = tuple(value[0] for value in basis_data)
    return np.asarray(
        [
            evaluate_tensor_bary3(component, *basis)
            for component in coordinates
        ],
        dtype=np.float64,
    )


def _mapping_and_jacobian(
    coordinates: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    basis_data: tuple[
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    basis = tuple(value[0] for value in basis_data)
    derivative = tuple(value[1] for value in basis_data)
    physical = _mapping_only(coordinates, basis_data)
    jacobian = np.empty((3, 3), dtype=np.float64)
    for physical_axis, component in enumerate(coordinates):
        jacobian[physical_axis, 0] = evaluate_tensor_bary3(
            component, derivative[0], basis[1], basis[2]
        )
        jacobian[physical_axis, 1] = evaluate_tensor_bary3(
            component, basis[0], derivative[1], basis[2]
        )
        jacobian[physical_axis, 2] = evaluate_tensor_bary3(
            component, basis[0], basis[1], derivative[2]
        )
    return physical, jacobian


def invert_map_newton_3d(
    x: object,
    y: object,
    z: object,
    target: object,
    *,
    initial_q: object | None = None,
    max_iterations: int = 30,
    physical_tolerance_factor: float = 1.0e-11,
    reference_tolerance: float = 1.0e-8,
    maximum_backtracking_steps: int = 12,
) -> InverseMapResult:
    """Invert one curved tensor-product element mapping with damped Newton."""
    coordinates = tuple(
        np.asarray(component, dtype=np.float64) for component in (x, y, z)
    )
    shapes = tuple(component.shape for component in coordinates)
    if any(component.ndim != 3 for component in coordinates) or len(set(shapes)) != 1:
        raise ValueError(
            "x, y, and z must be matching three-dimensional element arrays."
        )
    if any(size < 2 for size in shapes[0]):
        raise ValueError("Every element axis must contain at least two GLL nodes.")
    if not all(np.all(np.isfinite(component)) for component in coordinates):
        raise ValueError("Element coordinates must be finite.")
    target_arr = np.asarray(target, dtype=np.float64)
    if target_arr.shape != (3,) or not np.all(np.isfinite(target_arr)):
        raise ValueError("target must contain three finite physical coordinates.")
    if (
        not isinstance(max_iterations, Integral)
        or isinstance(max_iterations, (bool, np.bool_))
        or int(max_iterations) < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if physical_tolerance_factor <= 0.0 or reference_tolerance < 0.0:
        raise ValueError("Inverse-map tolerances must be positive or zero as appropriate.")

    element_shape = tuple(int(size) for size in shapes[0])
    node_sets, weight_sets = _reference_data(element_shape)
    stacked = np.stack(coordinates)
    lower = np.min(stacked, axis=(1, 2, 3))
    upper = np.max(stacked, axis=(1, 2, 3))
    element_scale = float(np.linalg.norm(upper - lower))
    if not np.isfinite(element_scale) or element_scale <= np.finfo(np.float64).tiny:
        return InverseMapResult(
            q=_readonly((np.nan, np.nan, np.nan), np.float64),
            converged=False,
            residual=float("inf"),
            iterations=0,
            reason="degenerate_element_scale",
        )
    physical_tolerance = physical_tolerance_factor * element_scale

    if initial_q is None:
        squared_distance = sum(
            (component - target_arr[axis]) ** 2
            for axis, component in enumerate(coordinates)
        )
        nearest = np.unravel_index(int(np.argmin(squared_distance)), element_shape)
        q = np.asarray(
            [node_sets[axis][nearest[axis]] for axis in range(3)],
            dtype=np.float64,
        )
    else:
        q = np.asarray(initial_q, dtype=np.float64).copy()
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError("initial_q must contain three finite coordinates.")

    residual_norm = float("inf")
    for iteration in range(int(max_iterations) + 1):
        basis_data = _basis_triplet(node_sets, weight_sets, q)
        physical, jacobian = _mapping_and_jacobian(coordinates, basis_data)
        residual_vector = physical - target_arr
        residual_norm = float(np.linalg.norm(residual_vector))
        if residual_norm <= physical_tolerance:
            inside = bool(
                np.all(q >= -1.0 - reference_tolerance)
                and np.all(q <= 1.0 + reference_tolerance)
            )
            if not inside:
                return InverseMapResult(
                    q=_readonly(q, np.float64),
                    converged=False,
                    residual=residual_norm,
                    iterations=iteration,
                    reason="outside_reference_element",
                )
            stable_q = q.copy()
            stable_q[
                (stable_q < -1.0) & (stable_q >= -1.0 - reference_tolerance)
            ] = -1.0
            stable_q[
                (stable_q > 1.0) & (stable_q <= 1.0 + reference_tolerance)
            ] = 1.0
            return InverseMapResult(
                q=_readonly(stable_q, np.float64),
                converged=True,
                residual=residual_norm,
                iterations=iteration,
                reason="converged",
            )
        if iteration == int(max_iterations):
            break
        try:
            update = np.linalg.solve(jacobian, residual_vector)
        except np.linalg.LinAlgError:
            return InverseMapResult(
                q=_readonly(q, np.float64),
                converged=False,
                residual=residual_norm,
                iterations=iteration,
                reason="singular_jacobian",
            )
        if not np.all(np.isfinite(update)):
            return InverseMapResult(
                q=_readonly(q, np.float64),
                converged=False,
                residual=residual_norm,
                iterations=iteration,
                reason="non_finite_newton_update",
            )

        accepted = False
        step = 1.0
        for _ in range(maximum_backtracking_steps + 1):
            trial_q = q - step * update
            trial_basis = _basis_triplet(node_sets, weight_sets, trial_q)
            trial_physical = _mapping_only(coordinates, trial_basis)
            trial_norm = float(np.linalg.norm(trial_physical - target_arr))
            if np.isfinite(trial_norm) and trial_norm < residual_norm:
                q = trial_q
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return InverseMapResult(
                q=_readonly(q, np.float64),
                converged=False,
                residual=residual_norm,
                iterations=iteration,
                reason="backtracking_failed",
            )

    return InverseMapResult(
        q=_readonly(q, np.float64),
        converged=False,
        residual=residual_norm,
        iterations=int(max_iterations),
        reason="maximum_iterations",
    )


def _grid_size(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    parsed = int(value)
    if parsed < 2:
        raise ValueError(f"{name} must be greater than or equal to 2.")
    return parsed


def _bin_shape(
    element_count: int,
    x_span: float,
    z_span: float,
) -> tuple[int, int]:
    target_bin_count = max(1, 4 * element_count)
    if x_span <= 0.0 or z_span <= 0.0:
        return 1, 1
    aspect = x_span / z_span
    x_bins = max(1, min(512, int(np.ceil(np.sqrt(target_bin_count * aspect)))))
    z_bins = max(1, min(512, int(np.ceil(target_bin_count / x_bins))))
    return x_bins, z_bins


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


def build_spectral_slice_interpolation_plan(
    data: object,
    *,
    nx: int,
    nz: int,
    y_target: float | None = None,
    physical_tolerance_factor: float = 1.0e-11,
    reference_tolerance: float = 1.0e-8,
    max_iterations: int = 30,
) -> SpectralSliceInterpolationPlan:
    """Build a reusable element-aware plan for a fixed physical x-z slice."""
    nx_value = _grid_size(nx, "nx")
    nz_value = _grid_size(nz, "nz")
    elements, geometry, element_shape, signatures = _validated_snapshot(
        data, require_concentration=True
    )
    polynomial_order = tuple(size - 1 for size in element_shape)

    all_y_min = min(float(np.min(coordinates[1])) for coordinates in geometry)
    all_y_max = max(float(np.max(coordinates[1])) for coordinates in geometry)
    if y_target is None:
        resolved_y = 0.5 * (all_y_min + all_y_max)
    else:
        resolved_y = float(y_target)
        if not np.isfinite(resolved_y):
            raise ValueError("y_target must be finite when supplied.")

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
    global_scale = float(
        np.linalg.norm(
            (
                np.max(aabbs[:, 1]) - np.min(aabbs[:, 0]),
                np.max(aabbs[:, 3]) - np.min(aabbs[:, 2]),
                np.max(aabbs[:, 5]) - np.min(aabbs[:, 4]),
            )
        )
    )
    aabb_tolerance = max(
        np.finfo(np.float64).eps * max(global_scale, 1.0) * 32.0,
        physical_tolerance_factor * max(global_scale, np.finfo(np.float64).tiny),
    )
    eligible = np.flatnonzero(
        (aabbs[:, 2] - aabb_tolerance <= resolved_y)
        & (resolved_y <= aabbs[:, 3] + aabb_tolerance)
    )
    if eligible.size == 0:
        raise ValueError(
            f"No spectral element y AABB intersects y_target={resolved_y:.16g}."
        )

    x_min = float(np.min(aabbs[eligible, 0]))
    x_max = float(np.max(aabbs[eligible, 1]))
    z_min = float(np.min(aabbs[eligible, 4]))
    z_max = float(np.max(aabbs[eligible, 5]))
    if not x_max > x_min or not z_max > z_min:
        raise ValueError(
            "Intersecting spectral elements do not define non-zero x and z spans."
        )
    Xi, Zi = np.meshgrid(
        np.linspace(x_min, x_max, nx_value, dtype=np.float64),
        np.linspace(z_min, z_max, nz_value, dtype=np.float64),
    )
    target_shape = tuple(int(value) for value in Xi.shape)

    x_bins, z_bins = _bin_shape(
        int(eligible.size), x_max - x_min, z_max - z_min
    )
    spatial_bins: list[list[int]] = [
        [] for _ in range(x_bins * z_bins)
    ]
    for element_index in eligible:
        bounds = aabbs[element_index]
        ix0 = _bin_coordinate(bounds[0] - aabb_tolerance, x_min, x_max, x_bins)
        ix1 = _bin_coordinate(bounds[1] + aabb_tolerance, x_min, x_max, x_bins)
        iz0 = _bin_coordinate(bounds[4] - aabb_tolerance, z_min, z_max, z_bins)
        iz1 = _bin_coordinate(bounds[5] + aabb_tolerance, z_min, z_max, z_bins)
        for iz in range(iz0, iz1 + 1):
            for ix in range(ix0, ix1 + 1):
                spatial_bins[iz * x_bins + ix].append(int(element_index))
    bin_candidates = tuple(tuple(sorted(set(values))) for values in spatial_bins)

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

    for flat_index, (x_value, z_value) in enumerate(
        zip(Xi.ravel(), Zi.ravel(), strict=True)
    ):
        ix = _bin_coordinate(float(x_value), x_min, x_max, x_bins)
        iz = _bin_coordinate(float(z_value), z_min, z_max, z_bins)
        candidates = bin_candidates[iz * x_bins + ix]
        candidate_pairs += len(candidates)
        accepted: list[tuple[float, int, InverseMapResult]] = []
        target = (float(x_value), resolved_y, float(z_value))
        for element_index in candidates:
            bounds = aabbs[element_index]
            if not (
                bounds[0] - aabb_tolerance <= x_value <= bounds[1] + aabb_tolerance
                and bounds[2] - aabb_tolerance
                <= resolved_y
                <= bounds[3] + aabb_tolerance
                and bounds[4] - aabb_tolerance
                <= z_value
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
        residual, element_index, best = min(
            accepted, key=lambda item: (item[0], item[1])
        )
        del residual
        owner[flat_index] = element_index
        valid_indices.append(flat_index)
        q_rows.append(best.q)

    if q_rows:
        q_values = np.vstack(q_rows).astype(np.float64, copy=False)
    else:
        q_values = np.empty((0, 3), dtype=np.float64)
    node_sets, weight_sets = _reference_data(element_shape)
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
        spatial_bin_shape=(x_bins, z_bins),
    )
    return SpectralSliceInterpolationPlan(
        target_shape=target_shape,
        Xi=_readonly(Xi, np.float64),
        Zi=_readonly(Zi, np.float64),
        y_target=resolved_y,
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


def validate_spectral_geometry(
    plan: SpectralSliceInterpolationPlan,
    data: object,
    *,
    source_file: object | None = None,
) -> None:
    """Verify exact element shapes and coordinate signatures for plan reuse."""
    source = "<unknown source>" if source_file is None else str(source_file)
    elements = _elements(data)
    if len(elements) != plan.element_count:
        raise SpectralGeometryMismatchError(
            f"Spectral geometry mismatch for {source}: expected "
            f"{plan.element_count} elements, got {len(elements)}."
        )
    for index, element in enumerate(elements):
        coordinates = _element_geometry(element, index)
        actual_shape = tuple(int(size) for size in coordinates[0].shape)
        if actual_shape != plan.element_shape:
            raise SpectralGeometryMismatchError(
                f"Spectral geometry mismatch for {source}, element {index}: "
                f"expected shape {plan.element_shape}, got {actual_shape}."
            )
        actual_signature = element_coordinate_signature(*coordinates)
        expected_signature = plan.element_coordinate_signatures[index]
        if actual_signature != expected_signature:
            raise SpectralGeometryMismatchError(
                f"Spectral geometry mismatch for {source}, element {index}: "
                f"expected coordinate signature {expected_signature}, got "
                f"{actual_signature}."
            )


def apply_spectral_slice_interpolation_plan(
    plan: SpectralSliceInterpolationPlan,
    data: object,
    *,
    source_file: object | None = None,
) -> NDArray[np.float64]:
    """Validate one frame and apply factorized element-local interpolation."""
    validate_spectral_geometry(plan, data, source_file=source_file)
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


def spectral_plan_metadata(
    plan: SpectralSliceInterpolationPlan,
) -> Mapping[str, float | int]:
    """Return immutable numeric grid metadata for a spectral plan."""
    return MappingProxyType(
        {
            "xmin": float(np.min(plan.Xi)),
            "xmax": float(np.max(plan.Xi)),
            "zmin": float(np.min(plan.Zi)),
            "zmax": float(np.max(plan.Zi)),
            "nx": int(plan.target_shape[1]),
            "nz": int(plan.target_shape[0]),
            "spectral_slice_y": float(plan.y_target),
            "spectral_element_n0": int(plan.element_shape[0]),
            "spectral_element_n1": int(plan.element_shape[1]),
            "spectral_element_n2": int(plan.element_shape[2]),
            "spectral_polynomial_order_q0": int(plan.polynomial_order[0]),
            "spectral_polynomial_order_q1": int(plan.polynomial_order[1]),
            "spectral_polynomial_order_q2": int(plan.polynomial_order[2]),
        }
    )
