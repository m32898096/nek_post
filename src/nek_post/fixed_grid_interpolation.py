"""Reusable interpolation geometry for stationary scattered x-z coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError, cKDTree


class FixedGridGeometryMismatchError(ValueError):
    """Raised when source geometry cannot safely reuse an interpolation plan."""


@dataclass(frozen=True)
class FixedGridInterpolationPlan:
    """NumPy-only geometry needed to repeatedly interpolate onto one grid."""

    method: str
    duplicate_decimals: int
    expected_raw_point_count: int
    finite_coordinate_mask: NDArray[np.bool_]
    expected_rounded_source_keys: NDArray[np.float64]
    duplicate_group_inverse_indices: NDArray[np.int64]
    duplicate_group_counts: NDArray[np.int64]
    deduplicated_x: NDArray[np.float64]
    deduplicated_z: NDArray[np.float64]
    target_shape: tuple[int, int]
    valid_target_mask: NDArray[np.bool_]
    target_simplex_indices: NDArray[np.int64] | None
    source_vertex_indices: NDArray[np.int64] | None
    barycentric_weights: NDArray[np.float64] | None
    nearest_source_indices: NDArray[np.int64] | None


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _source_arrays(
    x: object,
    z: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    z_arr = np.asarray(z, dtype=np.float64).ravel()
    if x_arr.shape != z_arr.shape:
        raise ValueError("x and z must have matching one-dimensional sizes.")
    return x_arr, z_arr


def _target_arrays(
    Xi: object,
    Zi: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    Xi_arr = np.asarray(Xi, dtype=np.float64)
    Zi_arr = np.asarray(Zi, dtype=np.float64)
    if Xi_arr.ndim != 2 or Zi_arr.ndim != 2 or Xi_arr.shape != Zi_arr.shape:
        raise ValueError("Xi and Zi must have matching two-dimensional shapes.")
    if Xi_arr.size == 0:
        raise ValueError("Xi and Zi must contain at least one target point.")
    return Xi_arr, Zi_arr


def _duplicate_decimals(value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError("duplicate_decimals must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("duplicate_decimals must be a non-negative integer.")
    return parsed


def _method(value: str) -> str:
    if value not in {"linear", "nearest"}:
        raise ValueError("method must be exactly 'linear' or 'nearest'.")
    return value


def build_fixed_grid_interpolation_plan(
    x: object,
    z: object,
    Xi: object,
    Zi: object,
    *,
    method: str = "linear",
    duplicate_decimals: int = 10,
) -> FixedGridInterpolationPlan:
    """Build reusable duplicate grouping and target interpolation weights."""
    method_value = _method(method)
    decimals = _duplicate_decimals(duplicate_decimals)
    x_arr, z_arr = _source_arrays(x, z)
    Xi_arr, Zi_arr = _target_arrays(Xi, Zi)

    finite_coordinate_mask = np.isfinite(x_arr) & np.isfinite(z_arr)
    if not np.any(finite_coordinate_mask):
        raise ValueError("No finite x-z source coordinates are available.")
    x_finite = x_arr[finite_coordinate_mask]
    z_finite = z_arr[finite_coordinate_mask]
    finite_keys = np.column_stack(
        (
            np.round(x_finite, decimals),
            np.round(z_finite, decimals),
        )
    )
    _, inverse, counts = np.unique(
        finite_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    inverse = np.asarray(inverse, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    deduplicated_x = np.bincount(inverse, weights=x_finite) / counts
    deduplicated_z = np.bincount(inverse, weights=z_finite) / counts
    unique_count = int(counts.size)
    if method_value == "nearest" and unique_count < 1:
        raise ValueError("Nearest interpolation requires a unique source point.")
    if method_value == "linear" and unique_count < 3:
        raise ValueError(
            "Linear interpolation requires at least three unique source points."
        )

    rounded_source_keys = np.full((x_arr.size, 2), np.nan, dtype=np.float64)
    rounded_source_keys[finite_coordinate_mask] = finite_keys
    target_finite = np.isfinite(Xi_arr) & np.isfinite(Zi_arr)
    if not np.any(target_finite):
        raise ValueError("Xi and Zi contain no finite target coordinates.")
    target_points = np.column_stack(
        (Xi_arr.ravel()[target_finite.ravel()], Zi_arr.ravel()[target_finite.ravel()])
    )
    source_points = np.column_stack((deduplicated_x, deduplicated_z))

    valid_target_mask = np.zeros(Xi_arr.size, dtype=bool)
    target_simplex_indices: NDArray[np.int64] | None = None
    source_vertex_indices: NDArray[np.int64] | None = None
    barycentric_weights: NDArray[np.float64] | None = None
    nearest_source_indices: NDArray[np.int64] | None = None

    if method_value == "linear":
        try:
            triangulation = Delaunay(source_points)
        except QhullError as exc:
            raise ValueError(
                "Linear interpolation requires at least three non-collinear "
                "unique source points."
            ) from exc
        finite_simplex = triangulation.find_simplex(target_points)
        target_simplex_indices = np.full(Xi_arr.size, -1, dtype=np.int64)
        target_simplex_indices[target_finite.ravel()] = finite_simplex
        valid_target_mask = target_simplex_indices >= 0
        valid_simplices = target_simplex_indices[valid_target_mask]
        source_vertex_indices = np.asarray(
            triangulation.simplices[valid_simplices],
            dtype=np.int64,
        )
        transforms = triangulation.transform[valid_simplices]
        valid_targets = np.column_stack(
            (
                Xi_arr.ravel()[valid_target_mask],
                Zi_arr.ravel()[valid_target_mask],
            )
        )
        transformed = np.einsum(
            "nij,nj->ni",
            transforms[:, :2, :],
            valid_targets - transforms[:, 2, :],
        )
        barycentric_weights = np.column_stack(
            (transformed, 1.0 - np.sum(transformed, axis=1))
        )
    else:
        tree = cKDTree(source_points)
        _, nearest = tree.query(target_points)
        nearest_source_indices = np.full(Xi_arr.size, -1, dtype=np.int64)
        nearest_source_indices[target_finite.ravel()] = nearest
        valid_target_mask = target_finite.ravel()

    return FixedGridInterpolationPlan(
        method=method_value,
        duplicate_decimals=decimals,
        expected_raw_point_count=int(x_arr.size),
        finite_coordinate_mask=_readonly(finite_coordinate_mask, np.bool_),
        expected_rounded_source_keys=_readonly(
            rounded_source_keys, np.float64
        ),
        duplicate_group_inverse_indices=_readonly(inverse, np.int64),
        duplicate_group_counts=_readonly(counts, np.int64),
        deduplicated_x=_readonly(deduplicated_x, np.float64),
        deduplicated_z=_readonly(deduplicated_z, np.float64),
        target_shape=tuple(int(value) for value in Xi_arr.shape),
        valid_target_mask=_readonly(
            valid_target_mask.reshape(Xi_arr.shape), np.bool_
        ),
        target_simplex_indices=(
            None
            if target_simplex_indices is None
            else _readonly(target_simplex_indices, np.int64)
        ),
        source_vertex_indices=(
            None
            if source_vertex_indices is None
            else _readonly(source_vertex_indices, np.int64)
        ),
        barycentric_weights=(
            None
            if barycentric_weights is None
            else _readonly(barycentric_weights, np.float64)
        ),
        nearest_source_indices=(
            None
            if nearest_source_indices is None
            else _readonly(nearest_source_indices, np.int64)
        ),
    )


def _format_key(keys: np.ndarray, finite: bool, position: int) -> str:
    if not finite:
        return "<non-finite>"
    return f"({keys[position, 0]:.16g}, {keys[position, 1]:.16g})"


def validate_fixed_grid_source_geometry(
    plan: FixedGridInterpolationPlan,
    x: object,
    z: object,
) -> None:
    """Verify that source positions can reuse a plan's duplicate mapping."""
    x_arr, z_arr = _source_arrays(x, z)
    actual_count = int(x_arr.size)
    if actual_count != plan.expected_raw_point_count:
        raise FixedGridGeometryMismatchError(
            "Fixed-grid source geometry point-count mismatch: "
            f"expected {plan.expected_raw_point_count}, got {actual_count}."
        )

    actual_finite = np.isfinite(x_arr) & np.isfinite(z_arr)
    actual_keys = np.full((actual_count, 2), np.nan, dtype=np.float64)
    actual_keys[actual_finite] = np.column_stack(
        (
            np.round(x_arr[actual_finite], plan.duplicate_decimals),
            np.round(z_arr[actual_finite], plan.duplicate_decimals),
        )
    )
    pattern_mismatch = actual_finite != plan.finite_coordinate_mask
    coordinate_mismatch = np.zeros(actual_count, dtype=bool)
    common_finite = actual_finite & plan.finite_coordinate_mask
    coordinate_mismatch[common_finite] = np.any(
        actual_keys[common_finite]
        != plan.expected_rounded_source_keys[common_finite],
        axis=1,
    )
    mismatch = np.flatnonzero(pattern_mismatch | coordinate_mismatch)
    if mismatch.size:
        position = int(mismatch[0])
        expected = _format_key(
            plan.expected_rounded_source_keys,
            bool(plan.finite_coordinate_mask[position]),
            position,
        )
        actual = _format_key(
            actual_keys,
            bool(actual_finite[position]),
            position,
        )
        raise FixedGridGeometryMismatchError(
            "Fixed-grid source geometry mismatch at source position "
            f"{position}: expected rounded key {expected}, got {actual}."
        )


def apply_fixed_grid_interpolation_plan(
    plan: FixedGridInterpolationPlan,
    x: object,
    z: object,
    values: object,
) -> NDArray[np.float64]:
    """Apply precomputed geometry to one frame of source values."""
    validate_fixed_grid_source_geometry(plan, x, z)
    values_arr = np.asarray(values, dtype=np.float64).ravel()
    if values_arr.size != plan.expected_raw_point_count:
        raise ValueError(
            "values point-count mismatch: "
            f"expected {plan.expected_raw_point_count}, got {values_arr.size}."
        )

    finite_source_values = values_arr[plan.finite_coordinate_mask]
    finite_value_mask = np.isfinite(finite_source_values)
    group_count = int(plan.duplicate_group_counts.size)
    value_counts = np.bincount(
        plan.duplicate_group_inverse_indices[finite_value_mask],
        minlength=group_count,
    )
    value_sums = np.bincount(
        plan.duplicate_group_inverse_indices[finite_value_mask],
        weights=finite_source_values[finite_value_mask],
        minlength=group_count,
    )
    deduplicated_values = np.full(group_count, np.nan, dtype=np.float64)
    groups_with_values = value_counts > 0
    deduplicated_values[groups_with_values] = (
        value_sums[groups_with_values] / value_counts[groups_with_values]
    )

    result = np.full(
        int(np.prod(plan.target_shape)),
        np.nan,
        dtype=np.float64,
    )
    valid_target_flat = plan.valid_target_mask.ravel()
    if plan.method == "linear":
        if plan.source_vertex_indices is None or plan.barycentric_weights is None:
            raise ValueError("Linear interpolation plan is missing weight arrays.")
        vertex_values = deduplicated_values[plan.source_vertex_indices]
        result[valid_target_flat] = np.sum(
            vertex_values * plan.barycentric_weights,
            axis=1,
        )
    else:
        if plan.nearest_source_indices is None:
            raise ValueError("Nearest interpolation plan is missing source indices.")
        result[valid_target_flat] = deduplicated_values[
            plan.nearest_source_indices[valid_target_flat]
        ]
    return result.reshape(plan.target_shape)
