"""Element-local GLL derivatives in native pymech storage order.

Scalar arrays have shape ``(nt, ns, nr)`` (pymech's ``lz, ly, lx``).
Reference coordinates ``(r, s, t)`` follow NumPy axes ``(2, 1, 0)`` and
ascending GLL nodes. These reference directions need not align with physical
``(x, y, z)``. The nodal geometry determines that transformation.

``J[..., i, j] = d(x_i)/d(reference_j)`` uses physical rows and reference
columns in ``(r, s, t)`` order. For row gradients, ``grad_x = grad_ref @ J^-1``.
This module solves the equivalent transposed linear system rather than forming
an inverse. Unlike spectral_interpolation's private pointwise Jacobian, which
uses array-axis columns ``(q0, q1, q2)``, these columns are reversed to match
Nek's ``(r, s, t)`` convention. All operations remain local to one element.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_coordinates, get_velocity
from nek_post.gll import gll_differentiation_matrix


DEFAULT_SINGULAR_RTOL = 1.0e-12
REFERENCE_ARRAY_AXES = (2, 1, 0)


def _real_array(values: object, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _scalar_array(values: object, name: str) -> NDArray[np.float64]:
    array = _real_array(values, name)
    if array.ndim != 3 or any(n < 2 for n in array.shape):
        raise ValueError(f"{name} must be a three-dimensional GLL array with >= 2 nodes per axis.")
    return array


def _components(values: object, name: str) -> tuple[NDArray[np.float64], ...]:
    try:
        components = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly three component arrays.") from exc
    if len(components) != 3:
        raise ValueError(f"{name} must contain exactly three component arrays.")
    arrays = tuple(_scalar_array(v, f"{name}[{i}]") for i, v in enumerate(components))
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"{name} component shapes must match.")
    return arrays


def reference_derivatives(values: object) -> NDArray[np.float64]:
    """Return ``(df/dr, df/ds, df/dt)`` as ``values.shape + (3,)``.

    Independent one-dimensional GLL operators contract axes 2, 1, and 0.
    Non-cubic element shapes are supported; no full 3-D operator is assembled.
    Inputs are preserved and calculations/results use float64.
    """
    array = _scalar_array(values, "values")
    derivatives = []
    with np.errstate(over="ignore", invalid="ignore"):
        for axis in REFERENCE_ARRAY_AXES:
            matrix = gll_differentiation_matrix(array.shape[axis])
            derivative = np.tensordot(matrix, array, axes=(1, axis))
            derivatives.append(np.moveaxis(derivative, 0, axis))
    result = np.stack(derivatives, axis=-1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Reference derivatives contain non-finite values.")
    return result


def _validated_jacobian(
    jacobian: object, singular_rtol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(singular_rtol, (bool, np.bool_)):
        raise ValueError("singular_rtol must be finite and in [0, 1).")
    try:
        tolerance = float(singular_rtol)
    except (TypeError, ValueError) as exc:
        raise ValueError("singular_rtol must be finite and in [0, 1).") from exc
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("singular_rtol must be finite and in [0, 1).")
    matrix = _real_array(jacobian, "jacobian")
    if matrix.ndim != 5 or matrix.shape[-2:] != (3, 3) or any(n < 2 for n in matrix.shape[:3]):
        raise ValueError("jacobian must have shape (nt, ns, nr, 3, 3), with >= 2 nodes per axis.")
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            determinant = np.linalg.det(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Jacobian factorization failed.") from exc
    invalid = (
        ~np.all(np.isfinite(singular_values), axis=-1)
        | (singular_values[..., -1] <= tolerance * singular_values[..., 0])
        | ~np.isfinite(determinant)
        | (determinant == 0.0)
    )
    if np.any(invalid):
        index = tuple(int(i) for i in np.argwhere(invalid)[0])
        raise ValueError(f"Singular, near-singular or invalid Jacobian at array index {index}.")
    if np.any(determinant > 0.0) and np.any(determinant < 0.0):
        raise ValueError("Jacobian determinant changes sign across GLL nodes (folded element).")
    return matrix, determinant


def element_jacobian(
    coordinates: object, *, singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(J, detJ)`` for component arrays ``(x, y, z)``.

    Shapes are ``(nt, ns, nr, 3, 3)`` and ``(nt, ns, nr)``. Columns of J
    correspond to ``(r, s, t)``. A consistently negative orientation is allowed
    and its determinant is preserved, never replaced by its absolute value.
    Singular values with ``sigma_min / sigma_max <= singular_rtol`` are
    rejected, independently of coordinate units, as are nodal sign changes.
    Validation is at GLL nodes; it does not certify injectivity between nodes.
    """
    arrays = _components(coordinates, "coordinates")
    jacobian = np.stack([reference_derivatives(a) for a in arrays], axis=-2)
    return _validated_jacobian(jacobian, singular_rtol)


def _solve_gradient(
    derivatives: NDArray[np.float64], jacobian: NDArray[np.float64],
) -> NDArray[np.float64]:
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = np.linalg.solve(
                jacobian.swapaxes(-1, -2), derivatives[..., None]
            )[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("Physical gradient transformation failed: singular Jacobian.") from exc
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Physical gradient contains non-finite values.")
    return result


def transform_reference_gradient(
    derivatives: object, jacobian: object, *,
    singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> NDArray[np.float64]:
    """Transform ``(..., 3)`` nodal row gradients using a reusable Jacobian.

    The three grid axes must match J. Reference components are ``(r, s, t)``;
    returned components are Cartesian ``(x, y, z)``. Neither input is changed.
    """
    matrix, _ = _validated_jacobian(jacobian, singular_rtol)
    reference = _real_array(derivatives, "derivatives")
    if reference.shape != matrix.shape[:-2] + (3,):
        raise ValueError("Reference derivative shape must match jacobian grid shape + (3,).")
    return _solve_gradient(reference, matrix)


def physical_gradient(
    values: object, coordinates: object, *,
    singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> NDArray[np.float64]:
    """Return ``(df/dx, df/dy, df/dz)`` in the original GLL grid layout."""
    jacobian, _ = element_jacobian(coordinates, singular_rtol=singular_rtol)
    derivatives = reference_derivatives(values)
    if derivatives.shape[:-1] != jacobian.shape[:-2]:
        raise ValueError("Scalar field shape must match coordinate shape.")
    return _solve_gradient(derivatives, jacobian)


def velocity_gradient_tensor(
    coordinates: object, velocity: object, *,
    singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> NDArray[np.float64]:
    """Return ``A[..., i, j] = d(velocity_i)/d(x_j)`` for ``(u, v, w)``.

    Coordinate and velocity inputs are three matching element component arrays
    (including pymech pos/vel arrays). Output shape is ``(nt, ns, nr, 3, 3)``;
    rows are ``(u, v, w)``, columns are physical ``(x, y, z)``. Geometry is
    calculated once, and all three gradients are solved together. No global
    averaging, projection, interpolation, or file access is performed.
    """
    jacobian, _ = element_jacobian(coordinates, singular_rtol=singular_rtol)
    components = _components(velocity, "velocity")
    if components[0].shape != jacobian.shape[:-2]:
        raise ValueError("Velocity field shape must match coordinate shape.")
    reference = np.stack([reference_derivatives(a) for a in components], axis=-2)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = np.linalg.solve(
                jacobian.swapaxes(-1, -2), reference.swapaxes(-1, -2)
            ).swapaxes(-1, -2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Velocity gradient transformation failed: singular Jacobian.") from exc
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Velocity gradient contains non-finite values.")
    return result


def element_velocity_gradient(
    element: Any, *, singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> NDArray[np.float64]:
    """Adapter for one reader element, using existing coordinate/velocity accessors."""
    return velocity_gradient_tensor(
        get_coordinates(element), get_velocity(element), singular_rtol=singular_rtol
    )
