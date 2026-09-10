"""Legendre--Gauss--Lobatto nodes, quadrature, and interpolation utilities."""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, roots_jacobi


def _readonly(array: object) -> NDArray[np.float64]:
    result = np.asarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _cached_gll_nodes(node_count: int) -> NDArray[np.float64]:
    if node_count == 2:
        return _readonly((-1.0, 1.0))
    interior, _ = roots_jacobi(node_count - 2, 1.0, 1.0)
    nodes = np.empty(node_count, dtype=np.float64)
    nodes[0] = -1.0
    nodes[-1] = 1.0
    nodes[1:-1] = interior
    return _readonly(nodes)


def gll_nodes(node_count: int) -> NDArray[np.float64]:
    """Return exactly ``node_count`` Legendre--Gauss--Lobatto nodes.

    The returned float64 array is deterministic and read-only.  A fresh view is
    returned so callers cannot make the cached base array writable.
    """
    if not isinstance(node_count, Integral) or isinstance(
        node_count, (bool, np.bool_)
    ):
        raise ValueError("node_count must be an integer greater than or equal to 2.")
    count = int(node_count)
    if count < 2:
        raise ValueError("node_count must be greater than or equal to 2.")
    result = _cached_gll_nodes(count).view()
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _cached_gll_quadrature_weights(
    node_count: int,
) -> NDArray[np.float64]:
    nodes = gll_nodes(node_count)
    polynomial_order = node_count - 1
    legendre_values = eval_legendre(polynomial_order, nodes)
    denominator = (
        polynomial_order
        * (polynomial_order + 1)
        * legendre_values**2
    )
    return _readonly(2.0 / denominator)


def gll_quadrature_weights(node_count: int) -> NDArray[np.float64]:
    """Return quadrature weights for exactly ``node_count`` GLL nodes.

    ``node_count`` is the number of nodes, not the polynomial order. Thus a
    Nek polynomial order of 7 uses ``node_count=8``. These integration weights
    are distinct from the barycentric weights used for interpolation.

    The returned float64 array is deterministic and read-only. A fresh view is
    returned so callers cannot make the cached base array writable.
    """
    if not isinstance(node_count, Integral) or isinstance(
        node_count, (bool, np.bool_)
    ):
        raise ValueError("node_count must be an integer greater than or equal to 2.")
    count = int(node_count)
    if count < 2:
        raise ValueError("node_count must be greater than or equal to 2.")
    result = _cached_gll_quadrature_weights(count).view()
    result.setflags(write=False)
    return result


def barycentric_weights(nodes: object) -> NDArray[np.float64]:
    """Return first-form barycentric weights for distinct interpolation nodes."""
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    if nodes_arr.ndim != 1 or nodes_arr.size < 2:
        raise ValueError("nodes must be a one-dimensional array with at least 2 values.")
    if not np.all(np.isfinite(nodes_arr)):
        raise ValueError("nodes must contain only finite values.")
    differences = nodes_arr[:, None] - nodes_arr[None, :]
    np.fill_diagonal(differences, 1.0)
    if np.any(differences == 0.0):
        raise ValueError("nodes must contain distinct values.")
    weights = 1.0 / np.prod(differences, axis=1, dtype=np.float64)
    weights /= np.max(np.abs(weights))
    return _readonly(weights)


def barycentric_basis_and_derivative(
    nodes: object,
    weights: object,
    q: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate barycentric Lagrange basis values and derivatives at ``q``."""
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if (
        nodes_arr.ndim != 1
        or nodes_arr.size < 2
        or weights_arr.shape != nodes_arr.shape
    ):
        raise ValueError("nodes and weights must be matching one-dimensional arrays.")
    if not np.all(np.isfinite(nodes_arr)) or not np.all(np.isfinite(weights_arr)):
        raise ValueError("nodes and weights must contain only finite values.")
    q_value = float(q)
    if not np.isfinite(q_value):
        raise ValueError("q must be finite.")

    exact = np.flatnonzero(q_value == nodes_arr)
    if exact.size:
        node_index = int(exact[0])
        basis = np.zeros(nodes_arr.size, dtype=np.float64)
        basis[node_index] = 1.0
        derivative = np.empty(nodes_arr.size, dtype=np.float64)
        other = np.arange(nodes_arr.size) != node_index
        derivative[other] = weights_arr[other] / (
            weights_arr[node_index]
            * (nodes_arr[node_index] - nodes_arr[other])
        )
        derivative[node_index] = -np.sum(derivative[other])
        return _readonly(basis), _readonly(derivative)

    difference = q_value - nodes_arr
    terms = weights_arr / difference
    denominator = np.sum(terms)
    if not np.isfinite(denominator) or denominator == 0.0:
        raise FloatingPointError(
            "Barycentric basis denominator is zero or non-finite."
        )
    basis = terms / denominator
    reciprocal_sum = np.sum(weights_arr / difference**2) / denominator
    derivative = basis * (reciprocal_sum - 1.0 / difference)
    # Preserve the derivative of the partition of unity in floating point.
    derivative[int(np.argmax(np.abs(basis)))] -= np.sum(derivative)
    return _readonly(basis), _readonly(derivative)


@lru_cache(maxsize=None)
def _cached_gll_interpolation_matrix(
    source_node_count: int,
    target_node_count: int,
) -> NDArray[np.float64]:
    source_nodes = gll_nodes(source_node_count)
    target_nodes = gll_nodes(target_node_count)
    weights = barycentric_weights(source_nodes)
    return _readonly(
        np.vstack(
            [
                barycentric_basis_and_derivative(source_nodes, weights, q)[0]
                for q in target_nodes
            ]
        )
    )


def gll_interpolation_matrix(
    source_node_count: int,
    target_node_count: int,
) -> NDArray[np.float64]:
    """Evaluate source GLL Lagrange basis functions at target GLL nodes.

    The shape is ``(target_node_count, source_node_count)``; each row contains
    the source basis evaluated at one target node. Both counts must be integers
    >= 2. Here ``polynomial_order = node_count - 1``: 8 -> 10 nodes resamples
    a P7 polynomial on the order-9 GLL nodal set, adding no solution information.

    The cached float64 matrix is deterministic and read-only. A fresh view is
    returned so callers cannot make the cached base array writable.
    """
    for name, count in (
        ("source_node_count", source_node_count),
        ("target_node_count", target_node_count),
    ):
        try:
            gll_nodes(count)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer greater than or equal to 2."
            ) from exc
    result = _cached_gll_interpolation_matrix(
        int(source_node_count), int(target_node_count)
    ).view()
    result.setflags(write=False)
    return result
