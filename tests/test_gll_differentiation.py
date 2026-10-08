"""Analytic polynomial and SEM identities for the public nodal operator."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from nek_post.gll import (
    gll_differentiation_matrix,
    gll_nodes,
    gll_quadrature_weights,
)


@pytest.mark.parametrize("node_count", (2, 3, 4, 8, 12, 16, 32))
def test_supported_polynomials_are_differentiated(node_count):
    nodes = gll_nodes(node_count)
    matrix = gll_differentiation_matrix(node_count)
    assert matrix.shape == (node_count, node_count)
    assert matrix.dtype == np.float64
    assert np.all(np.isfinite(matrix))
    for degree in range(node_count):
        expected = np.zeros_like(nodes) if degree == 0 else degree * nodes ** (degree - 1)
        np.testing.assert_allclose(matrix @ nodes**degree, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("node_count", (2, 5, 8, 16))
def test_matrix_matches_legendre_gll_formula_and_sbp_identity(node_count):
    nodes = gll_nodes(node_count)
    degree = node_count - 1
    legendre = eval_legendre(degree, nodes)
    expected = np.zeros((node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                expected[i, j] = legendre[i] / (legendre[j] * (nodes[i] - nodes[j]))
    expected[0, 0] = -degree * (degree + 1) / 4
    expected[-1, -1] = degree * (degree + 1) / 4
    matrix = gll_differentiation_matrix(node_count)
    np.testing.assert_allclose(matrix, expected, atol=2e-12, rtol=2e-12)
    weighted = gll_quadrature_weights(node_count)[:, None] * matrix
    boundary = np.zeros_like(matrix)
    boundary[0, 0], boundary[-1, -1] = -1, 1
    np.testing.assert_allclose(weighted + weighted.T, boundary, atol=2e-13, rtol=0)


def test_operator_cache_is_read_only_and_count_is_not_degree():
    a = gll_differentiation_matrix(np.int64(8))
    b = gll_differentiation_matrix(8)
    assert a.shape == (8, 8)
    assert np.shares_memory(a, b)
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0] = 0
    with pytest.raises(ValueError):
        a.setflags(write=True)


@pytest.mark.parametrize("node_count", (0, 1, -1, 2.5, True, np.bool_(False), "8", None))
def test_invalid_counts_are_rejected(node_count):
    with pytest.raises(ValueError, match="node_count"):
        gll_differentiation_matrix(node_count)
