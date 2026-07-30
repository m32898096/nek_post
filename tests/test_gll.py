from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_nodes,
)


@pytest.mark.parametrize("node_count", range(2, 17))
def test_gll_nodes_have_requested_count_endpoints_symmetry_and_float64(
    node_count: int,
) -> None:
    nodes = gll_nodes(node_count)

    assert nodes.shape == (node_count,)
    assert nodes.dtype == np.float64
    assert nodes[0] == -1.0
    assert nodes[-1] == 1.0
    np.testing.assert_allclose(nodes, -nodes[::-1], rtol=0.0, atol=2.0e-15)
    assert not nodes.flags.writeable


def test_n7_means_eight_gll_nodes() -> None:
    assert gll_nodes(8).size == 8


@pytest.mark.parametrize("node_count", (1, 0, -2, 2.5, True))
def test_gll_nodes_reject_invalid_counts(node_count: object) -> None:
    with pytest.raises(ValueError, match="node_count"):
        gll_nodes(node_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("node_count", (2, 5, 8, 16))
def test_barycentric_basis_has_kronecker_property_and_partition_of_unity(
    node_count: int,
) -> None:
    nodes = gll_nodes(node_count)
    weights = barycentric_weights(nodes)

    for index, q in enumerate(nodes):
        basis, derivative = barycentric_basis_and_derivative(nodes, weights, q)
        expected = np.zeros(node_count)
        expected[index] = 1.0
        np.testing.assert_array_equal(basis, expected)
        assert np.sum(basis) == 1.0
        assert np.sum(derivative) == pytest.approx(0.0, abs=3.0e-13)

    for q in np.linspace(-0.97, 0.93, 13):
        basis, derivative = barycentric_basis_and_derivative(nodes, weights, q)
        assert np.sum(basis) == pytest.approx(1.0, abs=4.0e-15)
        assert np.sum(derivative) == pytest.approx(0.0, abs=4.0e-13)


@pytest.mark.parametrize("node_count", (2, 4, 8, 16))
def test_derivative_matrix_differentiates_supported_polynomials(
    node_count: int,
) -> None:
    nodes = gll_nodes(node_count)
    weights = barycentric_weights(nodes)
    derivative_matrix = np.vstack(
        [
            barycentric_basis_and_derivative(nodes, weights, q)[1]
            for q in nodes
        ]
    )

    for degree in range(node_count):
        values = nodes**degree
        expected = (
            np.zeros_like(nodes)
            if degree == 0
            else degree * nodes ** (degree - 1)
        )
        np.testing.assert_allclose(
            derivative_matrix @ values,
            expected,
            rtol=2.0e-11,
            atol=2.0e-11,
        )


@pytest.mark.parametrize("node_count", (3, 6, 8, 12))
def test_barycentric_interpolation_reproduces_polynomials_through_degree_n(
    node_count: int,
) -> None:
    nodes = gll_nodes(node_count)
    weights = barycentric_weights(nodes)

    for degree in range(node_count):
        polynomial = Polynomial([(-0.3) ** power for power in range(degree + 1)])
        nodal_values = polynomial(nodes)
        for q in np.linspace(-1.0, 1.0, 21):
            basis, _ = barycentric_basis_and_derivative(nodes, weights, q)
            assert basis @ nodal_values == pytest.approx(
                polynomial(q), rel=3.0e-13, abs=3.0e-13
            )


def test_barycentric_weights_reject_duplicate_nodes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        barycentric_weights([-1.0, 0.0, 0.0, 1.0])
