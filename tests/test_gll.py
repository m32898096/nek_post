from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_interpolation_matrix,
    gll_nodes,
    gll_quadrature_weights,
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


@pytest.mark.parametrize("node_count", range(2, 17))
def test_gll_quadrature_weights_are_positive_symmetric_normalized_and_read_only(
    node_count: int,
) -> None:
    weights = gll_quadrature_weights(node_count)

    assert weights.shape == (node_count,)
    assert weights.dtype == np.float64
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights, weights[::-1], rtol=0.0, atol=3.0e-15)
    assert np.sum(weights) == pytest.approx(2.0, abs=3.0e-15)
    assert not weights.flags.writeable


def test_eight_node_quadrature_weights_match_matlab_reference() -> None:
    expected = np.asarray(
        [
            0.0357142857142857,
            0.210704227143506,
            0.341122692483504,
            0.412458794658704,
            0.412458794658704,
            0.341122692483504,
            0.210704227143506,
            0.0357142857142857,
        ]
    )

    np.testing.assert_allclose(
        gll_quadrature_weights(8),
        expected,
        rtol=2.0e-14,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("node_count", range(2, 11))
def test_gll_quadrature_integrates_monomials_through_degree_two_n_minus_three(
    node_count: int,
) -> None:
    nodes = gll_nodes(node_count)
    weights = gll_quadrature_weights(node_count)

    for degree in range(2 * node_count - 2):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        actual = float(weights @ nodes**degree)
        assert actual == pytest.approx(expected, rel=3.0e-13, abs=3.0e-14)


@pytest.mark.parametrize("node_count", (1, 0, -2, 2.5, True))
def test_gll_quadrature_weights_reject_invalid_node_counts(
    node_count: object,
) -> None:
    with pytest.raises(ValueError, match="node_count"):
        gll_quadrature_weights(node_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_node_count", "target_node_count"),
    ((8, 10), (2, 10), (5, 9), (11, 13), (10, 8)),
)
def test_gll_interpolation_matrix_shape_partition_of_unity_and_constant_exactness(
    source_node_count: int,
    target_node_count: int,
) -> None:
    matrix = gll_interpolation_matrix(source_node_count, target_node_count)

    assert matrix.shape == (target_node_count, source_node_count)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        matrix @ np.full(source_node_count, 2.5), 2.5, rtol=0.0, atol=3.0e-15
    )
    np.testing.assert_array_equal(matrix[0], np.eye(source_node_count)[0])
    np.testing.assert_array_equal(matrix[-1], np.eye(source_node_count)[-1])


@pytest.mark.parametrize("degree", range(8))
def test_gll_interpolation_matrix_reproduces_polynomials_through_degree_seven(
    degree: int,
) -> None:
    polynomial = Polynomial([(-0.7) ** power for power in range(degree + 1)])
    actual = gll_interpolation_matrix(8, 10) @ polynomial(gll_nodes(8))

    np.testing.assert_allclose(
        actual, polynomial(gll_nodes(10)), rtol=0.0, atol=3.0e-15
    )


@pytest.mark.parametrize("node_count", (2, 5, 8, 10))
def test_gll_interpolation_matrix_same_node_set_is_identity(node_count: int) -> None:
    np.testing.assert_array_equal(
        gll_interpolation_matrix(node_count, node_count), np.eye(node_count)
    )


def test_gll_interpolation_matrix_cache_is_read_only_and_deterministic() -> None:
    first = gll_interpolation_matrix(8, 10)
    second = gll_interpolation_matrix(np.int64(8), np.int64(10))

    assert first is not second
    assert np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        first[0, 0] = 2.0
    with pytest.raises(ValueError):
        first.setflags(write=True)


@pytest.mark.parametrize("name", ("source_node_count", "target_node_count"))
@pytest.mark.parametrize(
    "invalid_count", (0, 1, -2, 2.5, 8.0, True, np.bool_(False), "8", None, [8])
)
def test_gll_interpolation_matrix_rejects_invalid_node_counts(
    name: str,
    invalid_count: object,
) -> None:
    counts = {"source_node_count": 8, "target_node_count": 10}
    counts[name] = invalid_count  # type: ignore[assignment]
    with pytest.raises(ValueError, match=name):
        gll_interpolation_matrix(**counts)
