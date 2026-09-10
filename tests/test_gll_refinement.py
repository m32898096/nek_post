from __future__ import annotations

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.gll_refinement import refine_gll_tensor3


def _polynomial_values(
    shape: tuple[int, int, int],
    degrees: tuple[int, int, int],
) -> np.ndarray:
    q0, q1, q2 = np.meshgrid(
        *(gll_nodes(count) for count in shape), indexing="ij"
    )
    return (
        (1.0 + 0.3 * q0 ** degrees[0])
        * (0.5 - 0.7 * q1 ** degrees[1])
        * (1.2 + 0.2 * q2 ** degrees[2])
        + 0.4 * q0 * q1 - 0.6 * q1 * q2 + 0.8 * q0 * q2
    )


@pytest.mark.parametrize(
    ("source_shape", "target_node_count"),
    (((8, 8, 8), 10), ((2, 3, 4), 7), ((3, 5, 9), 10), ((11, 4, 6), 12)),
)
def test_refine_tensor_polynomial_exactness_preserves_axis_order(
    source_shape: tuple[int, int, int],
    target_node_count: int,
) -> None:
    degrees = tuple(count - 1 for count in source_shape)
    values = _polynomial_values(source_shape, degrees)
    original = values.copy()

    actual = refine_gll_tensor3(values, target_node_count=target_node_count)
    expected = _polynomial_values((target_node_count,) * 3, degrees)

    assert actual.shape == (target_node_count,) * 3
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4.0e-15)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(
        actual, refine_gll_tensor3(values, target_node_count=target_node_count)
    )


@pytest.mark.parametrize("dtype", (np.int64, np.float32, np.float64))
def test_refine_constant_tensor_is_exact_and_returns_float64(dtype: type) -> None:
    actual = refine_gll_tensor3(np.full((8, 8, 8), 3, dtype=dtype), target_node_count=10)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, 3.0, rtol=0.0, atol=5.0e-15)


def test_n7_to_n9_nodes_to_n7_nodes_polynomial_round_trip() -> None:
    values = _polynomial_values((8, 8, 8), (7, 7, 7))
    refined = refine_gll_tensor3(values, target_node_count=10)
    restored = refine_gll_tensor3(refined, target_node_count=8)

    assert restored.shape == (8, 8, 8)
    np.testing.assert_allclose(restored, values, rtol=0.0, atol=5.0e-15)


def test_refine_to_fewer_nodes_evaluates_source_polynomial() -> None:
    values = _polynomial_values((8, 8, 8), (7, 7, 7))

    np.testing.assert_allclose(
        refine_gll_tensor3(values, target_node_count=4),
        _polynomial_values((4, 4, 4), (7, 7, 7)),
        rtol=0.0,
        atol=4.0e-15,
    )


def test_refine_same_node_set_preserves_noncontiguous_read_only_tensor() -> None:
    values = np.arange(8**3, dtype=np.float64).reshape(8, 8, 8).transpose(2, 0, 1)
    values.setflags(write=False)

    actual = refine_gll_tensor3(values, target_node_count=np.int64(8))

    np.testing.assert_array_equal(actual, values)
    assert not np.shares_memory(actual, values)
    assert not values.flags.writeable


@pytest.mark.parametrize("shape", ((), (8,), (8, 8), (2, 8, 8, 8)))
def test_refine_rejects_non_three_dimensional_values(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        refine_gll_tensor3(np.zeros(shape), target_node_count=10)


@pytest.mark.parametrize("shape", ((0, 8, 8), (8, 1, 8), (8, 8, 1)))
def test_refine_rejects_too_few_source_nodes(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="at least 2 GLL nodes per axis"):
        refine_gll_tensor3(np.zeros(shape), target_node_count=10)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_refine_rejects_nonfinite_values(value: float) -> None:
    values = np.ones((8, 8, 8))
    values[1, 2, 3] = value
    with pytest.raises(ValueError, match="finite"):
        refine_gll_tensor3(values, target_node_count=10)


def test_refine_rejects_complex_values_without_discarding_imaginary_part() -> None:
    with pytest.raises(ValueError, match="real"):
        refine_gll_tensor3(np.full((2, 2, 2), 1.0 + 2.0j), target_node_count=10)


@pytest.mark.parametrize(
    "target_node_count", (0, 1, -2, 2.5, 10.0, True, np.bool_(False), "10", None, [10])
)
def test_refine_rejects_invalid_target_node_counts(target_node_count: object) -> None:
    with pytest.raises(ValueError, match="target_node_count"):
        refine_gll_tensor3(
            np.ones((8, 8, 8)),
            target_node_count=target_node_count,  # type: ignore[arg-type]
        )
