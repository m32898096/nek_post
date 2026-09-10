from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.gll_refinement import (
    RefinedGLLElement,
    refine_gll_tensor3,
    refine_nek_element_gll,
    refine_nek_elements_gll,
)


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


def _analytic_element_arrays(
    node_count: int,
    *,
    x_origin: float = 0.0,
    reference_axis: int = 0,
    degree: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = np.meshgrid(*(gll_nodes(node_count),) * 3, indexing="ij")
    x = x_origin + 0.5 * (reference[reference_axis] + 1.0)
    y = 0.2 + 0.7 * reference[(reference_axis + 1) % 3]
    z = -0.3 + 0.4 * reference[(reference_axis + 2) % 3]
    concentration = (
        (1.0 + 0.3 * (x / 2.0) ** degree)
        * (0.5 - 0.7 * y**degree)
        * (1.2 + 0.2 * z**degree)
        + 0.4 * x * y - 0.6 * y * z + 0.8 * x * z
    )
    return x, y, z, concentration


def _synthetic_element(
    node_count: int = 8,
    *,
    x_origin: float = 0.0,
    reference_axis: int = 0,
    degree: int = 7,
) -> SimpleNamespace:
    x, y, z, concentration = _analytic_element_arrays(
        node_count, x_origin=x_origin, reference_axis=reference_axis, degree=degree
    )
    return SimpleNamespace(pos=[x, y, z], temp=[concentration], scal=None)


def _refined_arrays(element: RefinedGLLElement) -> tuple[np.ndarray, ...]:
    return element.x, element.y, element.z, element.concentration


@pytest.mark.parametrize("reference_axis", (0, 1, 2))
def test_refine_nek_element_affine_coordinates_concentration_and_metadata(
    reference_axis: int,
) -> None:
    element = _synthetic_element(reference_axis=reference_axis)
    source_arrays = (*element.pos, element.temp[0])
    original = tuple(array.copy() for array in source_arrays)

    refined = refine_nek_element_gll(element, target_node_count=10)
    repeated = refine_nek_element_gll(element, target_node_count=10)
    expected = _analytic_element_arrays(10, reference_axis=reference_axis)

    assert isinstance(refined, RefinedGLLElement)
    assert refined.source_node_count == 8
    assert refined.source_polynomial_order == 7
    assert refined.target_node_count == 10
    assert refined.target_nodal_order == 9
    for actual, analytic, repeated_array, source, before in zip(
        _refined_arrays(refined), expected, _refined_arrays(repeated),
        source_arrays, original, strict=True,
    ):
        assert actual.shape == (10, 10, 10)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, analytic, rtol=0.0, atol=2.0e-15)
        np.testing.assert_array_equal(actual, repeated_array)
        np.testing.assert_array_equal(source, before)
        assert not np.shares_memory(actual, source)
        source[:] = 99.0
        np.testing.assert_array_equal(actual, repeated_array)


def test_refined_nek_element_arrays_and_metadata_are_immutable() -> None:
    refined = refine_nek_element_gll(_synthetic_element(), target_node_count=10)

    with pytest.raises(FrozenInstanceError):
        refined.source_node_count = 10  # type: ignore[misc]
    for array in _refined_arrays(refined):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0, 0, 0] = 0.0
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("source_node_count", (2, 4, 11))
def test_refine_nek_element_supports_other_isotropic_source_node_counts(
    source_node_count: int,
) -> None:
    degree = min(source_node_count - 1, 7)
    element = _synthetic_element(source_node_count, degree=degree)
    refined = refine_nek_element_gll(element, target_node_count=np.int64(10))

    assert refined.source_node_count == source_node_count
    assert refined.source_polynomial_order == source_node_count - 1
    assert type(refined.target_node_count) is int
    for actual, expected in zip(
        _refined_arrays(refined), _analytic_element_arrays(10, degree=degree), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_refine_nek_element_uses_concentration_scalar_fallback_and_float64() -> None:
    element = _synthetic_element()
    element.temp = None
    element.scal = [np.full((8, 8, 8), 0.5, dtype=np.float32)]
    element.pos = [array.astype(np.float32) for array in element.pos]

    refined = refine_nek_element_gll(element, target_node_count=10)

    assert all(array.dtype == np.float64 for array in _refined_arrays(refined))
    np.testing.assert_allclose(refined.concentration, 0.5, rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize("reference_axis", (0, 1, 2))
def test_independently_refined_adjacent_elements_agree_on_shared_face(
    reference_axis: int,
) -> None:
    left = refine_nek_element_gll(
        _synthetic_element(x_origin=0.0, reference_axis=reference_axis),
        target_node_count=10,
    )
    right = refine_nek_element_gll(
        _synthetic_element(x_origin=1.0, reference_axis=reference_axis),
        target_node_count=10,
    )

    for left_array, right_array in zip(
        _refined_arrays(left), _refined_arrays(right), strict=True
    ):
        left_face = np.take(left_array, -1, axis=reference_axis)
        right_face = np.take(right_array, 0, axis=reference_axis)
        assert left_face.shape == (10, 10)
        np.testing.assert_allclose(left_face, right_face, rtol=0.0, atol=2.0e-15)


def test_refine_nek_snapshot_preserves_element_order_and_individual_source_metadata() -> None:
    elements = [
        _synthetic_element(8, x_origin=2.0),
        _synthetic_element(4, x_origin=0.0, degree=3),
        _synthetic_element(8, x_origin=1.0),
    ]
    refined = refine_nek_elements_gll(
        SimpleNamespace(elem=iter(elements)), target_node_count=10
    )

    assert isinstance(refined, tuple)
    assert len(refined) == 3
    assert [element.source_node_count for element in refined] == [8, 4, 8]
    assert [element.x[0, 0, 0] for element in refined] == [2.0, 0.0, 1.0]
    for source, actual in zip(elements, refined, strict=True):
        expected = refine_nek_element_gll(source, target_node_count=10)
        for actual_array, expected_array in zip(
            _refined_arrays(actual), _refined_arrays(expected), strict=True
        ):
            np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.parametrize("field_index", range(4))
@pytest.mark.parametrize("shape", ((8, 8), (8, 8, 7)))
def test_refine_nek_element_rejects_invalid_or_mismatched_field_shapes(
    field_index: int,
    shape: tuple[int, ...],
) -> None:
    element = _synthetic_element()
    if field_index < 3:
        element.pos[field_index] = np.zeros(shape)
    else:
        element.temp[0] = np.zeros(shape)
    with pytest.raises(ValueError, match="three-dimensional|shape must match"):
        refine_nek_element_gll(element, target_node_count=10)


@pytest.mark.parametrize(
    ("shape", "message"),
    (((8, 7, 8), "isotropic"), ((1, 1, 1), "at least 2"), ((0, 0, 0), "at least 2")),
)
def test_refine_nek_element_rejects_invalid_source_shapes(
    shape: tuple[int, int, int], message: str,
) -> None:
    element = SimpleNamespace(pos=np.zeros((3, *shape)), temp=np.zeros((1, *shape)))
    with pytest.raises(ValueError, match=message):
        refine_nek_element_gll(element, target_node_count=10)


@pytest.mark.parametrize("field_index", range(4))
@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf, 1.0j, "bad"))
def test_refine_nek_element_rejects_nonfinite_complex_or_nonnumeric_fields(
    field_index: int, bad_value: object,
) -> None:
    element = _synthetic_element()
    invalid = np.full((8, 8, 8), bad_value)
    if field_index < 3:
        element.pos[field_index] = invalid
    else:
        element.temp[0] = invalid
    name = ("x", "y", "z", "concentration")[field_index]
    with pytest.raises(ValueError, match=f"{name} must .*finite|{name} must be a real"):
        refine_nek_element_gll(element, target_node_count=10)


@pytest.mark.parametrize("missing", ("coordinates", "concentration"))
def test_refine_nek_element_reports_missing_fields(missing: str) -> None:
    element = _synthetic_element()
    if missing == "coordinates":
        del element.pos
    else:
        element.temp = None
    with pytest.raises(ValueError, match=f"provide .*{missing}"):
        refine_nek_element_gll(element, target_node_count=10)


@pytest.mark.parametrize("target_node_count", (1, 0, -2, 10.0, True, None))
@pytest.mark.parametrize("snapshot", (False, True))
def test_nek_refinement_adapters_reject_invalid_target_node_counts(
    target_node_count: object, snapshot: bool,
) -> None:
    element = _synthetic_element()
    function = refine_nek_elements_gll if snapshot else refine_nek_element_gll
    source = SimpleNamespace(elem=[element]) if snapshot else element
    with pytest.raises(ValueError, match="target_node_count"):
        function(source, target_node_count=target_node_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("data", (object(), SimpleNamespace(elem=None), SimpleNamespace(elem=3)))
def test_refine_nek_snapshot_rejects_missing_or_noniterable_elements(data: object) -> None:
    with pytest.raises(ValueError, match="iterable 'elem'"):
        refine_nek_elements_gll(data, target_node_count=10)


def test_refine_nek_snapshot_rejects_empty_elements() -> None:
    with pytest.raises(ValueError, match="no spectral elements"):
        refine_nek_elements_gll(SimpleNamespace(elem=[]), target_node_count=10)


def test_refine_nek_snapshot_identifies_malformed_element_index() -> None:
    data = SimpleNamespace(elem=[_synthetic_element(), object()])
    with pytest.raises(ValueError, match="Element 1: .*coordinates"):
        refine_nek_elements_gll(data, target_node_count=10)
