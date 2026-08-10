from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from nek_post.cantero_equivalent_height import (
    apply_cantero_equivalent_height_plan,
    build_cantero_equivalent_height_plan,
    cantero_equivalent_height_path,
    compute_cantero_equivalent_height,
    load_cantero_equivalent_height_npz,
    save_cantero_equivalent_height_npz,
)
from nek_post.gll import gll_nodes


DEFAULT_X_EDGES = (-2.0, 0.5, 3.0)
DEFAULT_Y_EDGES = (0.0, 0.5, 1.5)
DEFAULT_Z_EDGES = (0.0, 0.4, 1.0)
DEFAULT_MAPPING = (2, 1, 0)


def _data(
    field_function,
    *,
    x_edges: tuple[float, ...] = DEFAULT_X_EDGES,
    y_edges: tuple[float, ...] = DEFAULT_Y_EDGES,
    z_edges: tuple[float, ...] = DEFAULT_Z_EDGES,
    shape: tuple[int, int, int] = (4, 4, 4),
    mapping: tuple[int, int, int] = DEFAULT_MAPPING,
) -> SimpleNamespace:
    reference = np.meshgrid(
        *(gll_nodes(node_count) for node_count in shape), indexing="ij"
    )
    physical_edges = (x_edges, y_edges, z_edges)
    elements = []
    for iz in range(len(z_edges) - 1):
        for iy in range(len(y_edges) - 1):
            for ix in range(len(x_edges) - 1):
                coordinates = []
                for physical_axis, edges in enumerate(physical_edges):
                    lower = edges[(ix, iy, iz)[physical_axis]]
                    upper = edges[(ix, iy, iz)[physical_axis] + 1]
                    q = reference[mapping[physical_axis]]
                    coordinates.append(
                        0.5 * (lower + upper) + 0.5 * (upper - lower) * q
                    )
                x, y, z = coordinates
                elements.append(
                    SimpleNamespace(
                        pos=tuple(coordinates),
                        temp=(np.asarray(field_function(x, y, z), dtype=np.float64),),
                    )
                )
    return SimpleNamespace(elem=elements)


def test_constant_field_and_composite_weight_normalization() -> None:
    plan, result = compute_cantero_equivalent_height(
        _data(lambda x, _y, _z: np.ones_like(x))
    )

    assert result.local_equivalent_height.shape == (
        result.y_coordinates.size,
        result.x_coordinates.size,
    )
    np.testing.assert_allclose(result.local_equivalent_height, 1.0, atol=3.0e-14)
    np.testing.assert_allclose(result.span_averaged_height, 1.0, atol=3.0e-14)
    assert result.spanwise_length == pytest.approx(1.5)
    assert plan.spanwise_quadrature_weights.shape == result.y_coordinates.shape
    assert result.y_coordinates.size == 7
    assert plan.spanwise_quadrature_weights.sum() == pytest.approx(
        result.spanwise_length,
        abs=3.0e-14,
    )


def test_z_dependent_field_uses_existing_z_gll_integration() -> None:
    _plan, result = compute_cantero_equivalent_height(
        _data(lambda _x, _y, z: z)
    )

    np.testing.assert_allclose(result.local_equivalent_height, 0.5, atol=3.0e-14)
    np.testing.assert_allclose(result.span_averaged_height, 0.5, atol=3.0e-14)


def test_spanwise_average_is_composite_gll_integral_divided_by_geometry_length() -> None:
    plan, result = compute_cantero_equivalent_height(
        _data(lambda _x, y, _z: y)
    )
    expected_local = np.broadcast_to(
        result.y_coordinates[:, None], result.local_equivalent_height.shape
    )

    np.testing.assert_allclose(result.local_equivalent_height, expected_local, atol=3.0e-14)
    np.testing.assert_allclose(result.span_averaged_height, 0.75, atol=3.0e-14)
    direct_average = (
        plan.spanwise_quadrature_weights @ result.local_equivalent_height
    ) / result.spanwise_length
    np.testing.assert_allclose(result.span_averaged_height, direct_average, atol=3.0e-14)


def test_full_nonconstant_analytical_field_preserves_y_x_orientation() -> None:
    _plan, result = compute_cantero_equivalent_height(
        _data(lambda x, y, z: 1.0 + x + 2.0 * y + 3.0 * z)
    )
    x, y = np.meshgrid(result.x_coordinates, result.y_coordinates)

    np.testing.assert_allclose(
        result.local_equivalent_height,
        2.5 + x + 2.0 * y,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        result.span_averaged_height,
        4.0 + result.x_coordinates,
        atol=4.0e-14,
    )


def test_nondefault_spanwise_length_comes_from_geometry() -> None:
    y_edges = (-2.0, -0.5, 2.0)
    _plan, result = compute_cantero_equivalent_height(
        _data(lambda _x, y, _z: y, y_edges=y_edges)
    )

    assert result.spanwise_length == pytest.approx(4.0)
    np.testing.assert_allclose(result.span_averaged_height, 0.0, atol=3.0e-14)


def test_plan_reuse_and_geometry_order_mismatch_rejection() -> None:
    first = _data(lambda x, _y, _z: np.ones_like(x))
    plan = build_cantero_equivalent_height_plan(first)
    second = _data(lambda x, _y, _z: 2.0 * np.ones_like(x))
    reused = apply_cantero_equivalent_height_plan(plan, second)

    np.testing.assert_allclose(reused.local_equivalent_height, 2.0, atol=3.0e-14)
    np.testing.assert_allclose(reused.span_averaged_height, 2.0, atol=3.0e-14)
    mismatched = _data(
        lambda x, _y, _z: np.ones_like(x), shape=(3, 4, 4)
    )
    with pytest.raises(ValueError, match="element shape"):
        apply_cantero_equivalent_height_plan(plan, mismatched)


def test_plan_reuse_rejects_reordered_identical_elements() -> None:
    data = _data(lambda x, _y, _z: np.ones_like(x))
    plan = build_cantero_equivalent_height_plan(data)
    reordered = SimpleNamespace(elem=list(reversed(data.elem)))

    with pytest.raises(
        ValueError,
        match=r"Incompatible geometry: element 0 coordinate signature changed",
    ):
        apply_cantero_equivalent_height_plan(plan, reordered)


def test_alternate_axis_mapping_uses_physical_gll_axes_for_both_integrals() -> None:
    x_edges = (-3.0, -1.0, 2.0)
    y_edges = (-2.0, -0.5, 3.0)
    z_edges = (0.5, 1.2, 2.5)
    mapping = (0, 2, 1)
    shape = (3, 4, 5)
    plan, result = compute_cantero_equivalent_height(
        _data(
            lambda x, y, z: 1.0 + x + 2.0 * y + 3.0 * z,
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            shape=shape,
            mapping=mapping,
        )
    )

    def expected_coordinates(
        edges: tuple[float, ...], node_count: int
    ) -> np.ndarray:
        nodes = gll_nodes(node_count)
        pieces = []
        for cell_index, (lower, upper) in enumerate(
            zip(edges[:-1], edges[1:], strict=True)
        ):
            values = 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes
            pieces.append(values if cell_index == 0 else values[1:])
        return np.concatenate(pieces)

    expected_x = expected_coordinates(x_edges, shape[mapping[0]])
    expected_y = expected_coordinates(y_edges, shape[mapping[1]])
    x, y = np.meshgrid(expected_x, expected_y)
    z_lower, z_upper = z_edges[0], z_edges[-1]
    z_length = z_upper - z_lower
    z_quadratic_integral = 1.5 * (z_upper**2 - z_lower**2)
    expected_local = z_length * (1.0 + x + 2.0 * y) + z_quadratic_integral
    y_mean = 0.5 * (y_edges[0] + y_edges[-1])
    expected_span_averaged = (
        z_length * (1.0 + expected_x + 2.0 * y_mean)
        + z_quadratic_integral
    )

    assert plan.z_integration_plan.physical_to_reference_axes == mapping
    assert plan.z_integration_plan.integration_reference_axis == mapping[2]
    assert plan.z_integration_plan.quadrature_weights.size == shape[mapping[2]]
    np.testing.assert_allclose(result.x_coordinates, expected_x)
    np.testing.assert_allclose(result.y_coordinates, expected_y)
    assert result.local_equivalent_height.shape == (expected_y.size, expected_x.size)
    np.testing.assert_allclose(result.local_equivalent_height, expected_local, atol=5.0e-14)
    np.testing.assert_allclose(
        result.span_averaged_height,
        expected_span_averaged,
        atol=5.0e-14,
    )
    assert result.spanwise_length == pytest.approx(y_edges[-1] - y_edges[0])
    assert plan.spanwise_quadrature_weights.shape == expected_y.shape
    assert plan.spanwise_quadrature_weights.sum() == pytest.approx(
        result.spanwise_length,
        abs=5.0e-14,
    )


def test_artifact_round_trip_uses_allow_pickle_free_schema(tmp_path: Path) -> None:
    plan, result = compute_cantero_equivalent_height(
        _data(lambda x, y, z: 1.0 + x + 2.0 * y + 3.0 * z)
    )
    path = cantero_equivalent_height_path(tmp_path, case="N7", index=79)
    assert path == (
        tmp_path
        / "N7"
        / "N7_f00079_cantero_equivalent_height.npz"
    )
    save_cantero_equivalent_height_npz(
        path,
        plan,
        result,
        case="N7",
        index=79,
        time=19.5,
        source_file="/data/Nek5000_data/case_N7/GC0.f00079",
    )
    loaded = load_cantero_equivalent_height_npz(path)

    assert loaded["case"] == "N7"
    assert loaded["index"] == 79
    assert loaded["time"] == 19.5
    assert loaded["source_file"] == "/data/Nek5000_data/case_N7/GC0.f00079"
    assert loaded["spanwise_length"] == pytest.approx(1.5)
    np.testing.assert_allclose(
        loaded["local_equivalent_height"], result.local_equivalent_height
    )
    np.testing.assert_allclose(
        loaded["span_averaged_height"], result.span_averaged_height
    )
    np.testing.assert_allclose(
        loaded["spanwise_quadrature_weights"].sum(),
        loaded["spanwise_length"],
    )
    np.testing.assert_array_equal(loaded["element_shape"], [4, 4, 4])
    np.testing.assert_array_equal(loaded["element_interval_counts"], [2, 2, 2])
    np.testing.assert_array_equal(loaded["physical_to_reference_axes"], [2, 1, 0])
