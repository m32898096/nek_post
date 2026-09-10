from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import barycentric_basis_and_derivative, barycentric_weights, gll_nodes
from nek_post.refined_gll_horizontal_slice import (
    apply_refined_gll_horizontal_slice_plan,
    build_refined_gll_horizontal_slice_plan,
)
from nek_post.spectral_interpolation import (
    SpectralGeometryMismatchError,
    evaluate_tensor_bary3,
    invert_map_newton_3d,
)


EDGES = ((-1.0, -0.2, 1.0), (0.0, 0.3, 0.8, 1.5), (-0.2, 0.1, 0.6))


def _polynomial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (1.0 + 0.3*x**7) * (0.5 - 0.2*y**7) * (1.0 + 0.7*z**7) + x*y*z**3


def _mesh(
    *, mapping: tuple[int, int, int] = (2, 1, 0), reverse: bool = False,
    node_count: int = 8, scale: float = 1.0, serialize_float32: bool = False,
) -> SimpleNamespace:
    q = np.meshgrid(*(gll_nodes(node_count),) * 3, indexing="ij")
    elements = []
    for iz in range(2):
        for iy in range(3):
            for ix in range(2):
                cell = (ix, iy, iz)
                coordinates = []
                for p in range(3):
                    lower, upper = np.asarray(EDGES[p][cell[p]:cell[p]+2]) * scale
                    reference = q[mapping[p]]
                    if reverse and (sum(cell) + p) % 2:
                        reference = -reference
                    c = 0.5*(lower+upper) + 0.5*(upper-lower)*reference
                    if serialize_float32:
                        c = c.astype(np.float32).astype(np.float64)
                    coordinates.append(c)
                elements.append(SimpleNamespace(
                    pos=coordinates, temp=[_polynomial(*coordinates)], cell=cell
                ))
    order = np.random.default_rng(21).permutation(len(elements))
    return SimpleNamespace(elem=[elements[i] for i in order])


def _expected_axis(edges: tuple[float, ...], node_count: int) -> np.ndarray:
    profiles = [
        0.5*(a+b) + 0.5*(b-a)*gll_nodes(node_count)
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    return np.r_[profiles[0], *(p[1:] for p in profiles[1:])]


@pytest.mark.parametrize("mapping", tuple(permutations(range(3))))
@pytest.mark.parametrize("periodic_y", (False, True))
def test_affine_tensor_plane_coordinates_counts_polynomial_and_orientation(
    mapping: tuple[int, int, int], periodic_y: bool,
) -> None:
    data = _mesh(mapping=mapping, reverse=True)
    plan = build_refined_gll_horizontal_slice_plan(
        data, target_node_count=10, z_target=0.04, periodic_y=periodic_y
    )
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    assert result.x.size == 2*9 + 1
    assert result.y.size == 3*9 + (not periodic_y)
    assert result.concentration.shape == (result.y.size, result.x.size)
    assert result.source_node_count == 8
    assert result.target_node_count == 10
    assert result.periodic_y is periodic_y
    assert result.y_period == pytest.approx(1.5, abs=1e-15)
    assert plan.physical_to_reference_axes == mapping
    assert plan.element_interval_counts == (2, 3, 2)
    assert plan.source_element_count == 12
    assert np.all(np.diff(plan.element_indices) > 0)
    assert all(data.elem[i].cell[2] == 0 for i in plan.element_indices)
    assert plan.duplicated_internal_x_nodes_removed == 1
    assert plan.duplicated_internal_y_nodes_removed == 2
    expected_y = _expected_axis(EDGES[1], 10)
    if periodic_y:
        expected_y = expected_y[:-1]
        assert result.y[-1] < EDGES[1][-1]
    np.testing.assert_allclose(result.x, _expected_axis(EDGES[0], 10), rtol=0., atol=1e-15)
    np.testing.assert_allclose(result.y, expected_y, rtol=0., atol=1e-15)
    for coordinate, edges in ((result.x, EDGES[0]), (result.y, EDGES[1])):
        assert np.all(np.diff(coordinate) > 0.0)
        for boundary in edges[1:-1]:
            assert np.count_nonzero(np.isclose(coordinate, boundary, rtol=0., atol=1e-15)) == 1
        assert np.ptp(np.diff(coordinate)) > 0.01  # retain non-uniform GLL sampling
    x, y = np.meshgrid(result.x, result.y, indexing="xy")
    np.testing.assert_allclose(result.concentration, _polynomial(x, y, 0.04), rtol=0., atol=5e-15)
    assert plan.maximum_shared_interface_coordinate_mismatch < 1e-15
    assert result.maximum_shared_interface_concentration_mismatch < 5e-15
    for node_count in (8, 10):
        assert np.min(np.abs(gll_nodes(node_count)[:, None] - plan.reference_q_z)) > 1e-3


@pytest.mark.parametrize("z_target", (0.1, -0.2, 0.6, float(-0.05 + 0.15*gll_nodes(8)[3])))
def test_exact_vertical_node_and_layer_boundary_have_one_deterministic_plane(
    z_target: float,
) -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=z_target)
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    assert plan.element_indices.size == 6
    x, y = np.meshgrid(result.x, result.y)
    np.testing.assert_allclose(
        result.concentration, _polynomial(x, y, z_target), rtol=0., atol=5e-15
    )
    if z_target == 0.1:
        assert all(data.elem[i].cell[2] == 0 for i in plan.element_indices)
        np.testing.assert_allclose(plan.reference_q_z, 1., rtol=0., atol=1e-15)


def test_plan_reuses_geometry_for_new_concentration_without_rebuilding() -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    first = apply_refined_gll_horizontal_slice_plan(data, plan)
    for element in data.elem:
        element.temp[0] = 2.0*element.temp[0] + 0.7
    second = apply_refined_gll_horizontal_slice_plan(data, plan)
    np.testing.assert_allclose(
        second.concentration, 2.*first.concentration+0.7, rtol=0., atol=8e-15
    )


@pytest.mark.parametrize("change", ("coordinates", "order", "count"))
def test_plan_rejects_geometry_or_order_changes(change: str) -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    if change == "coordinates":
        data.elem[0].pos[0][0, 0, 0] += 1e-12
    elif change == "order":
        data.elem.reverse()
    else:
        data.elem.pop()
    with pytest.raises(SpectralGeometryMismatchError, match="Geometry or element ordering"):
        apply_refined_gll_horizontal_slice_plan(data, plan)


@pytest.mark.parametrize("z_target", (-0.21, 0.61, np.nan, np.inf))
def test_invalid_vertical_target_is_rejected(z_target: float) -> None:
    with pytest.raises(ValueError, match="z_target"):
        build_refined_gll_horizontal_slice_plan(_mesh(), target_node_count=10, z_target=z_target)


@pytest.mark.parametrize("scale", (1e-8, 1.0, 1e8))
def test_tolerance_is_scale_aware_and_rejects_horizontal_warp(scale: float) -> None:
    data = _mesh(scale=scale)
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04*scale)
    assert plan.maximum_tensor_separability_deviation == 0.
    assert plan.coordinate_tolerances[0] == pytest.approx(64*np.finfo(float).eps*2.*scale)
    data.elem[0].pos[0] += 1e-6 * data.elem[0].pos[1]
    with pytest.raises(ValueError, match="not separable"):
        build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04*scale)


def test_repeated_interval_with_inconsistent_internal_coordinate_profile_is_rejected() -> None:
    data = _mesh()
    q = np.meshgrid(*(gll_nodes(8),)*3, indexing="ij")[2]
    data.elem[0].pos[0] += 0.01*(1.-q**2)
    with pytest.raises(ValueError, match="Repeated x cell profiles disagree"):
        build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)


@pytest.mark.parametrize("change", ("missing", "duplicate", "gap", "overlap"))
def test_invalid_structured_topology_is_rejected(change: str) -> None:
    data = _mesh()
    if change == "missing":
        data.elem.pop()
    elif change == "duplicate":
        data.elem.append(data.elem[0])
    else:
        for element in data.elem:
            if element.cell[0] == 1:
                element.pos[0] += 0.01 if change == "gap" else -0.01
    with pytest.raises(ValueError, match="incomplete|duplicate|gap or overlap"):
        build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)


def test_periodic_seam_retains_lower_row_without_averaging_upper_row() -> None:
    data = _mesh()
    for element in data.elem:
        element.temp = [element.pos[1].copy()]
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    np.testing.assert_allclose(
        result.concentration,
        np.broadcast_to(result.y[:, None], result.concentration.shape),
        rtol=0., atol=2e-15,
    )
    np.testing.assert_array_equal(result.concentration[0], 0.)


def test_discontinuous_internal_field_is_rejected_instead_of_averaged() -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    data.elem[plan.element_indices[0]].temp[0] += 0.01
    with pytest.raises(ValueError, match="concentration disagrees across shared interfaces"):
        apply_refined_gll_horizontal_slice_plan(data, plan)


def test_immutability_and_float64() -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    for obj in (plan, result):
        with pytest.raises(FrozenInstanceError):
            obj.z_target = 0.2
        for array in (obj.x, obj.y):
            assert array.dtype == np.float64
            with pytest.raises(ValueError):
                array[0] = 7.
            with pytest.raises(ValueError):
                array.setflags(write=True)
    assert result.concentration.dtype == np.float64
    for array in (
        result.concentration, plan.vertical_basis, plan.reference_q_z, plan.element_indices
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_stored_float32_geometry_uses_original_spectral_map_not_affine_replacement() -> None:
    data = _mesh(serialize_float32=True)
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    nodes = gll_nodes(8)
    weights = barycentric_weights(nodes)
    for row in (0, 2, 5):
        index = plan.element_indices[row]
        ix, iy, _ = plan.element_cell_indices[row]
        gx, gy = plan.x_local_to_global[ix, 3], plan.y_local_to_global[iy, 4]
        element = data.elem[index]
        inverse = invert_map_newton_3d(
            *element.pos, (result.x[gx], result.y[gy], 0.04), physical_tolerance_factor=1e-14
        )
        assert inverse.converged
        basis = [barycentric_basis_and_derivative(nodes, weights, q)[0] for q in inverse.q]
        direct = evaluate_tensor_bary3(element.temp[0], *basis)
        assert result.concentration[gy, gx] == pytest.approx(direct, rel=0., abs=5e-14)
    assert np.max(np.abs(result.x - _expected_axis(EDGES[0], 10))) > 1e-9


@pytest.mark.parametrize("target_node_count", (1, True, 10.0, 7))
def test_invalid_or_insufficient_target_node_count_is_rejected(target_node_count: object) -> None:
    with pytest.raises(ValueError, match="node_count"):
        build_refined_gll_horizontal_slice_plan(
            _mesh(), target_node_count=target_node_count, z_target=0.04
        )


@pytest.mark.parametrize("bad_field", (None, np.zeros((8, 8)), np.full((8, 8, 8), np.nan)))
def test_malformed_concentration_is_rejected(bad_field: object) -> None:
    data = _mesh()
    plan = build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)
    data.elem[plan.element_indices[0]].temp = [bad_field]
    with pytest.raises(ValueError, match="concentration"):
        apply_refined_gll_horizontal_slice_plan(data, plan)


def test_monotonic_nodal_samples_do_not_allow_a_folded_coordinate_polynomial() -> None:
    data = _mesh()
    profile = np.r_[np.arange(7) * 0.01, 1.0]
    assert np.all(np.diff(profile) > 0.0)
    for element in data.elem:
        lower, upper = EDGES[0][element.cell[0]:element.cell[0]+2]
        element.pos[0] = np.broadcast_to(
            lower + (upper-lower)*profile[None, None, :], (8, 8, 8)
        ).copy()
    with pytest.raises(ValueError, match="not strictly monotonic between GLL nodes"):
        build_refined_gll_horizontal_slice_plan(data, target_node_count=10, z_target=0.04)


@pytest.mark.parametrize("node_count", (2, 8))
def test_equal_source_and_target_counts_preserve_the_horizontal_source_nodes(
    node_count: int,
) -> None:
    data = _mesh(node_count=node_count)
    for element in data.elem:
        x, y, z = element.pos
        element.temp = [1.0 + x + 2.0*y + 3.0*z]
    plan = build_refined_gll_horizontal_slice_plan(
        data, target_node_count=node_count, z_target=0.04
    )
    result = apply_refined_gll_horizontal_slice_plan(data, plan)
    np.testing.assert_array_equal(result.x, _expected_axis(EDGES[0], node_count))
    x, y = np.meshgrid(result.x, result.y)
    np.testing.assert_allclose(result.concentration, 1.0+x+2.0*y+0.12, rtol=0., atol=4e-15)
