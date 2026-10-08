"""Analytic small meshes; no real snapshots or finite-difference references."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import gll_nodes, gll_quadrature_weights
from nek_post.l2_projection import (
    build_l2_projection_operator,
    build_structured_node_map,
    element_gll_weights,
    project_velocity_gradient,
    shared_node_discontinuity,
    structured_global_node_shape,
)
from nek_post.spectral_derivatives import velocity_gradient_tensor


def mesh(edges=((0., 1.), (0., 1.), (0., 1.)), shape=(3, 3, 3),
         mapping=(2, 1, 0), reverse=False, float32=False):
    reference = np.meshgrid(*(gll_nodes(n) for n in shape), indexing="ij")
    elements = []
    for cell in product(*(range(len(v) - 1) for v in edges)):
        pos = []
        for ax in range(3):
            low, high = edges[ax][cell[ax]:cell[ax] + 2]
            q = reference[mapping[ax]]
            if reverse and (sum(cell) + ax) % 2:
                q = -q
            pos.append((low + high)/2 + (high - low)/2*q)
        elements.append(SimpleNamespace(pos=np.asarray(pos, dtype=np.float32 if float32 else np.float64), cell=cell))
    return SimpleNamespace(elem=elements)


def local_values(data, shape=None):
    shape = data.elem[0].pos.shape[1:] if shape is None else shape
    values = np.empty((len(data.elem), *shape, 3, 3))
    for i in range(len(data.elem)):
        values[i] = np.arange(1, 10).reshape(3, 3) * (i + 1)
    return values


def coordinate_oracle(data, gradients, periodic_axes, edges):
    """Independent tiny coordinate grouping; exact binary endpoints only.

    Production mapping uses structured integers, never this coordinate hash.
    Analytic affine detJ is volume/8; GLL factors retain array-axis order.
    """
    groups = {}
    locations = {}
    shape = gradients.shape[1:4]
    wt, ws, wr = (gll_quadrature_weights(n) for n in shape)
    for ei, element in enumerate(data.elem):
        volume = np.prod([edges[ax][element.cell[ax]+1] - edges[ax][element.cell[ax]] for ax in range(3)])
        for node in np.ndindex(shape):
            coords = [float(element.pos[(ax, *node)]) for ax in range(3)]
            for ax, name in enumerate("xyz"):
                if name in periodic_axes and coords[ax] == edges[ax][-1]:
                    coords[ax] = edges[ax][0]
            key = tuple(coords)
            weight = volume/8 * wt[node[0]] * ws[node[1]] * wr[node[2]]
            mass, rhs = groups.get(key, (0., np.zeros((3, 3))))
            groups[key] = mass + weight, rhs + weight * gradients[(ei, *node)]
            locations[(ei, *node)] = key
    result = np.empty_like(gradients)
    for local, key in locations.items():
        mass, rhs = groups[key]
        result[local] = rhs/mass
    return result, len(groups)


def test_single_element_identity_and_analytic_stage1_gradient():
    data = mesh(shape=(4, 5, 6))
    x, y, z = data.elem[0].pos
    velocity = (x**2+y*z, y**2+x*z, z**2+x*y)
    gradients = velocity_gradient_tensor(data.elem[0].pos, velocity)[None]
    saved = gradients.copy()
    operator = build_l2_projection_operator(data, periodic_axes=())
    result = project_velocity_gradient(operator, gradients, mesh=data)
    assert result.values.shape == gradients.shape
    np.testing.assert_allclose(result.values, gradients, atol=5e-15)
    np.testing.assert_array_equal(gradients, saved)
    assert result.diagnostics.shared_node_count == 0
    np.testing.assert_array_equal(result.diagnostics.discontinuity_before, 0)
    np.testing.assert_array_equal(result.diagnostics.discontinuity_after, 0)


def test_two_nonuniform_elements_face_weighted_average_and_mass():
    edges = ((0., 1., 4.), (0., 2.), (0., 1.))
    data = mesh(edges)
    operator = build_l2_projection_operator(data, periodic_axes=(), chunk_size=1)
    gradients = local_values(data)
    gradients[1] *= 2  # contributor values 1 and 4, volume ratio 1:3
    result = project_velocity_gradient(operator, gradients, chunk_size=1)
    expected = np.arange(1, 10).reshape(3, 3) * 3.25
    np.testing.assert_allclose(result.values[0, :, :, -1], np.broadcast_to(expected, (3, 3, 3, 3)))
    np.testing.assert_array_equal(result.values[0, :, :, -1], result.values[1, :, :, 0])
    assert operator.mass.shape == (45,)
    assert np.isfinite(operator.mass).all() and (operator.mass > 0).all()
    assert operator.mass.sum() == pytest.approx(8.)
    wr = gll_quadrature_weights(3)
    np.testing.assert_allclose(operator.weights[0], wr[:, None, None]*wr[None, :, None]*wr[None, None, :]/4)
    np.testing.assert_allclose(operator.weights[1], 3*operator.weights[0])
    np.testing.assert_array_equal(result.diagnostics.discontinuity_after, 0)
    np.testing.assert_allclose(result.diagnostics.discontinuity_before, 3*np.arange(1, 10).reshape(3, 3))


@pytest.mark.parametrize("periodic", ((), ("x",), ("y",), ("x", "y"), ("x", "y", "z")))
def test_faces_edges_corners_and_periodic_averages_against_independent_oracle(periodic):
    edges = ((0., 1., 3.), (0., 2., 3.), (0., 1., 5.))
    data = mesh(edges)
    gradients = local_values(data)
    coords_before = [e.pos.copy() for e in data.elem]
    operator = build_l2_projection_operator(data, periodic_axes=periodic, chunk_size=3)
    expected, global_count = coordinate_oracle(data, gradients, periodic, edges)
    result = project_velocity_gradient(operator, gradients, chunk_size=2)
    np.testing.assert_allclose(result.values, expected, atol=1e-13, rtol=2e-14)
    assert operator.node_map.global_node_count == global_count
    np.testing.assert_array_equal(result.diagnostics.discontinuity_after, 0)
    ids = operator.node_map.node_ids(0, len(data.elem))
    assert ids.dtype == np.int32
    assert len(np.unique(ids)) == global_count
    shared = np.bincount(ids.ravel(), minlength=global_count) > 1
    assert shared.sum() == operator.node_map.shared_node_count
    # Explicit internal eight-element corner and four-element edge checks.
    lookup = {e.cell: i for i, e in enumerate(data.elem)}
    corner_ids = [ids[lookup[c], 2*(1-c[2]), 2*(1-c[1]), 2*(1-c[0])] for c in product((0, 1), repeat=3)]
    assert len(set(corner_ids)) == 1
    edge_ids = [ids[lookup[(0, cy, cz)], 2*(1-cz), 2*(1-cy), :] for cy, cz in product((0, 1), repeat=2)]
    for edge in edge_ids[1:]: np.testing.assert_array_equal(edge_ids[0], edge)
    for e, original in zip(data.elem, coords_before): np.testing.assert_array_equal(e.pos, original)


def test_single_element_combined_periodic_corners_and_distinct_z_planes():
    data = mesh(shape=(3, 4, 5))
    mapping = build_structured_node_map(data, periodic_axes=("y", "x"))
    ids = mapping.node_ids(0, 1)[0]
    assert mapping.periodic_axes == ("x", "y")
    assert mapping.global_shape_xyz == (4, 3, 3)
    np.testing.assert_array_equal(ids[:, :, 0], ids[:, :, -1])
    np.testing.assert_array_equal(ids[:, 0, :], ids[:, -1, :])
    assert len(set(ids[0, j, i] for j in (0, 3) for i in (0, 4))) == 1
    assert not np.intersect1d(ids[0], ids[-1]).size
    result = project_velocity_gradient(build_l2_projection_operator(data, periodic_axes=("x", "y")), local_values(data))
    np.testing.assert_array_equal(result.diagnostics.discontinuity_after, 0)


@pytest.mark.parametrize("mapping", ((2, 1, 0), (0, 2, 1), (1, 0, 2)))
def test_non_cubic_permuted_reversed_and_shuffled_elements(mapping):
    edges = ((0., 1., 3.), (0., 2., 3.), (0., 1., 5.))
    data = mesh(edges, shape=(3, 4, 5), mapping=mapping, reverse=True)
    np.random.default_rng(42).shuffle(data.elem)
    operator = build_l2_projection_operator(data, periodic_axes=("x", "y"), chunk_size=1)
    assert operator.node_map.physical_to_array_axes == mapping
    assert not operator.node_map.element_axis_increasing.all()
    expected, count = coordinate_oracle(data, local_values(data), ("x", "y"), edges)
    result = project_velocity_gradient(operator, local_values(data), chunk_size=3)
    np.testing.assert_allclose(result.values, expected, atol=2e-13)
    assert operator.node_map.global_node_count == count
    assert operator.mass.sum() == pytest.approx(45.)


@pytest.mark.parametrize("periodic", ((), ("x", "y")))
def test_constant_nine_component_preservation_float32_input_and_reuse(periodic):
    data = mesh(((0., 1., 4.), (0., 1., 2.), (0., 1.)), float32=True)
    operator = build_l2_projection_operator(data, periodic_axes=periodic)
    constants = np.array([[1., -2., .5], [3., -4., .125], [6., -7., 8.]], dtype=np.float32)
    gradients = np.broadcast_to(constants, (len(data.elem), 3, 3, 3, 3, 3))
    result = project_velocity_gradient(operator, gradients, mesh=data)
    assert result.values.dtype == np.float64
    np.testing.assert_allclose(result.values, gradients, atol=3e-15)
    np.testing.assert_array_equal(result.diagnostics.discontinuity_before, 0)
    np.testing.assert_array_equal(result.diagnostics.discontinuity_after, 0)
    second = project_velocity_gradient(operator, 2*gradients, mesh=deepcopy(data), diagnostics=False)
    np.testing.assert_allclose(second.values, 2*result.values, atol=8e-15)
    assert second.diagnostics is None
    for array in (operator.weights, operator.mass, operator.node_map.element_cell_indices):
        assert not array.flags.writeable


def test_local_curved_weights_use_analytic_nodal_determinant():
    t, s, r = np.meshgrid(*(gll_nodes(5) for _ in range(3)), indexing="ij")
    coordinates = np.stack((r + .15*s**2, s + .1*t**2, t + .1*r**2))
    w = gll_quadrature_weights(5)
    expected = (1 + .012*r*s*t)*w[:, None, None]*w[None, :, None]*w[None, None, :]
    np.testing.assert_allclose(element_gll_weights(coordinates), expected, atol=2e-15)
    # Curved global mapping must not be silently treated as an affine box.
    with pytest.raises(ValueError):
        build_structured_node_map(SimpleNamespace(elem=[SimpleNamespace(pos=coordinates)]), periodic_axes=())


def test_n7_h_count_without_full_size_arrays():
    assert structured_global_node_shape((272, 12, 8), (8, 8, 8), periodic_axes=("x", "y")) == (1904, 84, 57)
    assert 1904*84*57 == 9_116_352
    assert structured_global_node_shape((272, 12, 8), (8, 8, 8), periodic_axes=()) == (1905, 85, 57)
    assert 1905*85*57 == 9_229_725


@pytest.mark.parametrize("bad", (None, "xy", ("X",), ("t",), ("x", "x"), (False,), 0))
def test_periodicity_requires_explicit_valid_configuration(bad):
    with pytest.raises(ValueError, match="periodic_axes"):
        build_structured_node_map(mesh(), periodic_axes=bad)


@pytest.mark.parametrize("counts,nodes", (
    ((0, 1, 1), (3, 3, 3)), ((1, 1), (3, 3, 3)),
    ((True, 1, 1), (3, 3, 3)), ((1., 1, 1), (3, 3, 3)),
    ((1, 1, 1), (1, 3, 3)), ((1, 1, 1), (3, 3)),
    (None, (3, 3, 3)), ((2**63, 1, 1), (3, 3, 3)),
))
def test_invalid_or_overflowing_structured_dimensions(counts, nodes):
    with pytest.raises(ValueError):
        structured_global_node_shape(counts, nodes, periodic_axes=())


@pytest.mark.parametrize("kind", ("empty", "duplicate", "missing", "gap", "overlap", "nan", "inf", "singular", "mapping", "shape"))
def test_invalid_geometry_or_topology_rejected(kind):
    data = mesh(((0., 1., 2.),)*3)
    if kind == "empty": data.elem.clear()
    elif kind == "duplicate": data.elem.append(deepcopy(data.elem[0]))
    elif kind == "missing": data.elem.pop(0)
    elif kind in ("gap", "overlap"):
        for e in data.elem:
            if e.cell[0] == 1: e.pos[0] += .1 if kind == "gap" else -.1
    elif kind in ("nan", "inf"): data.elem[0].pos[0, 0, 0, 0] = np.nan if kind == "nan" else np.inf
    elif kind == "singular": data.elem[0].pos[0] = 0
    elif kind == "mapping": data.elem[0].pos = data.elem[0].pos.transpose(0, 3, 2, 1)
    elif kind == "shape": data.elem[0].pos = data.elem[0].pos[:, :2]
    with pytest.raises(ValueError):
        build_l2_projection_operator(data, periodic_axes=("x", "y"), chunk_size=1)


def test_expected_counts_reject_missing_outer_layer():
    data = mesh(((0., 1., 2.), (0., 1.), (0., 1.)))
    data.elem.pop()
    with pytest.raises(ValueError, match="Expected element counts"):
        build_l2_projection_operator(data, periodic_axes=(), expected_element_counts=(2, 1, 1))


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, 1+2j))
def test_nonfinite_and_complex_coordinate_weights_rejected(bad):
    coordinates = mesh().elem[0].pos.astype(complex if isinstance(bad, complex) else float)
    coordinates[0, 0, 0, 0] = bad
    with pytest.raises(ValueError): element_gll_weights(coordinates)


def test_folded_local_geometry_and_invalid_integration_weights(monkeypatch):
    import nek_post.l2_projection as module
    coordinates = mesh().elem[0].pos.copy()
    coordinates[0] **= 2  # derivative vanishes at x=0 and changes within extension
    with pytest.raises(ValueError): element_gll_weights(coordinates)
    for bad in (np.zeros((3, 3, 3)), np.full((3, 3, 3), np.nan), np.full((3, 3, 3), np.inf)):
        monkeypatch.setattr(module, "element_jacobian", lambda *args, _bad=bad, **kwargs: (None, _bad))
        with pytest.raises(ValueError, match="weights"): element_gll_weights(mesh().elem[0].pos)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, 1+2j))
def test_invalid_gradient_values_rejected_before_output_mutation(bad):
    operator = build_l2_projection_operator(mesh(), periodic_axes=())
    values = np.zeros((1, 3, 3, 3, 3, 3), dtype=complex if isinstance(bad, complex) else float)
    values[0, 0, 0, 0, 0, 0] = bad
    output = np.full(values.shape, 42.)
    with pytest.raises(ValueError): project_velocity_gradient(operator, values, out=output)
    np.testing.assert_array_equal(output, 42)
    with pytest.raises(ValueError): shared_node_discontinuity(operator.node_map, values)


def test_noncontiguous_input_preallocated_output_and_independent_jump_measurement():
    data = mesh(((0., 1., 2.), (0., 1.), (0., 1.)))
    operator = build_l2_projection_operator(data, periodic_axes=("x",))
    values = local_values(data)[..., ::-1, ::-1]
    assert not values.flags.c_contiguous
    output = np.empty(values.shape, np.float64)
    result = project_velocity_gradient(operator, values, out=output)
    assert result.values is output
    np.testing.assert_array_equal(shared_node_discontinuity(operator.node_map, output), 0)
    np.testing.assert_array_equal(result.diagnostics.discontinuity_before, shared_node_discontinuity(operator.node_map, values))


def test_output_validation_and_gradient_shape():
    operator = build_l2_projection_operator(mesh(), periodic_axes=())
    values = local_values(mesh())
    readonly = np.empty_like(values); readonly.setflags(write=False)
    for out in (values, np.empty_like(values, dtype=np.float32), readonly, np.empty((1,)), values[..., ::-1, :]):
        with pytest.raises(ValueError, match="out"): project_velocity_gradient(operator, values, out=out)
    for bad in (np.zeros((3, 3)), values.astype(str), values.astype(bool)):
        with pytest.raises(ValueError): project_velocity_gradient(operator, bad)
    with pytest.raises(ValueError, match="boolean"):
        project_velocity_gradient(operator, values, diagnostics="yes")


@pytest.mark.parametrize("chunk_size", (0, -1, True, 1.5))
def test_invalid_chunks_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        build_structured_node_map(mesh(), periodic_axes=(), chunk_size=chunk_size)
    operator = build_l2_projection_operator(mesh(), periodic_axes=())
    with pytest.raises(ValueError, match="chunk_size"):
        project_velocity_gradient(operator, local_values(mesh()), chunk_size=chunk_size)


def test_changed_geometry_order_count_and_invalid_slices_are_rejected():
    data = mesh(((0., 1., 2.), (0., 1.), (0., 1.)))
    operator = build_l2_projection_operator(data, periodic_axes=())
    reversed_mesh = SimpleNamespace(elem=data.elem[::-1])
    changed = deepcopy(data); changed.elem[0].pos[0] += .01
    shorter = SimpleNamespace(elem=data.elem[:1])
    for invalid in (reversed_mesh, changed, shorter):
        with pytest.raises(ValueError, match="differs"):
            project_velocity_gradient(operator, local_values(data), mesh=invalid)
    for start, stop in ((-1, 1), (0, 3), (2, 1), (False, 1)):
        with pytest.raises(ValueError): operator.node_map.node_ids(start, stop)
    assert operator.node_map.node_ids(1, 1).shape == (0, 3, 3, 3)


def test_float64_overflow_in_rhs_is_rejected():
    data = mesh(((0., 1., 2.), (0., 1.), (0., 1.)))
    operator = build_l2_projection_operator(data, periodic_axes=())
    # Amplifying geometry weights is only a controlled overflow fixture.
    from dataclasses import replace
    operator = replace(operator, weights=operator.weights * 1e308)
    with pytest.raises(FloatingPointError, match="non-finite"):
        project_velocity_gradient(operator, local_values(data)*100, diagnostics=False)


def test_weighted_orthogonality_integral_preservation_and_idempotence():
    data = mesh(((0., 1., 3.), (0., 1., 4.), (0., 1., 2.)))
    operator = build_l2_projection_operator(data, periodic_axes=("x", "y"))
    values = np.random.default_rng(7).normal(size=local_values(data).shape)
    result = project_velocity_gradient(operator, values)
    ids = operator.node_map.node_ids(0, len(data.elem)).ravel()
    residual = np.zeros((operator.node_map.global_node_count, 3, 3))
    np.add.at(residual, ids, (operator.weights[..., None, None]*(values-result.values)).reshape(-1, 3, 3))
    np.testing.assert_allclose(residual, 0, atol=3e-15)
    before = np.sum(operator.weights[..., None, None]*values, axis=(0, 1, 2, 3))
    after = np.sum(operator.weights[..., None, None]*result.values, axis=(0, 1, 2, 3))
    np.testing.assert_allclose(before, after, atol=1e-14)
    second = project_velocity_gradient(operator, result.values, diagnostics=False)
    np.testing.assert_allclose(second.values, result.values, atol=1e-15)


def test_float32_p7_geometry_and_bounded_preparation(monkeypatch):
    import nek_post.l2_projection as module
    edges = ((-17., -16.875, -16.75), (0., .125), (0., .0867, .1951))
    data = mesh(edges, shape=(8, 8, 8), float32=True)
    original = [e.pos.copy() for e in data.elem]
    validated = module._validated_geometry
    batches = []
    def observe(batch):
        batches.append(len(batch.elem))
        return validated(batch)
    monkeypatch.setattr(module, "_validated_geometry", observe)
    operator = build_l2_projection_operator(data, periodic_axes=("x", "y"), chunk_size=2, expected_element_counts=(2, 1, 2))
    assert max(batches) == 2
    assert operator.node_map.global_shape_xyz == (14, 7, 15)
    assert operator.mass.sum() == pytest.approx(.25*.125*.1951, rel=1e-6)
    for e, copy in zip(data.elem, original): np.testing.assert_array_equal(e.pos, copy)
    assert operator.storage_bytes >= operator.weights.nbytes + operator.mass.nbytes


@pytest.mark.parametrize("bad", (0., -1., np.nan, np.inf))
def test_invalid_quadrature_weights_rejected(monkeypatch, bad):
    import nek_post.l2_projection as module
    monkeypatch.setattr(module, "gll_quadrature_weights", lambda n: np.full(n, bad))
    with pytest.raises(ValueError, match="weights"):
        element_gll_weights(mesh().elem[0].pos)
