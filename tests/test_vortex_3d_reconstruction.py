"""Phase 2: analytical P7 sampling, bounded reconstruction, protected tiny VTUs."""
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from test_l2_projection import mesh
from test_lambda_ci_workflow import synthetic_file
from test_vortex_3d import run_args, velocity
from nek_post.gll import gll_nodes
from nek_post.gll_refinement import refine_gll_tensor3
from nek_post.hex_validation import validate_hexahedral_grid
from nek_post.io_nek import read_nek_file
from nek_post.l2_projection import build_l2_projection_operator, project_scalar_field, project_velocity_gradient
from nek_post.spectral_derivatives import element_velocity_gradient
from nek_post.spectral_interpolation import evaluate_tensor_bary3
from nek_post.gll import barycentric_weights, barycentric_basis_and_derivative
from nek_post.swirling_strength import compute_swirling_strength
from nek_post.vortex_3d import (
    compute_projected_vortex_fields, compute_native_vortex_fields, build_native_vortex_grid,
    write_binary_vtu,
)
from nek_post.vortex_3d_reconstruction import (
    _local, build_sampling_plan, build_reconstructed_vortex_grid,
)
from nek_post.vortex_3d_workflow import inspect_vortex_frame, run_single_frame_vortex


def polynomial_data(*, mapping=(2, 1, 0), reverse=False, multiple=False):
    edges = ((0., 1., 2.5), (0., .5, 2.), (0., 1.)) if multiple else ((0., 1.),)*3
    data = mesh(edges, shape=(8,)*3, mapping=mapping, reverse=reverse)
    for e in data.elem:
        x, y, z = e.pos
        e.vel = np.stack((x**7/7 + y*z, y**6/6 + x*z - 2*z, z**5/5 + x*y + 3*y))
    return data


def analytic_tensor(x, y, z):
    result = np.empty((*x.shape, 3, 3))
    result[..., 0, 0], result[..., 0, 1], result[..., 0, 2] = x**6, z, y
    result[..., 1, 0], result[..., 1, 1], result[..., 1, 2] = z, y**5, x-2
    result[..., 2, 0], result[..., 2, 1], result[..., 2, 2] = y, x+3, z**4
    return result


@pytest.mark.parametrize('count', [8, 10, 12, 14])
@pytest.mark.parametrize('mapping', list(permutations(range(3))))
def test_p7_tensor_polynomial_and_existing_point_evaluator(count, mapping):
    q = np.meshgrid(*(gll_nodes(8) for _ in range(3)), indexing='ij')
    def polynomial(q):
        a, b, c = (q[i] for i in mapping)
        return .3*a**7*b**6*c**5 - .8*a**2*b*c**7 + 1.2
    source = polynomial(q)
    reconstructed = refine_gll_tensor3(source, target_node_count=count)
    target = np.meshgrid(*(gll_nodes(count) for _ in range(3)), indexing='ij')
    np.testing.assert_allclose(reconstructed, polynomial(target), atol=4e-15, rtol=4e-15)
    if count == 8:
        np.testing.assert_array_equal(reconstructed, source)
    nodes = gll_nodes(8)
    weights = barycentric_weights(nodes)
    node = (count//3, count//2, count-2)
    rows = [barycentric_basis_and_derivative(nodes, weights, target[i][node])[0] for i in range(3)]
    assert reconstructed[node] == pytest.approx(evaluate_tensor_bary3(source, *rows), abs=2e-15)


@pytest.mark.parametrize('mapping', list(permutations(range(3))))
@pytest.mark.parametrize('reverse', [False, True])
def test_all_nine_gradients_and_phase1_arithmetic_unchanged(mapping, reverse):
    data = polynomial_data(mapping=mapping, reverse=reverse, multiple=True)
    raw = np.stack([element_velocity_gradient(e) for e in data.elem])
    operator = build_l2_projection_operator(data, periodic_axes=(), chunk_size=1)
    # Independent orchestration of the validated Phase 1 APIs, exact comparison.
    expected_omega = project_scalar_field(operator, raw[..., 2, 1]-raw[..., 1, 2], chunk_size=1)
    expected_tensor = project_velocity_gradient(operator, raw, diagnostics=False, chunk_size=1).values
    fields = compute_projected_vortex_fields(data, periodic_axes=(), chunk_size=1)
    np.testing.assert_array_equal(fields.gradients, expected_tensor)
    np.testing.assert_array_equal(fields.omega_x, expected_omega)
    for ei, e in enumerate(data.elem):
        np.testing.assert_allclose(fields.gradients[ei], analytic_tensor(*e.pos), atol=3e-11, rtol=3e-12)
    native = compute_native_vortex_fields(data, periodic_axes=(), chunk_size=1)
    np.testing.assert_array_equal(native.omega_x, expected_omega)
    np.testing.assert_array_equal(native.lambda_ci, compute_swirling_strength(expected_tensor, chunk_size=1).lambda_ci)


@pytest.mark.parametrize('count', [8, 10, 12])
@pytest.mark.parametrize('reverse', [False, True])
def test_all_nine_reconstructed_components_and_coordinate_mapping(monkeypatch, count, reverse):
    import nek_post.vortex_3d_reconstruction as module
    data = polynomial_data(mapping=(0, 2, 1), reverse=reverse)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    captured = []
    original = module.compute_swirling_strength
    def observe(tensor, **kw):
        captured.append(tensor.copy())
        return original(tensor, **kw)
    monkeypatch.setattr(module, 'compute_swirling_strength', observe)
    grid, diagnostics = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    xyz = np.stack([refine_gll_tensor3(a, target_node_count=count) for a in data.elem[0].pos], axis=-1)
    # Factorized coordinate evaluation preserves original coordinate polynomials;
    # the full tensor evaluator can differ in the last few float64 bits.
    np.testing.assert_allclose(grid.points, xyz.reshape(-1, 3), atol=8e-16, rtol=2e-15)
    local_tensor = _local(captured[0], fields.node_map, 0)
    np.testing.assert_allclose(local_tensor.reshape(-1, 3, 3), analytic_tensor(*grid.points.T), atol=4e-13)
    np.testing.assert_allclose(grid['omega_x'], 5., atol=5e-13)
    assert diagnostics['maximum_omega_route_difference'] < 3e-14
    if count == 8:
        native = build_native_vortex_grid(data, compute_native_vortex_fields(data, periodic_axes=()))
        np.testing.assert_array_equal(grid.points, native.points)
        np.testing.assert_array_equal(grid.cells, native.cells)
        np.testing.assert_array_equal(grid['omega_x'], native['omega_x'])
        np.testing.assert_array_equal(grid['lambda_ci'], native['lambda_ci'])


@pytest.mark.parametrize('count', [10, 12])
@pytest.mark.parametrize('matrix,omega,strength', [
    ([[0,0,0], [0,0,-2], [0,2,0]], 4., 2.),
    ([[0,0,0], [0,0,3], [0,-3,0]], -6., 3.),
    ([[1,0,0], [0,-2,0], [0,0,3]], 0., 0.),
    ([[0,0,0], [0,0,0], [0,0,0]], 0., 0.),
])
def test_analytical_rotation_strain_and_zero(count, matrix, omega, strength):
    data = velocity(mesh(shape=(8,)*3, reverse=True), matrix)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    grid, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    np.testing.assert_allclose(grid['omega_x'], omega, atol=3e-12)
    np.testing.assert_allclose(grid['lambda_ci'], strength, atol=1e-6)


@pytest.mark.parametrize('count', [10, 12])
def test_lambda_tensor_first_is_not_scalar_interpolation(count):
    data = mesh(shape=(8,)*3)
    x, y, z = data.elem[0].pos
    data.elem[0].vel = np.stack((0*x, -z, (x*x+.1)*y))
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    native = compute_native_vortex_fields(data, periodic_axes=())
    grid, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    np.testing.assert_allclose(grid['lambda_ci'], np.sqrt(grid.points[:, 0]**2+.1), atol=3e-14)
    scalar_first = refine_gll_tensor3(native.lambda_ci[0], target_node_count=count).ravel()
    assert np.max(np.abs(scalar_first-grid['lambda_ci'])) > 1e-5


@pytest.mark.parametrize('mapping', list(permutations(range(3))))
@pytest.mark.parametrize('count', [10, 12])
def test_nonuniform_permuted_reversed_multi_element_all_eight_vertices(mapping, count):
    data = polynomial_data(mapping=mapping, reverse=True, multiple=True)
    # Permute element storage too; topology must follow physical cell indices.
    data.elem = data.elem[::-1]
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    grid, diagnostics = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    assert grid.n_points == 4*count**3
    assert grid.n_cells == 4*(count-1)**3
    cells = grid.cell_connectivity.reshape(-1, 8)
    np.testing.assert_array_equal(np.unique(cells), np.arange(grid.n_points))
    p = grid.points[cells]
    bits = np.asarray([(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)])
    lo, hi = p.min(axis=1), p.max(axis=1)
    np.testing.assert_array_equal(p, np.where(bits[None], hi[:, None], lo[:, None]))
    assert np.all(hi > lo)
    validate_hexahedral_grid(grid)
    expected_strength = compute_swirling_strength(analytic_tensor(*grid.points.T)).lambda_ci
    np.testing.assert_allclose(grid['lambda_ci'], expected_strength, atol=1e-9, rtol=1e-10)
    assert diagnostics['checked_output_faces'] == 4
    assert max(diagnostics['maximum_output_face_differences'].values()) < 1e-12
    for ei, e in enumerate(data.elem):
        block = grid.points[ei*count**3:(ei+1)*count**3]
        for ax in range(3):
            expected = (e.pos[ax].min()+e.pos[ax].max())/2 + np.ptp(e.pos[ax])/2*gll_nodes(count)
            np.testing.assert_allclose(np.unique(block[:, ax]), expected, atol=8e-16)


@pytest.mark.parametrize('periodic', [(), ('x', 'y')])
@pytest.mark.parametrize('count', [10, 12])
def test_crop_exact_full_domain_values_and_periodic_contributions(periodic, count):
    data = mesh(((0., 1., 2.), (0., 1.), (0., 1.)), shape=(8,)*3,
                mapping=(1, 0, 2), reverse=True)
    for ei, e in enumerate(data.elem):
        x, y, z = e.pos
        rate = 2. if ei == 0 else -2.
        e.vel = np.stack((0*x, -rate*z, rate*y))
    fields = compute_projected_vortex_fields(data, periodic_axes=periodic)
    full, diag = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    cropped, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count, crop=(0., .9, .1, 1., 0., 1.))
    ids = {tuple(point): i for i, point in enumerate(full.points[:count**3])}
    used = [ids[tuple(point)] for point in cropped.points]
    np.testing.assert_array_equal(cropped.points, full.points[used])
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(cropped[name], full[name][used])
    assert cropped.bounds.x_max < .9 and cropped.bounds.y_min > .1
    if periodic:
        assert diag['checked_periodic_output_faces'] == 3  # x wrap plus each element's y wrap
        other_fields = compute_projected_vortex_fields(data, periodic_axes=())
        other, _ = build_reconstructed_vortex_grid(data, other_fields, sampling_nodes=count)
        seam = full.points[:, 0] == 0.
        assert np.max(np.abs(full['omega_x'][seam]-other['omega_x'][seam])) > 3.


def test_shared_trace_disagreement_and_omega_route_disagreement_fail_closed():
    data = polynomial_data(multiple=True)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    bad = fields.gradients.copy()
    bad[0, ..., 0, 0] += .1
    with pytest.raises(ValueError, match='Shared-face gradients disagree'):
        build_reconstructed_vortex_grid(data, replace(fields, gradients=bad), sampling_nodes=10)
    single = polynomial_data()
    fields = compute_projected_vortex_fields(single, periodic_axes=())
    with pytest.raises(ValueError, match='projected-curl and projected-gradient'):
        build_reconstructed_vortex_grid(single, replace(fields, omega_x=fields.omega_x+1), sampling_nodes=10)


@pytest.mark.parametrize('count', [10, 12])
def test_multi_element_crop_keeps_exact_element_local_samples(count):
    data = polynomial_data(mapping=(2,0,1), multiple=True, reverse=True)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    crop = (.1, 2.4, .1, 1.9, .1, .9)
    plan = build_sampling_plan(data, fields.node_map, count, crop=crop)
    full, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count)
    cropped, diagnostics = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count, crop=crop)
    for ei in range(4):
        first, last = plan.point_offsets[ei:ei+2]
        original = full.points[ei*count**3:(ei+1)*count**3]
        lookup = {tuple(point): ei*count**3+i for i, point in enumerate(original)}
        used = [lookup[tuple(point)] for point in cropped.points[first:last]]
        for name in ('omega_x', 'lambda_ci'):
            np.testing.assert_array_equal(cropped[name][first:last], full[name][used])
    assert diagnostics['checked_output_faces'] == 4


@pytest.mark.parametrize('count', [10, 12])
def test_bounded_element_evaluation_and_chunk_invariance(monkeypatch, count):
    import nek_post.vortex_3d_reconstruction as module
    data = polynomial_data(multiple=True, reverse=True)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    calls = []
    original = module.compute_swirling_strength
    def observe(array, **kwargs):
        calls.append((array.shape, kwargs['chunk_size']))
        return original(array, **kwargs)
    monkeypatch.setattr(module, 'compute_swirling_strength', observe)
    a, diag = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count, chunk_size=1)
    b, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=count, chunk_size=3)
    assert calls == [((count,count,count,3,3), 1)]*4 + [((count,count,count,3,3), 3)]*4
    assert diag['reconstruction_tensor_elements_at_once'] == 1
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize('value', [0, 7, -1, True, 8., '10', np.nan, 10**9])
def test_invalid_sampling_fails_before_payload_or_output(tmp_path, monkeypatch, value):
    import nek_post.vortex_3d_workflow as module
    case, _ = synthetic_file(tmp_path, shape=(8,)*3)
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **kw: pytest.fail('loaded payload'))
    with pytest.raises(ValueError, match='sampling|capacity'):
        run_single_frame_vortex(**run_args(case, tmp_path/'out'), sampling_nodes=value)
    assert not (tmp_path/'out').exists()


def test_explicit_sampling_rejects_non_p7_source(tmp_path):
    case, _ = synthetic_file(tmp_path)
    with pytest.raises(ValueError, match='native P7'):
        inspect_vortex_frame(**run_args(case, tmp_path/'out'), sampling_nodes=10)


@pytest.mark.parametrize('count', [8, 10, 12])
def test_preflight_estimates_and_no_payload_or_output(tmp_path, monkeypatch, count):
    import nek_post.vortex_3d_workflow as module
    case, source = synthetic_file(tmp_path, shape=(8,)*3)
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **kw: pytest.fail('loaded payload'))
    report = run_single_frame_vortex(**run_args(case, tmp_path/'out'), sampling_nodes=count, dry_run=True)
    assert report['full_local_point_count'] == 2*count**3
    assert report['full_linear_hex_count'] == 2*(count-1)**3
    assert report['native_local_point_count'] == 1024
    output = 40*2*count**3 + 80*2*(count-1)**3 + 1024**2
    assert report['estimated_output_bytes'] == output
    workspace = 0 if count == 8 else 1024*count**3
    assert report['estimated_ram_bytes'] == 8*8*1024+source.stat().st_size+240*1024+6*output+workspace+1024**3
    assert report['required_disk_bytes'] == 2*output+1024**2
    assert report['estimated_reconstruction_workspace_bytes'] == workspace
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('count', [10, 12])
def test_binary_workflow_coexists_with_native_and_records_provenance(tmp_path, count):
    case, source = synthetic_file(tmp_path, shape=(8,)*3, opposite=True)
    before = sha256(source.read_bytes()).hexdigest()
    args = run_args(case, tmp_path/'out')
    native = run_single_frame_vortex(**args)
    native_hashes = {k: sha256(Path(p).read_bytes()).hexdigest() for k, p in native['output_paths'].items()}
    actual = run_single_frame_vortex(**args, sampling_nodes=count)
    grid = pv.read(actual['output_paths']['vtu'])
    assert Path(actual['output_paths']['vtu']).name == f'GC0_f00001_gll{count}.vtu'
    assert f'/gll{count}/' in actual['output_directory']
    assert grid.n_points == 2*count**3 and grid.n_cells == 2*(count-1)**3
    for name in ('omega_x', 'lambda_ci'):
        assert grid[name].dtype == np.float64
    assert grid.points.dtype == np.float64
    assert grid.field_data['TIME_VALUE'][0] == .123456789
    metadata = json.loads(Path(actual['output_paths']['metadata']).read_text())
    assert metadata['reconstruction']['target_node_count'] == count
    assert metadata['reconstruction']['source_polynomial_degree'] == 7
    assert metadata['reconstruction']['checked_output_faces'] == 1
    assert metadata['binary_vtu_roundtrip_verified'] and metadata['source_unchanged']
    assert not metadata['visualization']['paraview_verified'] and not metadata['visualization']['tecplot_verified']
    assert 'I(P(A))' in metadata['numerical_definition']['lambda_ci']
    assert 'vortex_3d_reconstruction.py' in metadata['numerical_software_signature']
    assert sha256(source.read_bytes()).hexdigest() == before
    assert {k: sha256(Path(p).read_bytes()).hexdigest() for k, p in native['output_paths'].items()} == native_hashes
    hashes = {p: sha256(Path(p).read_bytes()).hexdigest() for p in actual['output_paths'].values()}
    with pytest.raises(FileExistsError):
        run_single_frame_vortex(**args, sampling_nodes=count)
    assert {p: sha256(Path(p).read_bytes()).hexdigest() for p in hashes} == hashes


def test_explicit_eight_matches_native_workflow_exactly(tmp_path):
    case, _ = synthetic_file(tmp_path, shape=(8,)*3)
    a = run_single_frame_vortex(**run_args(case, tmp_path/'native'))
    b = run_single_frame_vortex(**run_args(case, tmp_path/'explicit'), sampling_nodes=8)
    ga, gb = (pv.read(r['output_paths']['vtu']) for r in (a, b))
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(ga[name], gb[name])
    np.testing.assert_array_equal(ga.points, gb.points)
    np.testing.assert_array_equal(ga.cells, gb.cells)
    assert Path(a['output_paths']['vtu']).name == Path(b['output_paths']['vtu']).name


def test_vtk_borrowed_buffers_survive_gc_and_exact_roundtrip(tmp_path):
    data = polynomial_data(reverse=True)
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    grid, _ = build_reconstructed_vortex_grid(data, fields, sampling_nodes=10)
    assert np.shares_memory(grid.points, grid._reconstruction_buffers[0])
    assert np.shares_memory(grid.cell_connectivity, grid._reconstruction_buffers[1])
    gc.collect()
    path = tmp_path/'sampled.vtu'
    write_binary_vtu(path, grid, time=.777777777777)
    actual = pv.read(path)
    np.testing.assert_array_equal(actual.points, grid.points)
    np.testing.assert_array_equal(actual.cell_connectivity, grid.cell_connectivity)
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(actual[name], grid[name])
    assert actual.field_data['TIME_VALUE'][0] == .777777777777
    assert 'format="binary"' in path.read_text()


@pytest.mark.parametrize('factory', [list, tuple, iter, lambda values: (v for v in values)])
def test_dense_iterable_configuration_and_mutation(tmp_path, factory):
    case, _ = synthetic_file(tmp_path, shape=(8,)*3, opposite=True)
    axes, counts, crop = ['y', 'x'], [2,1,1], [0., .9, 0., 1., 0., 1.]
    common = dict(case=case, index=1, sampling_nodes=10)
    baseline = run_single_frame_vortex(**common, output_dir=tmp_path/'base',
        periodic_axes=tuple(axes), expected_element_counts=tuple(counts), crop=tuple(crop))
    def mutate(message):
        axes[:] = ['z']; counts[:] = [9,9,9]; crop[:] = [np.nan]*6
    actual = run_single_frame_vortex(**common, output_dir=tmp_path/'other', periodic_axes=factory(axes),
        expected_element_counts=factory(counts), crop=factory(crop), progress=mutate)
    assert actual['configuration'] == baseline['configuration']
    a, b = (pv.read(r['output_paths']['vtu']) for r in (actual, baseline))
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(a[name], b[name])


def test_plan_reuse_and_invalid_geometry_crop_or_fields():
    data = polynomial_data()
    fields = compute_projected_vortex_fields(data, periodic_axes=())
    a = build_sampling_plan(data, fields.node_map, 10)
    b = build_sampling_plan(data, fields.node_map, 10)
    assert a.spans == b.spans and a.point_offsets == b.point_offsets
    for crop in ((-1,1,0,1,0,1), (0,.001,0,.001,0,.001), (0,0,0,1,0,1)):
        with pytest.raises(ValueError, match='crop'):
            build_sampling_plan(data, fields.node_map, 10, crop=crop)
    bad = fields.gradients.copy()
    bad[0, 3,3,3,0,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        build_reconstructed_vortex_grid(data, replace(fields, gradients=bad), sampling_nodes=10, crop=(0,.1,0,.1,0,.1))
    data.elem[0].pos[0,3,3,3] += .1
    with pytest.raises(ValueError):
        build_sampling_plan(data, fields.node_map, 10)


def test_large_sampling_resource_gate_precedes_target_operator_allocation(tmp_path, monkeypatch):
    import nek_post.vortex_3d_workflow as workflow
    import nek_post.vortex_3d_reconstruction as reconstruction
    case, _ = synthetic_file(tmp_path, shape=(8,)*3)
    monkeypatch.setattr(workflow, 'available_memory_bytes', lambda: 2**30)
    monkeypatch.setattr(workflow, 'read_nek_file', lambda *a, **k: pytest.fail('loaded before resource gate'))
    monkeypatch.setattr(reconstruction, 'gll_interpolation_matrix', lambda *a: pytest.fail('built large operator'))
    args = run_args(case, tmp_path/'out')
    report = run_single_frame_vortex(**args, sampling_nodes=10000, dry_run=True)
    assert not report['resources_acceptable']
    with pytest.raises(MemoryError):
        run_single_frame_vortex(**args, sampling_nodes=10000)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('endian', ['little', 'big'])
@pytest.mark.parametrize('word_bytes', [4, 8])
def test_dense_one_based_map_permutation_preserves_scientific_fields(tmp_path, endian, word_bytes):
    from pymech.neksuite import writenek
    _, source = synthetic_file(tmp_path, shape=(8,)*3)
    data = read_nek_file(source)
    for ei, element in enumerate(data.elem):
        x, y, z = element.pos
        rate = 1.25 if ei == 0 else -2.5
        element.vel[:] = np.stack((.5*x, -rate*z, rate*y))
    data.wdsz, data.endian = word_bytes, endian
    base, permuted = tmp_path/'base', tmp_path/'permuted'
    base.mkdir(); permuted.mkdir()
    writenek(str(base/source.name), data)
    # writenek writes array records in elmap order; change both record order
    # and IDs together so global physical element identity is preserved.
    data.elmap[:] = (2, 1)
    writenek(str(permuted/source.name), data)
    reports = [run_single_frame_vortex(**run_args(case, tmp_path/f'out_{i}'), sampling_nodes=10)
               for i, case in enumerate((base, permuted))]
    a, b = (pv.read(r['output_paths']['vtu']) for r in reports)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.cell_connectivity, b.cell_connectivity)
    for name in ('omega_x', 'lambda_ci'):
        np.testing.assert_array_equal(a[name], b[name])
        assert np.ptp(a[name]) > .5
    assert reports[0]['element_map'] != reports[1]['element_map']


@pytest.mark.parametrize('ids', [(0,0), (1,1), (-1,2), (1,3)])
def test_dense_invalid_map_rejected_before_projected_computation(tmp_path, monkeypatch, ids):
    import struct
    import nek_post.vortex_3d_workflow as module
    case, source = synthetic_file(tmp_path, shape=(8,)*3)
    original = source.read_bytes()
    source.write_bytes(original[:136]+struct.pack('<2i', *ids)+original[144:])
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **kw: pytest.fail('invalid map reached reader'))
    monkeypatch.setattr(module, 'compute_projected_vortex_fields', lambda *a, **kw: pytest.fail('invalid map reached calculation'))
    with pytest.raises(ValueError, match='[Ee]lement map'):
        run_single_frame_vortex(**run_args(case, tmp_path/'out'), sampling_nodes=10)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure', ['write', 'source', 'publish'])
def test_dense_failure_retains_staging_and_never_publishes(tmp_path, monkeypatch, failure):
    import nek_post.vortex_3d_workflow as module
    case, source = synthetic_file(tmp_path, shape=(8,)*3)
    args = run_args(case, tmp_path/'out')
    def fail(*a, **kw):
        raise OSError('synthetic publication failure')
    if failure == 'write':
        monkeypatch.setattr(module, 'write_binary_vtu', fail)
    elif failure == 'publish':
        monkeypatch.setattr(module, '_publish_directory', fail)
    else:
        original = module.write_binary_vtu
        def change_source(*a, **kw):
            result = original(*a, **kw)
            with source.open('ab') as stream:
                stream.write(b'changed synthetic source')
            return result
        monkeypatch.setattr(module, 'write_binary_vtu', change_source)
    with pytest.raises((OSError, RuntimeError), match='failure|changed'):
        run_single_frame_vortex(**args, sampling_nodes=10)
    parent = tmp_path/'out'/'gll10'/case.name
    assert not (parent/'f00001').exists()
    assert len(list(parent.glob('.vortex-3d-f00001-*'))) == 1


def test_cli_sampling_default_and_dense_option():
    script = Path(__file__).resolve().parents[1]/'scripts/40_export_vortex_3d.py'
    spec = importlib.util.spec_from_file_location('phase2_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parse_args([]).sampling_nodes is None
    assert cli.parse_args(['--sampling-nodes', '12', '--dry-run']).sampling_nodes == 12
    with pytest.raises(SystemExit):
        cli.parse_args(['--sampling-nodes', '10', '--all-frames'])
