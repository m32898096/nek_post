"""Synthetic native 3D field, physical ordering and binary-file validation."""
from dataclasses import replace
import importlib.util
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
import pytest

from test_l2_projection import mesh, coordinate_oracle
from test_lambda_ci_workflow import synthetic_file
from nek_post.l2_projection import build_l2_projection_operator, project_velocity_gradient
from nek_post.spectral_derivatives import element_velocity_gradient
from nek_post.swirling_strength import compute_swirling_strength
from nek_post.vorticity import compute_projected_omega_x
from nek_post.vortex_3d import (
    build_native_vortex_grid, compute_native_vortex_fields, write_binary_vtu,
)
from nek_post.vortex_3d_workflow import inspect_vortex_frame, run_single_frame_vortex


def velocity(data, matrix):
    for element in data.elem:
        element.vel = np.einsum('ij,j...->i...', matrix, element.pos) + np.asarray([.2, -.3, .1])[:, None, None, None]
    return data


@pytest.mark.parametrize('matrix,omega,strength', [
    ([[0, 0, 0], [0, 0, -2], [0, 2, 0]], 4., 2.),
    ([[0, 0, 0], [0, 0, 3], [0, -3, 0]], -6., 3.),
    ([[0, 0, 0], [0, 0, 2], [0, 0, 0]], -2., 0.),
    ([[1, 0, 0], [0, -2, 0], [0, 0, 3]], 0., 0.),
    ([[.5, 0, 0], [0, 1, -4], [0, 1, 1]], 5., 2.),
])
@pytest.mark.parametrize('mapping,reverse', [((2, 1, 0), False), ((0, 2, 1), True), ((1, 0, 2), True)])
def test_linear_velocity_analytic_fields(matrix, omega, strength, mapping, reverse):
    data = velocity(mesh(shape=(8, 7, 6), mapping=mapping, reverse=reverse), matrix)
    fields = compute_native_vortex_fields(data, periodic_axes=())
    np.testing.assert_allclose(fields.omega_x, omega, atol=4e-12)
    # A defective shear/zero spectrum is deliberately not tolerance-filtered
    # by the kernel. Its tiny complex perturbations remain roundoff-sensitive.
    np.testing.assert_allclose(fields.lambda_ci, strength, atol=1e-6)
    assert fields.omega_x.dtype == fields.lambda_ci.dtype == np.float64


@pytest.mark.parametrize('periodic', [(), ('x', 'y')])
def test_shared_raw_gradient_single_pass_and_existing_pipelines(monkeypatch, periodic):
    import nek_post.vortex_3d as module
    data = mesh(((0., 1., 4.), (0., 1.), (0., 1.)), shape=(4, 5, 6), reverse=True)
    for ei, e in enumerate(data.elem):
        x, y, z = e.pos
        e.vel = np.stack((x*y, (ei + 1)*y*z + x*x, (-1)**ei*y*y + z*x))
    raw = np.stack([element_velocity_gradient(e) for e in data.elem])
    calls = []
    original = module.element_velocity_gradient
    def observe(e):
        calls.append(e)
        return original(e)
    monkeypatch.setattr(module, 'element_velocity_gradient', observe)
    fields = compute_native_vortex_fields(data, periodic_axes=periodic, chunk_size=1)
    assert len(calls) == len(data.elem)
    operator = build_l2_projection_operator(data, periodic_axes=periodic)
    projected = project_velocity_gradient(operator, raw, diagnostics=False, chunk_size=1).values
    expected = compute_swirling_strength(projected).lambda_ci
    np.testing.assert_array_equal(fields.lambda_ci, expected)
    # Independent physical-coordinate mass oracle; no direct-ID assumptions.
    curl = np.zeros_like(raw)
    curl[..., 0, 0] = raw[..., 2, 1] - raw[..., 1, 2]
    oracle, _ = coordinate_oracle(data, curl, periodic, ((0., 1., 4.), (0., 1.), (0., 1.)))
    np.testing.assert_allclose(fields.omega_x, oracle[..., 0, 0], atol=1e-13)
    np.testing.assert_allclose(fields.omega_x, compute_projected_omega_x(data, operator), atol=1e-13)


def test_nonlinear_eigenvalues_after_mass_weighted_projection():
    data = mesh(((0., 1., 4.), (0., 1.), (0., 1.)))
    for ei, element in enumerate(data.elem):
        x, y, z = element.pos
        rate = 2. if ei == 0 else -2.
        element.vel = np.stack((0*x, -rate*z, rate*y))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    # At x=1 masses have ratio 1:3. Projected rotation is -1, whereas
    # projecting the already-computed strengths would incorrectly give 2.
    for ei, e in enumerate(data.elem):
        interface = e.pos[0] == 1.
        np.testing.assert_allclose(fields.lambda_ci[ei][interface], 1., atol=1e-13)
        np.testing.assert_allclose(fields.omega_x[ei][interface], -2., atol=1e-13)


@pytest.mark.parametrize('mapping', [(2, 1, 0), (0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1)])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('shape', [(4, 5, 6), (8, 8, 8)])
def test_interior_gll_points_cell_winding_and_physical_point_order(mapping, reverse, shape):
    data = velocity(mesh(((0., 1., 3.), (0., 1.), (0., 2.)), shape=shape,
                         mapping=mapping, reverse=reverse), np.eye(3))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    # A coordinate-encoded signed sentinel detects scalar/point transposition.
    omega = np.stack([e.pos[0] + 10*e.pos[1] - 100*e.pos[2] for e in data.elem])
    fields = replace(fields, omega_x=omega, lambda_ci=omega**2)
    grid = build_native_vortex_grid(data, fields)
    local_points = int(np.prod(shape))
    local_cells = int(np.prod(np.asarray(shape)-1))
    assert grid.n_points == 2*local_points
    assert grid.n_cells == 2*local_cells
    np.testing.assert_array_equal(grid.points, np.concatenate([np.moveaxis(e.pos, 0, -1).reshape(-1, 3) for e in data.elem]))
    np.testing.assert_array_equal(grid['omega_x'], omega.ravel())
    np.testing.assert_array_equal(grid['lambda_ci'], (omega**2).ravel())
    # All nodes, including SEM interior nodes, participate in connectivity.
    cells = grid.cells.reshape(-1, 9)[:, 1:]
    np.testing.assert_array_equal(np.unique(cells), np.arange(grid.n_points))
    p = grid.points[cells]
    corner_bits = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    low, high = p.min(axis=1), p.max(axis=1)
    np.testing.assert_array_equal(p, np.where(corner_bits[None], high[:, None], low[:, None]))
    assert np.all(high > low)  # The entire cuboid J is positive diagonal.
    for ei, element in enumerate(data.elem):
        block = cells[ei*local_cells:(ei+1)*local_cells]
        assert block.min() >= ei*local_points and block.max() < (ei+1)*local_points
        profiles = [np.unique(a) for a in element.pos]
        intervals = [tuple(zip(v[:-1], v[1:])) for v in profiles]
        expected_bounds = {(*x, *y, *z) for x, y, z in product(*intervals)}
        lo, hi = low[ei*local_cells:(ei+1)*local_cells], high[ei*local_cells:(ei+1)*local_cells]
        actual_bounds = {tuple(v for pair in zip(a, b) for v in pair) for a, b in zip(lo, hi)}
        assert actual_bounds == expected_bounds
    assert np.all(grid.celltypes == pv.CellType.HEXAHEDRON)
    # Shared interface coordinates remain duplicate element-local point IDs.
    assert len(np.unique(grid.points, axis=0)) < grid.n_points


@pytest.mark.parametrize('mapping', [(2, 1, 0), (0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1)])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('periodic', [(), ('x', 'y')])
def test_p7_crops_are_exact_subsets_of_full_projected_fields(mapping, reverse, periodic):
    data = mesh(((0., 1., 4.), (0., .5, 2.), (0., .25, 1.)), shape=(8, 8, 8),
                mapping=mapping, reverse=reverse)
    for ei, element in enumerate(data.elem):
        x, y, z = element.pos
        rate = .5 + .2*ei
        element.vel = np.stack((.1*x*y+.2*z, -rate*z+x*z+.1*y*y,
                                rate*y+.2*x*y+.1*z*z))
    fields = compute_native_vortex_fields(data, periodic_axes=periodic)
    whole = build_native_vortex_grid(data, fields)
    assert np.ptp(whole['omega_x']) > 1 and np.ptp(whole['lambda_ci']) > .1
    for crop in ((0., 1.7, 0., 2., .08, 1.), (2.3, 4., 0., 2., .08, 1.)):
        reduced = build_native_vortex_grid(data, fields, crop=crop)
        bounds = np.asarray(crop).reshape(3, 2)
        inside = ((whole.points >= bounds[:, 0]) & (whole.points <= bounds[:, 1])).all(axis=1)
        cells = whole.cells.reshape(-1, 9)[:, 1:]
        retained = cells[inside[cells].all(axis=1)]
        used, inverse = np.unique(retained, return_inverse=True)
        np.testing.assert_array_equal(reduced.points, whole.points[used])
        np.testing.assert_array_equal(reduced.cells.reshape(-1, 9)[:, 1:], inverse.reshape(-1, 8))
        for name in ('omega_x', 'lambda_ci'):
            np.testing.assert_array_equal(reduced[name], whole[name][used])
    if periodic:
        # Check physical seam matching independently of the mapper's node IDs,
        # including duplicated SEM faces, edges and corners.
        for axis, top in ((0, 4.), (1, 2.)):
            bottom = whole.points[:, axis] == 0.
            upper = whole.points[:, axis] == top
            transverse = [a for a in range(3) if a != axis]
            a = np.flatnonzero(bottom)[np.lexsort(whole.points[bottom][:, transverse].T)]
            b = np.flatnonzero(upper)[np.lexsort(whole.points[upper][:, transverse].T)]
            np.testing.assert_array_equal(whole.points[a][:, transverse], whole.points[b][:, transverse])
            for name in ('omega_x', 'lambda_ci'):
                np.testing.assert_array_equal(whole[name][a], whole[name][b])


def test_p7_is_343_hexes_per_element_and_crop_is_native_only():
    data = velocity(mesh(shape=(8, 8, 8)), np.eye(3))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    whole = build_native_vortex_grid(data, fields)
    assert (whole.n_points, whole.n_cells) == (512, 343)
    crop = build_native_vortex_grid(data, fields, crop=(.3, .9, 0., 1., 0., 1.))
    assert 0 < crop.n_cells < 343
    assert .3 <= crop.bounds[0] < crop.bounds[1] <= .9
    assert crop.bounds[0] != .3 and crop.bounds[1] != .9
    assert set(map(tuple, crop.points)).issubset(set(map(tuple, whole.points)))


def test_float32_native_coordinates_and_discontinuous_shared_fields_preserved():
    data = velocity(mesh(((0., 1., 2.), (0., 1.), (0., 1.)),
                         shape=(8, 8, 8), float32=True), np.eye(3))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    # Visualization has no authority to reconcile anonymous scalar replicas.
    # Distinct values at coincident points must remain explicitly distinct.
    omega = fields.omega_x.copy()
    omega[1, :, :, 0] = -123.
    fields = replace(fields, omega_x=omega)
    grid = build_native_vortex_grid(data, fields)
    native = np.concatenate([np.moveaxis(e.pos, 0, -1).reshape(-1, 3) for e in data.elem])
    np.testing.assert_array_equal(grid.points, native.astype(np.float64))
    assert grid.points.dtype == np.float64
    interface = grid.points[:, 0] == 1.
    assert np.count_nonzero(interface) == 128
    assert np.count_nonzero(grid['omega_x'][interface] == -123.) == 64
    np.testing.assert_array_equal(grid['omega_x'], omega.ravel())


@pytest.mark.parametrize('crop', [(0, 0, 0, 1, 0, 1), (0, np.inf, 0, 1, 0, 1), (0, 1),
                                  (-1, 1, 0, 1, 0, 1), (.001, .002, 0, 1, 0, 1)])
def test_bad_crop_rejected(crop):
    data = velocity(mesh(), np.eye(3))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    with pytest.raises(ValueError, match='crop'):
        build_native_vortex_grid(data, fields, crop=crop)


@pytest.mark.parametrize('kind', ['nan_velocity', 'nan_geometry', 'singular', 'non_gll', 'missing_element', 'inconsistent_shared'])
def test_invalid_native_data_rejected(kind):
    data = velocity(mesh(((0., 1., 2.), (0., 1.), (0., 1.)), shape=(4, 4, 4)), np.eye(3))
    if kind == 'nan_velocity': data.elem[0].vel[1, 1, 1, 1] = np.nan
    if kind == 'nan_geometry': data.elem[0].pos[1, 1, 1, 1] = np.nan
    if kind == 'singular': data.elem[0].pos[2] = 0
    if kind == 'non_gll': data.elem[0].pos[0, :, :, 1] += .01
    if kind == 'missing_element': data.elem.pop()
    if kind == 'inconsistent_shared': data.elem[1].pos[0] += 1e-8
    with pytest.raises(ValueError):
        compute_native_vortex_fields(data, periodic_axes=(), expected_element_counts=(2, 1, 1))


@pytest.mark.parametrize('kind', ['order', 'shape', 'nan', 'dtype', 'negative_lambda'])
def test_bad_field_layout_or_order_rejected(kind):
    data = velocity(mesh(((0., 1., 2.), (0., 1.), (0., 1.))), np.eye(3))
    fields = compute_native_vortex_fields(data, periodic_axes=())
    if kind == 'order': data.elem.reverse()
    if kind == 'shape': fields = replace(fields, omega_x=fields.omega_x[..., 0])
    if kind == 'nan': fields.omega_x[0, 0, 0, 0] = np.inf
    if kind == 'dtype': fields = replace(fields, omega_x=fields.omega_x.astype(np.float32))
    if kind == 'negative_lambda': fields.lambda_ci[0, 0, 0, 0] = -1
    with pytest.raises(ValueError): build_native_vortex_grid(data, fields)


def test_true_binary_vtu_round_trip_and_collision(tmp_path):
    data = velocity(mesh(shape=(4, 5, 6), mapping=(0, 2, 1), reverse=True),
                    [[0, 0, 0], [0, 0, -2], [0, 2, 0]])
    fields = compute_native_vortex_fields(data, periodic_axes=())
    grid = build_native_vortex_grid(data, fields)
    path = write_binary_vtu(tmp_path/'GC0_f00001.vtu', grid, time=.123456789)
    xml = ET.parse(path).getroot()
    assert xml.attrib['type'] == 'UnstructuredGrid'
    assert all(a.attrib.get('format') in ('binary', 'appended') for a in xml.findall('.//DataArray'))
    actual = pv.read(path)
    for name in ('omega_x', 'lambda_ci'):
        assert actual.point_data[name].dtype == np.float64
        np.testing.assert_array_equal(actual[name], grid[name])
    np.testing.assert_array_equal(actual.points, grid.points)
    np.testing.assert_array_equal(actual.cells, grid.cells)
    assert actual.field_data['TIME_VALUE'][0] == .123456789
    before = path.read_bytes()
    with pytest.raises(FileExistsError): write_binary_vtu(path, grid, time=1.)
    assert path.read_bytes() == before
    with pytest.raises(ValueError): write_binary_vtu(tmp_path/'x.vtu', grid, time=np.nan)
    with pytest.raises(ValueError): write_binary_vtu(tmp_path/'x.vtr', grid, time=0.)


def test_vtu_writer_rejects_repository_and_detects_corrupt_roundtrip(tmp_path, monkeypatch):
    from nek_post.vortex_3d_workflow import REPOSITORY
    data = velocity(mesh(), np.eye(3))
    grid = build_native_vortex_grid(data, compute_native_vortex_fields(data, periodic_axes=()))
    with pytest.raises(ValueError, match='outside'):
        write_binary_vtu(REPOSITORY/'unsafe.vtu', grid, time=0.)
    save = pv.UnstructuredGrid.save
    def corrupt(self, path, **kwargs):
        copy = self.copy()
        copy.point_data['omega_x'][0] += 1.
        save(copy, path, **kwargs)
    monkeypatch.setattr(pv.UnstructuredGrid, 'save', corrupt)
    path = tmp_path/'corrupted.vtu'
    with pytest.raises(ValueError, match='round-trip changed omega_x'):
        write_binary_vtu(path, grid, time=0.)
    assert path.is_file()  # preserve failed staging evidence


def run_args(case, output):
    return dict(case=case, index=1, output_dir=output, periodic_axes=(), expected_element_counts=(2, 1, 1))


def test_header_only_dry_run_and_low_ram_never_load(tmp_path, monkeypatch):
    import nek_post.vortex_3d_workflow as module
    case, _ = synthetic_file(tmp_path)
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **k: pytest.fail('loaded payload'))
    args = run_args(case, tmp_path/'out')
    report = run_single_frame_vortex(**args, dry_run=True)
    assert report['time'] == .123456789
    assert report['geometry_validation'] == 'pending payload read'
    assert not (tmp_path/'out').exists()
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 1)
    assert not run_single_frame_vortex(**args, dry_run=True)['resources_acceptable']
    with pytest.raises(MemoryError): run_single_frame_vortex(**args)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind', ['unknown_ram', 'low_disk'])
def test_unavailable_resources_stop_before_payload(tmp_path, monkeypatch, kind):
    import nek_post.vortex_3d_workflow as module
    case, _ = synthetic_file(tmp_path)
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **k: pytest.fail('loaded payload'))
    if kind == 'unknown_ram':
        monkeypatch.setattr(module, 'available_memory_bytes', lambda: None)
    else:
        from types import SimpleNamespace
        monkeypatch.setattr(module.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=1))
    with pytest.raises((MemoryError, OSError)):
        run_single_frame_vortex(**run_args(case, tmp_path/'out'))
    assert not (tmp_path/'out').exists()


def test_single_frame_workflow_provenance_and_no_overwrite(tmp_path, monkeypatch):
    import nek_post.vortex_3d_workflow as module
    case, source = synthetic_file(tmp_path)
    original = source.read_bytes()
    report = run_single_frame_vortex(**run_args(case, tmp_path/'out'))
    assert source.read_bytes() == original
    assert report['source_unchanged'] and report['binary_vtu_roundtrip_verified']
    assert report['completion_status'] == 'completed'
    assert not report['visualization']['paraview_verified']
    assert not report['visualization']['tecplot_verified']
    paths = {k: Path(v) for k, v in report['output_paths'].items()}
    assert paths['vtu'].name == 'GC0_f00001.vtu'
    assert paths['vtu'].parent == tmp_path/'out'/'synthetic'/'f00001'
    assert json.loads(paths['metadata'].read_text()) == report
    assert len(list(paths['vtu'].parent.iterdir())) == 2
    grid = pv.read(paths['vtu'])
    assert (grid.n_points, grid.n_cells) == (54, 16)
    assert grid.field_data['TIME_VALUE'][0] == report['time']
    # Both fields use the same points and cells, not separately assembled files.
    assert set(grid.point_data) == {'omega_x', 'lambda_ci'}
    before = {key: p.read_bytes() for key, p in paths.items()}
    monkeypatch.setattr(module, 'read_nek_file', lambda *a, **k: pytest.fail('collision loaded payload'))
    with pytest.raises(FileExistsError): run_single_frame_vortex(**run_args(case, tmp_path/'out'))
    assert {key: p.read_bytes() for key, p in paths.items()} == before


@pytest.mark.parametrize('kind', ['source', 'repository', 'alias', 'dangling', 'frame_ancestor', 'file_ancestor'])
def test_unsafe_output_paths(tmp_path, kind):
    from nek_post.vortex_3d_workflow import REPOSITORY
    case, _ = synthetic_file(tmp_path)
    output = tmp_path/'bad'
    if kind == 'source': output = case/'new'
    if kind == 'repository': output = REPOSITORY/'new_export'
    if kind == 'alias': output.symlink_to(case, target_is_directory=True)
    if kind == 'dangling': output.symlink_to(tmp_path/'missing', target_is_directory=True)
    if kind == 'frame_ancestor':
        output.mkdir()
        (output/'saved.vtu').write_bytes(b'preserve')
        output = output/'nested'/'new'
    if kind == 'file_ancestor': output.write_text('preserve')
    with pytest.raises(ValueError): inspect_vortex_frame(**run_args(case, output))


@pytest.mark.parametrize('setting,value', [('index', 0), ('index', 100000), ('chunk_size', 0),
                                          ('periodic_axes', ('x', 'x')), ('expected_element_counts', (1, 1, 1))])
def test_bad_workflow_settings(tmp_path, setting, value):
    case, _ = synthetic_file(tmp_path)
    args = run_args(case, tmp_path/'out')
    args[setting] = value
    with pytest.raises(ValueError): inspect_vortex_frame(**args)


@pytest.mark.parametrize('failure', ['writer', 'source_change', 'publication_collision'])
def test_failed_export_preserves_staging_and_existing_output(tmp_path, monkeypatch, failure):
    import nek_post.vortex_3d_workflow as module
    case, source = synthetic_file(tmp_path)
    output = tmp_path/'out'
    original_write = module.write_binary_vtu
    def failing_write(path, grid, **kwargs):
        if failure == 'writer': raise OSError('simulated writer failure')
        result = original_write(path, grid, **kwargs)
        if failure == 'source_change':
            with source.open('ab') as stream: stream.write(b'changed')
        else:
            final = output/case.name/'f00001'
            final.mkdir()
            (final/'existing.txt').write_text('preserve')
        return result
    monkeypatch.setattr(module, 'write_binary_vtu', failing_write)
    with pytest.raises((OSError, RuntimeError)): run_single_frame_vortex(**run_args(case, output))
    stages = list((output/case.name).glob('.vortex-3d-*'))
    assert len(stages) == 1
    if failure == 'publication_collision':
        assert (output/case.name/'f00001'/'existing.txt').read_text() == 'preserve'
    else:
        assert not (output/case.name/'f00001').exists()


def test_cli_single_frame_synthetic(tmp_path, capsys):
    path = Path(__file__).resolve().parents[1]/'scripts/40_export_vortex_3d.py'
    spec = importlib.util.spec_from_file_location('vortex_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parse_args([]).index == 79
    for args in (['--index', '1,2'], ['--all-frames'], ['--overwrite'], ['--start-index', '1']):
        with pytest.raises(SystemExit): cli.parse_args(args)
    case, _ = synthetic_file(tmp_path)
    cli.main(['--case', str(case), '--index', '1', '--output-dir', str(tmp_path/'out'),
              '--expected-element-counts', '2', '1', '1', '--periodic-axes'])
    report = json.loads(capsys.readouterr().out)
    assert report['frame_index'] == 1 and report['completion_status'] == 'completed'
