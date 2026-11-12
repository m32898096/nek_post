"""Complete positive-x 3D volumes; synthetic Nek input only, no viewer filters."""
from itertools import permutations
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
import pytest

from test_l2_projection import mesh
from nek_post import vortex_3d_series as series
from nek_post import vortex_3d_workflow as single
from nek_post.io_nek import read_nek_file
from nek_post.vortex_3d import compute_native_vortex_fields, build_native_vortex_grid
from nek_post.vortex_3d_selection import (
    POSITIVE_X_BOUNDS, VolumeSelection, series_volume_selection, build_volume_output_plan,
    apply_volume_output_plan,
)


@pytest.fixture(autouse=True)
def ample_memory(monkeypatch):
    monkeypatch.setattr(single, 'available_memory_bytes', lambda: 10**15)


def volume_frames(tmp_path, *, edges=((-17., -3., 0., 2., 17.), (0., .5, 1.5), (0., .25, 1.)),
                  mapping=(2,1,0), reverse=True, shape=(8,8,8), times=(.03125, .278), name='case_N7_H'):
    from pymech.core import HexaData
    from pymech.neksuite import writenek
    geometry = mesh(edges, shape=shape, mapping=mapping, reverse=reverse)
    case = tmp_path/name
    case.mkdir()
    data = HexaData(3, len(geometry.elem), list(shape[::-1]), [3,3,1,1,0])
    data.wdsz, data.endian = 8, 'little'
    for index, timestamp in enumerate(times, 1):
        data.time, data.istep = timestamp, 13*index
        for element, original in zip(data.elem, geometry.elem):
            element.pos[:] = original.pos
            x, y, z = element.pos
            rate = index*(-4. if x.max() <= 0 else 2.)
            element.vel[:] = np.stack((.25*x*y, -rate*z+.2*y*z, rate*y+.3*z*x))
        writenek(str(case/f'GC0.f{index:05d}'), data)
    return case, tuple(len(e)-1 for e in edges), geometry


def run(case, counts, output, **kwargs):
    options = dict(indices=(1,), expected_element_counts=counts, periodic_axes=('x','y'))
    options.update(kwargs)
    return series.run_vortex_series(case, output, **options)


def saved_grid(result, frame=1):
    parent = Path(result['preflight']['output_root'])
    count = result['configuration'].get('sampling_nodes')
    suffix = f'_gll{count}' if count is not None and count > 8 else ''
    stem = parent/f'f{frame:05d}'/f'GC0_f{frame:05d}{suffix}'
    return pv.read(stem.with_suffix('.vtu')), json.loads(stem.with_suffix('.json').read_text())


@pytest.mark.parametrize('count', [None,8,10,12])
@pytest.mark.parametrize('mapping', list(permutations(range(3))))
def test_complete_positive_volume_matches_full_domain_at_identical_samples(tmp_path, count, mapping):
    case, counts, geometry = volume_frames(tmp_path, mapping=mapping, times=(.03125,))
    full_report = run(case, counts, tmp_path/'full', sampling_nodes=count, full_domain=True)
    half_report = run(case, counts, tmp_path/'positive', sampling_nodes=count)
    full, _ = saved_grid(full_report)
    positive, metadata = saved_grid(half_report)
    n = count or 8
    local_points, local_cells = n**3, (n-1)**3
    selected = [i for i, e in enumerate(geometry.elem) if e.pos[0].min() >= 0]
    assert isinstance(positive, pv.UnstructuredGrid)
    assert np.all(positive.celltypes == pv.CellType.HEXAHEDRON)
    assert positive.n_points == len(selected)*local_points
    assert positive.n_cells == len(selected)*local_cells
    assert list(positive.bounds) == list(POSITIVE_X_BOUNDS)
    assert set(positive.point_data) == {'omega_x','lambda_ci'}
    for name in positive.point_data:
        assert positive[name].dtype == np.float64
    cells = positive.cell_connectivity.reshape(-1,8)
    xyz = positive.points[cells]
    assert np.all(xyz[...,0] >= 0.) and np.all(xyz[...,0] <= 17.)
    assert np.all(xyz[...,1] >= 0.) and np.all(xyz[...,1] <= 1.5)
    assert np.all(xyz[...,2] >= 0.) and np.all(xyz[...,2] <= 1.)
    assert np.all(np.ptp(xyz, axis=1) > 0.)  # Every cell spans all three dimensions.
    assert np.sum(xyz[...,0].min(axis=1) == 0.) == counts[1]*counts[2]*(n-1)**2
    assert np.sum(positive.points[:,0] == 0.) == counts[1]*counts[2]*n**2
    full_cells = full.cell_connectivity.reshape(-1,8)
    for out_index, source_index in enumerate(selected):
        actual = slice(out_index*local_points,(out_index+1)*local_points)
        expected = slice(source_index*local_points,(source_index+1)*local_points)
        np.testing.assert_array_equal(positive.points[actual], full.points[expected])
        for name in ('omega_x','lambda_ci'):
            np.testing.assert_array_equal(positive[name][actual], full[name][expected])
        np.testing.assert_array_equal(
            cells[out_index*local_cells:(out_index+1)*local_cells]-out_index*local_points,
            full_cells[source_index*local_cells:(source_index+1)*local_cells]-source_index*local_points)
    np.testing.assert_array_equal(positive.field_data['TIME_VALUE'],full.field_data['TIME_VALUE'])
    assert metadata['output_selection']['mode'] == 'positive_x'
    assert metadata['output_selection']['retain_x0_boundary_vertices'] is True
    assert metadata['output_plan']['selected_element_count'] == len(selected)
    assert half_report['preflight']['estimated_ram_bytes'] < full_report['preflight']['estimated_ram_bytes']


@pytest.mark.parametrize('count',[8,10,12])
def test_positive_dry_run_counts_selected_mesh_and_reads_coordinates_only(tmp_path,monkeypatch,count):
    import nek_post.vortex_3d as numerical
    import nek_post.vortex_3d_reconstruction as reconstruction
    case, counts, geometry = volume_frames(tmp_path)
    calls = []
    original = single.read_nek_file
    def coordinates_only(*args, **kwargs):
        assert {'ux','uy','uz'} <= set(kwargs['skip_vars'])
        calls.append(args[0])
        return original(*args, **kwargs)
    def forbidden(*args,**kwargs): pytest.fail('dry-run computed a numerical field or 3D output mesh')
    monkeypatch.setattr(single, 'read_nek_file', coordinates_only)
    monkeypatch.setattr(numerical, 'element_velocity_gradient', forbidden)
    monkeypatch.setattr(reconstruction, 'reconstruct_element', forbidden)
    monkeypatch.setattr(single, 'build_native_vortex_grid', forbidden)
    monkeypatch.setattr(single, 'build_reconstructed_vortex_grid', forbidden)
    monkeypatch.setattr(series, 'validate_saved_grid', forbidden)
    destination = tmp_path/'output'
    report = run(case, counts, destination, indices=(1,2), sampling_nodes=count, dry_run=True)
    selected = sum(e.pos[0].min() >= 0 for e in geometry.elem)
    points, cells = selected*count**3, selected*(count-1)**3
    output_bytes = 40*points+80*cells+1024**2
    assert report['status'] == 'preflight_passed'
    assert len(calls) == 1  # One compact coordinate plan for the entire series.
    assert report['output_plan']['selected_local_point_count'] == points
    assert report['output_plan']['selected_linear_hex_count'] == cells
    assert report['required_disk_bytes'] == 3*(2*output_bytes+1024**2)+series.CONTROL_MARGIN
    assert all(f['estimated_output_bytes'] == output_bytes for f in report['frames'])
    assert not destination.exists()


@pytest.mark.parametrize('count',[8,10,12])
def test_only_positive_elements_are_reconstructed_but_all_native_gradients_are_used(tmp_path,monkeypatch,count):
    import nek_post.vortex_3d as numerical
    import nek_post.vortex_3d_reconstruction as reconstruction
    case, counts, geometry = volume_frames(tmp_path, times=(.03125,))
    raw, dense = [], []
    gradient, reconstruct = numerical.element_velocity_gradient, reconstruction.reconstruct_element
    def differentiate(element):
        raw.append(element.pos[0].min())
        return gradient(element)
    def sample(*args, **kwargs):
        dense.append(args[3])
        return reconstruct(*args, **kwargs)
    monkeypatch.setattr(numerical, 'element_velocity_gradient', differentiate)
    monkeypatch.setattr(reconstruction, 'reconstruct_element', sample)
    run(case, counts, tmp_path/'out', sampling_nodes=count)
    selected = sum(e.pos[0].min() >= 0 for e in geometry.elem)
    assert len(raw) == len(geometry.elem) and min(raw) < 0
    assert len(dense) == (0 if count == 8 else selected)


def test_x0_values_include_negative_side_projection_contributions(tmp_path):
    case, counts, _ = volume_frames(tmp_path, times=(.03125,))
    data = read_nek_file(case/'GC0.f00001')
    full = compute_native_vortex_fields(data, periodic_axes=(), expected_element_counts=counts)
    positive = build_native_vortex_grid(data, full, crop=POSITIVE_X_BOUNDS)
    trimmed = SimpleNamespace(elem=[e for e in data.elem if e.pos[0].min() >= 0])
    wrong_fields = compute_native_vortex_fields(trimmed, periodic_axes=(), expected_element_counts=(2,2,2))
    wrong = build_native_vortex_grid(trimmed, wrong_fields)
    for name in ('omega_x','lambda_ci'):
        boundary = positive.points[:,0] == 0.
        assert np.max(np.abs(positive[name][boundary]-wrong[name][boundary])) > .2


def test_unsafe_dense_target_fails_before_coordinate_or_target_gll_planning(tmp_path,monkeypatch):
    import nek_post.vortex_3d_reconstruction as reconstruction
    case, counts, _ = volume_frames(tmp_path,times=(.03125,))
    monkeypatch.setattr(single,'available_memory_bytes',lambda:8*1024**3)
    def forbidden(*args,**kwargs): pytest.fail('unsafe target reached coordinate/GLL planning')
    monkeypatch.setattr(single,'read_nek_file',forbidden)
    monkeypatch.setattr(reconstruction,'gll_interpolation_matrix',forbidden)
    output=tmp_path/'out'
    report=run(case,counts,output,sampling_nodes=10000,dry_run=True)
    assert report['status'] == 'invalid'
    assert any('Insufficient RAM for coordinate-only crop planning' in e for e in report['failures'])
    assert not output.exists()


@pytest.mark.parametrize('count',[None,8,10,12])
def test_missing_x0_element_boundary_is_reported_before_gradients(tmp_path,monkeypatch,count):
    case, counts, _ = volume_frames(tmp_path, edges=((-17.,-2.,17.),(0.,1.5),(0.,1.)), times=(.03125,))
    import nek_post.vortex_3d as numerical
    monkeypatch.setattr(numerical,'element_velocity_gradient',lambda *a:pytest.fail('differentiated unsupported positive crop'))
    out = tmp_path/'out'
    report = run(case, counts, out, sampling_nodes=count, dry_run=True)
    assert report['status'] == 'invalid'
    assert any('x=0 is not an element boundary' in e for e in report['failures'])
    with pytest.raises(ValueError,match='x=0 is not an element boundary'):
        run(case, counts, out, sampling_nodes=count)
    assert not out.exists()


@pytest.mark.parametrize('edges', [((-17.,0.,16.),(0.,1.5),(0.,1.)),
    ((-17.,0.,17.),(0.,2.),(0.,1.)), ((-17.,0.,17.),(0.,1.5),(0.,2.))])
def test_incompatible_physical_extent_is_not_silently_truncated(tmp_path,edges):
    case, counts, _ = volume_frames(tmp_path, edges=edges,times=(.03125,))
    report = run(case, counts, tmp_path/'out', dry_run=True)
    assert report['status'] == 'invalid'
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('count',[8,10,12])
def test_positive_region_is_preserved_by_pvd_and_strict_resume(tmp_path,monkeypatch,count):
    case, counts, _ = volume_frames(tmp_path)
    output = tmp_path/'series'
    first = run(case, counts, output, indices=(1,),sampling_nodes=count)
    root = Path(first['preflight']['output_root'])
    original = {p:single._file_hash(p) for p in (root/'f00001').iterdir()}
    resumed = run(case, counts, output, indices=(1,2),sampling_nodes=count,resume=True)
    assert [f['action'] for f in resumed['frames']] == ['reused','generated']
    assert {p:single._file_hash(p) for p in original} == original
    assert resumed['configuration']['crop'] == list(POSITIVE_X_BOUNDS)
    entries = ET.parse(resumed['preflight']['pvd_path']).getroot()[0]
    assert [float(e.get('timestep')) for e in entries] == [.03125,.278]
    for entry in entries:
        path = Path(resumed['preflight']['pvd_path']).parent/entry.get('file')
        grid = pv.read(path)
        assert isinstance(grid,pv.UnstructuredGrid)
        assert list(grid.bounds) == list(POSITIVE_X_BOUNDS)
        assert grid['omega_x'].dtype == grid['lambda_ci'].dtype == np.float64
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['output_selection'] == resumed['output_selection']
    manifest = json.loads((root/'series'/'manifest.json').read_text())
    assert manifest['output_selection'] == resumed['output_selection']
    with pytest.raises(ValueError,match='[Ii]ncompatible|foreign'):
        run(case, counts, output, sampling_nodes=count, full_domain=True, resume=True)
    monkeypatch.setattr(series,'validate_saved_grid',lambda *a:pytest.fail('dry-run decoded an output volume'))
    report = run(case, counts, output, indices=(1,2),sampling_nodes=count,resume=True,dry_run=True)
    assert report['status'] == 'preflight_passed'
    assert all(f['payload_validation'] == 'pending' for f in report['frames'])


@pytest.mark.parametrize('count',[8,10,12])
def test_coordinate_plan_is_reused_but_each_frame_projects_the_full_source(tmp_path,monkeypatch,count):
    case, counts, _ = volume_frames(tmp_path)
    original = single.read_nek_file
    calls=[]
    def observe(*args,**kwargs):
        calls.append({'ux','uy','uz'} <= set(kwargs['skip_vars']))
        return original(*args,**kwargs)
    monkeypatch.setattr(single,'read_nek_file',observe)
    result = run(case,counts,tmp_path/'out',indices=(1,2),sampling_nodes=count)
    assert result['status'] == 'completed'
    assert calls == [True,False,False]  # One coordinate plan, then two whole snapshots.


def test_crop_estimate_uses_actual_element_counts_not_an_assumed_half(tmp_path):
    edges = ((-17.,-2.,0.,1.,3.,17.),(0.,.5,1.5),(0.,.25,1.))
    case, counts, _ = volume_frames(tmp_path,edges=edges,times=(.03125,))
    report = run(case,counts,tmp_path/'out',sampling_nodes=12,dry_run=True)
    plan = report['output_plan']
    assert plan['selected_element_count'] == 3*2*2
    assert plan['selected_local_point_count'] == 12*12**3
    assert plan['selected_linear_hex_count'] == 12*11**3
    original = single.inspect_vortex_frame(case,1,tmp_path/'full',sampling_nodes=12,
                                           expected_element_counts=counts,periodic_axes=('x','y'))
    selected_bytes = 40*12*12**3+80*12*11**3+1024**2
    assert report['frames'][0]['estimated_output_bytes'] == selected_bytes
    assert report['estimated_ram_bytes'] == original['estimated_ram_bytes']-6*(original['estimated_output_bytes']-selected_bytes)
    assert original['native_local_point_count'] == prod(counts)*8**3
    config = single._configuration(('x','y'), counts, POSITIVE_X_BOUNDS, 256, 12)
    for available, required, selected_required in (
            ('available_ram_bytes', 'estimated_ram_bytes', report['estimated_ram_bytes']),
            ('available_disk_bytes', 'required_disk_bytes', 2*selected_bytes+1024**2)):
        budget = dict(original, available_ram_bytes=10**15, available_disk_bytes=10**15,
                      resources_acceptable=False)
        budget[available] = (original[required]+selected_required)//2
        apply_volume_output_plan(budget, plan, config, VolumeSelection('positive_x', POSITIVE_X_BOUNDS))
        assert budget['resources_acceptable'] is True  # Recompute the status after reducing output estimates.


def test_explicit_positive_crop_resumes_the_same_default_region(tmp_path):
    case, counts, _ = volume_frames(tmp_path,times=(.03125,))
    output = tmp_path/'out'
    initial = run(case,counts,output,sampling_nodes=10)
    resumed = run(case,counts,output,sampling_nodes=10,crop=POSITIVE_X_BOUNDS,resume=True)
    assert initial['output_selection'] == resumed['output_selection']
    assert resumed['frames'][0]['action'] == 'reused'


def test_generic_case_opt_in_and_single_frame_crop_remain_backward_compatible(tmp_path):
    case, counts, _ = volume_frames(tmp_path, times=(.03125,),name='different_case')
    full = run(case, counts, tmp_path/'full')
    assert full['output_selection']['mode'] == 'full'
    half = run(case, counts, tmp_path/'half',positive_x=True)
    assert half['output_selection']['mode'] == 'positive_x'
    # A legacy arbitrary crop is still a complete-subcell crop, without imposing
    # the new series preset's SEM-boundary requirement on the single-frame CLI.
    original = single.run_single_frame_vortex(case,1,tmp_path/'single',crop=(.1,16.9,0.,1.5,0.,1.),
                                             expected_element_counts=counts,periodic_axes=('x','y'))
    saved = pv.read(original['output_paths']['vtu'])
    assert saved.bounds.x_min >= .1 and saved.bounds.x_max <= 16.9
    assert saved.n_cells > 0 and saved.bounds.z_max == 1.


@pytest.mark.parametrize('shape,mapping', [((3,4,5),(2,1,0)),((4,3,5),(0,2,1))])
def test_native_non_p7_crop_estimates_use_original_coordinate_ordering(tmp_path,shape,mapping):
    case, counts, geometry = volume_frames(tmp_path,shape=shape,mapping=mapping,times=(.03125,))
    result = run(case,counts,tmp_path/'out')
    grid, metadata = saved_grid(result)
    selected = sum(e.pos[0].min() >= 0 for e in geometry.elem)
    assert grid.n_points == selected*prod(shape)
    assert grid.n_cells == selected*prod(n-1 for n in shape)
    assert metadata['selected_local_point_count'] == grid.n_points
    assert list(grid.bounds) == list(POSITIVE_X_BOUNDS)


@pytest.mark.parametrize('flags',[{'positive_x':True,'full_domain':True}, {'positive_x':True,'crop':POSITIVE_X_BOUNDS},
                                {'full_domain':True,'crop':POSITIVE_X_BOUNDS}])
def test_conflicting_volume_flags_are_rejected_before_input_io(tmp_path,flags):
    with pytest.raises(ValueError,match='Choose'):
        series.run_vortex_series(tmp_path/'missing_case',tmp_path/'out',indices=(1,),**flags)
    assert not (tmp_path/'out').exists()
