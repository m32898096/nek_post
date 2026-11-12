"""Generated tiny Nek snapshots, real VTUs, strict resume and injected crashes."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import threading
import weakref
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
import pytest

from test_lambda_ci_workflow import synthetic_file
from nek_post.io_nek import read_nek_file
from nek_post import vortex_3d_series as series
from nek_post import vortex_3d_workflow as single
from nek_post.vortex_3d_artifacts import grid_fingerprints, binary_vtu_fingerprint


@pytest.fixture(autouse=True)
def ample_memory(monkeypatch):
    monkeypatch.setattr(single, 'available_memory_bytes', lambda: 10**15)


def frames(tmp_path, *, times=(.03125, .278, .903), shape=(8, 8, 8)):
    from pymech.neksuite import writenek
    case, source = synthetic_file(tmp_path, shape=shape)
    data = read_nek_file(source)
    data.wdsz = 8
    for index, value in enumerate(times, 1):
        data.time, data.istep = value, 13*index
        for ei, e in enumerate(data.elem):
            x, y, z = e.pos
            rate = index*(1.25 if ei == 0 else -2.5)
            e.vel[:] = np.stack((x*x*y, -rate*z+x*y, rate*y+z*x))
        writenek(str(case/f'GC0.f{index:05d}'), data)
    return case, tmp_path/'results'


def run(case, output, **kwargs):
    options = dict(indices=(1, 2, 3), periodic_axes=('x', 'y'), expected_element_counts=(2, 1, 1))
    options.update(kwargs)
    return series.run_vortex_series(case, output, **options)


def root(case, output, count=None):
    return (output/f'gll{count}' if count is not None and count > 8 else output)/case.name


def frame_paths(case, output, index=1, count=None):
    name = f'GC0_f{index:05d}' + (f'_gll{count}' if count is not None and count > 8 else '')
    folder = root(case, output, count)/f'f{index:05d}'
    return {'vtu': folder/(name+'.vtu'), 'metadata': folder/(name+'.json')}


def hashes(paths):
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}


@pytest.mark.parametrize('options,expected', [
    ({}, tuple(range(1,82))), ({'indices':'3,1'}, (1,3)), ({'indices':'2-4,6'}, (2,3,4,6)),
    ({'indices':iter((2,1))}, (1,2)), ({'start_index':3,'end_index':3}, (3,)),
    ({'start_index':1,'end_index':4,'max_frames':2}, (1,2)), ({'indices':'99999'}, (99999,)),
])
def test_frame_selection(options, expected):
    assert series.frame_indices(**options) == expected


@pytest.mark.parametrize('options', [
    {'indices':'1,1'}, {'indices':'1-3,2'}, {'indices':'0'}, {'indices':'100000'},
    {'indices':'1-1000000000000'}, {'indices':'4-2'}, {'indices':'1,,2'}, {'indices':''},
    {'indices':[True]}, {'indices':[1.]}, {'indices':[]}, {'indices':1},
    {'start_index':2}, {'end_index':3}, {'start_index':True,'end_index':3},
    {'indices':'1','start_index':1,'end_index':2}, {'indices':'1','max_frames':0},
    {'indices':'1','max_frames':True}, {'indices':'1,1','max_frames':1},
])
def test_invalid_selection_is_rejected_before_expensive_work(options):
    with pytest.raises(ValueError):
        series.frame_indices(**options)


def test_dry_run_reads_only_headers_maps_and_checksums_and_creates_nothing(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    def forbidden(*a, **kw): pytest.fail('dry run decoded or computed a payload')
    monkeypatch.setattr(single, 'read_nek_file', forbidden)
    monkeypatch.setattr(series, 'validate_saved_grid', forbidden)
    before = hashes(case.iterdir())
    report = run(case, output, dry_run=True, indices='1,3', sampling_nodes=12)
    assert report['status'] == 'preflight_passed'
    assert report['selected_indices'] == [1,3]
    assert [f['time'] for f in report['frames']] == [.03125,.903]
    assert [f['action'] for f in report['frames']] == ['generate','generate']
    assert report['configuration']['sampling_nodes'] == 12
    assert not output.exists()
    assert hashes(case.iterdir()) == before


def test_missing_snapshot_and_default_81_selection_are_reported_read_only(tmp_path):
    case, output = frames(tmp_path)
    report = run(case, output, indices=None, dry_run=True)
    assert report['status'] == 'invalid' and len(report['frames']) == 81
    assert sum(not f['input_available'] for f in report['frames']) == 78
    with pytest.raises(ValueError, match='preflight failed'):
        run(case, output, indices=(1,4))
    assert not output.exists()


@pytest.mark.parametrize('times', [(0.,0.,1.), (0.,2.,1.)])
def test_duplicate_or_decreasing_physical_times_fail_before_outputs(tmp_path, times):
    case, output = frames(tmp_path, times=times)
    report = run(case, output, dry_run=True)
    assert any('timestamps' in f for f in report['failures'])
    with pytest.raises(ValueError, match='timestamps'):
        run(case, output)
    assert not output.exists()


@pytest.mark.parametrize('count', [None, 8, 10, 12])
def test_series_equals_single_frame_with_actual_time_and_binary_float64(tmp_path, count):
    case, output = frames(tmp_path)
    result = run(case, output, indices=(1,3), sampling_nodes=count, crop=(0.,1.8,0.,1.,0.,1.))
    assert result['status'] == 'completed' and result['pvd_entries'] == 2
    collection = ET.parse(result['preflight']['pvd_path']).getroot()[0]
    assert [float(e.get('timestep')) for e in collection] == [.03125,.903]
    for index, entry in zip((1,3), collection):
        paths = frame_paths(case, output, index, count)
        assert (Path(result['preflight']['pvd_path']).parent/entry.get('file')).resolve() == paths['vtu']
        reference = single.run_single_frame_vortex(case, index, tmp_path/f'single_{index}',
            sampling_nodes=count, periodic_axes=('x','y'), expected_element_counts=(2,1,1), crop=(0.,1.8,0.,1.,0.,1.))
        a, b = pv.read(paths['vtu']), pv.read(reference['output_paths']['vtu'])
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.cell_connectivity, b.cell_connectivity)
        for name in ('omega_x','lambda_ci'):
            assert a[name].dtype == np.float64
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(a.field_data['TIME_VALUE'], b.field_data['TIME_VALUE'])
        metadata = json.loads(paths['metadata'].read_text())
        assert metadata['array_sha256'] == grid_fingerprints(a)
        assert metadata['native_mesh_signature'] == reference['native_mesh_signature']
        if count is not None:
            assert metadata['reconstruction']['omega_route_audit_performed'] is (count > 8)
        assert 'format="binary"' in paths['vtu'].read_text()


def test_strict_resume_preserves_existing_artifacts_and_reconciles_nonconsecutive_pvd(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,3), sampling_nodes=10)
    protected = hashes([p for i in (1,3) for p in frame_paths(case,output,i,10).values()])
    calls = []
    original = single.run_single_frame_vortex
    def observe(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(single, 'run_single_frame_vortex', observe)
    report = run(case, output, indices=(1,2,3), sampling_nodes=10, resume=True)
    assert calls == [2]
    assert [f['action'] for f in report['frames']] == ['reused','generated','reused']
    assert hashes(Path(p) for p in protected) == protected
    entries = ET.parse(report['preflight']['pvd_path']).getroot()[0]
    assert [float(e.get('timestep')) for e in entries] == [.03125,.278,.903]
    before = hashes(p for p in root(case,output,10).rglob('*') if p.is_file())
    with pytest.raises(ValueError, match='--resume'):
        run(case, output, sampling_nodes=10)
    assert hashes(Path(p) for p in before) == before


def test_resume_dry_run_defers_payload_validation_and_changes_no_outputs(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    run(case, output)
    protected = hashes(p for p in output.rglob('*') if p.is_file())
    monkeypatch.setattr(series, 'validate_saved_grid', lambda *a: pytest.fail('dry run read VTU payload'))
    result = run(case, output, resume=True, dry_run=True)
    assert all(f['action'] == 'validate_then_skip' and f['payload_validation'] == 'pending' for f in result['frames'])
    assert hashes(Path(p) for p in protected) == protected


def test_series_default_root_and_explicit_output_parent(tmp_path, monkeypatch):
    assert series.DEFAULT_OUTPUT_ROOT == Path(
        '/data/Nek5000_data/results/h_refinement/vortex_3d_series')
    case, output = frames(tmp_path)
    default = tmp_path/'h_refinement/vortex_3d_series'
    monkeypatch.setattr(series, 'DEFAULT_OUTPUT_ROOT', default)
    report = run(case, None, indices=(1,), sampling_nodes=10, dry_run=True)
    assert Path(report['output_root']) == root(case, default, 10)
    custom = run(case, output, indices=(1,), sampling_nodes=10, dry_run=True)
    assert Path(custom['output_root']) == root(case, output, 10)
    assert not default.exists() and not output.exists()


@pytest.mark.parametrize('count', [None, 10])
def test_relocated_series_resumes_after_audited_metadata_rebase(tmp_path, monkeypatch, count):
    case, output = frames(tmp_path)
    run(case, output, indices=(1, 3), sampling_nodes=count)
    original_vtus = {i: frame_paths(case, output, i, count)['vtu'].read_bytes() for i in (1, 3)}
    original_pvd = next(output.rglob('*.pvd')).read_bytes()
    relocated = tmp_path/'h_refinement/vortex_3d_series'
    relocated.parent.mkdir()
    output.rename(relocated)
    failed = run(case, relocated, indices=(1, 3), sampling_nodes=count, resume=True, dry_run=True)
    assert failed['status'] == 'invalid'
    assert any('output_paths' in failure for failure in failed['failures'])

    controls = root(case, relocated, count)/'series'
    manifest_path = controls/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for index in (1, 3):
        paths = frame_paths(case, relocated, index, count)
        original = paths['metadata'].read_bytes()
        report = json.loads(original)
        report['output_relocation'] = dict(original_metadata_sha256=sha256(original).hexdigest(),
                                         original_output_paths=report['output_paths'])
        report['output_directory'] = str(paths['vtu'].parent)
        report['output_paths'] = {key: str(path) for key, path in paths.items()}
        paths['metadata'].write_text(json.dumps(report, indent=2)+'\n')
        manifest['frames'][str(index)]['metadata_sha256'] = sha256(paths['metadata'].read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    summary_path = controls/'processing_summary.json'
    summary = json.loads(summary_path.read_text())
    for key in ('log_path', 'summary_path'):
        summary[key] = str(relocated/Path(summary[key]).relative_to(output))
    preflight = summary['preflight']
    for key in ('output_root', 'pvd_path'):
        preflight[key] = str(relocated/Path(preflight[key]).relative_to(output))
    for frame in preflight['frames']:
        frame['output_paths'] = {key: str(relocated/Path(path).relative_to(output))
                                 for key, path in frame['output_paths'].items()}
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')

    def forbidden(*args, **kwargs):
        pytest.fail('relocation regenerated a completed scientific frame')
    monkeypatch.setattr(single, 'run_single_frame_vortex', forbidden)
    protected = hashes(p for p in relocated.rglob('*') if p.parent.name.startswith('f') and p.is_file())
    result = run(case, relocated, indices=(1, 3), sampling_nodes=count, resume=True)
    assert result['status'] == 'completed'
    assert [frame['action'] for frame in result['frames']] == ['reused', 'reused']
    assert result['preflight']['existing_indices'] == [1, 3]
    assert next(relocated.rglob('*.pvd')).read_bytes() == original_pvd
    assert hashes(Path(path) for path in protected) == protected
    assert not output.exists()
    for index, payload in original_vtus.items():
        assert frame_paths(case, relocated, index, count)['vtu'].read_bytes() == payload

    # Rebased paths still require the original numerical provenance and checksums.
    paths = frame_paths(case, relocated, 1, count)
    report = json.loads(paths['metadata'].read_text())
    report['numerical_software_signature']['vortex_3d.py'] = '0'*64
    paths['metadata'].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='signature'):
        run(case, relocated, indices=(1, 3), sampling_nodes=count, resume=True)


@pytest.mark.parametrize('damage', ['vtu','metadata_missing','metadata_json','time','index','config',
    'source_hash','implementation','arrays','completion','mesh','metadata_bool','software','scipy','python'])
def test_corrupted_or_incompatible_completed_frames_never_reused(tmp_path, damage):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    paths = frame_paths(case,output)
    if damage == 'vtu':
        with paths['vtu'].open('ab') as stream: stream.write(b'changed')
    elif damage == 'metadata_missing':
        paths['metadata'].rename(tmp_path/'preserved_metadata.json')
    elif damage == 'metadata_json':
        paths['metadata'].write_text('{broken')
    else:
        data = json.loads(paths['metadata'].read_text())
        if damage == 'time': data['time'] = .5
        if damage == 'index': data['frame_index'] = 2
        if damage == 'config': data['configuration']['periodic_axes'] = []
        if damage == 'source_hash': data['source_sha256'] = '0'*64
        if damage == 'implementation': data['numerical_software_signature']['gll.py'] = '0'*64
        if damage == 'arrays': data['array_sha256']['omega_x'] = '0'*64
        if damage == 'completion': data['completion_status'] = 'running'
        if damage == 'mesh': data['native_mesh_signature']['element_shape'] = [3,3,3]
        if damage == 'metadata_bool': data['frame_index'] = True
        if damage == 'software': data['software']['vtk'] = 'changed'
        if damage == 'scipy': data['software']['scipy'] = 'changed'
        if damage == 'python': data['python'] = 'changed'
        paths['metadata'].write_text(json.dumps(data))
    protected = hashes(p for p in output.rglob('*') if p.is_file())
    with pytest.raises(ValueError): run(case, output, indices=(1,), resume=True)
    assert hashes(Path(p) for p in protected) == protected


@pytest.mark.parametrize('setting,value', [('sampling_nodes',10), ('crop',(0.,1.,0.,1.,0.,1.)),
                                         ('periodic_axes',()), ('chunk_size',1)])
def test_configuration_changes_cannot_reuse_existing_series(tmp_path, setting, value):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    # Sampling densities intentionally use independent roots, so test metadata
    # incompatibility by attempting to adopt an existing native frame explicitly.
    if setting == 'sampling_nodes':
        value = 8
    with pytest.raises(ValueError, match='[Ii]ncompatible|differs'):
        run(case, output, indices=(1,), resume=True, **{setting:value})


def test_changed_source_or_series_signature_stops_resume(tmp_path, monkeypatch):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    original = series._signature
    monkeypatch.setattr(series, '_signature', lambda: {**original(), 'vortex_3d_series.py':'0'*64})
    with pytest.raises(ValueError, match='[Ii]ncompatible'):
        run(case, output, indices=(1,), resume=True)
    monkeypatch.setattr(series, '_signature', original)
    with (case/'GC0.f00001').open('ab') as stream: stream.write(b'changed source')
    with pytest.raises(ValueError, match='source|Source'):
        run(case, output, indices=(1,), resume=True)


def test_changed_full_native_geometry_fails_before_second_frame_gradients(tmp_path,monkeypatch):
    import nek_post.vortex_3d as numerical
    from pymech.neksuite import writenek
    case, output = frames(tmp_path)
    path = case/'GC0.f00002'
    data = read_nek_file(path)
    for element in data.elem: element.pos[0] *= 2
    writenek(str(path),data)
    calls=[]
    original = numerical.element_velocity_gradient
    def observe(element):
        calls.append(element)
        return original(element)
    monkeypatch.setattr(numerical,'element_velocity_gradient',observe)
    with pytest.raises(ValueError, match='geometry or element ordering'):
        run(case, output, indices=(1,2), crop=(0.,1.,0.,1.,0.,1.))
    assert frame_paths(case,output,1)['vtu'].exists()
    assert not frame_paths(case,output,2)['vtu'].parent.exists()
    assert len(calls) == 2  # Only the first frame's two elements were differentiated.


def test_resume_validates_actual_cells_even_with_consistent_modified_checksums(tmp_path):
    case, output = frames(tmp_path)
    run(case, output, indices=(1,))
    paths = frame_paths(case,output)
    data = pv.read(paths['vtu'])
    data.points[:,0] *= -1
    data.save(paths['vtu'],binary=True)
    meta = json.loads(paths['metadata'].read_text())
    meta['array_sha256'] = grid_fingerprints(data)
    meta['output_sha256']['vtu'] = single._file_hash(paths['vtu'])
    meta['visualization']['bounds'] = list(data.bounds)
    paths['metadata'].write_text(json.dumps(meta))
    controls = series._control_files(root(case,output))
    controls['manifest'].rename(tmp_path/'preserved_manifest.json')
    with pytest.raises(ValueError, match='Jacobian|vertices|degenerate'):
        run(case, output, indices=(1,), resume=True)


@pytest.mark.parametrize('stage_kind', ['incomplete','complete'])
def test_interrupted_writer_or_publication_recovers_without_overwrite(tmp_path, monkeypatch, stage_kind):
    case, output = frames(tmp_path)
    original_writer, original_publish = single.write_binary_vtu, single._publish_directory
    if stage_kind == 'incomplete':
        def fail(path, *args, **kwargs):
            with Path(path).open('xb') as stream: stream.write(b'partial synthetic output')
            raise OSError('writer interruption')
        monkeypatch.setattr(single,'write_binary_vtu',fail)
    else:
        monkeypatch.setattr(single,'_publish_directory',lambda *a: (_ for _ in ()).throw(OSError('publication interruption')))
    with pytest.raises(OSError): run(case,output,indices=(1,))
    assert not frame_paths(case,output)['vtu'].parent.exists()
    stages = list(root(case,output).glob('.vortex-3d-f00001-*'))
    assert len(stages) == 1
    protected = hashes(p for p in stages[0].iterdir())
    with pytest.raises(ValueError, match='staging'):
        run(case,output,indices=(1,),resume=True)
    monkeypatch.setattr(single,'write_binary_vtu',original_writer)
    monkeypatch.setattr(single,'_publish_directory',original_publish)
    if stage_kind == 'complete':
        monkeypatch.setattr(single,'run_single_frame_vortex',lambda *a,**kw:pytest.fail('recomputed complete staging'))
    report = run(case,output,indices=(1,),resume=True,recover_staging=True)
    assert report['frames'][0]['action'] == ('recovered' if stage_kind == 'complete' else 'generated')
    if stage_kind == 'incomplete':
        assert hashes(Path(p) for p in protected) == protected


@pytest.mark.parametrize('kind', ['pvd','manifest','pvd_replace'])
def test_published_frames_reconciled_after_interrupted_controls(tmp_path, monkeypatch, kind):
    case, output = frames(tmp_path)
    if kind == 'pvd_replace': run(case,output,indices=(1,))
    pvd = series._control_files(root(case,output))['pvd']
    old = pvd.read_bytes() if pvd.exists() else None
    original_pvd, original_json, original_replace = series._publish_pvd, series._atomic_json, series.os.replace
    if kind == 'pvd':
        monkeypatch.setattr(series,'_publish_pvd',lambda *a:(_ for _ in ()).throw(OSError('PVD interruption')))
    elif kind == 'manifest':
        def fail(path,*args):
            if Path(path).name == 'manifest.json': raise OSError('checkpoint interruption')
            return original_json(path,*args)
        monkeypatch.setattr(series,'_atomic_json',fail)
    else:
        def fail(src,dst):
            if Path(dst).suffix == '.pvd': raise OSError('atomic PVD replacement interruption')
            return original_replace(src,dst)
        monkeypatch.setattr(series.os,'replace',fail)
    index = 2 if kind == 'pvd_replace' else 1
    with pytest.raises(OSError): run(case,output,indices=(index,))
    assert frame_paths(case,output,index)['vtu'].exists()
    if kind != 'manifest': assert (pvd.read_bytes() if pvd.exists() else None) == old
    protected = hashes(frame_paths(case,output,index).values())
    monkeypatch.setattr(series,'_publish_pvd',original_pvd)
    monkeypatch.setattr(series,'_atomic_json',original_json)
    monkeypatch.setattr(series.os,'replace',original_replace)
    monkeypatch.setattr(single,'run_single_frame_vortex',lambda *a,**kw:pytest.fail('recomputed published frame'))
    report = run(case,output,indices=(index,),resume=True)
    assert report['status'] == 'completed' and report['pvd_entries'] == index
    assert hashes(Path(p) for p in protected) == protected


@pytest.mark.parametrize('kind',['duplicate_time','duplicate_file','wrong_time','foreign_file','absolute','unsorted'])
def test_invalid_pvd_is_preserved_and_rejected(tmp_path,kind):
    case, output = frames(tmp_path)
    run(case,output)
    pvd = series._control_files(root(case,output))['pvd']
    tree = ET.parse(pvd); entries = tree.getroot()[0]
    if kind == 'duplicate_time': entries[1].set('timestep',entries[0].get('timestep'))
    if kind == 'duplicate_file': entries[1].set('file',entries[0].get('file'))
    if kind == 'wrong_time': entries[0].set('timestep','.123')
    if kind == 'foreign_file': entries[0].set('file',str(Path('../../../')/'GC0.f00001'))
    if kind == 'absolute': entries[0].set('file',str(frame_paths(case,output)['vtu']))
    if kind == 'unsorted': entries[:] = list(entries)[::-1]
    tree.write(pvd)
    original = pvd.read_bytes()
    with pytest.raises(ValueError): run(case,output,resume=True)
    assert pvd.read_bytes() == original


@pytest.mark.parametrize('kind',['ram','disk','file_size'])
def test_resource_gates_precede_payload_reads_and_outputs(tmp_path, monkeypatch,kind):
    case, output = frames(tmp_path)
    if kind == 'ram': monkeypatch.setattr(single,'available_memory_bytes',lambda:1)
    if kind == 'disk': monkeypatch.setattr(series.shutil,'disk_usage',lambda p:type('D',(),{'free':1})())
    if kind == 'file_size': monkeypatch.setattr(series,'_filesystem_limits',lambda p:{'effective_file_size_limit_bytes':1})
    monkeypatch.setattr(single,'read_nek_file',lambda *a,**kw:pytest.fail('resource failure decoded source'))
    monkeypatch.setattr(series,'validate_saved_grid',lambda *a:pytest.fail('resource failure decoded VTU'))
    report = run(case,output,dry_run=True)
    assert report['status'] == 'invalid'
    with pytest.raises(ValueError,match='[Ii]nsufficient|file-size'):
        run(case,output)
    assert not output.exists()


def test_disk_and_ram_estimates_and_max_frames_are_conservative(tmp_path):
    case, output = frames(tmp_path)
    report = run(case,output,dry_run=True,sampling_nodes=12)
    header = single.inspect_vortex_frame(case,1,output,periodic_axes=('x','y'),expected_element_counts=(2,1,1),sampling_nodes=12)
    assert report['required_disk_bytes'] == 4*header['required_disk_bytes']+series.CONTROL_MARGIN
    assert report['estimated_ram_bytes'] == header['estimated_ram_bytes']
    assert run(case,output,max_frames=1,dry_run=True)['selected_indices'] == [1]


@pytest.mark.parametrize('location',['repository','raw','symlink','control_symlink','frame_ancestor'])
def test_output_path_safety(tmp_path,location):
    case, output = frames(tmp_path)
    if location == 'repository': output = single.REPOSITORY/'forbidden-series-output'
    if location == 'raw': output = case/'generated'
    if location == 'symlink':
        output.symlink_to(case,target_is_directory=True)
    if location == 'control_symlink':
        folder = root(case,output); folder.mkdir(parents=True)
        (folder/'series').symlink_to(case,target_is_directory=True)
    if location == 'frame_ancestor':
        run(case,output,indices=(1,))
        output = frame_paths(case,output)['vtu'].parent/'nested'
    with pytest.raises(ValueError): run(case,output,indices=(1,),dry_run=True)


def test_frame_objects_are_released_between_calls(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    refs = []
    original_write, original_run = single.write_binary_vtu, single.run_single_frame_vortex
    def write(path,grid,**kw):
        refs.append(weakref.ref(grid))
        return original_write(path,grid,**kw)
    def produce(*args,**kw):
        assert all(ref() is None for ref in refs)
        return original_run(*args,**kw)
    monkeypatch.setattr(single,'write_binary_vtu',write)
    monkeypatch.setattr(single,'run_single_frame_vortex',produce)
    run(case,output,sampling_nodes=10)
    assert len(refs) == 3 and all(ref() is None for ref in refs)


def test_competing_writers_refresh_after_persistent_lock(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    entered, release, second_preflight = threading.Event(), threading.Event(), threading.Event()
    original_preflight, original_run = series._preflight, single.run_single_frame_vortex
    initial_count = 0
    generated = []
    def inspect(*args,**kw):
        nonlocal initial_count
        result = original_preflight(*args,**kw)
        if not kw['deep']:
            initial_count += 1
            if initial_count == 2: second_preflight.set()
        return result
    def produce(*args,**kw):
        generated.append(args[1])
        return original_run(*args,**kw)
    def progress(message):
        if json.loads(message)['message'] == 'Frame started':
            entered.set()
            assert release.wait(10)
    monkeypatch.setattr(series,'_preflight',inspect)
    monkeypatch.setattr(single,'run_single_frame_vortex',produce)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run,case,output,indices=(1,),resume=True,progress=progress)
        assert entered.wait(10)
        lock = root(case,output)/'.vortex-3d-series.lock'
        inode = lock.stat().st_ino
        second = pool.submit(run,case,output,indices=(1,),resume=True)
        try:
            assert second_preflight.wait(10)
        finally:
            release.set()
        assert first.result(15)['status'] == second.result(15)['status'] == 'completed'
    assert generated == [1] and lock.stat().st_ino == inode


def test_stale_lock_inode_reused_and_released_after_failure(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    folder = root(case,output); folder.mkdir(parents=True)
    lock = folder/'.vortex-3d-series.lock'; lock.write_text('inactive historical PID\n')
    inode = lock.stat().st_ino
    original = single.run_single_frame_vortex
    monkeypatch.setattr(single,'run_single_frame_vortex',lambda *a,**kw:(_ for _ in ()).throw(OSError('injected')))
    with pytest.raises(OSError): run(case,output,indices=(1,))
    monkeypatch.setattr(single,'run_single_frame_vortex',original)
    assert run(case,output,indices=(1,))['status'] == 'completed'
    assert lock.stat().st_ino == inode


def test_series_cli_help_options_and_synthetic_execution(tmp_path,capsys):
    script = Path(__file__).resolve().parents[1]/'scripts/41_export_vortex_3d_series.py'
    spec = importlib.util.spec_from_file_location('vortex_series_cli',script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    args = cli.parse_args([])
    assert args.case == 'N7_H' and args.indices is None and args.sampling_nodes is None
    assert args.output_dir == series.DEFAULT_OUTPUT_ROOT
    case, output = frames(tmp_path)
    assert cli.main(['--case',str(case),'--output-dir',str(output),'--indices','1,3',
                     '--expected-element-counts','2','1','1','--sampling-nodes','10','--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['selected_indices'] == [1,3]
    assert not output.exists()
    assert cli.main(['--case',str(case),'--output-dir',str(output),'--indices','1',
                     '--expected-element-counts','2','1','1']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'
    with pytest.raises(ValueError,match='requires --resume'):
        run(case,output,recover_staging=True)


def completed_stage(case, output, monkeypatch):
    original = single._publish_directory
    monkeypatch.setattr(single,'_publish_directory',lambda *a:(_ for _ in ()).throw(OSError('injected publication interruption')))
    with pytest.raises(OSError): run(case,output,indices=(1,))
    monkeypatch.setattr(single,'_publish_directory',original)
    return next(root(case,output).glob('.vortex-3d-f00001-*'))


@pytest.mark.parametrize('kind',['multiple','incompatible','changed_before_move','extra_before_move','symlink_before_move'])
def test_unsafe_complete_staging_is_preserved_and_not_published(tmp_path,monkeypatch,kind):
    case, output = frames(tmp_path)
    stage = completed_stage(case,output,monkeypatch)
    if kind == 'multiple': shutil.copytree(stage,stage.with_name(stage.name+'-copy'))
    if kind == 'incompatible':
        path = next(stage.glob('*.json'))
        meta = json.loads(path.read_text()); meta['time'] = .4
        path.write_text(json.dumps(meta))
    def progress(message):
        if kind == 'changed_before_move' and json.loads(message)['message'] == 'Frame started':
            with next(stage.glob('*.vtu')).open('ab') as stream: stream.write(b'changed after validation')
        if kind == 'extra_before_move' and json.loads(message)['message'] == 'Frame started':
            (stage/'unexpected.vtu').write_bytes(b'unvalidated extra artifact')
        if kind == 'symlink_before_move' and json.loads(message)['message'] == 'Frame started':
            preserved = stage.with_name(stage.name+'-preserved')
            stage.rename(preserved)
            stage.symlink_to(preserved,target_is_directory=True)
    with pytest.raises((ValueError,RuntimeError),match='Multiple|metadata|staging changed'):
        run(case,output,indices=(1,),resume=True,recover_staging=True,progress=progress)
    assert stage.exists() and not frame_paths(case,output)['vtu'].parent.exists()


def test_recovery_allocates_no_new_scientific_disk_space(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    completed_stage(case,output,monkeypatch)
    report = run(case,output,indices=(1,),resume=True,recover_staging=True,dry_run=True)
    assert report['pending_frame_disk_allowances'] == []
    assert report['required_disk_bytes'] == series.CONTROL_MARGIN
    assert report['frames'][0]['action'] == 'validate_then_recover'


@pytest.mark.parametrize('damage',['ascii','float32','negative_lambda','nan','timestamp','extra_array','ghost_cells','extra_field'])
def test_content_validation_independent_of_consistently_rewritten_metadata(tmp_path,damage):
    case, output = frames(tmp_path)
    run(case,output,indices=(1,))
    paths = frame_paths(case,output)
    grid = pv.read(paths['vtu'])
    if damage == 'float32': grid['omega_x'] = grid['omega_x'].astype(np.float32)
    if damage == 'negative_lambda': grid['lambda_ci'][0] = -1.
    if damage == 'nan': grid['omega_x'][0] = np.nan
    if damage == 'timestamp': grid.field_data['TIME_VALUE'][0] += .5
    if damage == 'extra_array': grid['extra'] = np.ones(grid.n_points)
    if damage == 'ghost_cells': grid.cell_data['vtkGhostType'] = np.ones(grid.n_cells,dtype=np.uint8)
    if damage == 'extra_field': grid.field_data['extra'] = np.asarray([1.])
    grid.save(paths['vtu'],binary=damage!='ascii')
    metadata = json.loads(paths['metadata'].read_text())
    metadata['array_sha256'] = grid_fingerprints(grid)
    metadata['output_sha256']['vtu'] = single._file_hash(paths['vtu'])
    paths['metadata'].write_text(json.dumps(metadata))
    series._control_files(root(case,output))['manifest'].rename(tmp_path/'preserved_checkpoint.json')
    with pytest.raises(ValueError): run(case,output,indices=(1,),resume=True)


def test_binary_xml_headers_are_scanned_across_chunk_boundaries(tmp_path):
    prefix = b'<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
    payload = prefix+b'A'*(1024**2-len(prefix)-5)
    payload += 7*b'<DataArray type="Float64" format="binary">AAAA</DataArray>'
    payload += b'</UnstructuredGrid></VTKFile>'
    path = tmp_path/'headers.vtu'; path.write_bytes(payload)
    assert binary_vtu_fingerprint(path) == sha256(payload).hexdigest()


@pytest.mark.parametrize('contents',[b'<VTKFile',b'<VTKFile type="PolyData"/>',
    b'<VTKFile type="UnstructuredGrid"><DataArray format="ascii"/>',
    b'<VTKFile type="UnstructuredGrid"><DataArray format="appended"/>',
    b'<VTKFile type="UnstructuredGrid"><DataArray broken=>'])
def test_binary_format_validation_rejects_invalid_headers(tmp_path,contents):
    path = tmp_path/'invalid.vtu'; path.write_bytes(contents)
    with pytest.raises(ValueError): binary_vtu_fingerprint(path)


def test_wrong_writer_serialization_is_rejected_before_frame_publication(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    def ascii_writer(path,grid,*,time):
        grid.field_data['TIME_VALUE'] = np.asarray([time])
        grid.save(path,binary=False)
    monkeypatch.setattr(single,'write_binary_vtu',ascii_writer)
    with pytest.raises(ValueError,match='binary'):
        run(case,output,indices=(1,))
    assert not frame_paths(case,output)['vtu'].parent.exists()
    assert len(list(root(case,output).glob('.vortex-3d-f00001-*'))) == 1


def test_invalid_raw_map_in_later_frame_blocks_entire_preflight(tmp_path,monkeypatch):
    import struct
    case, output = frames(tmp_path)
    source = case/'GC0.f00002'; raw = source.read_bytes()
    source.write_bytes(raw[:136]+struct.pack('<2i',0,0)+raw[144:])
    monkeypatch.setattr(single,'read_nek_file',lambda *a,**kw:pytest.fail('invalid map reached reader'))
    with pytest.raises(ValueError,match='map'):
        run(case,output)
    assert not output.exists()


def test_live_remaining_disk_check_stops_before_next_frame(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    disk = series.shutil.disk_usage
    low = False
    def space(path):
        return type('D',(),{'free':1})() if low else disk(path)
    def progress(message):
        nonlocal low
        if json.loads(message)['message'] == 'Frame completed': low = True
    monkeypatch.setattr(series.shutil,'disk_usage',space)
    with pytest.raises(OSError,match='remaining frames'):
        run(case,output,progress=progress)
    assert frame_paths(case,output,1)['vtu'].exists()
    assert not frame_paths(case,output,2)['vtu'].parent.exists()


def test_saved_payload_resource_gate_and_malformed_xml_stop_without_changes(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    run(case,output,indices=(1,))
    protected = hashes(p for p in output.rglob('*') if p.is_file())
    monkeypatch.setattr(single,'available_memory_bytes',lambda:1)
    monkeypatch.setattr(series,'validate_saved_grid',lambda *a:pytest.fail('read without sufficient RAM'))
    with pytest.raises(ValueError,match='Insufficient RAM'):
        run(case,output,indices=(1,),resume=True)
    assert hashes(Path(p) for p in protected) == protected
    monkeypatch.setattr(single,'available_memory_bytes',lambda:10**15)
    pvd = series._control_files(root(case,output))['pvd']; pvd.write_text('<VTKFile')
    with pytest.raises(ValueError,match='Malformed PVD XML'):
        run(case,output,indices=(1,),resume=True)
    assert pvd.read_text() == '<VTKFile'


def test_normalizes_generator_configuration_once_and_isolates_mutation(tmp_path):
    class Once:
        def __init__(self,values): self.values,self.calls=values,0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            return iter(self.values)
    case, output = frames(tmp_path)
    axes,counts,crop = ['y','x'],[2,1,1],[0.,1.8,0.,1.,0.,1.]
    a,b,c = Once(axes),Once(counts),Once(crop)
    def mutate(message):
        axes[:] = ['z']; counts[:] = [9,9,9]; crop[:] = [np.nan]*6
    result = series.run_vortex_series(case,output,indices=iter((1,3)),periodic_axes=a,
        expected_element_counts=b,crop=c,sampling_nodes=10,progress=mutate)
    assert a.calls == b.calls == c.calls == 1
    assert result['configuration']['periodic_axes'] == ['x','y']
    assert result['configuration']['expected_element_counts'] == [2,1,1]
    assert result['configuration']['crop'] == [0.,1.8,0.,1.,0.,1.]


def test_implementation_change_while_waiting_stops_before_computation(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    signature = series._signature
    @contextmanager
    def changed_lock(path):
        monkeypatch.setattr(series,'_signature',lambda:{**signature(),'changed':'0'*64})
        yield
    monkeypatch.setattr(series,'_pvd_update_lock',changed_lock)
    monkeypatch.setattr(single,'read_nek_file',lambda *a,**kw:pytest.fail('computed changed implementation'))
    with pytest.raises(RuntimeError,match='changed while waiting'):
        run(case,output,indices=(1,))
    assert not output.exists()


def test_filesystem_and_process_file_limits_are_detected(tmp_path,monkeypatch):
    monkeypatch.setattr(series.resource,'getrlimit',lambda value:(4096,8192))
    monkeypatch.setattr(series.shutil,'which',lambda name:'/usr/bin/stat')
    monkeypatch.setattr(series.subprocess,'run',lambda *a,**kw:type('R',(),{'returncode':0,'stdout':'vfat\n'})())
    limits = series._filesystem_limits(tmp_path)
    assert limits['process_file_size_limit_bytes'] == 4096
    assert limits['filesystem_file_size_limit_bytes'] == 2**32-1
    assert limits['effective_file_size_limit_bytes'] == 4096


def test_control_resource_allowances_scale_for_large_inventories():
    assert series._control_allowance(81,81)[1] == series.CONTROL_MARGIN
    per_file, disk = series._control_allowance(99999,99999)
    assert per_file > 1024**3 and disk >= 4*per_file


def test_oversized_frame_inventory_stops_before_header_io(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    monkeypatch.setattr(single,'_inspect_vortex_frame',lambda *a,**kw:pytest.fail('oversized inventory reached header IO'))
    with pytest.raises(ValueError,match='control-metadata budget'):
        run(case,output,indices='1-99999',dry_run=True)
    assert not output.exists()


@pytest.mark.parametrize('damage',['definition','ranges_missing','range_nan','visual','counts'])
def test_cold_adoption_requires_complete_provenance_even_without_checkpoint(tmp_path,damage):
    case, output = frames(tmp_path)
    run(case,output,indices=(1,))
    paths = frame_paths(case,output)
    metadata = json.loads(paths['metadata'].read_text())
    if damage == 'definition': metadata['numerical_definition']['omega_x'] = 'incorrect'
    if damage == 'ranges_missing': del metadata['field_ranges']
    if damage == 'range_nan': metadata['field_ranges']['lambda_ci']['min'] = float('nan')
    if damage == 'visual': del metadata['visualization']['cells']
    if damage == 'counts': metadata['full_linear_hex_count'] += 1
    paths['metadata'].write_text(json.dumps(metadata))
    series._control_files(root(case,output))['manifest'].rename(tmp_path/'preserved_manifest.json')
    report = run(case,output,indices=(1,),resume=True,dry_run=True)
    assert report['status'] == 'invalid'
    assert report['frames'][0]['status'] == 'incompatible'


@pytest.mark.parametrize('count',[8,10])
def test_dry_run_never_claims_a_completed_reconstruction_audit(tmp_path,count):
    case, output = frames(tmp_path)
    report = single.inspect_vortex_frame(case,1,output,sampling_nodes=count,
        expected_element_counts=(2,1,1),periodic_axes=('x','y'))
    assert report['reconstruction']['omega_route_audit_performed'] is False
    if count == 8:
        assert 'not performed' in report['reconstruction']['omega_x']


def test_foreign_control_metadata_is_not_overwritten(tmp_path):
    case, output = frames(tmp_path)
    folder = root(case,output)/'series'; folder.mkdir(parents=True)
    path = folder/'processing_summary.json'; path.write_text('{"unrelated":"protected"}')
    with pytest.raises(ValueError,match='foreign series checkpoint'):
        run(case,output,indices=(1,))
    assert path.read_text() == '{"unrelated":"protected"}'


def test_source_change_during_header_inspection_fails_read_only_preflight(tmp_path,monkeypatch):
    case, output = frames(tmp_path)
    original = single._inspect_vortex_frame
    def inspect(*args,**kwargs):
        report = original(*args,**kwargs)
        with Path(report['source_file']).open('ab') as stream: stream.write(b'changed after header read')
        return report
    monkeypatch.setattr(single,'_inspect_vortex_frame',inspect)
    monkeypatch.setattr(single,'read_nek_file',lambda *a,**kw:pytest.fail('decoded unstable snapshot'))
    report = run(case,output,indices=(1,),dry_run=True)
    assert report['status'] == 'invalid'
    assert any('changed while inspecting' in value for value in report['failures'])
    assert not output.exists()
