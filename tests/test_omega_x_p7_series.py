"""Small deterministic saved-scalar fixtures; no external production dependencies."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import io
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from nek_post import omega_x_p7_series as series
from nek_post import omega_x_p7_slice as single
from nek_post import omega_x_workflow as workflow
from nek_post.gll import gll_nodes
from nek_post.lambda_ci_export import write_single_frame_pvd
from nek_post.rectilinear_export import RectilinearScalarField, write_rectilinear_scalar_vtr
from test_lambda_ci_workflow import synthetic_file
from test_omega_x_p7_slice import sample


def native_frames(tmp_path, indices=(1, 2), *, nested=True, times=None):
    from pymech.neksuite import readnek, writenek
    case, source = synthetic_file(tmp_path, shape=(8, 8, 8))
    data = readnek(str(source))
    inputs = tmp_path/'native'; inputs.mkdir()
    nodes = gll_nodes(8)
    coords = (np.r_[(nodes+1)/2, 1+(nodes[1:]+1)/2], (nodes+1)/2, (nodes+1)/2)
    zz, yy, xx = np.meshgrid(*coords[::-1], indexing='ij')
    scalar = .5+(xx*(xx-2))**3+.2*yy*(yy-1)-.5*zz**6
    for offset,index in enumerate(indices):
        data.time = index*.125+index**2*.00001 if times is None else times[offset]; data.istep = index*13
        source = case/f'GC0.f{index:05d}'; writenek(str(source), data)
        folder = (inputs/case.name if nested else inputs)/f'f{index:05d}'; folder.mkdir(parents=True)
        paths = workflow._paths(folder, case.name, index)
        field = RectilinearScalarField(coords, np.ascontiguousarray(scalar*(.1+index/81)), 'omega_x')
        write_rectilinear_scalar_vtr(paths['vtr'], field, time=data.time)
        write_single_frame_pvd(paths['pvd'], paths['vtr'], time=data.time)
        metadata = dict(schema_version=1, field='omega_x', completion_status='completed', case=case.name,
            frame_index=index, time=data.time, istep=data.istep, source_file=str(source),
            source_sha256=workflow._file_hash(source), source_unchanged=True,
            configuration={'periodic_axes':['x','y'], 'expected_element_counts':[2,1,1], 'expected_orders':[8,8,8]},
            numerical_software_signature=workflow._signature(),
            omega_x={'min':float(field.values.min()),'max':float(field.values.max())},
            visualization=dict(shape_xyz=[15,8,8], bounds_xyz=[[0.,2.],[0.,1.],[0.,1.]],
                coordinate_sha256=[workflow._coordinate_hash(c) for c in coords], native_gll=True,
                periodic_endpoints_preserved=True, interpolation='none'),
            output_paths={k:str(p) for k,p in paths.items()},
            output_sha256={k:workflow._file_hash(paths[k]) for k in ('vtr','pvd')})
        paths['metadata'].write_text(json.dumps(metadata))
    return case, inputs, tmp_path/'slices'


def run(case, inputs, output, **kwargs):
    return series.run_omega_x_p7_series(case, inputs, output,
        expected_element_counts=(2,1,1), expected_bounds=((0.,2.),(0.,1.),(0.,1.)),
        samples_per_element=5, **kwargs)


def contents(root):
    return {str(p):workflow._file_hash(p) for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def fast_figures(monkeypatch):
    # Use genuine tiny PNG/PDFs; numerical fields and normal decoder checks stay real.
    from PIL import Image
    from matplotlib.figure import Figure
    image = io.BytesIO(); Image.new('RGB',(32,32),'white').save(image,format='PNG')
    pdf = io.BytesIO(); fig = Figure(figsize=(1,1)); fig.savefig(pdf,format='pdf')
    def figures(samples, paths, **kwargs):
        for key,path in paths.items():
            if key.endswith('_png'):
                with path.open('xb') as f: f.write(image.getvalue())
            elif key.endswith('_pdf'):
                with path.open('xb') as f: f.write(pdf.getvalue())
        return {'test_fixture':True, 'smoothing':False}
    monkeypatch.setattr(single, '_figures', figures)


@pytest.mark.parametrize('location', ['frame', 'nested', 'alias', 'root_equal', 'renamed_frame'])
def test_output_root_safety_rejects_frame_ancestry_before_preflight(tmp_path, monkeypatch, fast_figures, location):
    case, inputs, output = native_frames(tmp_path, indices=(79,))
    run(case, inputs, output, indices=[79])
    frame = output/case.name/'f00079'
    if location == 'renamed_frame':
        frame = frame.rename(frame.with_name('renamed-scientific-frame'))
    parent = frame if location == 'frame' else frame/'nested'/'deeper'
    if location == 'alias':
        parent = tmp_path/'alias'
        parent.symlink_to(frame, target_is_directory=True)
    if location == 'root_equal':
        case = tmp_path/'sources'/'f00079'
        case.mkdir(parents=True)
        parent = frame.parent  # Appending this case name makes the final root exactly the frame.
    before = contents(output)
    directories = sorted(str(p) for p in output.rglob('*'))
    def forbidden(*args, **kwargs):
        pytest.fail('Unsafe output ancestry reached preflight')
    monkeypatch.setattr(series, '_preflight', forbidden)
    try:
        with pytest.raises(ValueError, match='scientific frame'):
            run(case, inputs, parent, indices=[79], resume=True)
    finally:
        assert contents(output) == before
        assert sorted(str(p) for p in output.rglob('*')) == directories


def test_output_root_safety_accepts_case_root_and_independent_sibling(tmp_path, monkeypatch, fast_figures):
    case, inputs, output = native_frames(tmp_path, indices=(79,))
    run(case, inputs, output, indices=[79])
    before = contents(output)
    monkeypatch.setattr(single, 'DEFAULT_OUTPUT_ROOT', output)
    assert series._locations(case, inputs, None)[2] == output/case.name
    independent = output/case.name/'f00079-independent'
    assert series._locations(case, inputs, independent)[2] == independent/case.name
    assert not independent.exists() and contents(output) == before


@pytest.mark.parametrize('scenario', ['partial', 'all', 'incompatible', 'incomplete_final',
                                    'incomplete_stage', 'complete_stage', 'moved_stage'])
def test_lock_time_refresh_with_real_waiting_writer(tmp_path, monkeypatch, fast_figures, scenario):
    case, inputs, output = native_frames(tmp_path)
    root = output/case.name
    local = threading.local()
    writer1_locked = threading.Event()
    writer2_waiting = threading.Event()
    release_writer1 = threading.Event()
    writer2_locked = threading.Event()
    calls = []; sampled = []; preserved = {}
    real_lock = series._pvd_update_lock
    real_preflight = series._preflight
    real_sample = series._sample
    real_publish = series._publish_directory
    real_figures = single._figures
    if scenario == 'moved_stage':
        def interrupted(*args):
            raise OSError('Seed complete staging before both preflights')
        with monkeypatch.context() as patch:
            patch.setattr(series, '_publish_directory', interrupted)
            with pytest.raises(OSError, match='Seed'):
                run(case, inputs, output, indices=[1])

    @contextmanager
    def synchronized_lock(path):
        if local.writer == 2:
            writer2_waiting.set()
        with real_lock(path):
            if local.writer == 1:
                writer1_locked.set()
                assert release_writer1.wait(30), 'Writer 2 did not finish initial preflight'
            else:
                writer2_locked.set()
            try:
                yield
            finally:
                if local.writer == 1:
                    if scenario == 'incompatible':
                        metadata = next((root/'f00001').glob('*_validation.json'))
                        report = json.loads(metadata.read_text())
                        report['interpolation_signature']['gll.py'] = '0'*64
                        metadata.write_text(json.dumps(report))
                    elif scenario == 'incomplete_final':
                        next((root/'f00001').glob('*_p7.pdf')).unlink()
                    for p in root.rglob('*'):
                        if p.is_file() and ('f00001' in p.parts or 'f00002' in p.parts or
                                            '.omega-x-p7-staging' in p.parts):
                            preserved[p] = workflow._file_hash(p)

    def preflight(*args, **kwargs):
        if local.writer == 2:
            calls.append(writer2_locked.is_set())
        return real_preflight(*args, **kwargs)

    def sample(*args, **kwargs):
        if local.writer == 2:
            sampled.append(args[2]['index'])
        return real_sample(*args, **kwargs)

    def publish(stage, final):
        if local.writer == 1 and scenario == 'complete_stage':
            raise OSError('Simulated stop after complete staging')
        return real_publish(stage, final)

    def figures(samples, paths, **kwargs):
        if local.writer == 1 and scenario == 'incomplete_stage':
            (next(iter(paths.values())).parent/'incomplete-marker').write_bytes(b'preserve')
            raise OSError('Simulated interrupted rendering')
        return real_figures(samples, paths, **kwargs)

    monkeypatch.setattr(series, '_pvd_update_lock', synchronized_lock)
    monkeypatch.setattr(series, '_preflight', preflight)
    monkeypatch.setattr(series, '_sample', sample)
    monkeypatch.setattr(series, '_publish_directory', publish)
    monkeypatch.setattr(single, '_figures', figures)

    def writer(number):
        local.writer = number
        selection = [1, 2] if number == 2 or scenario == 'all' else [1]
        return run(case, inputs, output, indices=selection, resume=True, recover_staging=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(writer, 1)
        second = None
        try:
            assert writer1_locked.wait(30), 'Writer 1 did not acquire the real lock'
            second = pool.submit(writer, 2)
            assert writer2_waiting.wait(30), 'Writer 2 did not reach the real lock'
        finally:
            release_writer1.set()
        if scenario in ('incomplete_stage', 'complete_stage'):
            with pytest.raises(OSError, match='Simulated'):
                first.result(timeout=30)
        else:
            first.result(timeout=30)
        if scenario in ('incompatible', 'incomplete_final'):
            with pytest.raises(ValueError, match='signature' if scenario == 'incompatible' else 'Incomplete'):
                second.result(timeout=30)
            assert sampled == []
            assert not (root/'f00002').exists()
        else:
            result = second.result(timeout=30)
            expected = {'partial':['reused','generated'], 'all':['reused','reused'],
                        'moved_stage':['reused','generated'],
                        'incomplete_stage':['generated','generated'],
                        'complete_stage':['recovered','generated']}[scenario]
            assert [f['action'] for f in result['frames']] == expected
            assert sampled == [i+1 for i,a in enumerate(expected) if a == 'generated']
            assert result['generate_frames'] == expected.count('generated')
            assert result['reuse_frames'] == expected.count('reused')
            assert result['recover_frames'] == expected.count('recovered')
            assert (result['estimated_new_output_bytes'] == 0) == (scenario == 'all')
            assert result['plan_build_count'] == 1
        for path, digest in preserved.items():
            if scenario == 'complete_stage':
                path = root/'f00001'/path.name
            assert workflow._file_hash(path) == digest
        assert calls == [False, True], 'Execution must use a complete preflight refreshed under the lock'


@pytest.mark.parametrize('dry_run', [True, False])
def test_invocation_signature_binds_reports_and_checkpoints_to_source(tmp_path, fast_figures, dry_run):
    case, inputs, output = native_frames(tmp_path, indices=(1,))
    result = run(case, inputs, output, indices=[1], dry_run=dry_run)
    expected = {name:workflow._file_hash(Path(series.__file__).parent/name) for name in series._signature()}
    assert result['interpolation_signature'] == expected == series._signature()
    if not dry_run:
        checkpoint = json.loads(Path(result['log_path']).read_text())
        assert checkpoint['interpolation_signature'] == expected


def test_failed_checkpoint_retains_invocation_signature(tmp_path, monkeypatch, fast_figures):
    case, inputs, output = native_frames(tmp_path, indices=(1,))
    expected = series._signature()
    def interrupted(*args):
        running = json.loads(next((output/case.name/'logs').glob('*.json')).read_text())
        assert running['status'] == 'running' and running['interpolation_signature'] == expected
        raise OSError('Checkpoint interruption')
    monkeypatch.setattr(series, '_export', interrupted)
    with pytest.raises(OSError, match='Checkpoint interruption'):
        run(case, inputs, output, indices=[1])
    failed = json.loads(next((output/case.name/'logs').glob('*.json')).read_text())
    assert failed['status'] == 'failed' and failed['interpolation_signature'] == expected


@pytest.mark.parametrize('resource', ['ram', 'disk'])
def test_lock_time_refresh_rechecks_resources_before_native_reader(tmp_path, monkeypatch, resource):
    from vtkmodules import vtkIOXML
    case, inputs, output = native_frames(tmp_path, indices=(1,))
    before = contents(inputs)
    real_lock = series._pvd_update_lock
    @contextmanager
    def depleted_lock(path):
        with real_lock(path):
            if resource == 'ram':
                monkeypatch.setattr(workflow, 'available_memory_bytes', lambda:1)
            else:
                monkeypatch.setattr(series.shutil, 'disk_usage', lambda p:SimpleNamespace(free=1))
            if resource == 'ram':
                monkeypatch.setattr(vtkIOXML, 'vtkXMLRectilinearGridReader',
                                    lambda:pytest.fail('Lock-time RAM rejection loaded VTR'))
            yield
    monkeypatch.setattr(series, '_pvd_update_lock', depleted_lock)
    with pytest.raises(MemoryError if resource == 'ram' else OSError):
        run(case, inputs, output, indices=[1])
    assert not (output/case.name/'f00001').exists()
    assert not (output/case.name/'logs').exists() and contents(inputs) == before


@pytest.mark.parametrize('indices', ['1,1','2-1','0','82','1,,2',[],[True],[1,1]])
def test_invalid_duplicate_or_out_of_range_selection(indices):
    with pytest.raises(ValueError): series.frame_indices(indices)


def test_selection_exact_numbering_and_conflicting_range():
    assert series.frame_indices('79,1-2') == (1,2,79)
    assert series.frame_indices() == tuple(range(1,82))
    with pytest.raises(ValueError): series.frame_indices('1',start_index=1)


@pytest.mark.parametrize('plane',[.75,.63])
def test_cached_plan_exactly_matches_uncached_and_is_invocation_local(plane):
    coords,values,expected = sample(lambda x,y,z:x**7-3*z**6+2*y**5,y=plane)
    plan = single.build_p7_slice_plan(coords,element_counts=(3,2,2),y=plane)
    matrices = [id(b) for _,b in plan.x_segments.values()]
    actual = single.reconstruct_p7_midplane(coords,values,element_counts=(3,2,2),y=plane,
        time=expected.time,plan=plan)
    np.testing.assert_array_equal(actual.p7,expected.p7)
    np.testing.assert_array_equal(actual.linear,expected.linear)
    assert matrices == [id(b) for _,b in plan.x_segments.values()]
    changed = list(coords); changed[0] = changed[0].copy(); changed[0][1] += 1e-7
    with pytest.raises(ValueError,match='Cached P7'):
        single.reconstruct_p7_midplane(changed,values,element_counts=(3,2,2),y=plane,plan=plan)


@pytest.mark.parametrize('nested',[False,True])
def test_dry_run_order_grid_and_no_writes(tmp_path,nested):
    case,inputs,output = native_frames(tmp_path,nested=nested)
    before = contents(inputs)
    result = run(case,inputs,output,indices=[2,1],dry_run=True)
    assert result['status']=='validated' and result['plan_build_count']==1
    assert [f['index'] for f in result['frames']]==[1,2]
    assert result['frames'][0]['time'] < result['frames'][1]['time']
    assert result['generate_frames']==2 and result['estimated_ram_bytes']>0
    assert result['estimated_new_output_bytes']>0 and not output.exists()
    assert contents(inputs)==before


@pytest.mark.parametrize('times',[(.5,.5),(.7,.4)])
def test_actual_times_must_increase_even_with_consistent_provenance(tmp_path,times):
    case,inputs,output=native_frames(tmp_path,times=times)
    with pytest.raises(ValueError,match='Actual timestamps'):
        run(case,inputs,output,indices=[1,2],dry_run=True)
    assert not output.exists()


@pytest.mark.parametrize('change',['missing','duplicate','number','timestamp','source_hash','source_bytes','vtr_hash','signature','geometry'])
def test_bad_native_inputs_rejected_without_writes(tmp_path,change):
    case,inputs,output = native_frames(tmp_path)
    folder = inputs/case.name/'f00002'; paths = workflow._paths(folder,case.name,2)
    if change=='missing': shutil.rmtree(folder)
    elif change=='duplicate': shutil.copytree(folder,inputs/'duplicate'/'f00002')
    elif change=='number': folder.rename(folder.with_name('f2'))
    elif change=='source_bytes':
        with (case/'GC0.f00002').open('ab') as f:f.write(b'changed')
    else:
        m=json.loads(paths['metadata'].read_text())
        if change=='timestamp':m['time']+=1
        elif change=='source_hash':m['source_sha256']='0'*64
        elif change=='vtr_hash':m['output_sha256']['vtr']='0'*64
        elif change=='signature':m['numerical_software_signature']['gll.py']='0'*64
        elif change=='geometry':
            from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader,vtkXMLRectilinearGridWriter
            r=vtkXMLRectilinearGridReader();r.SetFileName(str(paths['vtr']));r.Update();g=r.GetOutput()
            g.GetXCoordinates().SetValue(1,g.GetXCoordinates().GetValue(1)+1e-7)
            m['visualization']['coordinate_sha256'][0]=workflow._coordinate_hash(
                __import__('vtkmodules.util.numpy_support',fromlist=['vtk_to_numpy']).vtk_to_numpy(g.GetXCoordinates()))
            w=vtkXMLRectilinearGridWriter();w.SetFileName(str(paths['vtr']));w.SetInputData(g);assert w.Write()==1
            m['output_sha256']['vtr']=workflow._file_hash(paths['vtr'])
        paths['metadata'].write_text(json.dumps(m))
    before=contents(inputs)
    with pytest.raises((ValueError,OSError)):run(case,inputs,output,indices=[1,2],dry_run=True)
    assert not output.exists() and contents(inputs)==before


def test_signed_vtr_resume_checksums_and_no_recomputation(tmp_path,monkeypatch,fast_figures):
    case,inputs,output=native_frames(tmp_path); before=contents(inputs)
    def forbidden(*a,**k):pytest.fail('Series attempted native velocity/curl/projection/eigenvalues')
    monkeypatch.setattr(workflow,'read_nek_file',forbidden)
    monkeypatch.setattr(workflow,'compute_projected_omega_x',forbidden)
    monkeypatch.setattr(np.linalg,'eigvals',forbidden)
    first=run(case,inputs,output,indices=[2,1])
    assert first['status']=='completed_subset' and first['pvd_entries']==0
    assert first['plan_build_count']==1 and not Path(first['series_pvd']).exists()
    for f in first['frames']:
        m=json.loads(Path(f['output_paths']['metadata']).read_text())
        assert m['diagnostics']['p7']['min']<0<m['diagnostics']['p7']['max']
        assert m['vtr_readback']['dimensions_xyz']==[8,1,5]
    existing={p:h for p,h in contents(output).items() if '/f000' in p}
    monkeypatch.setattr(single,'reconstruct_p7_midplane',forbidden)
    resumed=run(case,inputs,output,indices=[1,2],resume=True)
    assert all(f['action']=='reused' for f in resumed['frames']) and resumed['plan_build_count']==1
    assert all(workflow._file_hash(Path(p))==h for p,h in existing.items()) and contents(inputs)==before
    with pytest.raises(FileExistsError):run(case,inputs,output,indices=[1,2])


def test_legacy_frame79_explicit_assessment_and_preservation(tmp_path,monkeypatch,fast_figures):
    case,inputs,output=native_frames(tmp_path,indices=(79,))
    first=run(case,inputs,output,indices=[79]);p=Path(first['frames'][0]['output_paths']['metadata'])
    m=json.loads(p.read_text())
    for k in ('schema_version','configuration','interpolation_signature'):m.pop(k)
    p.write_text(json.dumps(m)); before=contents(p.parent)
    with pytest.raises(ValueError,match='Unapproved legacy'):run(case,inputs,output,indices=[79],resume=True,dry_run=True)
    monkeypatch.setattr(series,'APPROVED_LEGACY_REPORTS',{(case.name,79):workflow._file_hash(p)})
    real_sample = series._sample
    assessments = []
    def assess(*args, **kwargs):
        assessments.append(args[2]['index'])
        return real_sample(*args, **kwargs)
    monkeypatch.setattr(series, '_sample', assess)
    resumed=run(case,inputs,output,indices=[79],resume=True)
    assert assessments == [79]  # Initial assessment is retained through lock-time revalidation.
    assert resumed['frames'][0]['legacy_compatibility']=='exact scalar/coordinate equality'
    assert contents(p.parent)==before and 'schema_version' not in json.loads(p.read_text())
    original=single.reconstruct_p7_midplane
    def wrong(*a,**k):
        s=original(*a,**k);return replace(s,p7=s.p7+1e-4)
    monkeypatch.setattr(single,'reconstruct_p7_midplane',wrong)
    with pytest.raises(ValueError,match='Legacy P7 scalar'):
        run(case,inputs,output,indices=[79],resume=True,dry_run=True)


@pytest.mark.parametrize('change',['output_hash','output_bytes','interpolation_signature','configuration','source','time','incomplete'])
def test_bad_existing_slice_never_overwritten(tmp_path,change,fast_figures):
    case,inputs,output=native_frames(tmp_path,indices=(1,))
    first=run(case,inputs,output,indices=[1]);paths={k:Path(p) for k,p in first['frames'][0]['output_paths'].items()}
    m=json.loads(paths['metadata'].read_text())
    if change=='output_bytes':paths['vtr'].write_bytes(b'corrupt')
    elif change=='incomplete':paths['p7_pdf'].unlink()
    else:
        if change=='output_hash':m['output_sha256']['vtr']='0'*64
        elif change=='interpolation_signature':m['interpolation_signature']['gll.py']='0'*64
        elif change=='configuration':m['configuration']['samples_per_element']=29
        elif change=='source':m['source_snapshot_sha256']='0'*64
        elif change=='time':m['time']+=.1
        paths['metadata'].write_text(json.dumps(m))
    before=contents(output)
    with pytest.raises(ValueError):run(case,inputs,output,indices=[1],resume=True)
    assert contents(output)==before and not Path(first['series_pvd']).exists()


@pytest.mark.parametrize('available',[1,None,-1])
def test_low_unknown_ram_rejected_before_reader(tmp_path,monkeypatch,available):
    from vtkmodules import vtkIOXML
    case,inputs,output=native_frames(tmp_path)
    monkeypatch.setattr(workflow,'available_memory_bytes',lambda:available)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('RAM rejection loaded VTR'))
    with pytest.raises(MemoryError):run(case,inputs,output,indices=[1,2],dry_run=True)
    assert not output.exists()


def test_low_disk_and_protected_output(tmp_path,monkeypatch):
    case,inputs,output=native_frames(tmp_path)
    monkeypatch.setattr(series.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    with pytest.raises(OSError,match='disk space'):run(case,inputs,output,indices=[1,2],dry_run=True)
    assert not output.exists()
    with pytest.raises(ValueError,match='outside'):run(case,inputs,inputs,indices=[1])


def test_late_ram_drop_stops_before_next_native_reader(tmp_path,monkeypatch):
    from vtkmodules import vtkIOXML
    case,inputs,output=native_frames(tmp_path)
    def progress(message):
        monkeypatch.setattr(workflow,'available_memory_bytes',lambda:1)
        monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('Late RAM drop loaded VTR'))
    with pytest.raises(MemoryError):run(case,inputs,output,indices=[1,2],dry_run=True,progress=progress)
    assert not output.exists()


def test_completed_staging_recovery_and_incomplete_staging_preserved(tmp_path,monkeypatch,fast_figures):
    case,inputs,output=native_frames(tmp_path)
    original=series._publish_directory
    def interrupted(stage,final):raise OSError('injected publish interruption')
    monkeypatch.setattr(series,'_publish_directory',interrupted)
    with pytest.raises(OSError,match='injected'):run(case,inputs,output,indices=[1,2])
    root=output/case.name; staged=list((root/'.omega-x-p7-staging').iterdir())
    assert len(staged)==1 and not (root/'f00001').exists()
    staged_before=contents(staged[0]); partial=root/'.omega-x-p7-staging'/'f00002.partial';partial.mkdir();(partial/'keep').write_text('keep')
    monkeypatch.setattr(series,'_publish_directory',original)
    dry=run(case,inputs,output,indices=[1,2],resume=True,recover_staging=True,dry_run=True)
    assert dry['recover_frames']==1 and staged[0].exists()
    resumed=run(case,inputs,output,indices=[1,2],resume=True,recover_staging=True)
    assert [f['action'] for f in resumed['frames']]==['recovered','generated']
    assert (partial/'keep').read_text()=='keep'
    assert all(workflow._file_hash(root/'f00001'/Path(p).name)==h for p,h in staged_before.items())


def test_duplicate_complete_staging_is_not_guessed(tmp_path,monkeypatch,fast_figures):
    case,inputs,output=native_frames(tmp_path,indices=(1,))
    monkeypatch.setattr(series,'_publish_directory',lambda *a:(_ for _ in ()).throw(OSError('stop')))
    with pytest.raises(OSError):run(case,inputs,output,indices=[1])
    root=output/case.name;stage=next((root/'.omega-x-p7-staging').iterdir());shutil.copytree(stage,stage.with_name('f00001.copy'))
    with pytest.raises(ValueError,match='Duplicate complete staged'):
        run(case,inputs,output,indices=[1],resume=True,recover_staging=True,dry_run=True)


def test_full81_pvd_gate_failure_and_resume_roundtrip(tmp_path,monkeypatch,fast_figures):
    case,inputs,output=native_frames(tmp_path,indices=tuple(range(1,82)))
    # Existing decoder regressions cover real PDFs; keep this 81-frame gate test bounded.
    monkeypatch.setattr(series,'_pdf_readable',lambda p:None)
    original=series._publish_pvd
    monkeypatch.setattr(series,'_publish_pvd',lambda *a:(_ for _ in ()).throw(OSError('final PVD interrupted')))
    with pytest.raises(OSError,match='final PVD'):run(case,inputs,output)
    target=output/case.name/f'{case.name}_omega_x_p7_slice_series.pvd'
    assert not target.exists() and len(list(target.parent.glob('f[0-9]*')))==81
    protected={p:h for p,h in contents(output).items() if '/f000' in p}
    monkeypatch.setattr(series,'_publish_pvd',original)
    resumed=run(case,inputs,output,resume=True)
    assert resumed['status']=='completed' and resumed['pvd_entries']==81 and resumed['generate_frames']==0
    entries=ET.parse(target).findall('Collection/DataSet');times=[float(e.get('timestep')) for e in entries]
    assert len(entries)==81 and np.all(np.diff(times)>0) and len({e.get('file') for e in entries})==81
    assert all(not Path(e.get('file')).is_absolute() and (target.parent/e.get('file')).is_file() for e in entries)
    assert all(workflow._file_hash(Path(p))==h for p,h in protected.items())
    before=target.read_bytes();run(case,inputs,output,resume=True,dry_run=True);assert target.read_bytes()==before
    with pytest.raises(ValueError,match='Complete PVD'):series._publish_pvd(target,resumed['frames'][:-1])
    target.unlink()
    def racing_link(source,destination):
        Path(destination).write_bytes(b'keep concurrent writer')
        raise FileExistsError('injected publication race')
    monkeypatch.setattr(series.os,'link',racing_link)
    with pytest.raises(FileExistsError):series._publish_pvd(target,resumed['frames'])
    assert target.read_bytes()==b'keep concurrent writer' and not list(target.parent.glob('.*.tmp'))


def test_cli_defaults_and_errors_without_output(tmp_path,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/39_compute_omega_x_p7_slice_series.py'
    spec=importlib.util.spec_from_file_location('omega_p7_series_cli',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    args=cli.parse_args([]);assert args.case=='N7_H' and not args.resume and not args.dry_run
    assert cli.main(['--case',str(tmp_path/'missing'),'--input-dir',str(tmp_path/'missing')])==1
    assert json.loads(capsys.readouterr().out)['status']=='invalid'
