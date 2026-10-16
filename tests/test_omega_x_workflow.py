"""Small native P7 files, safe publication/resume/backup, and CLI checks."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from test_lambda_ci_workflow import synthetic_file
from nek_post.omega_x_workflow import run_omega_x_series


def case_files(tmp_path):
    from pymech.neksuite import readnek,writenek
    case,source=synthetic_file(tmp_path,shape=(8,8,8))
    data=readnek(str(source))
    for e in data.elem:
        x,y,z=e.pos;e.vel[:]=np.stack((0*x,8*z*(x-.5),2*y))
    for index,t in [(1,.13),(2,1.27)]:
        data.time=t;data.istep=index*13;writenek(str(case/f'GC0.f{index:05d}'),data)
    return case,tmp_path/'results'


def run(case,root,**kwargs):
    return run_omega_x_series(case,root,expected_element_counts=(2,1,1),**kwargs)


def test_cli_selection_defaults_and_conflicting_flags():
    path=Path(__file__).resolve().parents[1]/'scripts/36_compute_omega_x.py'
    spec=importlib.util.spec_from_file_location('omega_cli',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=module.parse_args(['--case','N7_H','--indices','79','--resume','--dry-run'])
    assert args.case=='N7_H' and args.indices=='79' and args.resume and args.dry_run
    assert module.parse_args([]).indices is None
    with pytest.raises(SystemExit):module.parse_args(['--overwrite','--resume'])


def test_header_only_dry_run_no_writes_and_no_numerics(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path)
    monkeypatch.setattr(module,'read_nek_file',lambda *a,**k:pytest.fail('dry run loaded payload'))
    plan=run(case,root,indices='2,1',dry_run=True)
    assert not root.exists() and plan['resources_acceptable']
    assert [f['time'] for f in plan['frames']]==[.13,1.27]
    assert plan['frames'][0]['estimated_visual_shape_xyz']==[15,8,8]
    for extra in ({'indices':[1],'resume':True,'overwrite':True},{'start_index':1},
                  {'indices':[1],'start_index':1,'end_index':2}):
        with pytest.raises(ValueError):run(case,root,dry_run=True,**extra)
    with pytest.raises(ValueError):run(case,case/'forbidden',indices=[1],dry_run=True)
    assert not (case/'forbidden').exists()


def test_full_workflow_signed_readback_resume_and_source_preservation(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path);before={p:p.read_bytes() for p in case.iterdir()}
    report=run(case,root,indices=[2,1]);assert report['status']=='completed'
    assert [f['time'] for f in report['frames']]==[.13,1.27]
    for frame in report['frames']:
        assert frame['omega_x']['min']<0<frame['omega_x']['max']
        assert frame['visualization']['shape_xyz']==[15,8,8]
        assert frame['validation']=='passed'
    files={p:p.read_bytes() for p in (root/case.name).glob('f*/*') if p.is_file()}
    with pytest.raises(FileExistsError):run(case,root,indices=[1])
    monkeypatch.setattr(module,'compute_projected_omega_x',lambda *a,**k:pytest.fail('resume recomputed omega'))
    monkeypatch.setattr(module,'read_nek_file',lambda *a,**k:pytest.fail('resume loaded velocities'))
    resumed=run(case,root,indices=[1,2],resume=True)
    assert all(f['status']=='reused' for f in resumed['frames'])
    assert {p:p.read_bytes() for p in files}==files
    assert {p:p.read_bytes() for p in before}==before
    broken=Path(resumed['frames'][0]['output_paths']['vtr']);broken.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):run(case,root,indices=[1],resume=True)
    assert broken.read_bytes()==b'corrupt'


def test_explicit_overwrite_preserves_old_frame_and_rolls_back_on_failure(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path)
    initial=run(case,root,indices=[1]);paths=initial['frames'][0]['output_paths']
    before={Path(p).name:Path(p).read_bytes() for p in paths.values()}
    report=run(case,root,indices=[1],overwrite=True)
    backup=Path(report['frames'][0]['preserved_previous_frame'])
    assert {p.name:p.read_bytes() for p in backup.iterdir()}==before
    current={Path(p).name:Path(p).read_bytes() for p in paths.values()}
    original=module._publish_directory
    def interrupted(stage,final):
        if Path(stage).parent.name=='.omega-x-staging':raise OSError('injected publication failure')
        return original(stage,final)
    monkeypatch.setattr(module,'_publish_directory',interrupted)
    with pytest.raises(OSError,match='injected'):run(case,root,indices=[1],overwrite=True)
    assert {Path(p).name:Path(p).read_bytes() for p in paths.values()}==current
    unknown=Path(paths['metadata']).parent/'keep.dat';unknown.write_text('preserve')
    with pytest.raises(FileExistsError,match='unrelated'):run(case,root,indices=[1],overwrite=True)
    assert unknown.read_text()=='preserve'


def test_invalid_data_resource_limits_and_symlinks_stop_safely(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path)
    monkeypatch.setattr(module,'available_memory_bytes',lambda:0)
    with pytest.raises(MemoryError):run(case,root,indices=[1])
    assert not root.exists()
    monkeypatch.undo()
    target=tmp_path/'unrelated';target.mkdir();frame=root/case.name/'f00001';frame.parent.mkdir(parents=True);frame.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='redirect'):run(case,root,indices=[1],overwrite=True)
    assert not list(target.iterdir())


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('dry_run',[False,True])
def test_resume_ram_rejection_precedes_vtr_reader_and_computation(tmp_path,monkeypatch,mixed,dry_run):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path)
    run(case,root,indices=[1] if mixed else [1,2])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(module,'available_memory_bytes',lambda:1)
    def forbidden(*args,**kwargs):pytest.fail('memory-intensive operation ran before RAM rejection')
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',forbidden)
    monkeypatch.setattr(module,'read_nek_file',forbidden)
    monkeypatch.setattr(module,'compute_projected_omega_x',forbidden)
    if dry_run:
        report=run(case,root,indices=[1,2],resume=True,dry_run=True)
        assert report['resources_acceptable'] is False
        assert report['estimated_ram_bytes']>report['available_ram_bytes']==1
    else:
        with pytest.raises(MemoryError,match='RAM|memory|Resource|resource'):
            run(case,root,indices=[1,2],resume=True)
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


def test_reused_frames_nonzero_sequential_peak_and_unchanged_values(tmp_path,monkeypatch):
    import weakref
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);run(case,root,indices=[1,2])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    initial=run(case,root,indices=[1,2],resume=True,dry_run=True)
    peak=max(f['estimated_validation_ram_bytes'] for f in initial['frames'])
    assert initial['estimated_computation_ram_bytes']==0
    assert initial['estimated_ram_bytes']==peak>0
    assert peak<sum(f['estimated_validation_ram_bytes'] for f in initial['frames'])
    monkeypatch.setattr(module,'available_memory_bytes',lambda:peak+1)
    monkeypatch.setattr(module,'read_nek_file',lambda *a,**k:pytest.fail('resume read native velocities'))
    original=vtkIOXML.vtkXMLRectilinearGridReader;references=[];active=[]
    def observed():
        reader=original();references.append(weakref.ref(reader))
        active.append(sum(ref() is not None for ref in references))
        return reader
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',observed)
    report=run(case,root,indices=[1,2],resume=True)
    assert report['status']=='completed' and all(f['status']=='reused' for f in report['frames'])
    assert len(active)>=2 and max(active)==1
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


def test_mixed_resume_accounts_for_both_phases_and_preserves_old_frame(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);run(case,root,indices=[1])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan=run(case,root,indices=[1,2],resume=True,dry_run=True)
    reused,new=plan['frames']
    assert plan['estimated_ram_bytes']==max(f['estimated_ram_bytes'] for f in plan['frames'])
    assert plan['estimated_ram_bytes']>=new['estimated_computation_ram_bytes']>0
    assert plan['estimated_ram_bytes']>=new['estimated_validation_ram_bytes']>0
    assert reused['estimated_validation_ram_bytes']>0 and new['retained_validation_bytes']>0
    with monkeypatch.context() as low:
        low.setattr(module,'available_memory_bytes',lambda:reused['estimated_validation_ram_bytes']+1)
        low.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('mixed batch loaded VTR before batch RAM check'))
        rejected=run(case,root,indices=[1,2],resume=True,dry_run=True)
        assert rejected['resources_acceptable'] is False
    calls=[];original=module.compute_projected_omega_x
    def compute(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'available_memory_bytes',lambda:plan['estimated_ram_bytes']+1)
    monkeypatch.setattr(module,'compute_projected_omega_x',compute)
    report=run(case,root,indices=[1,2],resume=True)
    assert report['status']=='completed' and [f['status'] for f in report['frames']]==['reused','generated']
    assert calls==[1]
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('available',[None,float('nan'),float('inf'),-1,True,'unknown','probe_error'])
def test_unknown_resume_ram_fails_closed_without_vtr_loading(tmp_path,monkeypatch,available):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);run(case,root,indices=[1])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    def probe():
        if available=='probe_error':raise OSError('simulated unavailable memory probe')
        return available
    monkeypatch.setattr(module,'available_memory_bytes',probe)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('unknown RAM loaded VTR'))
    report=run(case,root,indices=[1],resume=True,dry_run=True)
    assert report['resources_acceptable'] is False and report['available_ram_bytes'] is None
    assert report['estimated_ram_bytes']>0 and 'Cannot establish available RAM' in report['resource_errors'][0]
    with pytest.raises(MemoryError,match='Cannot establish available RAM'):
        run(case,root,indices=[1],resume=True)
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


def test_ram_drop_at_validator_entry_is_rejected_before_reader(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);run(case,root,indices=[1])
    available=[10**15];original=module.validate_omega_x_frame
    monkeypatch.setattr(module,'available_memory_bytes',lambda:available[0])
    def validate(*args,**kwargs):
        available[0]=1
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'validate_omega_x_frame',validate)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('reader ignored late RAM drop'))
    report=run(case,root,indices=[1],resume=True,dry_run=True)
    assert report['resources_acceptable'] is False and report['available_ram_bytes']==1
    assert report['frames'][0]['resume_validation']=='pending'


def test_ram_rechecked_before_native_snapshot_read(tmp_path,monkeypatch):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path)
    available=[10**15];calls=[];original=module.read_nek_header
    def header(*args,**kwargs):
        calls.append(1)
        result=original(*args,**kwargs)
        if len(calls)==3:available[0]=1
        return result
    monkeypatch.setattr(module,'available_memory_bytes',lambda:available[0])
    monkeypatch.setattr(module,'read_nek_header',header)
    monkeypatch.setattr(module,'read_nek_file',lambda *a,**k:pytest.fail('native reader ignored late RAM drop'))
    with pytest.raises(MemoryError,match='Loading native snapshot.*insufficient RAM'):
        run(case,root,indices=[1])
    assert not (root/case.name/'f00001').exists()


@pytest.mark.parametrize('retained_field',[False,True])
def test_direct_validator_gates_reader_and_retained_readback_field(tmp_path,monkeypatch,retained_field):
    from types import SimpleNamespace
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);run(case,root,indices=[1])
    plan=run(case,root,indices=[1],resume=True,dry_run=True)
    frame=plan['frames'][0];directory=Path(frame['output_paths']['vtr']).parent
    metadata=json.loads(Path(frame['output_paths']['metadata']).read_text())
    expected=None
    if retained_field:
        expected=SimpleNamespace(values=np.zeros((8,8,15),dtype=np.float64),
            coordinates=(np.linspace(0,2,15),np.linspace(0,1,8),np.linspace(0,1,8)))
    # Enough for a reused-file read is still insufficient while its source
    # native field is retained for exact array comparisons.
    budget=frame['estimated_validation_ram_bytes']+1 if retained_field else 1
    monkeypatch.setattr(module,'available_memory_bytes',lambda:budget)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('direct validator bypassed RAM gate'))
    with pytest.raises(MemoryError,match='Validating omega_x frame.*insufficient RAM'):
        module.validate_omega_x_frame(directory,case=case.name,index=1,time_value=.13,
            config=plan['configuration'],source_sha256=metadata['source_sha256'],expected_field=expected)


def predecessor_frame(tmp_path,workflow_digest=None):
    import nek_post.omega_x_workflow as module
    case,root=case_files(tmp_path)
    initial=run(case,root,indices=[1])
    paths={k:Path(v) for k,v in initial['frames'][0]['output_paths'].items()}
    metadata=json.loads(paths['metadata'].read_text())
    metadata['numerical_software_signature']['omega_x_workflow.py']=workflow_digest or module._RAM_PREFLIGHT_PREDECESSOR_SHA256
    paths['metadata'].write_text(json.dumps(metadata))
    return case,root,paths,metadata


@pytest.mark.parametrize('workflow_digest',[
    '9f261baadf8e888ef216f83dc5cdcd2f9a4336e33c5b200340c60e22551d3736',
    '91e50314111ed4a4048ab445af41be57d7268eacdd1066251e74ee42434f713f',
])
def test_exact_ram_predecessor_resumes_without_metadata_rewrite(tmp_path,monkeypatch,workflow_digest):
    import nek_post.omega_x_workflow as module
    case,root,paths,metadata=predecessor_frame(tmp_path,workflow_digest)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert module._approved_ram_workflow(module._signature()['omega_x_workflow.py'])
    monkeypatch.setattr(module,'read_nek_file',lambda *a,**k:pytest.fail('legacy resume loaded native velocities'))
    monkeypatch.setattr(module,'compute_projected_omega_x',lambda *a,**k:pytest.fail('legacy resume recomputed omega'))
    report=run(case,root,indices=[1],resume=True)
    assert report['status']=='completed' and report['frames'][0]['status']=='reused'
    assert {p:p.read_bytes() for p in before}==before
    # The narrow software exception does not bypass output checksums.
    paths['vtr'].write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        run(case,root,indices=[1],resume=True)


@pytest.mark.parametrize('change',['unknown_workflow','gll.py','spectral_derivatives.py',
    'l2_projection.py','vorticity.py','rectilinear_export.py','missing_key','extra_key','unreviewed_successor'])
def test_ram_compatibility_rejects_other_signature_changes(tmp_path,monkeypatch,change):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root,paths,metadata=predecessor_frame(tmp_path)
    signature=metadata['numerical_software_signature']
    if change=='unknown_workflow':signature['omega_x_workflow.py']='0'*64
    elif change.endswith('.py'):
        # Preserve the historical signature and simulate a changed current
        # numerical module without editing any scientific source file.
        changed_current=module._signature();changed_current[change]='0'*64
        monkeypatch.setattr(module,'_signature',lambda:changed_current)
    elif change=='missing_key':signature.pop('gll.py')
    elif change=='extra_key':signature['unknown.py']='0'*64
    else:
        # Change the workflow bytes and its full hash consistently: even a
        # nonnumerical edit is not the specifically reviewed successor.
        from hashlib import sha256
        source_path=Path(module.__file__)
        changed=source_path.read_bytes()+b'\n# unreviewed change\n'
        original_read=Path.read_bytes;original_signature=module._signature
        def read(path):return changed if path==source_path else original_read(path)
        def current():
            result=original_signature();result['omega_x_workflow.py']=sha256(changed).hexdigest()
            return result
        monkeypatch.setattr(Path,'read_bytes',read)
        monkeypatch.setattr(module,'_signature',current)
    paths['metadata'].write_text(json.dumps(metadata))
    before={p:p.read_bytes() for p in paths.values()}
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('incompatible signature loaded VTR'))
    with pytest.raises(ValueError,match='signature'):
        run(case,root,indices=[1],resume=True)
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('settings',[
    {'periodic_axes':('x',)},
    {'expected_element_counts':(1,2,1)},
    {'expected_orders':(4,8,8)},
])
def test_ram_predecessor_rejects_changed_projection_settings(tmp_path,monkeypatch,settings):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case,root,paths,_=predecessor_frame(tmp_path)
    before={p:p.read_bytes() for p in paths.values()}
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('changed settings loaded VTR'))
    options={'expected_element_counts':(2,1,1),**settings}
    with pytest.raises(ValueError,match='metadata|snapshot'):
        module.run_omega_x_series(case,root,indices=[1],resume=True,**options)
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('change',['source_sha256','time','vtr','pvd'])
def test_ram_predecessor_retains_source_time_and_artifact_guards(tmp_path,monkeypatch,change):
    from vtkmodules import vtkIOXML
    case,root,paths,metadata=predecessor_frame(tmp_path)
    if change in ('vtr','pvd'):
        paths[change].write_bytes(paths[change].read_bytes()+b'\nchanged\n')
    else:
        metadata[change]='0'*64 if change=='source_sha256' else .14
        paths['metadata'].write_text(json.dumps(metadata))
    before={p:p.read_bytes() for p in paths.values()}
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('failed provenance loaded VTR'))
    with pytest.raises(ValueError,match='metadata|checksum'):
        run(case,root,indices=[1],resume=True)
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('change',['coordinates','timestamp','scalar','topology'])
def test_ram_predecessor_retains_vtr_scientific_validation(tmp_path,change):
    import nek_post.omega_x_workflow as module
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader,vtkXMLRectilinearGridWriter
    case,root,paths,metadata=predecessor_frame(tmp_path)
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(paths['vtr']));reader.Update()
    grid=reader.GetOutput()
    if change=='coordinates':grid.GetXCoordinates().SetValue(1,.01)
    elif change=='timestamp':grid.GetFieldData().GetArray('TIME_VALUE').SetValue(0,.14)
    elif change=='scalar':grid.GetPointData().GetArray('omega_x').SetValue(0,float('nan'))
    else:grid.SetDimensions(14,8,8)
    writer=vtkXMLRectilinearGridWriter();writer.SetFileName(str(paths['vtr']));writer.SetInputData(grid)
    assert writer.Write()==1
    del writer,grid,reader
    # Even if a synthetic fixture's outer checksum is refreshed, the original
    # physical-data checks must still reject incompatible content.
    metadata['output_sha256']['vtr']=module._file_hash(paths['vtr'])
    paths['metadata'].write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='coordinates|timestamp|scalar|topology'):
        run(case,root,indices=[1],resume=True)


def test_real_predecessor_three_frames_readonly_resume(monkeypatch):
    import nek_post.omega_x_workflow as module
    from vtkmodules import vtkIOXML
    case=Path('/data/Nek5000_data/case_N7_H')
    root=module.DEFAULT_OUTPUT_ROOT/case.name
    indices=(41,55,79)
    paths=[p for i in indices for p in module._paths(root/f'f{i:05d}',case.name,i).values()]
    if not all(p.is_file() for p in paths) or not all((case/f'GC0.f{i:05d}').is_file() for i in indices):
        pytest.skip('Optional existing N7_H frame 41/55/79 integration data is unavailable')
    before={p:module._file_hash(p) for p in paths}
    for i in indices:
        metadata=json.loads(module._paths(root/f'f{i:05d}',case.name,i)['metadata'].read_text())
        assert metadata['numerical_software_signature']['omega_x_workflow.py']==(
            '9f261baadf8e888ef216f83dc5cdcd2f9a4336e33c5b200340c60e22551d3736')
    def forbidden(*args,**kwargs):pytest.fail('Existing-frame integration test attempted native load or recomputation')
    monkeypatch.setattr(module,'read_nek_file',forbidden)
    monkeypatch.setattr(module,'compute_projected_omega_x',forbidden)
    with monkeypatch.context() as low:
        low.setattr(module,'available_memory_bytes',lambda:1)
        low.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',forbidden)
        rejected=module.run_omega_x_series(indices=indices,resume=True,dry_run=True)
        assert not rejected['resources_acceptable'] and rejected['estimated_ram_bytes']>0
    original_reader=vtkIOXML.vtkXMLRectilinearGridReader;original_require=module._require_ram
    checked=[];readers=[]
    def require(required,operation):
        available=original_require(required,operation)
        checked.append(operation)
        return available
    def reader():
        assert checked and checked[-1].startswith('Validating omega_x frame')
        checked.clear();readers.append(1)
        return original_reader()
    monkeypatch.setattr(module,'_require_ram',require)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',reader)
    report=module.run_omega_x_series(indices=[79,41,55],resume=True,dry_run=True)
    assert report['resources_acceptable'] and len(readers)==3
    assert [f['index'] for f in report['frames']]==list(indices)
    assert [f['time'] for f in report['frames']]==[10.0035071079,13.50377813401,19.50240408807]
    assert all(f['action']=='reused' and f['resume_validation']=='passed' for f in report['frames'])
    assert report['estimated_ram_bytes']==max(f['estimated_validation_ram_bytes'] for f in report['frames'])>0
    assert {p:module._file_hash(p) for p in paths}==before
