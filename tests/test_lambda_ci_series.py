"""Small analytic frames only; production snapshots are never pytest inputs."""
from pathlib import Path
import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from test_lambda_ci_plotting import mesh,tensor
from nek_post.lambda_ci_series import (
    expand_frame_indices,series_configuration,inspect_slice_series,run_slice_series,
    validate_frame_result,frame_manifest,_plans,_publish_directory,
)
from nek_post.lambda_ci_plotting import build_tensor_slice_plan,evaluate_tensor_slice_plan,evaluate_lambda_ci_slice
from nek_post.l2_projection import build_structured_node_map


def frames(tmp_path,count=3):
    from pymech.neksuite import writenek
    case=tmp_path/'synthetic';case.mkdir()
    times=[.03125,.278,.903]
    for index in range(1,count+1):
        data=mesh();data.time=times[index-1];data.istep=index*10
        data.elem[0].vel[:]*=index
        # Shared velocity need not be continuous: projected gradients are.
        writenek(str(case/f'GC0.f{index:05d}'),data)
    return case,tmp_path/'results'


def config():
    return series_configuration(x_slice=.3,periodic_axes=(),expected_element_counts=(2,2,2),subdivisions_per_element=8)


def options():
    return dict(x_slice=.3,periodic_axes=(),expected_element_counts=(2,2,2),subdivisions_per_element=8)


def test_frame_selection_and_cli():
    assert expand_frame_indices(1,3)==[1,2,3]
    assert expand_frame_indices(indices='3,1-2,2')==[1,2,3]
    assert expand_frame_indices(indices=[3,1],max_frames=1)==[1]
    for args in ({'indices':'3-1'},{'indices':[]},{'indices':[True]},{'start_index':0},{'max_frames':0}):
        with pytest.raises(ValueError):expand_frame_indices(**args)
    path=Path(__file__).resolve().parents[1]/'scripts/35_compute_lambda_ci_slice_series.py'
    spec=importlib.util.spec_from_file_location('series_cli',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=module.parse_args([])
    assert (args.start_index,args.end_index,args.subdivisions_per_element,args.vmin,args.vmax)==(1,81,28,0,10)
    assert module.parse_args(['--indices','17,33,49','--resume']).resume


def test_cached_interpolation_equals_single_frame_and_rejects_changed_order():
    data=mesh(True);mapping=build_structured_node_map(data,periodic_axes=())
    plan=build_tensor_slice_plan(data,mapping,normal_axis='y',plane_value=.75,samples_per_element=29,x_min=.333)
    matrix_ids=[tuple(id(m) for m in entry[3:]) for entry in plan.entries]
    for multiplier in (1,2):
        a=tensor(data)*multiplier
        cached=evaluate_tensor_slice_plan(a,plan)
        single=evaluate_lambda_ci_slice(data,mapping,a,normal_axis='y',plane_value=.75,samples_per_element=29,x_min=.333)
        np.testing.assert_array_equal(cached.lambda_ci,single.lambda_ci)
        assert [tuple(id(m) for m in entry[3:]) for entry in plan.entries]==matrix_ids
    data.elem[0],data.elem[1]=data.elem[1],data.elem[0]
    with pytest.raises(ValueError,match='ordering'):mapping.validate_mesh(data)


def test_dry_run_actual_times_no_output_and_invalid_config(tmp_path):
    case,root=frames(tmp_path)
    report=run_slice_series(case,root,end_index=3,dry_run=True,**options())
    assert [f['time'] for f in report['snapshots']]==[.03125,.278,.903]
    assert report['expected_output_counts']==dict(png=6,pdf=6,vtr=6,pvd_entries_each=3)
    assert not root.exists()
    for extra in ({'vmax':0},{'subdivisions_per_element':True},{'chunk_size':0}):
        with pytest.raises(ValueError):run_slice_series(case,root,end_index=3,dry_run=True,**(options()|extra))
    with pytest.raises(ValueError):run_slice_series(case,case/'forbidden',end_index=3,**options())
    assert not (case/'forbidden').exists()


def test_sequence_operator_reuse_resume_sorted_pvd_and_source_protection(tmp_path,monkeypatch):
    import nek_post.lambda_ci_series as module
    case,root=frames(tmp_path)
    source_bytes={p.name:p.read_bytes() for p in case.iterdir()}
    builds=[];original=module.build_l2_projection_operator
    def build(*args,**kwargs):
        builds.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'build_l2_projection_operator',build)
    result=run_slice_series(case,root,indices=[2,3],save_samples=True,log_name='batch',**options())
    assert builds==[1] and result['operator_build_count']==1 and result['slice_plan_build_count']==2
    assert [s['operator_reused'] for s in result['frames']]==[False,True]
    first=next(root.glob('f00002/plots/*.png')).read_bytes()
    resumed=run_slice_series(case,root,indices=[2,3],resume=True,log_name='resume',**options())
    assert builds==[1] and resumed['operator_build_count']==0
    assert all(s['status']=='reused' for s in resumed['frames'])
    assert next(root.glob('f00002/plots/*.png')).read_bytes()==first
    result=run_slice_series(case,root,indices=[1],resume=True,log_name='earlier',**options())
    for path in root.glob('*series.pvd'):
        entries=ET.parse(path).findall('Collection/DataSet')
        assert [float(e.get('timestep')) for e in entries]==[.03125,.278,.903]
        assert len({e.get('file') for e in entries})==3
        assert all((path.parent/e.get('file')).is_file() for e in entries)
    assert {p.name:p.read_bytes() for p in case.iterdir()}==source_bytes
    assert result['final_verification']['counts']==dict(png=2,pdf=2,vtr=2)
    # Existence alone is not a valid skip; preserve corrupted data for review.
    broken=next(root.glob('f00002/plots/*.pdf'));broken.write_bytes(b'%PDF broken')
    with pytest.raises(ValueError,match='checksum|PDF'):
        run_slice_series(case,root,indices=[2],resume=True,dry_run=True,**options())
    assert broken.read_bytes()==b'%PDF broken'


def test_atomic_publish_no_overwrite_and_explicit_staging_recovery(tmp_path,monkeypatch):
    import nek_post.lambda_ci_series as module
    case,root=frames(tmp_path,count=1)
    original=module._publish_directory
    def interrupted(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(module,'_publish_directory',interrupted)
    with pytest.raises(KeyboardInterrupt):run_slice_series(case,root,indices=[1],log_name='interrupt',**options())
    assert not (root/'f00001').exists()
    stage=next((root/'.stage5-staging').iterdir())
    assert next(stage.glob('plots/*paraview.json')).is_file()
    with pytest.raises(RuntimeError,match='recover-staging'):
        run_slice_series(case,root,indices=[1],resume=True,dry_run=True,**options())
    monkeypatch.setattr(module,'_publish_directory',original)
    report=run_slice_series(case,root,indices=[1],resume=True,recover_staging=True,log_name='recovery',**options())
    assert report['frames'][0]['status']=='recovered' and report['operator_build_count']==0
    other=tmp_path/'other';other.mkdir();(other/'preserve').write_text('yes')
    with pytest.raises(FileExistsError):_publish_directory(other,root/'f00001')
    assert (other/'preserve').read_text()=='yes'


def test_incompatible_mesh_stops_before_second_frame_publication(tmp_path):
    from pymech.neksuite import writenek
    case,root=frames(tmp_path,count=2)
    second=mesh();second.time=.278;second.elem[0].pos[0]+=1e-4
    writenek(str(case/'GC0.f00002'),second)
    with pytest.raises(ValueError,match='geometry or element ordering'):
        run_slice_series(case,root,end_index=2,log_name='mismatch',**options())
    assert (root/'f00001').is_dir() and not (root/'f00002').exists()
    summary=json.loads((root/'logs/mismatch_summary.json').read_text())
    assert summary['status']=='failed' and summary['operator_build_count']==1


def test_incomplete_result_and_duplicate_physical_time_are_rejected(tmp_path):
    from pymech.neksuite import writenek
    case,root=frames(tmp_path,count=2)
    frame=root/'f00001';frame.mkdir(parents=True);original=frame/'keep.txt';original.write_text('keep')
    with pytest.raises(ValueError,match='Incomplete frame'):
        run_slice_series(case,root,indices=[1],resume=True,dry_run=True,**options())
    assert original.read_text()=='keep'
    duplicate=mesh();duplicate.time=.03125;writenek(str(case/'GC0.f00002'),duplicate)
    with pytest.raises(ValueError,match='timestamps'):
        run_slice_series(case,root,end_index=2,resume=True,dry_run=True,**options())


def test_legacy_single_frame_result_can_be_reused_without_recomputation(tmp_path):
    from nek_post.lambda_ci_plotting import run_lambda_ci_slice_plots
    case,root=frames(tmp_path,count=1)
    run_lambda_ci_slice_plots(case,1,x_slice=.3,periodic_axes=(),expected_element_counts=(2,2,2),
                             subdivisions_per_element=8,output_dir=root/'f00001'/'plots',export_paraview=True,save_samples=True)
    before={str(p):p.read_bytes() for p in root.glob('f00001/**/*') if p.is_file()}
    report=run_slice_series(case,root,indices=[1],resume=True,log_name='legacy',**options())
    assert report['frames'][0]['legacy_validated'] and report['operator_build_count']==0
    assert {str(p):p.read_bytes() for p in root.glob('f00001/**/*') if p.is_file()}==before


@pytest.mark.parametrize('y_midplane,case_name',[(.75,'synthetic'),(.6,'synthetic'),(.6,'case_y_midplane')])
def test_explicit_series_identity_and_completed_summary(tmp_path,y_midplane,case_name):
    from nek_post.lambda_ci_series import verify_series_outputs,_read_snapshot
    case,root=frames(tmp_path,count=2)
    case=case.rename(tmp_path/case_name)
    settings=options()|{'y_midplane':y_midplane}
    report=run_slice_series(case,root,indices=[2,1],log_name='identity',**settings)
    assert report['status']=='completed' and report['final_verification']['validation']=='passed'
    for path in (root/'processing_summary.json',root/'logs/identity_summary.json'):
        assert json.loads(path.read_text())['status']=='completed'
    cfg=config()|{'y_midplane':y_midplane}
    pvds={}
    for key,coordinate in (('y_midplane',y_midplane),('x_slice',.3)):
        sample=frame_manifest(case.name,1,cfg,root/'f00001')[key+'_vtr']
        suffix=sample.stem.split('_lambda_ci_',1)[1]
        path=root/f'{case.name}_lambda_ci_{suffix}_series.pvd';pvds[key]=path
        entries=ET.parse(path).findall('Collection/DataSet')
        assert [float(e.get('timestep')) for e in entries]==[.03125,.278]
        assert len({e.get('file') for e in entries})==2
        for index,entry in enumerate(entries,1):
            target=frame_manifest(case.name,index,cfg,root/f'f{index:05d}')[key+'_vtr']
            assert (path.parent/entry.get('file')).resolve()==target.resolve()
            from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
            from vtkmodules.util.numpy_support import vtk_to_numpy
            reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(target));reader.Update()
            getter=reader.GetOutput().GetYCoordinates if key=='y_midplane' else reader.GetOutput().GetXCoordinates
            np.testing.assert_array_equal(vtk_to_numpy(getter()),[coordinate])
    # The other plane's valid file and correct time must not satisfy this series.
    tree=ET.parse(pvds['x_slice'])
    wrong=ET.parse(pvds['y_midplane']).find('Collection/DataSet').get('file')
    tree.find('Collection/DataSet').set('file',wrong);tree.write(pvds['x_slice'])
    data=_read_snapshot(case/'GC0.f00001',coordinates_only=True)
    mapping=build_structured_node_map(data,periodic_axes=())
    with pytest.raises(ValueError,match='Missing PVD frame/actual timestamp'):
        verify_series_outputs(root,case.name,report['preflight']['snapshots'],cfg,_plans(data,mapping,cfg))


def test_stale_legacy_append_does_not_block_published_frame_resume(tmp_path,monkeypatch):
    import nek_post.lambda_ci_series as module
    case,root=frames(tmp_path,count=2)
    run_slice_series(case,root,indices=[2],log_name='later',**options())
    stale={}
    for pvd in root.glob('*series.pvd'):
        for name in (pvd.name+'.append','.'+pvd.name+'.unknown.tmp'):
            path=root/name;path.write_bytes(b'unknown interrupted writer; preserve')
            stale[path]=path.read_bytes()
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(module,'_update_series',interrupted)
        with pytest.raises(KeyboardInterrupt):
            run_slice_series(case,root,indices=[1],log_name='interrupted_pvd',**options())
    before={p:p.read_bytes() for p in root.glob('f*/**/*') if p.is_file()}
    assert (root/'f00001').is_dir()
    report=run_slice_series(case,root,indices=[2,1],resume=True,log_name='repaired',**options())
    assert report['status']=='completed' and report['operator_build_count']==0
    assert all(s['status']=='reused' for s in report['frames'])
    assert json.loads((root/'processing_summary.json').read_text())['status']=='completed'
    assert {p:p.read_bytes() for p in before}==before
    assert {p:p.read_bytes() for p in stale}==stale
    for path in root.glob('*series.pvd'):
        entries=ET.parse(path).findall('Collection/DataSet')
        assert [float(e.get('timestep')) for e in entries]==[.03125,.278]
        assert len({e.get('file') for e in entries})==2


@pytest.mark.parametrize('existing',[False,True])
@pytest.mark.parametrize('failure',['write','fsync','xml','entries','replace'])
def test_pvd_failed_update_preserves_destination_and_unknown_temporaries(tmp_path,monkeypatch,existing,failure):
    import nek_post.lambda_ci_slice_export as module
    a=tmp_path/'a.vtr';a.write_bytes(b'fixture A')
    b=tmp_path/'b.vtr';b.write_bytes(b'fixture B')
    path=tmp_path/'series.pvd'
    if existing:module.append_slice_series_pvd(path,a,time=.278)
    before=path.read_bytes() if existing else None
    unknown=tmp_path/'.series.pvd.unknown.tmp';unknown.write_bytes(b'preserve unknown temp')
    def fail(*args,**kwargs):raise OSError('injected update failure')
    if failure=='write':
        # A partial write must close/unlink only this invocation's temporary file.
        original=module.os.fdopen
        class BrokenStream:
            def __init__(self,fd,*args,**kwargs):self.stream=original(fd,*args,**kwargs)
            def __enter__(self):return self
            def __exit__(self,*exc):self.stream.close()
            def fileno(self):return self.stream.fileno()
            def write(self,payload):self.stream.write(payload[:20]);raise OSError('injected update failure')
        monkeypatch.setattr(module.os,'fdopen',BrokenStream)
    elif failure=='xml':
        monkeypatch.setattr(module.ET,'tostring',lambda *args,**kwargs:b'<broken XML')
    elif failure=='entries':
        original=module.ET.tostring
        def wrong_entries(root,**kwargs):
            altered=ET.fromstring(original(root,**kwargs))
            altered.find('Collection/DataSet').set('timestep','99.0')
            return original(altered,**kwargs)
        monkeypatch.setattr(module.ET,'tostring',wrong_entries)
    else:
        operation='link' if failure=='replace' and not existing else failure
        monkeypatch.setattr(module.os,operation,fail)
    error=ET.ParseError if failure=='xml' else ValueError if failure=='entries' else OSError
    with pytest.raises(error):
        module.append_slice_series_pvd(path,b,time=.03125,sort_by_time=True)
    assert (path.read_bytes() if path.exists() else None)==before
    assert unknown.read_bytes()==b'preserve unknown temp'
    assert set(tmp_path.glob('.series.pvd.*.tmp'))=={unknown}
    assert a.read_bytes()==b'fixture A' and b.read_bytes()==b'fixture B'


def test_pvd_cooperating_concurrent_writers_keep_both_entries(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import nek_post.lambda_ci_slice_export as module
    a=tmp_path/'a.vtr';a.write_bytes(b'fixture A')
    b=tmp_path/'b.vtr';b.write_bytes(b'fixture B')
    path=tmp_path/'series.pvd'
    publishing=Event();release=Event();second_started=Event()
    original_replace=module.os.replace;original_link=module.os.link
    def publish(operation,source,destination):
        if not publishing.is_set():
            publishing.set()
            if not release.wait(10):raise TimeoutError('test did not release writer')
        return operation(source,destination)
    def second():
        second_started.set()
        return module.append_slice_series_pvd(path,b,time=.03125,sort_by_time=True)
    monkeypatch.setattr(module.os,'replace',lambda a,b:publish(original_replace,a,b))
    monkeypatch.setattr(module.os,'link',lambda a,b:publish(original_link,a,b))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(module.append_slice_series_pvd,path,a,time=.278,sort_by_time=True)
        try:
            assert publishing.wait(10)
            later=pool.submit(second)
            assert second_started.wait(10)
        finally:release.set()
        first.result(timeout=10);later.result(timeout=10)
    entries=ET.parse(path).findall('Collection/DataSet')
    assert [float(e.get('timestep')) for e in entries]==[.03125,.278]
    assert {e.get('file') for e in entries}=={'a.vtr','b.vtr'}
    before=path.read_bytes()
    for reference,t in ((a,.903),(b,.03125),(a,.278)):
        with pytest.raises(FileExistsError):module.append_slice_series_pvd(path,reference,time=t)
    assert path.read_bytes()==before
