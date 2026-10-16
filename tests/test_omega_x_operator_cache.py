"""Invocation-local operator reuse, exact geometry guards and retained RAM."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import nek_post.omega_x_workflow as workflow
from test_omega_x_workflow import case_files,run


def test_build_once_and_release_between_invocations(tmp_path,monkeypatch):
    case,root=case_files(tmp_path)
    original=workflow.build_l2_projection_operator;built=[];used=[]
    compute=workflow.compute_projected_omega_x
    def build(*args,**kwargs):
        operator=original(*args,**kwargs);built.append(operator);return operator
    def calculate(data,operator,**kwargs):
        used.append(operator);return compute(data,operator,**kwargs)
    monkeypatch.setattr(workflow,'build_l2_projection_operator',build)
    monkeypatch.setattr(workflow,'compute_projected_omega_x',calculate)
    first=run(case,root,indices=[1,2])
    assert len(built)==1 and used==[built[0],built[0]]
    assert first['operator_build_count']==1
    assert [f['operator_reused'] for f in first['frames']]==[False,True]
    second=run(case,tmp_path/'another-output',indices=[1])
    assert second['operator_build_count']==1 and len(built)==2 and built[0] is not built[1]


@pytest.mark.parametrize('change',['coordinates','element_order','orientation','node_shape','element_count'])
def test_mesh_mismatch_stops_without_rebuild_or_second_calculation(tmp_path,monkeypatch,change):
    case,root=case_files(tmp_path)
    original_read=workflow.read_nek_file;original_build=workflow.build_l2_projection_operator
    original_compute=workflow.compute_projected_omega_x;builds=[];computed=[]
    def read(path,**kwargs):
        data=original_read(path,**kwargs)
        if Path(path).name=='GC0.f00002':
            if change=='coordinates':
                for element in data.elem:element.pos[0]+=np.float32(.001)
            elif change=='element_order':data.elem.reverse()
            elif change=='orientation':data.elem[0].pos=data.elem[0].pos[...,::-1].copy()
            elif change=='node_shape':data.elem[0].pos=data.elem[0].pos[:,:,:,:-1].copy()
            else:data.elem.pop()
        return data
    def build(*args,**kwargs):builds.append(1);return original_build(*args,**kwargs)
    def compute(*args,**kwargs):computed.append(1);return original_compute(*args,**kwargs)
    monkeypatch.setattr(workflow,'read_nek_file',read)
    monkeypatch.setattr(workflow,'build_l2_projection_operator',build)
    monkeypatch.setattr(workflow,'compute_projected_omega_x',compute)
    with pytest.raises(ValueError,match='geometry|ordering|mismatch'):
        run(case,root,indices=[1,2])
    assert builds==[1] and computed==[1]
    assert (root/case.name/'f00001').is_dir() and not (root/case.name/'f00002').exists()


@pytest.mark.parametrize('change',['periodic','element_counts','orders','global_map'])
def test_cached_configuration_and_mapping_are_explicitly_checked(tmp_path,change):
    case,_=case_files(tmp_path)
    data=workflow.read_nek_file(case/'GC0.f00001')
    operator=workflow.build_l2_projection_operator(data,periodic_axes=('x','y'),expected_element_counts=(2,1,1))
    config={'periodic_axes':['x','y'],'expected_element_counts':[2,1,1],'expected_orders':[8,8,8]}
    if change=='periodic':config['periodic_axes']=['x']
    elif change=='element_counts':config['expected_element_counts']=[1,2,1]
    elif change=='orders':config['expected_orders']=[7,8,8]
    else:operator=replace(operator,node_map=replace(operator.node_map,global_shape_xyz=(1,2,3)))
    with pytest.raises(ValueError,match='configuration|mapping'):
        workflow._validate_cached_operator(operator,data,config)


def test_cached_curl_and_projection_equal_fresh_operator_exactly(tmp_path,monkeypatch):
    from pymech.neksuite import readnek,writenek
    case,root=case_files(tmp_path)
    later=readnek(str(case/'GC0.f00002'))
    for element in later.elem:
        element.vel[1]*=np.float32(1.25);element.vel[2]*=np.float32(-.75)
    writenek(str(case/'GC0.f00002'),later)
    compute=workflow.compute_projected_omega_x;compared=[]
    def compare(data,operator,**kwargs):
        cached=compute(data,operator,**kwargs)
        fresh=workflow.build_l2_projection_operator(data,periodic_axes=('x','y'),expected_element_counts=(2,1,1))
        uncached=compute(data,fresh,**kwargs)
        assert cached.dtype==uncached.dtype==np.float64
        assert np.array_equal(operator.weights,fresh.weights) and np.array_equal(operator.mass,fresh.mass)
        assert np.array_equal(cached,uncached)
        compared.append(cached.copy())
        return cached
    monkeypatch.setattr(workflow,'compute_projected_omega_x',compare)
    report=run(case,root,indices=[1,2])
    assert report['operator_build_count']==1 and len(compared)==2
    assert not np.array_equal(compared[0],compared[1])


def test_completed_resume_never_constructs_an_operator(tmp_path,monkeypatch):
    case,root=case_files(tmp_path);run(case,root,indices=[1,2])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(workflow,'build_l2_projection_operator',lambda *a,**k:pytest.fail('Resume built an operator'))
    report=run(case,root,indices=[1,2],resume=True)
    assert report['operator_build_count']==0 and all(f['status']=='reused' for f in report['frames'])
    assert report['preflight']['estimated_operator_cache_bytes']==0
    assert {p:p.read_bytes() for p in before}==before


def three_frames(tmp_path):
    from pymech.neksuite import readnek,writenek
    case,root=case_files(tmp_path)
    third=readnek(str(case/'GC0.f00002'));third.time=2.4;third.istep=39
    writenek(str(case/'GC0.f00003'),third)
    return case,root


def test_mixed_middle_resume_retains_operator_and_accounts_for_validation_ram(tmp_path,monkeypatch):
    case,root=three_frames(tmp_path);run(case,root,indices=[2])
    before={p:p.read_bytes() for p in (root/case.name/'f00002').iterdir()}
    plan=run(case,root,indices=[1,2,3],resume=True,dry_run=True)
    a,b,c=plan['frames'];cache=b['estimated_operator_cache_bytes']
    assert cache>0 and a['estimated_operator_cache_bytes']==c['estimated_operator_cache_bytes']==cache
    assert b['estimated_computation_ram_bytes']==0
    assert b['estimated_validation_ram_bytes']==workflow._validation_ram_bytes((15,8,8),
        Path(b['output_paths']['vtr']).stat().st_size,retained_bytes=cache)
    assert plan['estimated_ram_bytes']==max(f['estimated_ram_bytes'] for f in plan['frames'])
    original=workflow.validate_omega_x_frame;retained=[]
    def validate(*args,**kwargs):
        retained.append((kwargs['index'],kwargs.get('retained_operator_bytes',0)))
        return original(*args,**kwargs)
    monkeypatch.setattr(workflow,'validate_omega_x_frame',validate)
    report=run(case,root,indices=[1,2,3],resume=True)
    assert report['operator_build_count']==1
    assert [f['status'] for f in report['frames']]==['generated','reused','generated']
    assert any(index==2 and size>0 for index,size in retained)
    assert {p:p.read_bytes() for p in before}==before


def test_late_low_ram_after_cache_stops_before_next_vtr_or_snapshot(tmp_path,monkeypatch):
    from vtkmodules import vtkIOXML
    case,root=three_frames(tmp_path);run(case,root,indices=[2])
    before={p:p.read_bytes() for p in (root/case.name/'f00002').iterdir()}
    available=[10**15];stopped=[False]
    original_reader=vtkIOXML.vtkXMLRectilinearGridReader;original_read=workflow.read_nek_file
    def progress(message):
        if message.startswith('{'):
            available[0]=1;stopped[0]=True
    def reader():
        assert not stopped[0],'VTR reader ran after cache RAM rejection'
        return original_reader()
    def read(*args,**kwargs):
        assert not stopped[0],'Native reader ran after cache RAM rejection'
        return original_read(*args,**kwargs)
    monkeypatch.setattr(workflow,'available_memory_bytes',lambda:available[0])
    monkeypatch.setattr(workflow,'read_nek_file',read)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',reader)
    with pytest.raises(MemoryError,match='insufficient RAM'):
        run(case,root,indices=[1,2,3],resume=True,progress=progress)
    assert stopped[0] and not (root/case.name/'f00003').exists()
    assert {p:p.read_bytes() for p in before}==before


def test_direct_vtr_validation_budgets_retained_operator(tmp_path,monkeypatch):
    import json
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);report=run(case,root,indices=[1])
    frame=report['frames'][0];metadata=json.loads(Path(frame['output_paths']['metadata']).read_text())
    plan=run(case,root,indices=[1],resume=True,dry_run=True)
    enough_without_cache=plan['estimated_ram_bytes']+1
    monkeypatch.setattr(workflow,'available_memory_bytes',lambda:enough_without_cache)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('Ignored retained operator'))
    with pytest.raises(MemoryError,match='Validating omega_x frame'):
        workflow.validate_omega_x_frame(Path(frame['output_paths']['vtr']).parent,case=case.name,index=1,
            time_value=.13,config=plan['configuration'],source_sha256=metadata['source_sha256'],
            retained_operator_bytes=frame['operator_storage_bytes'])


@pytest.mark.parametrize('invalid',[-1,1.5,True,None])
def test_invalid_retained_cache_budget_cannot_bypass_vtr_gate(tmp_path,monkeypatch,invalid):
    import json
    from vtkmodules import vtkIOXML
    case,root=case_files(tmp_path);report=run(case,root,indices=[1])
    frame=report['frames'][0];metadata=json.loads(Path(frame['output_paths']['metadata']).read_text())
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('Invalid budget reached reader'))
    with pytest.raises(ValueError,match='retained_operator_bytes'):
        workflow.validate_omega_x_frame(Path(frame['output_paths']['vtr']).parent,case=case.name,index=1,
            time_value=.13,config=metadata['configuration'],source_sha256=metadata['source_sha256'],
            retained_operator_bytes=invalid)
