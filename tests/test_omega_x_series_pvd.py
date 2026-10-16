"""Saved-volume validation, actual times and atomic no-overwrite series XML."""
from pathlib import Path
import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from nek_post import omega_x_series_pvd as series
from nek_post import omega_x_workflow as workflow
from test_omega_x_workflow import case_files,run


def production(tmp_path, *, nested=False):
    case,root=case_files(tmp_path)
    outputs=run(case,root,indices=[1,2])
    tree=root/case.name
    if nested:
        outer=tmp_path/'nested';outer.mkdir();shutil.move(str(tree),str(outer/case.name));tree=outer
        for path in tree.rglob('*.json'):
            meta=json.loads(path.read_text())
            meta['output_paths']={k:str(v) for k,v in workflow._paths(path.parent,case.name,meta['frame_index']).items()}
            path.write_text(json.dumps(meta))
    return case,tree


def build(case,root,**kwargs):
    return series.build_omega_x_series_pvd(case,root,indices=[2,1],expected_element_counts=(2,1,1),
        expected_bounds=((0.,2.),(0.,1.),(0.,1.)),**kwargs)


def bytes_in(root):
    return {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('nested',[False,True])
def test_actual_times_order_relative_paths_stats_and_existing_artifact_preservation(tmp_path,monkeypatch,nested):
    case,root=production(tmp_path,nested=nested)
    before=bytes_in(root);source_before=bytes_in(case)
    monkeypatch.setattr(workflow,'compute_projected_omega_x',lambda *a,**k:pytest.fail('Validation computed omega'))
    monkeypatch.setattr(workflow,'read_nek_file',lambda *a,**k:pytest.fail('Validation read native arrays'))
    report=build(case,root)
    assert report['status']=='completed' and report['pvd_entries']==2
    assert [f['index'] for f in report['frames']]==[1,2]
    assert [f['time'] for f in report['frames']]==[.13,1.27]
    target=Path(report['series_pvd']);entries=ET.parse(target).findall('Collection/DataSet')
    assert all(not Path(e.get('file')).is_absolute() and (target.parent/e.get('file')).is_file() for e in entries)
    assert all(('synthetic/' in e.get('file'))==nested for e in entries)
    assert {p:p.read_bytes() for p in before}==before and bytes_in(case)==source_before
    for frame in report['frames']:
        stats=frame['statistics']
        assert stats['min']<0<stats['max'] and stats['rms']>0
        assert stats['min']<=stats['mean']<=stats['max']
        from test_lambda_ci_workflow import read_vtr
        from vtkmodules.util.numpy_support import vtk_to_numpy
        grid=read_vtr(Path(frame['output_paths']['vtr']))
        values=vtk_to_numpy(grid.GetPointData().GetArray('omega_x'))
        np.testing.assert_allclose([stats['mean'],stats['rms']],
            [np.mean(values),np.sqrt(np.mean(values**2))],rtol=1e-14,atol=1e-14)
    with pytest.raises(FileExistsError,match='overwrite'):build(case,root)
    assert series.validate_series_pvd(target,report['frames'])==2
    completed_before=bytes_in(root)
    audited=build(case,root,dry_run=True)
    assert audited['status']=='validated' and bytes_in(root)==completed_before


def test_dry_run_creates_nothing_in_new_output_directory(tmp_path):
    case,root=production(tmp_path);before=bytes_in(root);dest=tmp_path/'not-created'
    report=build(case,root,output_dir=dest,dry_run=True)
    assert report['status']=='validated' and not dest.exists() and bytes_in(root)==before


@pytest.mark.parametrize('change',['missing_frame','duplicate_frame','missing_metadata','missing_checksum',
    'timestamp','source_path','source_hash','output_path','periodic','signature','bad_reference',
    'corrupt_vtr','unexpected_file','malformed_metadata','invalid_pvd_header','corrupt_vtr_refreshed_checksum'])
def test_invalid_production_reports_errors_and_never_publishes(tmp_path,change):
    case,root=production(tmp_path);folder=root/'f00002';paths=workflow._paths(folder,case.name,2)
    if change=='missing_frame':shutil.rmtree(folder)
    elif change=='duplicate_frame':shutil.copytree(folder,root/'other'/'f00002')
    elif change=='missing_metadata':paths['metadata'].unlink()
    elif change=='corrupt_vtr':paths['vtr'].write_bytes(b'corrupted')
    elif change=='unexpected_file':(folder/'extra.vtr').write_bytes(b'extra')
    elif change=='malformed_metadata':paths['metadata'].write_text('[]')
    else:
        metadata=json.loads(paths['metadata'].read_text())
        if change=='missing_checksum':metadata.pop('source_sha256')
        elif change=='timestamp':metadata['time']=2.0
        elif change=='source_path':metadata['source_file']=str(case/'GC0.f00001')
        elif change=='source_hash':metadata['source_sha256']='0'*64
        elif change=='output_path':metadata['output_paths']['vtr']='wrong.vtr'
        elif change=='periodic':metadata['configuration']['periodic_axes']=['x']
        elif change=='signature':metadata['numerical_software_signature']['omega_x_workflow.py']='0'*64
        elif change=='bad_reference':
            xml=ET.parse(paths['pvd']);xml.find('Collection/DataSet').set('file','missing.vtr');xml.write(paths['pvd'])
            metadata['output_sha256']['pvd']=workflow._file_hash(paths['pvd'])
        elif change=='invalid_pvd_header':
            xml=ET.parse(paths['pvd']);xml.getroot().set('type','WrongType');xml.write(paths['pvd'])
            metadata['output_sha256']['pvd']=workflow._file_hash(paths['pvd'])
        elif change=='corrupt_vtr_refreshed_checksum':
            paths['vtr'].write_bytes(b'corrupted')
            metadata['output_sha256']['vtr']=workflow._file_hash(paths['vtr'])
        paths['metadata'].write_text(json.dumps(metadata))
    before=bytes_in(root)
    report=build(case,root)
    assert report['status']=='invalid' and report['errors']
    assert not (root/f'{case.name}_omega_x_series.pvd').exists() and bytes_in(root)==before
    if change=='missing_frame':assert report['missing_frames']==[2]
    if change=='duplicate_frame':assert report['duplicate_frames']==[2]


@pytest.mark.parametrize('change',['nan','wrong_field','float32','grid','zero','one_sided'])
def test_scientific_grid_scalar_checks_and_valid_zero_or_one_sided_fields(tmp_path,change):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader,vtkXMLRectilinearGridWriter
    from vtkmodules.util.numpy_support import vtk_to_numpy
    case,root=production(tmp_path);paths=workflow._paths(root/'f00002',case.name,2)
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(paths['vtr']));reader.Update();grid=reader.GetOutput()
    field=grid.GetPointData().GetArray('omega_x');values=vtk_to_numpy(field)
    if change=='nan':values[0]=np.nan
    elif change=='wrong_field':field.SetName('wrong')
    elif change=='float32':
        from vtkmodules.util.numpy_support import numpy_to_vtk
        grid.GetPointData().RemoveArray('omega_x')
        replacement=numpy_to_vtk(values.astype(np.float32),deep=True);replacement.SetName('omega_x')
        grid.GetPointData().AddArray(replacement)
    elif change=='grid':grid.GetXCoordinates().SetValue(1,.01)
    elif change=='zero':values[:]=0
    else:values[:]=3.5
    writer=vtkXMLRectilinearGridWriter();writer.SetFileName(str(paths['vtr']));writer.SetInputData(grid)
    assert writer.Write()==1
    metadata=json.loads(paths['metadata'].read_text())
    metadata['output_sha256']['vtr']=workflow._file_hash(paths['vtr'])
    if change=='grid':
        metadata['visualization']['coordinate_sha256'][0]=workflow._coordinate_hash(vtk_to_numpy(grid.GetXCoordinates()))
    if change in ('zero','one_sided'):metadata['omega_x']={'min':float(values.min()),'max':float(values.max())}
    paths['metadata'].write_text(json.dumps(metadata))
    del writer,field,values,grid,reader
    report=build(case,root)
    assert report['status']==('completed' if change in ('zero','one_sided') else 'invalid')
    if report['status']=='completed':
        statistics=report['frames'][1]['statistics'];expected=0 if change=='zero' else 3.5
        assert statistics=={'min':expected,'max':expected,'mean':expected,'rms':expected}


def test_atomic_failure_and_racing_existing_output_never_replace(tmp_path,monkeypatch):
    case,root=production(tmp_path);before=bytes_in(root)
    def failed(*args,**kwargs):raise OSError('injected link failure')
    with monkeypatch.context() as patch:
        patch.setattr(series.os,'link',failed)
        with pytest.raises(OSError,match='injected'):build(case,root)
    assert bytes_in(root)==before and not list(root.glob('*.tmp'))
    original=series.os.link;target=root/f'{case.name}_omega_x_series.pvd'
    def race(source,destination):
        target.write_bytes(b'preserve concurrent output');return original(source,destination)
    monkeypatch.setattr(series.os,'link',race)
    with pytest.raises(FileExistsError):build(case,root)
    assert target.read_bytes()==b'preserve concurrent output'


def test_changed_original_artifact_aborts_before_publication(tmp_path,monkeypatch):
    case,root=production(tmp_path);original=series._diagnostics;changed=[False]
    def diagnose(path,**kwargs):
        result=original(path,**kwargs)
        if not changed[0]:
            metadata=path.with_suffix('.json');metadata.write_bytes(metadata.read_bytes()+b'\n');changed[0]=True
        return result
    monkeypatch.setattr(series,'_diagnostics',diagnose)
    with pytest.raises(RuntimeError,match='artifacts changed'):build(case,root)
    assert not (root/f'{case.name}_omega_x_series.pvd').exists()


def test_duplicate_selection_and_low_ram_are_rejected_before_vtr_reader(tmp_path,monkeypatch):
    from vtkmodules import vtkIOXML
    case,root=production(tmp_path)
    with pytest.raises(ValueError,match='Duplicate'):series.validate_omega_x_production(case,root,indices=[1,1])
    monkeypatch.setattr(workflow,'available_memory_bytes',lambda:1)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('Low RAM loaded VTR'))
    with pytest.raises(MemoryError):build(case,root,dry_run=True)


def test_cli_options_and_failure_report(tmp_path,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/37_validate_omega_x_series.py'
    spec=importlib.util.spec_from_file_location('omega_series_cli',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    args=cli.parse_args(['--case','N7_H','--input-dir',str(tmp_path),'--dry-run'])
    assert args.case=='N7_H' and args.dry_run and args.input_dir==tmp_path
    assert cli.main(['--case',str(tmp_path/'missing'),'--input-dir',str(tmp_path)])==1
    assert json.loads(capsys.readouterr().out)['status']=='invalid'


def test_series_destination_cannot_pollute_frame_outputs(tmp_path):
    case,root=production(tmp_path);before=bytes_in(root)
    with pytest.raises(ValueError,match='inside frame outputs'):
        build(case,root,output_dir=root/'f00001')
    assert bytes_in(root)==before
