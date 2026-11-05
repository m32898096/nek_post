"""Self-contained scalar SEM reconstruction; no external production dependency."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from nek_post.gll import gll_nodes, barycentric_weights, barycentric_basis_and_derivative
from nek_post import omega_x_p7_slice as module


def mesh():
    def axis(edges):
        pieces=[(a+b)/2+(b-a)/2*gll_nodes(8) for a,b in zip(edges[:-1],edges[1:])]
        return np.concatenate([pieces[0],*(p[1:] for p in pieces[1:])])
    return tuple(axis(edges) for edges in ((-1.,0.,.5,1.),(0.,.75,1.5),(0.,.4,1.)))


def sample(function,**kwargs):
    coordinates=mesh()
    z,y,x=np.meshgrid(*coordinates[::-1],indexing='ij')
    values=np.asarray(function(x,y,z)+np.zeros_like(x),dtype=np.float64)
    result=module.reconstruct_p7_midplane(coordinates,values,element_counts=(3,2,2),time=19.50240408807,**kwargs)
    return coordinates,values,result


@pytest.mark.parametrize('constant',[-4.,0.,3.5])
def test_constant_preservation(constant):
    _,_,result=sample(lambda x,y,z:constant)
    np.testing.assert_allclose(result.p7,constant,rtol=0,atol=4e-14)
    np.testing.assert_allclose(result.linear,constant,rtol=0,atol=4e-14)
    assert result.diagnostics['maximum_shared_trace_difference']<1e-12


@pytest.mark.parametrize('degree',range(8))
@pytest.mark.parametrize('plane',[.75,.63])
def test_tensor_polynomials_through_degree_seven(degree,plane):
    # All three coordinates participate; a scalar trilinear substitute cannot
    # reproduce the higher-degree cases or an interior y plane correctly.
    f=lambda x,y,z:(1+x**degree)*(1-.2*y**degree)*(1+.3*z**degree)-2
    _,_,result=sample(f,y=plane)
    zz,xx=np.meshgrid(result.z,result.x,indexing='ij')
    np.testing.assert_allclose(result.p7,f(xx,plane,zz),rtol=2e-13,atol=2e-13)
    assert result.diagnostics['maximum_shared_trace_difference']<2e-12
    if degree>1:
        assert result.diagnostics['max_abs_difference']>1e-5


def test_native_nodal_roundtrip_and_shared_element_coordinates():
    coordinates,values,result=sample(lambda x,y,z:x**7-3*z**6+2*y**5)
    # Recover a complete positive-x element from its 29 samples, using eight
    # independent equispaced targets. Evaluate that polynomial back at all
    # original GLL nodes in both directions, including nonuniform interior nodes.
    block=result.p7[:29,27:56][::4,::4]
    uniform=np.linspace(-1,1,8);weights=barycentric_weights(uniform)
    basis=np.vstack([barycentric_basis_and_derivative(uniform,weights,q)[0] for q in gll_nodes(8)])
    np.testing.assert_allclose(basis@block@basis.T,values[:8,7,14:22],atol=2e-13,rtol=2e-13)
    assert result.p7.shape==(57,56)
    assert np.all(result.x>0) and result.x[-1]==1
    assert result.z[0]==0 and result.z[-1]==1
    assert len(np.unique(result.x))==len(result.x)
    assert len(np.unique(result.z))==len(result.z)
    assert result.x[27]==.5 and result.z[28]==.4
    assert result.diagnostics['exact_native_y_indices']==[7]
    assert result.diagnostics['intersected_y_elements']==[0,1]
    assert result.diagnostics['maximum_plane_mapping_residual']==0
    assert coordinates[1][7]==.75


def test_linear_representation_uses_identical_physical_targets():
    f=lambda x,y,z:2*x-3*y+4*z-1
    _,_,result=sample(f,y=.63)
    zz,xx=np.meshgrid(result.z,result.x,indexing='ij')
    np.testing.assert_allclose(result.linear,f(xx,.63,zz),atol=3e-15)
    np.testing.assert_allclose(result.p7,result.linear,atol=1e-14)
    assert result.diagnostics['intersected_y_elements']==[0]
    assert result.diagnostics['exact_native_y_indices']==[]


def test_overshoot_reported_and_not_clipped():
    # GLL samples of a degree-7 polynomial have an extremum between nodes.
    f=lambda x,y,z:1-(x-.26)**2
    coordinates,values,result=sample(f)
    assert result.diagnostics['maximum_local_overshoot_above']>1e-5
    assert result.diagnostics['overshooting_patch_count']>0
    assert result.diagnostics['overshoot_clipped'] is False
    assert result.p7.max()>values[:,7,7:15].max()
    np.testing.assert_allclose(result.p7,f(result.x[None,:],.75,result.z[:,None])+np.zeros_like(result.p7),atol=2e-14)


@pytest.mark.parametrize('change',['count','nongll','reversed','dtype','nonfinite','shape','out_of_bounds','empty_crop','samples'])
def test_invalid_geometry_and_input_rejected(change):
    coordinates=list(mesh());values=np.ones(tuple(map(len,coordinates))[::-1]);options={}
    counts=(3,2,2)
    if change=='count':counts=(2,2,2)
    elif change=='nongll':coordinates[0]=coordinates[0].copy();coordinates[0][1]+=.001
    elif change=='reversed':coordinates[0]=coordinates[0][::-1]
    elif change=='dtype':values=values.astype(np.float32)
    elif change=='nonfinite':values[0,0,0]=np.nan
    elif change=='shape':values=values[:,:,:-1]
    elif change=='out_of_bounds':options['y']=1.6
    elif change=='empty_crop':options['x_min']=2.
    elif change=='samples':options['samples_per_element']=1
    with pytest.raises(ValueError):
        module.reconstruct_p7_midplane(coordinates,values,element_counts=counts,**options)


def test_signed_vtr_roundtrip_order_time_and_no_overwrite(tmp_path):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    _,_,result=sample(lambda x,y,z:3*x-4*z)
    path=tmp_path/'slice.vtr';module.write_omega_slice_vtr(path,result)
    checked=module.validate_slice_vtr(path,result)
    assert checked['dimensions_xyz']==[56,1,57]
    assert checked['point_count']==56*57
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(path));reader.Update();grid=reader.GetOutput()
    scalar=vtk_to_numpy(grid.GetPointData().GetScalars())
    assert scalar.dtype==np.float64 and scalar.min()<0<scalar.max()
    assert grid.GetPointData().GetScalars().GetName()=='omega_x'
    for k,i in ((0,2),(13,5),(56,55)):
        assert scalar[k*56+i]==result.p7[k,i]
        assert grid.GetPoint(k*56+i)==(result.x[i],.75,result.z[k])
    assert grid.GetFieldData().GetArray('TIME_VALUE').GetValue(0)==19.50240408807
    before=path.read_bytes()
    with pytest.raises(FileExistsError):module.write_omega_slice_vtr(path,result)
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='coordinates'):
        module.validate_slice_vtr(path,replace(result,x=result.x+.01))


def test_readback_ram_gate_precedes_vtr_reader(tmp_path,monkeypatch):
    from vtkmodules import vtkIOXML
    _,_,result=sample(lambda x,y,z:x-z)
    path=tmp_path/'slice.vtr';module.write_omega_slice_vtr(path,result)
    monkeypatch.setattr(module.workflow,'available_memory_bytes',lambda:1)
    monkeypatch.setattr(vtkIOXML,'vtkXMLRectilinearGridReader',lambda:pytest.fail('RAM rejection loaded VTR'))
    with pytest.raises(MemoryError,match='insufficient RAM'):module.validate_slice_vtr(path,result)


@pytest.mark.parametrize('failure',['missing','duplicate','configuration','collision','protected','symlink','ram'])
def test_runner_refuses_bad_sources_and_preserves_originals(tmp_path,monkeypatch,failure):
    root=tmp_path/'original';root.mkdir();output=tmp_path/'new'
    case=tmp_path/'case_N7_H';case.mkdir()
    def forbidden(*args,**kwargs):pytest.fail('Rejected inputs reached scientific computation/read')
    monkeypatch.setattr(module.workflow,'compute_projected_omega_x',forbidden)
    monkeypatch.setattr(module.workflow,'read_nek_file',forbidden)
    monkeypatch.setattr(module,'reconstruct_p7_midplane',forbidden)
    monkeypatch.setattr(module,'read_nek_header',forbidden)
    config={'periodic_axes':['x','y'],'expected_element_counts':[272,12,8],'expected_orders':[8,8,8]}
    if failure=='collision':(output/case.name/'f00079').mkdir(parents=True)
    elif failure=='protected':output=root
    elif failure=='symlink':
        output.mkdir();(output/case.name).symlink_to(root,target_is_directory=True)
    elif failure in ('configuration','ram'):
        metadata=root/f'{case.name}_f00079_omega_x.json'
        metadata.write_text(json.dumps({'configuration':config if failure=='ram' else {}}))
        metadata.with_suffix('.vtr').write_bytes(b'never load this VTR')
        if failure=='ram':monkeypatch.setattr(module.workflow,'available_memory_bytes',lambda:1)
    elif failure=='duplicate':
        for name in ('a','b'):
            (root/name).mkdir();(root/name/f'{case.name}_f00079_omega_x.json').write_text('{}')
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    expected=MemoryError if failure=='ram' else FileExistsError if failure=='collision' else ValueError
    with pytest.raises(expected):module.run_omega_x_p7_slice(case=case,input_dir=root,output_dir=output)
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    assert not (output/case.name/'f00079').exists() or failure=='collision'


def test_cli_errors_are_explicit_without_output(tmp_path,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/38_plot_omega_x_p7_midplane.py'
    spec=importlib.util.spec_from_file_location('omega_p7_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    output=tmp_path/'output'
    assert cli.main(['--input-dir',str(tmp_path/'missing'),'--output-dir',str(output)])==1
    assert json.loads(capsys.readouterr().out)['status']=='invalid'
    assert not output.exists()


def test_controlled_figures_decodable_and_one_color_scale(tmp_path):
    from nek_post.lambda_ci_series import _image_readable,_pdf_readable
    _,_,result=sample(lambda x,y,z:2*x**7-3*z**3)
    paths={f'{key}_{fmt}':tmp_path/f'{key}.{fmt}' for key in ('linear','p7','comparison') for fmt in ('png','pdf')}
    style=module._figures(result,paths,case='synthetic',frame=79,dpi=60)
    assert style['vmin']==-style['vmax']
    assert style['vmax']==max(abs(result.p7).max(),abs(result.linear).max())
    for key,path in paths.items():
        if key.endswith('png'):assert min(_image_readable(path))>100
        else:_pdf_readable(path)
    before={p:p.read_bytes() for p in paths.values()}
    with pytest.raises(FileExistsError):module._figures(result,paths,case='synthetic',frame=79,dpi=60)
    assert {p:p.read_bytes() for p in paths.values()}==before
