"""Analytic tensors and small meshes; no production snapshot or VTR needed."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.l2_projection import build_structured_node_map
from nek_post.lambda_ci_plotting import (
    TECPLOT_RGB_POINTS,build_tecplot_colormap,sample_projected_tensor_slice,
    evaluate_lambda_ci_slice,plot_lambda_ci_slice,run_lambda_ci_slice_plots,
)


def mesh(reverse=False,shape=(8,8,8)):
    from pymech.core import HexaData
    data=HexaData(3,8,list(shape[::-1]),[3,3,1,1,0])
    data.wdsz,data.endian,data.time,data.istep=4,'little',.12345,3
    t,s,r=np.meshgrid(*(gll_nodes(n) for n in shape),indexing='ij')
    for ei,e in enumerate(data.elem):
        ix,iy,iz=ei%2,(ei//2)%2,ei//4
        e.pos[:]=np.stack((ix-1+(r+1)/2,iy+(s+1)/2,iz+(t+1)/2))
        x,y,z=e.pos;e.vel[:]=np.stack((-2*y,2*x,.5*z))
        if reverse and ei%2:e.pos[:]=e.pos[:,::-1,::-1,::-1];e.vel[:]=e.vel[:,::-1,::-1,::-1]
    return data


def tensor(data,kind='rotation'):
    result=[]
    for e in data.elem:
        x,y,z=e.pos;a=np.zeros((*x.shape,3,3))
        if kind=='rotation':a[...,0,1]=-(2+x+y*y+z**3);a[...,1,0]=-a[...,0,1];a[...,2,2]=.5
        elif kind=='polynomial':
            for i in range(3):
                for j in range(3):a[...,i,j]=(i+1)*x**7+(j+1)*y**6+(3*i+j+1)*z**5
        else:a[...,0,1]=-x;a[...,1,0]=x
        result.append(a)
    return np.array(result)


def test_exact_tecplot_control_points_and_continuous_colormap():
    cmap=build_tecplot_colormap()
    for channel,idx in zip(('red','green','blue'),range(3)):
        np.testing.assert_array_equal(cmap._segmentdata[channel][:,0],[q for q,_ in TECPLOT_RGB_POINTS])
        np.testing.assert_array_equal(cmap._segmentdata[channel][:,1],[rgb[idx] for _,rgb in TECPLOT_RGB_POINTS])
    # Matplotlib samples the exact piecewise-linear definition into a LUT.
    # Its evaluated colors can differ by at most one LUT interval's RGB slope.
    positions=np.array([q for q,_ in TECPLOT_RGB_POINTS]);colors=np.array([rgb for _,rgb in TECPLOT_RGB_POINTS])
    quantization_bound=np.max(np.abs(np.diff(colors,axis=0))/np.diff(positions)[:,None])/(cmap.N-1)
    for q,rgb in TECPLOT_RGB_POINTS:np.testing.assert_allclose(cmap(q)[:3],rgb,atol=quantization_bound,rtol=0)


@pytest.mark.parametrize('axis,value,crop',[('x',.371,None),('x',0.,None),('y',.75,0.),('y',1.,.333)])
@pytest.mark.parametrize('reverse',[False,True])
def test_planes_orientation_exact_crop_and_analytic_eigenvalues(axis,value,crop,reverse):
    data=mesh(reverse);mapping=build_structured_node_map(data,periodic_axes=())
    sampled=evaluate_lambda_ci_slice(data,mapping,tensor(data),normal_axis=axis,plane_value=value,
                                    samples_per_element=12,x_min=crop,eigenvalue_chunk_rows=3)
    h,z=np.meshgrid(sampled.horizontal,sampled.z)
    expected=2+(value if axis=='x' else h)+(h*h if axis=='x' else value*value)+z**3
    np.testing.assert_allclose(sampled.lambda_ci,expected,atol=5e-13,rtol=5e-14)
    np.testing.assert_array_equal(sampled.lambda_ci_squared,sampled.lambda_ci**2)
    assert sampled.lambda_ci.shape==(23,len(sampled.horizontal))
    assert sampled.normal_axis==axis and sampled.plane_value==value
    assert sampled.horizontal_axis==('y' if axis=='x' else 'x')
    if crop is not None:assert sampled.horizontal[0]==crop and np.all(sampled.horizontal>=crop)
    assert sampled.sampling_diagnostics['maximum_shared_tensor_difference']<2e-12


@pytest.mark.parametrize('axis,value',[('x',.33),('y',.75)])
def test_all_nine_components_and_noncubic_axis_ordering(axis,value):
    data=mesh(True,shape=(6,7,8));mapping=build_structured_node_map(data,periodic_axes=())
    plane=sample_projected_tensor_slice(data,mapping,tensor(data,'polynomial'),normal_axis=axis,plane_value=value,samples_per_element=15)
    h,z=np.meshgrid(plane.horizontal,plane.z);x=value if axis=='x' else h;y=h if axis=='x' else value
    for i in range(3):
        for j in range(3):np.testing.assert_allclose(plane.tensors[...,i,j],(i+1)*x**7+(j+1)*y**6+(3*i+j+1)*z**5,atol=5e-11,rtol=3e-13)


def test_eigenvalue_calculation_follows_tensor_interpolation():
    data=mesh();mapping=build_structured_node_map(data,periodic_axes=())
    samples=evaluate_lambda_ci_slice(data,mapping,tensor(data,'nonlinear'),normal_axis='x',plane_value=-.5,samples_per_element=9)
    np.testing.assert_allclose(samples.lambda_ci,.5,atol=2e-15)
    samples=evaluate_lambda_ci_slice(data,mapping,tensor(data,'nonlinear'),normal_axis='x',plane_value=0.,samples_per_element=9)
    np.testing.assert_array_equal(samples.lambda_ci,0.)


def test_non_polynomial_scalar_is_recomputed_not_interpolated():
    data=mesh();mapping=build_structured_node_map(data,periodic_axes=())
    a=tensor(data,'nonlinear')
    # Shift the zero crossing inside an element, away from original GLL nodes.
    for e,values in zip(data.elem,a):values[...,0,1]=-(e.pos[0]+.5);values[...,1,0]=e.pos[0]+.5
    samples=evaluate_lambda_ci_slice(data,mapping,a,normal_axis='x',plane_value=-.5,samples_per_element=9)
    np.testing.assert_allclose(samples.lambda_ci,0,atol=2e-16)
    assert np.min(np.abs(data.elem[0].pos[0]+.5))>0


def test_invalid_parameters_nonfinite_and_shared_traces():
    data=mesh();mapping=build_structured_node_map(data,periodic_axes=());a=tensor(data)
    base=dict(normal_axis='y',plane_value=.75)
    for change in ({'normal_axis':'z'},{'plane_value':np.nan},{'plane_value':3.},{'samples_per_element':1},
                   {'samples_per_element':True},{'x_min':np.inf},{'x_min':2.},{'normal_axis':'x','x_min':0}):
        with pytest.raises(ValueError):sample_projected_tensor_slice(data,mapping,a,**(base|change))
    with pytest.raises(ValueError):sample_projected_tensor_slice(data,mapping,a[0],**base)
    a[0,0,0,0,0,0]=np.nan
    with pytest.raises(ValueError):sample_projected_tensor_slice(data,mapping,a,**base)
    a=tensor(data);a[1,...,0,1]+=1
    with pytest.raises(ValueError,match='traces'):sample_projected_tensor_slice(data,mapping,a,normal_axis='x',plane_value=0)


def test_figures_and_no_overwrite(tmp_path):
    data=mesh();mapping=build_structured_node_map(data,periodic_axes=())
    samples=evaluate_lambda_ci_slice(data,mapping,tensor(data),normal_axis='y',plane_value=.75,x_min=0,samples_per_element=12)
    png,pdf=tmp_path/'plot.png',tmp_path/'plot.pdf'
    plot_lambda_ci_slice(samples,png,pdf,case='synthetic',frame=79,time=.12345,vmin=0,vmax=10)
    assert png.read_bytes().startswith(b'\x89PNG') and pdf.read_bytes().startswith(b'%PDF')
    before=png.read_bytes()
    with pytest.raises(FileExistsError):plot_lambda_ci_slice(samples,png,pdf,case='synthetic',frame=79,time=.12345)
    assert png.read_bytes()==before
    with pytest.raises(ValueError):plot_lambda_ci_slice(samples,tmp_path/'bad.png',tmp_path/'bad.pdf',case='synthetic',frame=79,time=.12345,vmin=10,vmax=0)


def test_complete_workflow_two_slices_and_sample_reuse(tmp_path,monkeypatch):
    from pymech.neksuite import writenek
    import nek_post.lambda_ci_plotting as module
    data=mesh();case=tmp_path/'synthetic';case.mkdir();source=case/'GC0.f00001';writenek(str(source),data)
    before=source.read_bytes();out=tmp_path/'plots'
    args=dict(y_midplane=.75,x_slice=.3,x_min=0,samples_per_element=12,output_dir=out,
              save_samples=True,periodic_axes=(),expected_element_counts=(2,2,2))
    dry=run_lambda_ci_slice_plots(case,1,**args,dry_run=True)
    assert not out.exists() and dry['no_vtr_scalar_read'] and dry['color_range']==[0,10]
    report=run_lambda_ci_slice_plots(case,1,**args)
    assert source.read_bytes()==before and report['protected_files_unchanged']
    assert report['slices']['y_midplane']['shape']==[23,12]
    assert report['slices']['x_slice']['shape']==[23,23]
    for key in ('y_midplane','x_slice'):
        with np.load(report['output_paths'][f'{key}_samples'],allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays['lambda_ci'],2,atol=1e-5)
            np.testing.assert_array_equal(arrays['lambda_ci_squared'],arrays['lambda_ci']**2)
            assert arrays['time']==data.time
    assert json.loads(Path(report['output_paths']['metadata']).read_text())['planes']['x_min']==0
    with pytest.raises(FileExistsError):run_lambda_ci_slice_plots(case,1,**args)
    with pytest.raises(ValueError):run_lambda_ci_slice_plots(case,1,**(args|{'output_dir':case/'unsafe'}))
    with pytest.raises(ValueError):run_lambda_ci_slice_plots(case,1,**(args|{'samples_per_element':True}))


def test_cli_parameters():
    path=Path(__file__).resolve().parents[1]/'scripts/34_plot_lambda_ci_slices.py'
    spec=importlib.util.spec_from_file_location('plot_cli',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    args=m.parse_args([])
    assert (args.y_midplane,args.x_slice,args.x_min,args.samples_per_element,args.vmin,args.vmax)==(.75,15,0,None,0,10)
    assert args.subdivisions_per_element==28 and args.export_paraview
    assert m.parse_args(['--samples-per-element','48','--x-min','.2','--save-samples']).save_samples


@pytest.mark.parametrize('axis,plane,dimensions',[('y',.75,(29,1,57)),('x',.371,(1,57,57))])
def test_plane_vtr_reader_round_trip_and_x_fast_order(tmp_path,axis,plane,dimensions):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from nek_post.lambda_ci_slice_export import write_lambda_ci_slice_vtr
    data=mesh(True);mapping=build_structured_node_map(data,periodic_axes=())
    samples=evaluate_lambda_ci_slice(data,mapping,tensor(data),normal_axis=axis,plane_value=plane,
                                    samples_per_element=29,x_min=0 if axis=='y' else None)
    path=tmp_path/'slice.vtr'
    write_lambda_ci_slice_vtr(path,samples,time=.12345)
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(path));reader.Update();grid=reader.GetOutput()
    assert grid.GetDimensions()==dimensions
    assert grid.GetNumberOfPoints()==np.prod(dimensions)
    assert grid.GetNumberOfCells()==(len(samples.z)-1)*(len(samples.horizontal)-1)
    physical=[vtk_to_numpy(getter()) for getter in (grid.GetXCoordinates,grid.GetYCoordinates,grid.GetZCoordinates)]
    np.testing.assert_array_equal(physical[0 if axis=='x' else 1],[plane])
    np.testing.assert_array_equal(physical[1 if axis=='x' else 0],samples.horizontal)
    np.testing.assert_array_equal(physical[2],samples.z)
    for name,values in [('lambda_ci',samples.lambda_ci),('lambda_ci_squared',samples.lambda_ci_squared)]:
        np.testing.assert_array_equal(vtk_to_numpy(grid.GetPointData().GetArray(name)),values.ravel())
    assert vtk_to_numpy(grid.GetFieldData().GetArray('TIME_VALUE'))[0]==.12345
    before=path.read_bytes()
    with pytest.raises(FileExistsError):write_lambda_ci_slice_vtr(path,samples,time=.12345)
    assert path.read_bytes()==before


def test_pvd_append_preserves_prior_entries_and_relative_paths(tmp_path):
    import xml.etree.ElementTree as ET
    from nek_post.lambda_ci_slice_export import append_slice_series_pvd,check_slice_series_entry
    case=tmp_path/'case';frames=case/'f00079'/'slices';frames.mkdir(parents=True)
    a=frames/'a.vtr';a.write_bytes(b'existing synthetic dataset')
    b=frames/'b.vtr';b.write_bytes(b'second synthetic dataset')
    pvd=case/'series.pvd'
    append_slice_series_pvd(pvd,a,time=19.50240408807)
    first=ET.parse(pvd).find('Collection/DataSet').attrib.copy()
    assert first['file']=='f00079/slices/a.vtr' and float(first['timestep'])==19.50240408807
    append_slice_series_pvd(pvd,b,time=20.5)
    entries=ET.parse(pvd).findall('Collection/DataSet')
    assert len(entries)==2 and entries[0].attrib==first
    before=pvd.read_bytes()
    for reference,time in ((a,21),(b,20.5),(a,19.50240408807)):
        with pytest.raises(FileExistsError):append_slice_series_pvd(pvd,reference,time=time)
    assert pvd.read_bytes()==before
    check_slice_series_entry(case/'new.pvd',case/'future.vtr',time=30)
    assert not (case/'new.pvd').exists()
    with pytest.raises(ValueError):append_slice_series_pvd(case/'missing.pvd',case/'missing.vtr',time=30)


def test_paraview_complete_workflow_defaults_and_existing_plot_protection(tmp_path):
    from pymech.neksuite import writenek
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    import xml.etree.ElementTree as ET
    data=mesh();case=tmp_path/'synthetic';case.mkdir();writenek(str(case/'GC0.f00001'),data)
    out=tmp_path/'results'/'synthetic'/'f00001'/'plots';out.mkdir(parents=True)
    old=out/'synthetic_f00001_lambda_ci_high_order_slices.json';old.write_text('preserve previous output')
    args=dict(y_midplane=.75,x_slice=.3,x_min=0,output_dir=out,export_paraview=True,
              save_samples=True,periodic_axes=(),expected_element_counts=(2,2,2))
    dry=run_lambda_ci_slice_plots(case,1,**args,dry_run=True)
    assert dry['subdivisions_per_element']==28 and dry['samples_per_element']==29
    assert Path(dry['output_paths']['y_midplane_vtr']).parent==out.parent/'slices'
    report=run_lambda_ci_slice_plots(case,1,**args)
    assert old.read_text()=='preserve previous output'
    for key in ('y_midplane','x_slice'):
        reader=vtkXMLRectilinearGridReader();reader.SetFileName(report['output_paths'][f'{key}_vtr']);reader.Update()
        with np.load(report['output_paths'][f'{key}_samples']) as arrays:
            np.testing.assert_array_equal(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('lambda_ci')),arrays['lambda_ci'].ravel())
        pvd=Path(report['series_paths'][key]);entry=ET.parse(pvd).find('Collection/DataSet')
        assert float(entry.get('timestep'))==data.time
        assert (pvd.parent/entry.get('file')).resolve()==Path(report['output_paths'][f'{key}_vtr']).resolve()
    with pytest.raises(FileExistsError):run_lambda_ci_slice_plots(case,1,**args)
    for invalid in (True,0,2.5):
        with pytest.raises(ValueError):run_lambda_ci_slice_plots(case,1,**(args|{'subdivisions_per_element':invalid}),dry_run=True)


def test_continuous_paraview_preset_and_no_overwrite(tmp_path):
    from nek_post.lambda_ci_slice_export import write_tecplot_paraview_preset
    path=tmp_path/'tecplot.json';write_tecplot_paraview_preset(path)
    preset=json.loads(path.read_text())[0]
    assert preset['ColorSpace']=='RGB'
    np.testing.assert_array_equal(preset['RGBPoints'],[v for q,rgb in TECPLOT_RGB_POINTS for v in (q,*rgb)])
    assert 'IndexedColors' not in preset
    with pytest.raises(FileExistsError):write_tecplot_paraview_preset(path)


@pytest.mark.parametrize('axis,plane',[('y',.75),('x',.371)])
def test_linear_reference_comparison_uses_same_physical_points(tmp_path,axis,plane):
    from nek_post.lambda_ci_export import RectilinearLambdaFields,write_rectilinear_lambda_vtr
    from nek_post.lambda_ci_slice_export import sample_linear_vtr_reference
    data=mesh();mapping=build_structured_node_map(data,periodic_axes=())
    samples=evaluate_lambda_ci_slice(data,mapping,tensor(data),normal_axis=axis,plane_value=plane,samples_per_element=11,x_min=0 if axis=='y' else None)
    coords=(np.array([-1.,-.2,1.]),np.array([0.,.6,2.]),np.array([0.,.8,2.]))
    z,y,x=np.meshgrid(*coords[::-1],indexing='ij');scalar=2+x+2*y+3*z
    path=tmp_path/'reference.vtr';write_rectilinear_lambda_vtr(path,RectilinearLambdaFields(coords,scalar,scalar**2),time=.12345)
    linear=sample_linear_vtr_reference(path,samples,time=.12345,chunk_rows=3)
    h,z=np.meshgrid(samples.horizontal,samples.z)
    np.testing.assert_allclose(linear,2+(h if axis=='y' else plane)+2*(plane if axis=='y' else h)+3*z,atol=3e-15)
    with pytest.raises(ValueError,match='timestamp'):sample_linear_vtr_reference(path,samples,time=99.)


def test_paraview_sibling_directory_cannot_redirect_into_original_data(tmp_path):
    case=tmp_path/'original';case.mkdir()
    frame=tmp_path/'result'/'f00079';frame.mkdir(parents=True)
    (frame/'slices').symlink_to(case,target_is_directory=True)
    with pytest.raises(ValueError,match='outside original data'):
        run_lambda_ci_slice_plots(case,79,output_dir=frame/'plots',export_paraview=True,dry_run=True)
    assert not list(case.iterdir())
