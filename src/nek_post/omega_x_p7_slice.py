"""P7 reconstruction of a saved, already projected scalar VTR; no new curl/L2."""
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nek_post import omega_x_workflow as workflow
from nek_post.gll import gll_nodes, barycentric_weights, barycentric_basis_and_derivative
from nek_post.gll_directional_integration import _element_geometry_tolerance, _affine_coordinate_values
from nek_post.lambda_ci_plotting import _physical_basis, _segments
from nek_post.lambda_ci_series import _publish_directory, _image_readable, _pdf_readable
from nek_post.io_nek import read_nek_header


DEFAULT_OUTPUT_ROOT = Path('/data/Nek5000_data/results/h_refinement/omega_x_highorder_slices')


@dataclass(frozen=True)
class OmegaSlice:
    x: np.ndarray
    z: np.ndarray
    y: float
    time: float
    p7: np.ndarray
    linear: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class P7SlicePlan:
    """Invocation-local stationary geometry and the existing P7 basis matrices."""
    coordinates: tuple
    element_counts: tuple
    y: float
    x_min: float
    samples_per_element: int
    residuals: tuple
    containing: tuple
    x: np.ndarray
    z: np.ndarray
    x_segments: dict
    z_segments: dict
    identity: np.ndarray

    def validate(self, coordinates, element_counts, y, x_min, samples_per_element):
        if (tuple(element_counts)!=self.element_counts or y!=self.y or x_min!=self.x_min
                or samples_per_element!=self.samples_per_element or len(coordinates)!=3
                or any(np.asarray(a).dtype!=np.float64 or not np.array_equal(a,b)
                       for a,b in zip(coordinates,self.coordinates))):
            raise ValueError('Cached P7 plan geometry/topology/configuration differs.')


def build_p7_slice_plan(coordinates, *, element_counts, y=.75, x_min=0., samples_per_element=29):
    """Prepare only the coordinate/basis work previously done per reconstruction."""
    count=workflow._positive_integer(samples_per_element,'samples_per_element',2)
    if len(element_counts)!=3 or any(not isinstance(c,int) or isinstance(c,bool) or c<1 for c in element_counts):
        raise ValueError('Three positive element counts are required.')
    if len(coordinates)!=3 or not np.isfinite([y,x_min]).all():
        raise ValueError('Three coordinates and finite plane/crop required.')
    profiles=[];residuals=[]
    coordinates=tuple(np.asarray(c).copy() for c in coordinates)
    for coordinate,n in zip(coordinates,element_counts):
        p,r=recover_gll_profiles(coordinate,n);profiles.append(p);residuals.append(r)
    px,py,pz=profiles
    containing=tuple((cell,*_physical_basis(line,float(y))) for cell,line in py.items() if line[0]<=y<=line[-1])
    if not containing:raise ValueError('Slice lies outside the native mesh; extrapolation forbidden.')
    positive={cell:p for cell,p in px.items() if p[-1]>x_min}
    x,x_segments=_segments(positive,count,float(x_min));z,z_segments=_segments(pz,count)
    nodes=gll_nodes(8);weights=barycentric_weights(nodes)
    identity=np.vstack([barycentric_basis_and_derivative(nodes,weights,q)[0] for q in nodes])
    for a in (*coordinates,x,z,identity,*(b for _,b,_ in containing),
              *(b for _,b in x_segments.values()),*(b for _,b in z_segments.values())):
        a.setflags(write=False)
    return P7SlicePlan(coordinates,tuple(element_counts),float(y),float(x_min),count,
                       tuple(residuals),containing,x,z,x_segments,z_segments,identity)


def recover_gll_profiles(coordinate, count):
    """Recover SEM intervals, not individual VTK cells, at stride 8-1."""
    count=workflow._positive_integer(count,'element_count')
    vector = np.asarray(coordinate)
    if (vector.dtype != np.float64 or vector.ndim != 1 or len(vector) != 7*count+1
            or not np.isfinite(vector).all() or not np.all(np.diff(vector)>0)):
        raise ValueError('Native coordinate shape/type/orientation is incompatible with P7 SEM grouping.')
    profiles = {}; maximum = 0.
    for cell in range(count):
        line = vector[7*cell:7*cell+8]
        residual = float(np.max(np.abs(line-_affine_coordinate_values(line,gll_nodes(8)))))
        if residual > _element_geometry_tolerance(line):
            raise ValueError(f'Invalid GLL element grouping/profile at interval {cell}.')
        profiles[cell] = line
        maximum = max(maximum,residual)
    return profiles,maximum


def reconstruct_p7_midplane(coordinates, values, *, element_counts, y=.75, x_min=0.,
                            samples_per_element=29, time=0., plan=None):
    """Interpolate a canonical physical (z,y,x) scalar, then crop strictly x>x_min.

    Dense coordinates use the same uniform reference sampling and physical
    coordinate mapping as lambda_ci slices. Linear and P7 share these targets.
    """
    count=workflow._positive_integer(samples_per_element,'samples_per_element',2)
    if len(element_counts)!=3 or any(not isinstance(c,int) or isinstance(c,bool) or c<1 for c in element_counts):
        raise ValueError('Three positive element counts are required.')
    if not np.isfinite([y,x_min,time]).all():raise ValueError('Finite plane/crop/time required.')
    if plan is None:
        plan=build_p7_slice_plan(coordinates,element_counts=element_counts,y=y,x_min=x_min,samples_per_element=count)
    plan.validate(coordinates,element_counts,y,x_min,count)
    residuals=plan.residuals
    array=np.asarray(values)
    if array.dtype!=np.float64 or array.shape!=tuple(len(c) for c in coordinates)[::-1] or not np.isfinite(array).all():
        raise ValueError('Projected scalar must be finite float64 in (Nz,Ny,Nx) order.')
    containing=plan.containing
    x,z,x_segments,z_segments=plan.x,plan.z,plan.x_segments,plan.z_segments
    output=np.full((len(z),len(x)),np.nan)
    max_jump=0.;below=above=0.;overshooting=0;nodal_error=0.;support_low=np.inf;support_high=-np.inf
    identity=plan.identity
    for iy,by,_ in containing:
        for iz,(sz,bz) in z_segments.items():
            for ix,(sx,bx) in x_segments.items():
                local=array[7*iz:7*iz+8,7*iy:7*iy+8,7*ix:7*ix+8]
                trace=np.einsum('y,zyx->zx',by,local,optimize=True)
                nodal_error=max(nodal_error,float(np.max(np.abs(identity@trace@identity.T-trace))))
                dense=bz@trace@bx.T
                previous=output[sz,sx];valid=np.isfinite(previous)
                if valid.any():
                    jump=float(np.max(np.abs(previous[valid]-dense[valid])));max_jump=max(max_jump,jump)
                    if jump>2e-12*max(1.,float(np.max(np.abs(dense)))):
                        raise ValueError('Shared SEM element traces disagree.')
                previous[~valid]=dense[~valid]
                lo,hi=float(trace.min()),float(trace.max());support_low=min(support_low,lo);support_high=max(support_high,hi)
                selected=dense[:,x[sx]>x_min]
                if selected.size:
                    low=max(0.,lo-float(selected.min()));high=max(0.,float(selected.max())-hi)
                    below=max(below,low);above=max(above,high)
                    overshooting+=int(max(low,high)>1e-12*max(1.,abs(lo),abs(hi)))
    keep=x>x_min;x=x[keep];output=output[:,keep]
    if len(x)<2 or not np.isfinite(output).all():raise ValueError('Incomplete slice coverage/crop.')
    interpolator=RegularGridInterpolator(tuple(coordinates[::-1]),array,method='linear',bounds_error=True)
    linear=np.empty_like(output)
    for start in range(0,len(z),16):
        zz,xx=np.meshgrid(z[start:start+16],x,indexing='ij')
        points=np.stack((zz,np.full_like(xx,y),xx),axis=-1)
        linear[start:start+16]=interpolator(points)
    difference=output-linear
    def extrema(a):
        low=np.unravel_index(np.argmin(a),a.shape);high=np.unravel_index(np.argmax(a),a.shape)
        return {'min':float(a.min()),'max':float(a.max()),'min_location_xz':[float(x[low[1]]),float(z[low[0]])],
                'max_location_xz':[float(x[high[1]]),float(z[high[0]])]}
    diagnostics={'shape_zx':list(output.shape),'samples_per_element':count,'polynomial_degree':7,
        'element_counts_xyz':list(element_counts),'native_nodes_per_element':8,'native_stride':7,
        'intersected_y_elements':[cell for cell,_,_ in containing],
        'exact_native_y_indices':np.flatnonzero(coordinates[1]==y).tolist(),
        'y_element_boundaries':coordinates[1][::7].tolist(),'maximum_coordinate_gll_residual_xyz':residuals,
        'maximum_plane_mapping_residual':max(r for _,_,r in containing),
        'maximum_shared_trace_difference':max_jump,'maximum_native_nodal_reproduction_error':nodal_error,
        'native_y_trace_support_extrema':[support_low,support_high],
        'maximum_local_overshoot_below':below,'maximum_local_overshoot_above':above,
        'overshooting_patch_count':overshooting,'overshoot_clipped':False,
        'p7':extrema(output),'linear':extrema(linear),
        'max_abs_difference':float(np.max(np.abs(difference))),
        'rms_difference':float(np.sqrt(np.mean(difference*difference))),
        'mean_difference':float(difference.mean()),
        'p7_rms':float(np.sqrt(np.mean(output*output))),
        'comparison':'trilinear native VTR versus element-local P7 on identical physical targets'}
    return OmegaSlice(x,z,float(y),float(time),np.ascontiguousarray(output),np.ascontiguousarray(linear),diagnostics)


def write_omega_slice_vtr(path,samples):
    """A true 2D physical x-z plane, with signed float64 active omega_x."""
    from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
    from vtkmodules.util.numpy_support import numpy_to_vtk
    path=Path(path)
    if path.suffix!='.vtr' or not np.isfinite([samples.time,samples.y]).all():
        raise ValueError('Require VTR path and finite plane/time.')
    for vector in (samples.x,samples.z):
        if np.ndim(vector)!=1 or len(vector)<2 or not np.isfinite(vector).all() or not np.all(np.diff(vector)>0):
            raise ValueError('Slice coordinate vectors must increase strictly.')
    if samples.p7.shape!=(len(samples.z),len(samples.x)) or samples.linear.shape!=samples.p7.shape:
        raise ValueError('Invalid slice scalar shapes.')
    grid=vtkRectilinearGrid();grid.SetDimensions(len(samples.x),1,len(samples.z))
    for name,setter,vector in zip('XYZ',(grid.SetXCoordinates,grid.SetYCoordinates,grid.SetZCoordinates),
                                  (samples.x,np.array([samples.y]),samples.z)):
        a=numpy_to_vtk(np.asarray(vector,dtype=np.float64),deep=True);a.SetName(name);setter(a)
    for name,values in (('omega_x',samples.p7),('omega_x_linear',samples.linear),
                         ('omega_x_p7_minus_linear',samples.p7-samples.linear)):
        if not np.isfinite(values).all():raise ValueError('Nonfinite output scalar.')
        a=numpy_to_vtk(np.ascontiguousarray(values,dtype=np.float64).ravel(),deep=True);a.SetName(name)
        grid.GetPointData().AddArray(a)
    grid.GetPointData().SetActiveScalars('omega_x')
    a=numpy_to_vtk(np.array([samples.time]),deep=True);a.SetName('TIME_VALUE');grid.GetFieldData().AddArray(a)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb'):pass
    writer=vtkXMLRectilinearGridWriter();writer.SetFileName(str(path));writer.SetInputData(grid)
    writer.SetDataModeToAppended();writer.EncodeAppendedDataOff();writer.SetCompressorTypeToNone()
    if writer.Write()!=1 or writer.GetErrorCode():raise OSError('Failed to write slice VTR.')
    return path


def validate_slice_vtr(path,samples):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    retained=samples.p7.nbytes+samples.linear.nbytes
    required=workflow._validation_ram_bytes((len(samples.x),1,len(samples.z)),Path(path).stat().st_size,
                                           retained_bytes=retained)
    workflow._require_ram(required,'Validating reconstructed omega_x slice')
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(path));reader.Update();grid=reader.GetOutput()
    if reader.GetErrorCode() or grid.GetDimensions()!=(len(samples.x),1,len(samples.z)):
        raise ValueError('Slice VTR read-back/dimensions differ.')
    for getter,expected in zip((grid.GetXCoordinates,grid.GetYCoordinates,grid.GetZCoordinates),
                               (samples.x,np.array([samples.y]),samples.z)):
        vector=vtk_to_numpy(getter())
        if vector.dtype!=np.float64 or not np.array_equal(vector,expected):raise ValueError('Slice coordinates differ.')
    expected={'omega_x':samples.p7,'omega_x_linear':samples.linear,'omega_x_p7_minus_linear':samples.p7-samples.linear}
    if grid.GetPointData().GetScalars().GetName()!='omega_x' or grid.GetPointData().GetNumberOfArrays()!=3:
        raise ValueError('Incorrect active slice scalar.')
    for name,values in expected.items():
        scalar=vtk_to_numpy(grid.GetPointData().GetArray(name))
        if scalar.dtype!=np.float64 or not np.array_equal(scalar,values.ravel()):raise ValueError('Slice scalar/order differs.')
    if grid.GetFieldData().GetArray('TIME_VALUE').GetValue(0)!=samples.time:raise ValueError('Slice time differs.')
    for point in (0,len(samples.x)-1,grid.GetNumberOfPoints()-1):
        if grid.GetPoint(point)!=(float(samples.x[point%len(samples.x)]),samples.y,float(samples.z[point//len(samples.x)])):
            raise ValueError('Slice point ordering differs.')
    return {'dimensions_xyz':list(grid.GetDimensions()),'point_count':grid.GetNumberOfPoints(),'validation':'passed'}


def _figures(samples,paths,*,case,frame,dpi=220):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    amplitude=max(float(np.max(np.abs(samples.p7))),float(np.max(np.abs(samples.linear))),1e-12)
    common={'cmap':'RdBu_r','vmin':-amplitude,'vmax':amplitude,'shading':'nearest','antialiased':False,'rasterized':True}
    def draw(ax,values,title):
        artist=ax.pcolormesh(samples.x,samples.z,values,**common)
        ax.set(xlabel='x',ylabel='z',xlim=(0.,float(samples.x[-1])),ylim=(float(samples.z[0]),float(samples.z[-1])),aspect='equal',title=title)
        return artist
    title=f'{case}, frame {frame}, t={samples.time:.12g}, y={samples.y:g}'
    for key,values in (('linear',samples.linear),('p7',samples.p7)):
        fig=Figure(figsize=(18,3.1),layout='constrained');FigureCanvasAgg(fig)
        ax=fig.subplots();artist=draw(ax,values,f'{title}\n{key.upper()} representation')
        fig.colorbar(artist,ax=ax,label=r'$\omega_x$',shrink=.8)
        for fmt in ('png','pdf'):
            with paths[key+'_'+fmt].open('xb') as stream:fig.savefig(stream,format=fmt,dpi=dpi)
        fig.clear()
    fig=Figure(figsize=(28,3.8),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    labels=('Native VTR piecewise linear',f'Element-local P7 ({samples.diagnostics["samples_per_element"]} samples/edge)')
    for ax,values,label in zip(axes,(samples.linear,samples.p7),labels):
        artist=draw(ax,values,label)
    fig.suptitle(title);fig.colorbar(artist,ax=list(axes),label=r'$\omega_x$',shrink=.7)
    for fmt in ('png','pdf'):
        with paths['comparison_'+fmt].open('xb') as stream:fig.savefig(stream,format=fmt,dpi=dpi)
    fig.clear()
    return {'colormap':'RdBu_r','vmin':-amplitude,'vmax':amplitude,'aspect':'equal',
            'rendering':'nearest sample cells, no smoothing/filtering','dpi':dpi}


def load_projected_omega_frame(case_dir,frame,inputs,*,expected_element_counts=(272,12,8),
                               expected_bounds=((-17.,17.),(0.,1.5),(0.,1.)),required_ram=None):
    """Shared saved-scalar validation/read; never load a native velocity payload."""
    metadata=json.loads(inputs['metadata'].read_text())
    if not isinstance(metadata,dict):raise ValueError('Native metadata must be a JSON object.')
    config=metadata.get('configuration')
    if config!={'periodic_axes':['x','y'],'expected_element_counts':list(expected_element_counts),'expected_orders':[8,8,8]}:
        raise ValueError('Require the validated native P7 configuration.')
    shape=tuple(7*n+1 for n in expected_element_counts)
    required=workflow._validation_ram_bytes(shape,inputs['vtr'].stat().st_size) if required_ram is None else required_ram
    workflow._require_ram(required,'Saved omega_x P7 reconstruction/figures')
    before={str(p):workflow._file_hash(p) for p in inputs.values()}
    source=case_dir/f'GC0.f{frame:05d}';header=read_nek_header(source)
    if (header.nb_dims!=3 or header.nb_vars[0]!=3 or header.nb_vars[1]!=3
            or header.nb_files!=1 or header.nb_elems_file!=header.nb_elems
            or source.is_symlink() or header.nb_elems!=int(np.prod(expected_element_counts)) or tuple(header.orders)!=(8,8,8)
            or header.wdsz not in (4,8) or not np.isfinite(header.time)):
        raise ValueError('Incompatible native N7_H snapshot header.')
    source_hash=workflow._file_hash(source)
    if (metadata.get('source_file')!=str(source) or metadata.get('istep')!=header.istep
            or metadata.get('output_paths')!={k:str(p) for k,p in inputs.items()}):
        raise ValueError('Source/output path provenance differs.')
    workflow.validate_omega_x_frame(inputs['vtr'].parent,case=case_dir.name,index=frame,
        time_value=float(header.time),config=config,source_sha256=source_hash)
    info=metadata['visualization']
    if (info['bounds_xyz']!=[list(b) for b in expected_bounds] or not info.get('native_gll')
            or not info.get('periodic_endpoints_preserved') or info.get('interpolation')!='none'
            or metadata.get('source_unchanged') is not True):
        raise ValueError('N7_H native geometry/provenance differs.')
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    workflow._require_ram(required,'Loading validated omega_x VTR for P7 reconstruction')
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(inputs['vtr']));reader.Update();grid=reader.GetOutput()
    if reader.GetErrorCode() or grid.GetDimensions()!=shape:
        raise ValueError('Native omega_x VTR read failed or dimensions changed.')
    coordinates=tuple(vtk_to_numpy(getter()).copy() for getter in (grid.GetXCoordinates,grid.GetYCoordinates,grid.GetZCoordinates))
    values=vtk_to_numpy(grid.GetPointData().GetArray('omega_x')).copy().reshape(grid.GetDimensions()[::-1])
    del grid,reader
    if [[float(c[0]),float(c[-1])] for c in coordinates]!=[list(b) for b in expected_bounds]:
        raise ValueError('Native physical coordinate bounds differ.')
    return header,metadata,coordinates,values,source_hash,before


def run_omega_x_p7_slice(case='N7_H',frame=79,input_dir=None,output_dir=None,*,y=.75,samples_per_element=29):
    """Validate one saved frame, reconstruct, compare and atomically publish."""
    case_dir=workflow.resolve_case(case);frame=workflow._positive_integer(frame,'frame')
    root=(workflow.DEFAULT_OUTPUT_ROOT/case_dir.name if input_dir is None else Path(input_dir)).resolve()
    output_root=(DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)).resolve()
    target=output_root/case_dir.name/f'f{frame:05d}'
    repository=Path(__file__).resolve().parents[2]
    if any(target.resolve().is_relative_to(p.resolve()) for p in
           (root,case_dir,repository,workflow.LAMBDA_ROOT,workflow.DEFAULT_OUTPUT_ROOT)):
        raise ValueError('Slice outputs must be outside production, snapshots, repository and lambda_ci results.')
    if target.exists() or target.is_symlink():raise FileExistsError(f'Refusing overwrite: {target}')
    found=list(root.rglob(f'{case_dir.name}_f{frame:05d}_omega_x.json'))
    if len(found)!=1:raise ValueError('Missing or duplicate omega_x frame metadata.')
    inputs=workflow._paths(found[0].parent,case_dir.name,frame)
    count=workflow._positive_integer(samples_per_element,'samples_per_element',2)
    upper_points=(272*(count-1)+1)*(8*(count-1)+1)
    required=workflow._validation_ram_bytes((1905,85,57),inputs['vtr'].stat().st_size)+64*8*upper_points
    header,metadata,coordinates,values,source_hash,before=load_projected_omega_frame(case_dir,frame,inputs,required_ram=required)
    source=case_dir/f'GC0.f{frame:05d}'
    samples=reconstruct_p7_midplane(coordinates,values,element_counts=(272,12,8),y=y,time=float(header.time),samples_per_element=count)
    del values
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.f{frame:05d}.staging.',dir=target.parent))
    plane=f'{y:g}'.replace('.','p')
    stem=f'{case_dir.name}_f{frame:05d}_omega_x_y{plane}_xgt0_n{count}'
    paths={'vtr':stage/(stem+'_p7.vtr'),'metadata':stage/(stem+'_validation.json')}
    for key in ('linear','p7','comparison'):
        for fmt in ('png','pdf'):paths[key+'_'+fmt]=stage/(stem+'_'+key+'.'+fmt)
    write_omega_slice_vtr(paths['vtr'],samples);checked=validate_slice_vtr(paths['vtr'],samples)
    style=_figures(samples,paths,case=case_dir.name,frame=frame)
    for key,path in paths.items():
        if key.endswith('_png'):_image_readable(path)
        elif key.endswith('_pdf'):_pdf_readable(path)
    if any(workflow._file_hash(Path(p))!=h for p,h in before.items()) or workflow._file_hash(source)!=source_hash:
        raise RuntimeError('Original scientific inputs changed; publication stopped.')
    report={'status':'completed','case':case_dir.name,'frame':frame,'time':float(header.time),'plane_y':float(y),
        'source_artifact_sha256':before,'source_snapshot_sha256':source_hash,'estimated_ram_bytes':required,
        'source_numerical_signature':metadata['numerical_software_signature'],
        'diagnostics':samples.diagnostics,'rendering':style,'vtr_readback':checked,
        'output_paths':{k:str(target/p.name) for k,p in paths.items()},
        'output_sha256':{k:workflow._file_hash(p) for k,p in paths.items() if k!='metadata'},
        'scientific_limit':'P7 representation of the existing projected scalar, not a higher-order DNS or accuracy proof'}
    with paths['metadata'].open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    _publish_directory(stage,target)
    return report
