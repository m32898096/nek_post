"""Dense SEM tensor sampling and direct Python lambda_ci slice figures.

Source tensors are globally projected native GLL data. Target points are
uniform in each element's reference coordinates, not interpolated VTR scalars.
The output image uses nearest sample cells without scalar smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from nek_post.fields import get_coordinates
from nek_post.gll import gll_nodes, barycentric_weights, barycentric_basis_and_derivative
from nek_post.swirling_strength import compute_swirling_strength


TECPLOT_RGB_POINTS = (
    (0.0, (0.12941176470588237, 0.4, 0.6745098039215687)),
    (0.1667, (0.403921568627451, 0.6627450980392157, 0.8117647058823529)),
    (0.3333, (0.8196078431372549, 0.8980392156862745, 0.9411764705882353)),
    (0.5, (0.9686274509803922, 0.9686274509803922, 0.9686274509803922)),
    (0.6667, (0.9921568627450981, 0.8588235294117647, 0.7803921568627451)),
    (0.8333, (0.9372549019607843, 0.5411764705882353, 0.3843137254901961)),
    (1.0, (0.6980392156862745, 0.09411764705882353, 0.16862745098039217)),
)


def build_tecplot_colormap():
    """Piecewise-linear continuous RGB controls supplied by the user."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('TecplotBlueRed', TECPLOT_RGB_POINTS, N=65536)


def _count(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 2:
        raise ValueError(f'{name} must be an integer >= 2.')
    return int(value)


@lru_cache(maxsize=None)
def _dense_basis(source_count, sample_count):
    nodes=gll_nodes(source_count);weights=barycentric_weights(nodes)
    matrix=np.vstack([barycentric_basis_and_derivative(nodes,weights,q)[0] for q in np.linspace(-1,1,sample_count)])
    matrix.setflags(write=False)
    return matrix


def _axis_line(element, physical_axis, array_axis):
    raw=np.asarray(get_coordinates(element)[physical_axis])
    sl=[0,0,0];sl[array_axis]=slice(None)
    line=raw[tuple(sl)].astype(np.float64)
    shape=[1,1,1];shape[array_axis]=len(line)
    if not np.array_equal(raw,np.broadcast_to(line.reshape(shape),raw.shape)):
        raise ValueError('Dense plotting requires exactly separable rectilinear coordinates.')
    if not np.isfinite(line).all() or not (np.all(np.diff(line)>0) or np.all(np.diff(line)<0)):
        raise ValueError('Coordinate profiles must be finite and strictly monotone.')
    return line


def _physical_basis(line, value):
    nodes=gll_nodes(len(line));weights=barycentric_weights(nodes)
    if not min(line[0],line[-1])<=value<=max(line[0],line[-1]):
        raise ValueError('Physical point lies outside its element; extrapolation forbidden.')
    exact=np.flatnonzero(line==value)
    if exact.size: q=nodes[exact[0]]
    else:
        def residual(q):
            return float(barycentric_basis_and_derivative(nodes,weights,q)[0]@line-value)
        q=brentq(residual,-1,1,xtol=1e-14,rtol=4*np.finfo(float).eps)
    basis=barycentric_basis_and_derivative(nodes,weights,q)[0]
    residual=abs(float(basis@line)-value)
    if residual>1e-11*max(1.,float(np.ptp(line))): raise ValueError('Physical plane inversion failed.')
    return basis,residual


@dataclass(frozen=True)
class TensorSliceSamples:
    normal_axis: str
    plane_value: float
    horizontal_axis: str
    horizontal: np.ndarray
    z: np.ndarray
    tensors: np.ndarray
    samples_per_element: int
    intersected_element_count: int
    maximum_shared_tensor_difference: float
    maximum_mapping_residual: float


@dataclass(frozen=True)
class LambdaSliceSamples:
    normal_axis: str
    plane_value: float
    horizontal_axis: str
    horizontal: np.ndarray
    z: np.ndarray
    lambda_ci: np.ndarray
    lambda_ci_squared: np.ndarray
    sampling_diagnostics: dict


def _segments(profiles, sample_count, minimum=None):
    """Physical axis and element-local basis rows, including an exact crop edge."""
    parts=[];segments={};offset=0
    for cell,p in sorted(profiles.items()):
        matrix=_dense_basis(len(p),sample_count)
        coordinate=matrix@p
        if minimum is not None:
            keep=coordinate>=minimum;coordinate=coordinate[keep];matrix=matrix[keep]
            if not len(coordinate): continue
            if p[0]<minimum<p[-1] and coordinate[0]>minimum:
                row,_=_physical_basis(p,minimum)
                coordinate=np.r_[minimum,coordinate];matrix=np.vstack([row,matrix])
        if parts:
            if parts[-1][-1]!=coordinate[0]: raise ValueError('Missing or mismatching element coordinate interface.')
            offset-=1
        segments[cell]=(slice(offset,offset+len(coordinate)),matrix)
        parts.append(coordinate);offset+=len(coordinate)
    if not parts: raise ValueError('Requested crop contains no slice area.')
    vector=np.concatenate([parts[0],*(p[1:] for p in parts[1:])])
    if len(vector)<2 or not np.all(np.diff(vector)>0): raise ValueError('Sample coordinates are not increasing.')
    return vector,segments


def build_tensor_slice_plan(data, node_map, *, normal_axis,
                                  plane_value, samples_per_element=32, x_min=None):
    """Prepare reusable physical coordinates and element-local P7 operators.

    Input shape: (nelem,nt,ns,nr,3,3), A[i,j]=du_i/dx_j. Physical x/y/z to
    array axes and per-element orientation come from the validated node map.
    normal_axis='y' yields x-z; 'x' yields y-z. x_min is valid only for y slices.
    Internal shared nodes are merged only after checking tensor consistency;
    periodic physical endpoints remain distinct. An interior x_min is evaluated
    exactly and added to the samples, rather than snapping to a stored node.
    """
    count=_count(samples_per_element,'samples_per_element')
    if normal_axis not in ('x','y') or not np.isfinite(plane_value): raise ValueError('Invalid slice plane.')
    if x_min is not None and (normal_axis!='y' or not np.isfinite(x_min)): raise ValueError('x_min is finite and valid only for y slices.')
    node_map.validate_mesh(data)
    normal='xyz'.index(normal_axis);horizontal=1 if normal==0 else 0
    axes=node_map.physical_to_array_axes
    profiles_h={};profiles_z={};candidates=[]
    for ei,element in enumerate(data.elem):
        ln=_axis_line(element,normal,axes[normal])
        if not min(ln[0],ln[-1])<=plane_value<=max(ln[0],ln[-1]): continue
        lh=_axis_line(element,horizontal,axes[horizontal]);lz=_axis_line(element,2,axes[2])
        if x_min is not None and max(lh[0],lh[-1])<=x_min: continue
        for a,line,profiles in ((horizontal,lh,profiles_h),(2,lz,profiles_z)):
            p=line if node_map.element_axis_increasing[ei,a] else line[::-1]
            cell=int(node_map.element_cell_indices[ei,a])
            if cell in profiles and not np.array_equal(profiles[cell],p): raise ValueError('Original coordinate replicas differ.')
            profiles[cell]=p
        candidates.append((ei,ln))
    if not candidates: raise ValueError('Plane/crop does not intersect the mesh; extrapolation forbidden.')
    h,hs=_segments(profiles_h,count,x_min);z,zs=_segments(profiles_z,count)
    entries=[];max_residual=0.
    for ei,ln in candidates:
        sh,bh=hs[int(node_map.element_cell_indices[ei,horizontal])]
        sz,bz=zs[int(node_map.element_cell_indices[ei,2])]
        if not node_map.element_axis_increasing[ei,horizontal]:bh=bh[:,::-1]
        if not node_map.element_axis_increasing[ei,2]:bz=bz[:,::-1]
        bn,residual=_physical_basis(ln,float(plane_value));max_residual=max(max_residual,residual)
        for matrix in (bh,bz,bn):matrix.setflags(write=False)
        entries.append((ei,sz,sh,bz,bh,bn))
    h.setflags(write=False);z.setflags(write=False)
    return TensorSlicePlan(node_map,normal_axis,float(plane_value),'xyz'[horizontal],h,z,count,
                          tuple(entries),(axes[2],axes[horizontal],axes[normal],3,4),max_residual)


@dataclass(frozen=True)
class TensorSlicePlan:
    node_map: object
    normal_axis: str
    plane_value: float
    horizontal_axis: str
    horizontal: np.ndarray
    z: np.ndarray
    samples_per_element: int
    entries: tuple
    tensor_axes: tuple
    maximum_mapping_residual: float


def sample_tensor_slice_plan(projected_gradients, plan):
    """Apply a prepared plan; caller must validate each frame's mesh before reuse."""
    gradients=np.asarray(projected_gradients)
    shape=(plan.node_map.element_count,*plan.node_map.element_shape,3,3)
    if gradients.shape!=shape or gradients.dtype.kind not in 'fiu':raise ValueError(f'Projected tensors must have real shape {shape}.')
    tensor=np.full((len(plan.z),len(plan.horizontal),3,3),np.nan)
    max_jump=0.
    for ei,sz,sh,bz,bh,bn in plan.entries:
        values=np.transpose(gradients[ei],plan.tensor_axes)
        if not np.isfinite(values).all():raise ValueError('Non-finite projected slice tensor.')
        local=np.einsum('at,bs,r,tsrij->abij',bz,bh,bn,values,optimize=True)
        previous=tensor[sz,sh];valid=np.isfinite(previous[...,0,0])
        if valid.any():
            jump=float(np.max(np.abs(previous[valid]-local[valid])));max_jump=max(max_jump,jump)
            if jump>2e-12*max(1.,float(np.max(np.abs(local)))):raise ValueError('Projected shared element traces disagree.')
        previous[~valid]=local[~valid]
    if not np.isfinite(tensor).all():raise ValueError('Incomplete or non-finite slice coverage.')
    return TensorSliceSamples(plan.normal_axis,plan.plane_value,plan.horizontal_axis,plan.horizontal,plan.z,
                             tensor,plan.samples_per_element,len(plan.entries),max_jump,plan.maximum_mapping_residual)


def sample_projected_tensor_slice(data, node_map, projected_gradients, *, normal_axis,
                                  plane_value, samples_per_element=32, x_min=None):
    """Existing one-frame interface; construct a plan and evaluate the nine components."""
    plan=build_tensor_slice_plan(data,node_map,normal_axis=normal_axis,plane_value=plane_value,
                                samples_per_element=samples_per_element,x_min=x_min)
    return sample_tensor_slice_plan(projected_gradients,plan)


def evaluate_tensor_slice_plan(projected_gradients,plan,*,eigenvalue_chunk_rows=16):
    """Apply cached tensor interpolation, then the existing chunked eigenvalue API."""
    sampled=sample_tensor_slice_plan(projected_gradients,plan)
    return _lambda_samples(sampled,eigenvalue_chunk_rows)


def _lambda_samples(sampled,eigenvalue_chunk_rows):
    result=compute_swirling_strength(sampled.tensors,chunk_size=eigenvalue_chunk_rows)
    diagnostics=dict(samples_per_element=sampled.samples_per_element,
        intersected_element_count=sampled.intersected_element_count,
        maximum_shared_tensor_difference=sampled.maximum_shared_tensor_difference,
        maximum_mapping_residual=sampled.maximum_mapping_residual,
        eigenvalue_chunk_rows=eigenvalue_chunk_rows,shape=list(result.lambda_ci.shape),
        plotted_lambda_ci_min=float(result.lambda_ci.min()),plotted_lambda_ci_max=float(result.lambda_ci.max()))
    return LambdaSliceSamples(sampled.normal_axis,sampled.plane_value,sampled.horizontal_axis,
                              sampled.horizontal,sampled.z,result.lambda_ci,result.lambda_ci_squared,diagnostics)

def evaluate_lambda_ci_slice(data,node_map,projected_gradients,*,normal_axis,plane_value,
                             samples_per_element=32,x_min=None,eigenvalue_chunk_rows=16):
    """P7 tensor interpolation followed by eigenvalues on the dense slice only."""
    sampled=sample_projected_tensor_slice(data,node_map,projected_gradients,normal_axis=normal_axis,
        plane_value=plane_value,samples_per_element=samples_per_element,x_min=x_min)
    return _lambda_samples(sampled,eigenvalue_chunk_rows)


def plot_lambda_ci_slice(samples: LambdaSliceSamples,png_path,pdf_path,*,case,frame,time,vmin=0.,vmax=10.,dpi=180):
    """Render dense SEM-evaluated samples with nearest cells, physical aspect=1.

    PDF rasterizes only the dense scalar layer; labels/colorbar remain vector.
    No scalar interpolation, smoothing or image filtering is added by plotting.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if not np.isfinite([vmin,vmax,time]).all() or vmin>=vmax:raise ValueError('Invalid color range or time.')
    _count(dpi,'dpi')
    png_path,pdf_path=Path(png_path),Path(pdf_path)
    if png_path.suffix!='.png' or pdf_path.suffix!='.pdf':raise ValueError('Figure paths must end in PNG/PDF.')
    if any(p.exists() or p.is_symlink() for p in (png_path,pdf_path)):raise FileExistsError('Figure output already exists.')
    h,z=samples.horizontal,samples.z
    if len(h)<2 or len(z)<2 or not all(np.isfinite(a).all() and np.all(np.diff(a)>0) for a in (h,z)):
        raise ValueError('Figure coordinates must be increasing and finite.')
    if samples.lambda_ci.shape!=(len(z),len(h)) or not np.isfinite(samples.lambda_ci).all() or np.any(samples.lambda_ci<0):
        raise ValueError('Invalid sampled scalar field.')
    ratio=float(np.ptp(h)/np.ptp(z));width=14 if ratio>5 else 8
    fig=Figure(figsize=(width,width/ratio+1.6));FigureCanvasAgg(fig)
    # Reserve label margins explicitly; a fixed physical aspect otherwise
    # allows an appended colorbar's labels to extend beyond the PDF page.
    fig.subplots_adjust(left=.07,right=.87,bottom=.18,top=.82)
    ax=fig.subplots()
    artist=ax.pcolormesh(h,z,samples.lambda_ci,shading='nearest',cmap=build_tecplot_colormap(),
                        vmin=vmin,vmax=vmax,rasterized=True,antialiased=False)
    ax.set(xlabel=samples.horizontal_axis,ylabel='z',xlim=(h[0],h[-1]),ylim=(z[0],z[-1]),aspect='equal',
           title=f'{case}, frame {frame}, time={time:.12g}\n{samples.normal_axis}={samples.plane_value:g}; {samples.horizontal_axis} in [{h[0]:g}, {h[-1]:g}]')
    color_axis=make_axes_locatable(ax).append_axes('right',size='3%',pad=.12)
    ticks=[vmin,(vmin+vmax)/2,vmax] if ratio>5 else None
    fig.colorbar(artist,cax=color_axis,label=r'$\lambda_{ci}$',ticks=ticks)
    for p,fmt in ((png_path,'png'),(pdf_path,'pdf')):
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:fig.savefig(stream,format=fmt,dpi=dpi)
    return png_path,pdf_path


def run_lambda_ci_slice_plots(case_dir,index=79,*,y_midplane=.75,x_slice=15.,x_min=0.,
                              samples_per_element=None,vmin=0.,vmax=10.,output_dir=None,
                              subdivisions_per_element=28,export_paraview=False,
                              chunk_size=256,eigenvalue_chunk_rows=16,save_samples=False,
                              dry_run=False,progress=None,periodic_axes=('x','y'),
                              expected_element_counts=(272,12,8)):
    """Compute one projected snapshot once and render both high-order slices.

    No existing VTR field is read. Dry run inspects the source header and
    conservative resources only. Optional NPZs save high-order slice samples
    for future plotting without repeating the numerical pipeline.
    """
    import json
    import resource
    import time
    from nek_post.io_nek import read_nek_file,read_nek_header
    from nek_post.lambda_ci_workflow import DEFAULT_OUTPUT_ROOT,inspect_single_frame,_file_hash
    from nek_post.spectral_derivatives import element_velocity_gradient
    from nek_post.l2_projection import build_l2_projection_operator,project_velocity_gradient
    started=time.perf_counter()
    if samples_per_element is None:
        if not isinstance(subdivisions_per_element,(int,np.integer)) or isinstance(subdivisions_per_element,(bool,np.bool_)) or subdivisions_per_element < 1:
            raise ValueError('subdivisions_per_element must be a positive integer.')
        subdivisions=int(subdivisions_per_element)
        n=subdivisions + 1
    else:
        n=_count(samples_per_element,'samples_per_element')
    if not np.isfinite([y_midplane,x_slice,x_min,vmin,vmax]).all() or vmin>=vmax:
        raise ValueError('Plane coordinates, crop and color range must be finite/valid.')
    if not isinstance(eigenvalue_chunk_rows,(int,np.integer)) or isinstance(eigenvalue_chunk_rows,(bool,np.bool_)) or eigenvalue_chunk_rows<1:
        raise ValueError('eigenvalue_chunk_rows must be a positive integer.')
    case_dir=Path(case_dir).resolve()
    if not isinstance(index,(int,np.integer)) or isinstance(index,(bool,np.bool_)) or index<1:raise ValueError('Invalid frame index.')
    frame_dir=DEFAULT_OUTPUT_ROOT/case_dir.name/f'f{index:05d}'
    out=Path(output_dir).resolve() if output_dir is not None else frame_dir/'plots'
    if out.is_relative_to(case_dir) or out.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Scientific output must be outside original data and the repository.')
    def token(v):
        s=format(float(v),'.15g')
        if '.' not in s and 'e' not in s:s+='.0'
        return s.replace('-','m').replace('+','p').replace('.','p')
    stem=f'{case_dir.name}_f{index:05d}_lambda_ci'
    yname='y_midplane' if y_midplane==.75 else f'y{token(y_midplane)}'
    names={'y_midplane':f'{stem}_{yname}_xge{token(x_min)}','x_slice':f'{stem}_x{format(float(x_slice),"g").replace("-","m").replace(".","p")}_slice'}
    paths={f'{key}_{fmt}':out/f'{name}.{fmt}' for key,name in names.items() for fmt in ('png','pdf')}
    if save_samples:paths.update({f'{key}_samples':out/f'{name}.npz' for key,name in names.items()})
    if export_paraview:
        # Fresh names preserve the previous 32-point Python-only products.
        names={'y_midplane':f'{stem}_{yname}' + (f'_xge{token(x_min)}' if x_min != 0 else ''),
               'x_slice':f'{stem}_x{format(float(x_slice),"g").replace("-","m").replace(".","p")}'}
        paths={f'{key}_{fmt}':out/f'{name}.{fmt}' for key,name in names.items() for fmt in ('png','pdf')}
        if save_samples:paths.update({f'{key}_samples':out/f'{name}.npz' for key,name in names.items()})
        paths['colormap_preset']=out/f'{stem}_tecplot_continuous.json'
        slice_dir=out.parent/'slices'
        paths.update({f'{key}_vtr':slice_dir/f'{name}.vtr' for key,name in names.items()})
        series={key:out.parent.parent/f'{case_dir.name}_lambda_ci_{name[len(stem)+1:]}_series.pvd' for key,name in names.items()}
        from nek_post.lambda_ci_slice_export import write_lambda_ci_slice_vtr,append_slice_series_pvd,check_slice_series_entry,write_tecplot_paraview_preset
    paths['metadata']=out/f'{stem}_high_order_slices{"_paraview" if export_paraview else ""}.json'
    destinations=list(paths.values()) + (list(series.values()) if export_paraview else [])
    for p in destinations:
        resolved=p.resolve()
        if resolved.is_relative_to(case_dir) or resolved.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('All scientific outputs must be outside original data and the repository.')
    for p in paths.values():
        if p.exists() or p.is_symlink():raise FileExistsError(f'Refusing overwrite: {p}')
    plan=inspect_single_frame(case_dir,index,out,chunk_size=chunk_size,periodic_axes=periodic_axes,expected_element_counts=expected_element_counts)
    if export_paraview:
        for key,p in series.items():
            check_slice_series_entry(p,paths[f'{key}_vtr'],time=plan['time'])
    dense_nodes=(expected_element_counts[0]*(n-1)+1)*(expected_element_counts[2]*(n-1)+1)
    estimated_ram=plan['estimated_ram_bytes']+dense_nodes*9*8
    report={'source_file':plan['source_file'],'time':plan['time'],'istep':plan['istep'],'frame_index':index,
            'element_count':plan['element_count'],'native_nodes_xyz':plan['header_orders_xyz'],
            'samples_per_element':n,'subdivisions_per_element':n-1,'native_density_multiplier':[n/k for k in plan['header_orders_xyz']],
            'native_interval_multiplier':[(n-1)/(k-1) for k in plan['header_orders_xyz']],
            'reference_sampling':'uniform linspace(-1,1,N) per element edge; source GLL barycentric basis',
            'planes':{'y_midplane':y_midplane,'x_slice':x_slice,'x_min':x_min},'color_range':[vmin,vmax],
            'tecplot_rgb_points':TECPLOT_RGB_POINTS,'estimated_ram_bytes':estimated_ram,'available_ram_bytes':plan['available_ram_bytes'],
            'resources_acceptable':bool(plan['resources_acceptable'] and estimated_ram<plan['available_ram_bytes']),
            'output_paths':{k:str(v) for k,v in paths.items()},'no_vtr_scalar_read':True,'no_smoothing':True}
    if export_paraview:report['series_paths']={key:str(p) for key,p in series.items()}
    if dry_run:return report
    if not report['resources_acceptable']:raise MemoryError('Insufficient memory/disk; snapshot not loaded.')
    protected=[Path(plan['source_file'])]
    if frame_dir.exists():protected.extend(p for p in frame_dir.rglob('*') if p.is_file())
    hashes={str(p):_file_hash(p) for p in protected}
    def notify(message):
        if progress:progress(message)
    stage=time.perf_counter();header=read_nek_header(plan['source_file'])
    skip=('pressure','temperature')+tuple(f's{i:02d}' for i in range(1,header.nb_vars[4]+1))
    data=read_nek_file(plan['source_file'],dtype=f'float{header.wdsz*8}',skip_vars=skip)
    if data.time!=header.time or len(data.elem)!=header.nb_elems:raise ValueError('Header/reader mismatch.')
    timings={'read':time.perf_counter()-stage}
    notify('Computing spectral gradients once for both slices')
    stage=time.perf_counter();raw=np.empty((len(data.elem),*reversed(header.orders),3,3),dtype=np.float64)
    for ei,element in enumerate(data.elem):
        raw[ei]=element_velocity_gradient(element)
        if (ei+1)%4096==0:notify(f'Gradients: {ei+1}/{len(data.elem)}')
    timings['spectral_gradients']=time.perf_counter()-stage
    notify('Preparing the existing global L2 operator')
    stage=time.perf_counter();operator=build_l2_projection_operator(data,periodic_axes=periodic_axes,chunk_size=chunk_size,expected_element_counts=expected_element_counts)
    timings['projection_preparation']=time.perf_counter()-stage
    notify('Projecting all nine gradient components')
    stage=time.perf_counter();projected=project_velocity_gradient(operator,raw,diagnostics=True,chunk_size=chunk_size)
    del raw
    mapping=operator.node_map;del operator
    report['projection_discontinuity_after']=projected.diagnostics.discontinuity_after.tolist()
    timings['l2_projection']=time.perf_counter()-stage
    slices={}
    for key,normal,value,crop in (('y_midplane','y',y_midplane,x_min),('x_slice','x',x_slice,None)):
        notify(f'Evaluating dense P7 tensors and eigenvalues: {normal}={value}')
        stage=time.perf_counter()
        slices[key]=evaluate_lambda_ci_slice(data,mapping,projected.values,normal_axis=normal,plane_value=value,
            samples_per_element=n,x_min=crop,eigenvalue_chunk_rows=eigenvalue_chunk_rows)
        timings[f'{key}_sampling_and_eigenvalues']=time.perf_counter()-stage
    del projected,data,mapping
    report['slices']={key:(s.sampling_diagnostics|{'normal_axis':s.normal_axis,'plane_value':s.plane_value,
        'horizontal_axis':s.horizontal_axis,'horizontal_bounds':[float(s.horizontal[0]),float(s.horizontal[-1])],
        'z_bounds':[float(s.z[0]),float(s.z[-1])],
        'color_clipped_fraction':float(np.mean((s.lambda_ci<vmin)|(s.lambda_ci>vmax)))}) for key,s in slices.items()}
    if any(_file_hash(Path(p))!=h for p,h in hashes.items()):raise RuntimeError('Protected file changed; export stopped.')
    for p in paths.values():
        if p.exists() or p.is_symlink():raise FileExistsError(f'Output appeared during calculation: {p}')
    notify('Rendering Tecplot-color PNG/PDF directly from high-order samples')
    stage=time.perf_counter()
    for key,s in slices.items():
        plot_lambda_ci_slice(s,paths[f'{key}_png'],paths[f'{key}_pdf'],case=case_dir.name,frame=index,time=header.time,vmin=vmin,vmax=vmax)
        if export_paraview:
            write_lambda_ci_slice_vtr(paths[f'{key}_vtr'],s,time=header.time)
        if save_samples:
            with paths[f'{key}_samples'].open('xb') as f:
                np.savez_compressed(f,horizontal=s.horizontal,z=s.z,lambda_ci=s.lambda_ci,lambda_ci_squared=s.lambda_ci_squared,
                                    normal_axis=s.normal_axis,plane_value=s.plane_value,horizontal_axis=s.horizontal_axis,
                                    time=header.time,frame=index,samples_per_element=n)
    if export_paraview:
        write_tecplot_paraview_preset(paths['colormap_preset'])
        for key,p in series.items():
            append_slice_series_pvd(p,paths[f'{key}_vtr'],time=header.time)
    timings['render_and_save']=time.perf_counter()-stage
    if any(_file_hash(Path(p))!=h for p,h in hashes.items()):raise RuntimeError('Protected source/result changed during rendering.')
    report.update(protected_sha256=hashes,protected_files_unchanged=True,stage_seconds=timings,
                  total_seconds=time.perf_counter()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                  output_file_bytes={k:p.stat().st_size for k,p in paths.items() if k!='metadata'},
                  display_method='Matplotlib pcolormesh shading=nearest, antialiased=False, aspect=equal; PDF scalar layer rasterized, no scalar filtering')
    with paths['metadata'].open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report
