"""Sequential, resumable dense lambda_ci slices with stationary-mesh caches.

Only workflow orchestration lives here. Differentiation, projection, tensor
interpolation and eigenvalues use the established numerical APIs unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import ctypes
import errno
import fcntl
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.l2_projection import build_structured_node_map, build_l2_projection_operator, project_velocity_gradient
from nek_post.lambda_ci_plotting import (
    TECPLOT_RGB_POINTS, build_tensor_slice_plan, evaluate_tensor_slice_plan, plot_lambda_ci_slice,
)
from nek_post.lambda_ci_slice_export import (
    write_lambda_ci_slice_vtr, write_tecplot_paraview_preset, append_slice_series_pvd, _collection,
)
from nek_post.lambda_ci_workflow import DEFAULT_OUTPUT_ROOT, _file_hash, available_memory_bytes
from nek_post.spectral_derivatives import element_velocity_gradient


SCHEMA_VERSION = 1


def expand_frame_indices(start_index=1, end_index=81, indices=None, max_frames=None):
    """Sorted unique positive indices; accept comma-separated integers/ranges."""
    if indices is None:
        if not 1 <= start_index <= end_index:
            raise ValueError('Require 1 <= start-index <= end-index.')
        values = list(range(start_index, end_index + 1))
    elif isinstance(indices, str):
        values = []
        for token in indices.split(','):
            parts = token.strip().split('-')
            if len(parts) == 1:
                values.append(int(parts[0]))
            elif len(parts) == 2 and 1 <= int(parts[0]) <= int(parts[1]):
                values.extend(range(int(parts[0]), int(parts[1]) + 1))
            else:
                raise ValueError('Invalid frame selection.')
    else:
        values = list(indices)
    if not values or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 1 for v in values):
        raise ValueError('Frame indices must be positive integers.')
    values = sorted(set(int(v) for v in values))
    if max_frames is not None:
        if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
            raise ValueError('max-frames must be a positive integer.')
        values = values[:max_frames]
    return values


def series_configuration(*, y_midplane=.75, x_slice=15., x_min=0., subdivisions_per_element=28,
                         vmin=0., vmax=10., periodic_axes=('x', 'y'), expected_element_counts=(272, 12, 8)):
    if not np.isfinite([y_midplane, x_slice, x_min, vmin, vmax]).all() or vmin >= vmax:
        raise ValueError('Invalid planes, crop or color range.')
    if not isinstance(subdivisions_per_element, (int, np.integer)) or isinstance(subdivisions_per_element, (bool, np.bool_)) or subdivisions_per_element < 1:
        raise ValueError('subdivisions-per-element must be a positive integer.')
    if len(expected_element_counts) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in expected_element_counts):
        raise ValueError('Expected element counts must be a positive integer triple.')
    if len(set(periodic_axes)) != len(periodic_axes) or any(a not in 'xyz' for a in periodic_axes):
        raise ValueError('Invalid periodic axes.')
    return dict(y_midplane=float(y_midplane), x_slice=float(x_slice), x_min=float(x_min),
                subdivisions_per_element=int(subdivisions_per_element), vmin=float(vmin), vmax=float(vmax),
                periodic_axes=list(periodic_axes), expected_element_counts=list(expected_element_counts),
                tecplot_rgb_points=[[q, list(rgb)] for q, rgb in TECPLOT_RGB_POINTS])


def _token(value):
    return format(value, 'g').replace('-', 'm').replace('.', 'p')


def frame_manifest(case, index, config, frame_dir):
    frame_dir = Path(frame_dir)
    stem = f'{case}_f{index:05d}_lambda_ci'
    y = 'y_midplane' if config['y_midplane'] == .75 else f"y{_token(config['y_midplane'])}"
    if config['x_min'] != 0:
        y += '_xge' + _token(config['x_min'])
    names = {'y_midplane': f'{stem}_{y}', 'x_slice': f"{stem}_x{_token(config['x_slice'])}"}
    paths = {f'{key}_{ext}': frame_dir / ('slices' if ext == 'vtr' else 'plots') / f'{name}.{ext}'
             for key, name in names.items() for ext in ('png', 'pdf', 'vtr')}
    paths['metadata'] = frame_dir / 'plots' / f'{stem}_high_order_slices_paraview.json'
    paths['colormap_preset'] = frame_dir / 'plots' / f'{stem}_tecplot_continuous.json'
    return paths


def _plans(data, mapping, config):
    count = config['subdivisions_per_element'] + 1
    return {
        'y_midplane': build_tensor_slice_plan(data, mapping, normal_axis='y', plane_value=config['y_midplane'], samples_per_element=count, x_min=config['x_min']),
        'x_slice': build_tensor_slice_plan(data, mapping, normal_axis='x', plane_value=config['x_slice'], samples_per_element=count),
    }


def _read_snapshot(source, *, coordinates_only=False):
    header = read_nek_header(source)
    skip = ('pressure', 'temperature') + tuple(f's{i:02d}' for i in range(1, header.nb_vars[4] + 1))
    if coordinates_only:
        skip += ('ux', 'uy', 'uz')
    data = read_nek_file(source, dtype=f'float{header.wdsz*8}', skip_vars=skip)
    if data.time != header.time or len(data.elem) != header.nb_elems:
        raise ValueError('Header/reader mismatch.')
    return data


def _signature():
    base = Path(__file__).parent
    names = ('gll.py', 'spectral_derivatives.py', 'l2_projection.py', 'swirling_strength.py', 'lambda_ci_plotting.py')
    return {n: _file_hash(base / n) for n in names}


def _rss():
    # Current RSS (not lifetime high-water mark) permits memory-growth audits.
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp, path)


def _image_readable(path):
    from PIL import Image
    with Image.open(path) as image:
        size = image.size
        image.verify()
    if min(size) < 1:
        raise ValueError('Empty PNG.')
    return list(size)


def _pdf_readable(path):
    # Poppler decodes the page to a tiny PNG on stdout, creating no output file.
    result = subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '8', '-png', str(path)],
                            capture_output=True, check=False, timeout=30)
    if result.returncode or not result.stdout.startswith(b'\x89PNG') or result.stderr:
        raise ValueError(f'PDF cannot be decoded: {path}: {result.stderr.decode(errors="replace")}')


def validate_frame_result(frame_dir, *, case, index, time_value, config, plans, source_sha256=None):
    """Validate completed modern/legacy outputs; existence alone never skips work."""
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    paths = frame_manifest(case, index, config, frame_dir)
    for key, path in paths.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f'Incomplete frame {index}: missing/invalid {key}: {path}')
    metadata = json.loads(paths['metadata'].read_text())
    modern = 'schema_version' in metadata
    if metadata.get('time') != time_value or metadata.get('frame_index') != index:
        raise ValueError('Diagnostic frame/timestamp mismatch.')
    if modern:
        if metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('completion_status') != 'completed' or metadata.get('configuration') != config:
            raise ValueError('Incomplete/incompatible diagnostics/configuration.')
        if metadata.get('numerical_software_signature') != _signature():
            raise ValueError('Numerical implementation changed; explicit compatibility review required.')
        stored_hash = metadata.get('source_sha256')
        expected_hashes=metadata.get('output_sha256',{})
        if set(expected_hashes)!=set(paths)-{'metadata'}:
            raise ValueError('Completed diagnostics lack output checksums.')
        for key,digest in expected_hashes.items():
            if _file_hash(paths[key])!=digest:raise ValueError(f'Completed output checksum differs: {key}')
    else:
        if not metadata.get('protected_files_unchanged') or not metadata.get('no_vtr_scalar_read') or not metadata.get('no_smoothing'):
            raise ValueError('Legacy result lacks successful scientific diagnostics.')
        if metadata.get('planes') != {'y_midplane': config['y_midplane'], 'x_slice': config['x_slice'], 'x_min': config['x_min']} or metadata.get('color_range') != [config['vmin'], config['vmax']] or metadata.get('samples_per_element') != config['subdivisions_per_element'] + 1 or metadata.get('tecplot_rgb_points') != config['tecplot_rgb_points']:
            raise ValueError('Legacy output configuration differs.')
        stored_hash = metadata.get('protected_sha256', {}).get(metadata.get('source_file'))
    if source_sha256 is not None and stored_hash != source_sha256:
        raise ValueError('Source SHA-256 differs from completed result.')
    preset = json.loads(paths['colormap_preset'].read_text())[0]
    if preset.get('ColorSpace') != 'RGB' or preset.get('RGBPoints') != [v for q, rgb in TECPLOT_RGB_POINTS for v in (q, *rgb)]:
        raise ValueError('Continuous Tecplot preset differs.')
    statistics = {}
    for key, plan in plans.items():
        _image_readable(paths[f'{key}_png']);_pdf_readable(paths[f'{key}_pdf'])
        reader = vtkXMLRectilinearGridReader();reader.SetFileName(str(paths[f'{key}_vtr']));reader.Update()
        grid = reader.GetOutput()
        expected_coordinates = ((plan.horizontal, np.array([plan.plane_value]), plan.z) if plan.normal_axis == 'y'
                                else (np.array([plan.plane_value]), plan.horizontal, plan.z))
        expected_dimensions = tuple(len(c) for c in expected_coordinates)
        if reader.GetErrorCode() or grid.GetDimensions() != expected_dimensions or grid.GetNumberOfCells() != (len(plan.horizontal)-1)*(len(plan.z)-1):
            raise ValueError(f'Invalid VTR dimensions/topology for {key}.')
        for getter, expected in zip((grid.GetXCoordinates, grid.GetYCoordinates, grid.GetZCoordinates), expected_coordinates):
            if not np.array_equal(vtk_to_numpy(getter()), expected):
                raise ValueError('VTR sample coordinates differ from requested mesh/plane.')
        timestamp = grid.GetFieldData().GetArray('TIME_VALUE')
        if timestamp is None or float(vtk_to_numpy(timestamp)[0]) != time_value:
            raise ValueError('VTR timestamp differs from actual snapshot.')
        values = []
        for name in ('lambda_ci', 'lambda_ci_squared'):
            field = grid.GetPointData().GetArray(name)
            if field is None or field.GetNumberOfComponents() != 1:
                raise ValueError(f'Missing scalar {name}.')
            array = vtk_to_numpy(field)
            if len(array) != np.prod(expected_dimensions) or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError('Invalid scalar shape or NaN/Inf/negative data.')
            values.append(array)
        if not np.array_equal(values[0]**2, values[1]):
            raise ValueError('Squared field differs from lambda_ci**2.')
        lo, hi = float(values[0].min()), float(values[0].max())
        info = metadata.get('slices', {}).get(key, {})
        if info.get('plotted_lambda_ci_min') != lo or info.get('plotted_lambda_ci_max') != hi:
            raise ValueError('Diagnostic min/max differ from VTR values.')
        optional = paths[f'{key}_png'].with_suffix('.npz')
        if optional.exists():
            with np.load(optional, allow_pickle=False) as arrays:
                if float(arrays['time']) != time_value or not np.array_equal(arrays['horizontal'], plan.horizontal) or not np.array_equal(arrays['z'], plan.z):
                    raise ValueError('NPZ coordinates/timestamp differ.')
                for name, expected in zip(('lambda_ci', 'lambda_ci_squared'), values):
                    if not np.array_equal(arrays[name].ravel(), expected):
                        raise ValueError('Sample/VTR scalar values differ.')
        statistics[key] = {'min': lo, 'max': hi, 'dimensions': list(expected_dimensions), 'points': len(values[0])}
    return {'index': index, 'time': time_value, 'slices': statistics,
            'runtime_seconds': metadata.get('total_seconds'), 'legacy_validated': not modern,
            'source_sha256': stored_hash, 'validation': 'passed'}


def inspect_slice_series(case_dir, output_root, indices, config, *, resume=False, recover_staging=False):
    """No velocity derivatives/projection/eigenvalues: headers, mesh, resume, resources."""
    case_dir, output_root = Path(case_dir).resolve(), Path(output_root).resolve()
    if not case_dir.is_dir() or output_root.is_relative_to(case_dir) or output_root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Input must exist; scientific outputs must be outside data/repository.')
    if not shutil.which('pdftoppm'):
        raise RuntimeError('Poppler pdftoppm is required for PDF completion validation.')
    snapshots = [];times = []
    for index in indices:
        source = case_dir / f'GC0.f{index:05d}'
        if not source.is_file():
            raise FileNotFoundError(source)
        h = read_nek_header(source)
        if h.nb_dims != 3 or h.nb_vars[0] != 3 or h.nb_vars[1] != 3 or h.nb_files != 1 or h.nb_elems_file != h.nb_elems or h.nb_elems != int(np.prod(config['expected_element_counts'])) or tuple(h.orders) != (8,8,8) or h.wdsz not in (4,8) or not np.isfinite(h.time):
            raise ValueError(f'Invalid/incompatible snapshot header: {source}')
        stat = source.stat()
        snapshots.append({'index':index, 'time':float(h.time), 'istep':int(h.istep), 'source':str(source),
                          'source_sha256':_file_hash(source), 'source_size':stat.st_size, 'source_mtime_ns':stat.st_mtime_ns})
        times.append(float(h.time))
    if not np.all(np.diff(times) > 0):
        raise ValueError('Physical timestamps must be unique and increasing in frame-index order.')
    # The original nodal coordinates, not previous scalar VTRs, establish plans.
    reference = _read_snapshot(snapshots[0]['source'], coordinates_only=True)
    mapping = build_structured_node_map(reference, periodic_axes=tuple(config['periodic_axes']),
                                        expected_element_counts=tuple(config['expected_element_counts']))
    plans = _plans(reference, mapping, config)
    del reference
    ancestor = output_root
    while not ancestor.exists():ancestor = ancestor.parent
    disk = shutil.disk_usage(ancestor).free
    bytes_per_frame = sum(p.horizontal.size*p.z.size*16 + (p.horizontal.size+p.z.size)*8 for p in plans.values()) + 8*1024**2
    states=[]
    for item in snapshots:
        frame_dir=output_root/f"f{item['index']:05d}"
        if frame_dir.resolve().is_relative_to(case_dir) or frame_dir.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('Output frame resolves inside original data/repository.')
        if frame_dir.exists() or frame_dir.is_symlink():
            if not resume:
                raise FileExistsError(f'Existing frame {frame_dir}; use --resume for full validation.')
            existing_mesh=_read_snapshot(item['source'],coordinates_only=True)
            mapping.validate_mesh(existing_mesh);del existing_mesh
            state=validate_frame_result(frame_dir,case=case_dir.name,index=item['index'],time_value=item['time'],config=config,plans=plans,source_sha256=item['source_sha256'])
            state['status']='reused'
        else:
            state={'index':item['index'],'time':item['time'],'status':'pending'}
        stages=list((output_root/'.stage5-staging').glob(f"f{item['index']:05d}.*"))
        if stages and not recover_staging and state['status']=='pending':
            raise RuntimeError(f'Unpublished staging data for frame {item["index"]}; use --recover-staging to validate/recover or preserve it and recompute.')
        states.append(state)
    # Validate existing collections before any expensive production work.
    validated_indices={s['index'] for s in states if s['status']=='reused'}
    example=frame_manifest(case_dir.name,indices[0],config,output_root/f'f{indices[0]:05d}')
    for key in ('y_midplane','x_slice'):
        suffix=example[key+'_vtr'].stem.split('_lambda_ci_',1)[1]
        path=output_root/f'{case_dir.name}_lambda_ci_{suffix}_series.pvd'
        _,collection=_collection(path)
        entries=list(collection)
        ordered=[float(e.get('timestep')) for e in entries]
        if ordered!=sorted(set(ordered)):
            raise ValueError('Existing PVD has unsorted or duplicated timestamps.')
        references=set()
        for entry in entries:
            target=(path.parent/entry.get('file')).resolve()
            if target in references:raise ValueError('Existing PVD has duplicate references.')
            references.add(target)
            try:frame_index=int(target.parent.parent.name.removeprefix('f'))
            except ValueError as exc:raise ValueError('PVD reference is outside a frame/slices directory.') from exc
            expected=frame_manifest(case_dir.name,frame_index,config,output_root/f'f{frame_index:05d}')[key+'_vtr'].resolve()
            if target!=expected:raise ValueError('PVD reference differs from the requested slice manifest.')
            source=case_dir/f'GC0.f{frame_index:05d}'
            actual=float(read_nek_header(source).time)
            if actual!=float(entry.get('timestep')):raise ValueError('PVD timestamp differs from actual source.')
            if frame_index not in validated_indices:
                validate_frame_result(output_root/f'f{frame_index:05d}',case=case_dir.name,index=frame_index,time_value=actual,config=config,plans=plans,source_sha256=_file_hash(source))
                validated_indices.add(frame_index)
    pending=sum(s['status']=='pending' for s in states)
    required_disk=pending*bytes_per_frame+2*bytes_per_frame
    local_nodes=mapping.element_count*int(np.prod(mapping.element_shape))
    estimated_ram=local_nodes*(2*9*8+8+8*4)+mapping.global_node_count*8*3+1024**3
    if disk < required_disk or available_memory_bytes() < estimated_ram:
        raise MemoryError('Insufficient disk or memory for sequential processing.')
    return dict(case_dir=str(case_dir),output_root=str(output_root),configuration=config,snapshots=snapshots,frames=states,
                requested_frames=len(indices),reused_frames=len(indices)-pending,pending_frames=pending,
                expected_output_counts={'png':2*len(indices),'pdf':2*len(indices),'vtr':2*len(indices),'pvd_entries_each':len(indices)},
                available_disk_bytes=disk,required_disk_bytes=required_disk,estimated_ram_bytes=estimated_ram,
                available_ram_bytes=available_memory_bytes(),time_range=[times[0],times[-1]],
                mesh_element_count=mapping.element_count,mesh_element_counts=list(mapping.element_interval_counts),
                physical_to_array_axes=list(mapping.physical_to_array_axes),
                slice_dimensions={k:([len(p.horizontal),1,len(p.z)] if p.normal_axis=='y' else [1,len(p.horizontal),len(p.z)]) for k,p in plans.items()},
                plans=plans)


@contextmanager
def _run_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'.stage5.lock').is_symlink():raise ValueError('Output lock must not be a symlink.')
    with (root/'.stage5.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another series process holds the output lock.') from exc
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
        yield


def _publish_directory(stage, final):
    """Linux atomic directory publication with RENAME_NOREPLACE."""
    final.parent.mkdir(parents=True,exist_ok=True)
    libc=ctypes.CDLL(None,use_errno=True)
    if not hasattr(libc,'renameat2'):
        raise RuntimeError('Atomic no-replace directory publishing requires renameat2.')
    result=libc.renameat2(-100,os.fsencode(stage),-100,os.fsencode(final),1)
    if result:
        error=ctypes.get_errno()
        if error==errno.EEXIST:raise FileExistsError(final)
        raise OSError(error,os.strerror(error))


def _update_series(root, case, index, time_value, config):
    paths=frame_manifest(case,index,config,root/f'f{index:05d}')
    series={key:root/f"{case}_lambda_ci_{paths[key+'_vtr'].stem.split('_lambda_ci_',1)[1]}_series.pvd" for key in ('y_midplane','x_slice')}
    for key,path in series.items():
        _,collection=_collection(path)
        target=paths[key+'_vtr'].resolve()
        matches=[e for e in collection if float(e.get('timestep'))==time_value or (path.parent/e.get('file')).resolve()==target]
        if matches:
            if len(matches)!=1 or float(matches[0].get('timestep'))!=time_value or (path.parent/matches[0].get('file')).resolve()!=target:
                raise ValueError('Conflicting/duplicate PVD timestamp or reference.')
        else:
            append_slice_series_pvd(path,target,time=time_value,sort_by_time=True)
    return {k:str(p) for k,p in series.items()}


def _export_frame(stage, final, *, case, item, config, slices, operator_build_count, plans_build_count, timings, initialization_seconds, save_samples):
    from importlib.metadata import version
    paths=frame_manifest(case,item['index'],config,stage)
    for key,s in slices.items():
        plot_lambda_ci_slice(s,paths[key+'_png'],paths[key+'_pdf'],case=case,frame=item['index'],time=item['time'],vmin=config['vmin'],vmax=config['vmax'])
        write_lambda_ci_slice_vtr(paths[key+'_vtr'],s,time=item['time'])
        if save_samples:
            with paths[key+'_png'].with_suffix('.npz').open('xb') as f:
                np.savez_compressed(f,horizontal=s.horizontal,z=s.z,lambda_ci=s.lambda_ci,lambda_ci_squared=s.lambda_ci_squared,
                                    time=item['time'],frame=item['index'],normal_axis=s.normal_axis,plane_value=s.plane_value,
                                    horizontal_axis=s.horizontal_axis,samples_per_element=config['subdivisions_per_element']+1)
    write_tecplot_paraview_preset(paths['colormap_preset'])
    # Reader round-trip equality is independently checked against in-memory samples.
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    for key,s in slices.items():
        reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(paths[key+'_vtr']));reader.Update()
        for name,expected in [('lambda_ci',s.lambda_ci),('lambda_ci_squared',s.lambda_ci_squared)]:
            if not np.array_equal(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(name)),expected.ravel()):
                raise ValueError('VTR writer changed sampled scalars.')
    metadata=dict(schema_version=SCHEMA_VERSION,completion_status='completed',configuration=config,
                  time=item['time'],istep=item['istep'],frame_index=item['index'],source_file=item['source'],source_sha256=item['source_sha256'],
                  numerical_software_signature=_signature(),software={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':version('matplotlib'),'vtk':version('vtk')},
                  output_paths={k:str(final/p.relative_to(stage)) for k,p in paths.items()},
                  output_sha256={k:_file_hash(p) for k,p in paths.items() if k!='metadata'},
                  operator_build_count=operator_build_count,slice_plan_build_count=plans_build_count,
                  operator_reused=operator_build_count==1,initialization_seconds=initialization_seconds,
                  stage_seconds=timings,total_seconds=sum(timings.values()),peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                  slices={key:s.sampling_diagnostics for key,s in slices.items()},newly_generated=True,
                  no_vtr_scalar_read=True,no_smoothing=True,sampled_scalar_vtr_exact=True)
    with paths['metadata'].open('x') as f:json.dump(metadata,f,indent=2,allow_nan=False);f.write('\n')


def verify_series_outputs(root, case, snapshots, config, plans):
    """Role-based manifest audit, actual times, all reader checks and source hashes."""
    frames=[]
    for item in snapshots:
        state=validate_frame_result(root/f"f{item['index']:05d}",case=case,index=item['index'],time_value=item['time'],config=config,plans=plans,source_sha256=item['source_sha256'])
        if _file_hash(Path(item['source']))!=item['source_sha256']:
            raise RuntimeError(f'Original source changed: {item["source"]}')
        frames.append(state)
    expected={item['index']:item['time'] for item in snapshots}
    series=_update_series(root,case,snapshots[-1]['index'],snapshots[-1]['time'],config)
    for key,path in series.items():
        entries=ET.parse(path).findall('Collection/DataSet')
        times=[float(e.get('timestep')) for e in entries]
        if times!=sorted(times) or len(set(times))!=len(times):raise ValueError('PVD times are unsorted/duplicated.')
        refs=[(Path(path).parent/e.get('file')).resolve() for e in entries]
        if len(set(refs))!=len(refs):raise ValueError('Duplicate PVD references.')
        for index,t in expected.items():
            manifest=frame_manifest(case,index,config,root/f'f{index:05d}')
            target=manifest[key+'_vtr'].resolve()
            matching=[e for e in entries if (Path(path).parent/e.get('file')).resolve()==target and float(e.get('timestep'))==t]
            if len(matching)!=1:raise ValueError('Missing PVD frame/actual timestamp.')
        if len(snapshots)==81 and len(entries)!=81:raise ValueError('Full production requires exactly 81 PVD entries.')
    return {'frames':frames,'counts':{'png':len(frames)*2,'pdf':len(frames)*2,'vtr':len(frames)*2},
            'pvd_entry_counts':{key:len(ET.parse(path).findall('Collection/DataSet')) for key,path in series.items()},'validation':'passed'}


def run_slice_series(case_dir, output_root=None, *, start_index=1,end_index=81,indices=None,max_frames=None,
                     y_midplane=.75,x_slice=15.,x_min=0.,subdivisions_per_element=28,vmin=0.,vmax=10.,
                     resume=False,dry_run=False,recover_staging=False,save_samples=False,chunk_size=256,eigenvalue_chunk_rows=16,
                     periodic_axes=('x','y'),expected_element_counts=(272,12,8),log_name=None,progress=None):
    """Preflight, sequential cache reuse, staged publication, checkpoint and final audit."""
    if not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size<1 or not isinstance(eigenvalue_chunk_rows,int) or isinstance(eigenvalue_chunk_rows,bool) or eigenvalue_chunk_rows<1:
        raise ValueError('Chunk sizes must be positive integers.')
    case_dir=Path(case_dir).resolve()
    root=(DEFAULT_OUTPUT_ROOT/case_dir.name if output_root is None else Path(output_root)).resolve()
    repository=Path(__file__).resolve().parents[2]
    for destination in (root,root/'logs',root/'.stage5-staging'):
        target=destination.resolve()
        if target.is_relative_to(case_dir) or target.is_relative_to(repository):
            raise ValueError('Scientific and operational outputs must be outside original data/repository.')
    selection=expand_frame_indices(start_index,end_index,indices,max_frames)
    config=series_configuration(y_midplane=y_midplane,x_slice=x_slice,x_min=x_min,subdivisions_per_element=subdivisions_per_element,vmin=vmin,vmax=vmax,periodic_axes=periodic_axes,expected_element_counts=expected_element_counts)
    if dry_run:
        plan=inspect_slice_series(case_dir,root,selection,config,resume=resume,recover_staging=recover_staging)
        plan.pop('plans')
        return plan
    started=time.perf_counter()
    with _run_lock(root):
        plan=inspect_slice_series(case_dir,root,selection,config,resume=resume,recover_staging=recover_staging)
        plans=plan.pop('plans')
        protected={str(p):_file_hash(p) for p in root.glob('f*/**/*') if p.is_file()}
        root.joinpath('logs').mkdir(exist_ok=True)
        run_id=log_name or time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        if Path(run_id).name!=run_id:raise ValueError('log-name must be a filename stem.')
        log_path=root/'logs'/f'{run_id}.jsonl'
        summary_path=root/'logs'/f'{run_id}_summary.json'
        with log_path.open('x',buffering=1) as log:
            summary={'status':'running','pid':os.getpid(),'configuration':config,'requested_indices':selection,'frames':[],
                     'log_path':str(log_path),'summary_path':str(summary_path),'operator_build_count':0,'slice_plan_build_count':2,
                     'preflight':plan,'protected_existing_sha256':protected,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            def event(message,**fields):
                entry={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':message,**fields}
                log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
                if progress:progress(json.dumps(entry,allow_nan=False))
            def checkpoint():
                summary['total_seconds']=time.perf_counter()-started
                summary['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                _atomic_json(summary_path,summary)
                _atomic_json(root/'processing_summary.json',summary)
            event('Preflight passed',frames=len(selection),reused=plan['reused_frames'],pending=plan['pending_frames'])
            checkpoint()
            operator=None;initialization_seconds=0.;processed=0
            try:
                for item,state in zip(plan['snapshots'],plan['frames']):
                    final=root/f"f{item['index']:05d}"
                    if state['status']=='reused':
                        _update_series(root,case_dir.name,item['index'],item['time'],config)
                        summary['frames'].append(state);event('Verified existing frame reused',index=item['index'],time=item['time'])
                        checkpoint();continue
                    if recover_staging:
                        recovered=False
                        for stage in sorted((root/'.stage5-staging').glob(f"f{item['index']:05d}.*")):
                            try:validated=validate_frame_result(stage,case=case_dir.name,index=item['index'],time_value=item['time'],config=config,plans=plans,source_sha256=item['source_sha256'])
                            except (ValueError,OSError,KeyError) as exc:
                                event('Incomplete staging preserved; new staging will be used',index=item['index'],stage=str(stage),reason=str(exc));continue
                            _publish_directory(stage,final);_update_series(root,case_dir.name,item['index'],item['time'],config)
                            validated['status']='recovered';summary['frames'].append(validated);recovered=True
                            event('Completed staged frame recovered',index=item['index']);checkpoint();break
                        if recovered:continue
                    frame_started=time.perf_counter()
                    event('Frame started',index=item['index'],time=item['time'])
                    tick=time.perf_counter();data=_read_snapshot(item['source']);timings={'read':time.perf_counter()-tick}
                    tick=time.perf_counter()
                    if operator is None:
                        operator=build_l2_projection_operator(data,periodic_axes=tuple(config['periodic_axes']),chunk_size=chunk_size,expected_element_counts=tuple(config['expected_element_counts']))
                        # Reuse the two plans built during this process's preflight.
                        next(iter(plans.values())).node_map.validate_mesh(data)
                        summary['operator_build_count']=1
                        initialization_seconds=time.perf_counter()-tick
                        event('Stationary operator and two slice plans built',seconds=initialization_seconds,operator_build_count=1)
                    operator.node_map.validate_mesh(data)
                    timings['mesh_validation']=time.perf_counter()-tick-initialization_seconds if processed==0 else time.perf_counter()-tick
                    tick=time.perf_counter()
                    raw=np.empty((len(data.elem),*operator.node_map.element_shape,3,3),dtype=np.float64)
                    for ei,element in enumerate(data.elem):raw[ei]=element_velocity_gradient(element)
                    timings['spectral_gradients']=time.perf_counter()-tick
                    tick=time.perf_counter();projected=project_velocity_gradient(operator,raw,diagnostics=True,chunk_size=chunk_size)
                    del raw,data
                    timings['l2_projection']=time.perf_counter()-tick
                    if np.any(projected.diagnostics.discontinuity_after!=0):raise ValueError('Projected shared-node discontinuity is nonzero.')
                    tick=time.perf_counter()
                    slices={key:evaluate_tensor_slice_plan(projected.values,p,eigenvalue_chunk_rows=eigenvalue_chunk_rows) for key,p in plans.items()}
                    del projected
                    timings['slice_sampling_and_eigenvalues']=time.perf_counter()-tick
                    if _file_hash(Path(item['source']))!=item['source_sha256']:raise RuntimeError('Source changed during frame computation.')
                    stage=root/'.stage5-staging'/f"f{item['index']:05d}.{uuid.uuid4().hex}"
                    stage.mkdir(parents=True)
                    tick=time.perf_counter()
                    _export_frame(stage,final,case=case_dir.name,item=item,config=config,slices=slices,
                                  operator_build_count=summary['operator_build_count'],plans_build_count=summary['slice_plan_build_count'],
                                  timings=timings,initialization_seconds=initialization_seconds,save_samples=save_samples)
                    timings['export']=time.perf_counter()-tick
                    # Metadata remains private staging until all checks pass.
                    metadata_path=frame_manifest(case_dir.name,item['index'],config,stage)['metadata']
                    metadata=json.loads(metadata_path.read_text());metadata['stage_seconds']=timings;metadata['total_seconds']=time.perf_counter()-frame_started;metadata['operator_reused']=processed>0
                    _atomic_json(metadata_path,metadata)
                    validated=validate_frame_result(stage,case=case_dir.name,index=item['index'],time_value=item['time'],config=config,plans=plans,source_sha256=item['source_sha256'])
                    _publish_directory(stage,final)
                    _update_series(root,case_dir.name,item['index'],item['time'],config)
                    del slices
                    gc.collect()
                    processed+=1;validated.update(status='generated',operator_reused=processed>1,observed_rss_bytes=_rss())
                    summary['frames'].append(validated)
                    event('Frame completed and verified',**validated,completed=len(summary['frames']),requested=len(selection))
                    checkpoint()
                event('Final manifest/reader/source verification started')
                summary['final_verification']=verify_series_outputs(root,case_dir.name,plan['snapshots'],config,plans)
                for path,digest in protected.items():
                    if _file_hash(Path(path))!=digest:raise RuntimeError(f'Protected existing result changed: {path}')
                summary['protected_existing_unchanged']=True;summary['status']='completed'
                summary['initialization_seconds']=initialization_seconds
                event('Run completed and verified',completed=len(summary['frames']),operator_build_count=summary['operator_build_count'])
                checkpoint()
            except BaseException as exc:
                summary['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
                summary['error']=f'{type(exc).__name__}: {exc}'
                event('Run stopped safely',error=summary['error']);checkpoint();raise
            return summary
