"""Safe native-volume omega_x workflow; no lambda_ci computation or output writes."""
from __future__ import annotations

from pathlib import Path
from math import prod
import json
import resource
import shutil
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.l2_projection import build_l2_projection_operator, structured_global_node_shape, _positive_integer, _periodic_axes
from nek_post.lambda_ci_export import write_single_frame_pvd
from nek_post.lambda_ci_series import expand_frame_indices, _publish_directory
from nek_post.lambda_ci_slice_export import _pvd_update_lock
from nek_post.lambda_ci_workflow import DEFAULT_OUTPUT_ROOT as LAMBDA_ROOT, _file_hash, _nearest_directory, available_memory_bytes
from nek_post.rectilinear_export import assemble_rectilinear_scalar_field, write_rectilinear_scalar_vtr
from nek_post.vorticity import compute_projected_omega_x

DEFAULT_OUTPUT_ROOT = Path('/data/Nek5000_data/results/h_refinement/omega_x')
RAM_SAFETY_MARGIN_BYTES = 1024**3
VTR_VALIDATION_PAYLOAD_COPIES = 4
# Explicitly reviewed orchestration-only transitions. The successor digest
# normalizes only its own 64-hex literal to zeros, avoiding a self-hash cycle.
# Any further workflow edit must fail closed until separately reviewed.
_RAM_PREFLIGHT_PREDECESSOR_SHA256 = '9f261baadf8e888ef216f83dc5cdcd2f9a4336e33c5b200340c60e22551d3736'
_CACHE_PREDECESSOR_SHA256 = '91e50314111ed4a4048ab445af41be57d7268eacdd1066251e74ee42434f713f'
_RAM_PREFLIGHT_WORKFLOW_SHA256 = '3b68a0982eb3dab6d3f15f83dc3770fce2a854e9b59007110f77fb163575e42f'


def _available_ram():
    """Use the established host/cgroup probe; unknown availability fails closed."""
    try:
        value = available_memory_bytes()
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError('available RAM must be a nonnegative integer byte count')
        return int(value), None
    except Exception as exc:
        return None, f'Cannot establish available RAM: {type(exc).__name__}: {exc}'


def _require_ram(required, operation):
    available, error = _available_ram()
    if error:
        raise MemoryError(f'{operation}: {error}')
    if required >= available:
        raise MemoryError(f'{operation}: insufficient RAM; require {required} bytes, available {available} bytes.')
    return available


def _validation_ram_bytes(shape, file_bytes, *, retained_bytes=0):
    """Peak for one uncompressed float64 VTR, not a sum over sequential frames.

    Budget four payloads for file/decode buffers, VTK data, conversion and
    validation temporaries, plus the existing 1-GiB safety margin. NumPy views
    often avoid copies, but the budget does not depend on that optimization.
    Stage read-back also retains the in-memory native field being compared.
    """
    payload = 8 * (prod(shape) + sum(shape))
    return retained_bytes + VTR_VALIDATION_PAYLOAD_COPIES * max(int(file_bytes), payload) + RAM_SAFETY_MARGIN_BYTES


def _operator_ram_bytes(local_nodes, global_nodes, element_count, visual_shape):
    """One weights/mass payload plus compact mapping and Python signatures."""
    return 8*(local_nodes + global_nodes + sum(visual_shape)) + element_count*(3*8 + 3 + 256)


def _mesh_validation_ram_bytes(element_count, element_shape):
    # validate_mesh forms an element-reference tuple and hashes one element:
    # three float64 coordinate arrays plus one contiguous byte buffer.
    return element_count*8 + 4*prod(element_shape)*8


def _validate_cached_operator(operator, data, config):
    """Exact element-order geometry signatures preserve the prepared mapping."""
    mapping = operator.node_map
    if (mapping.periodic_axes != _periodic_axes(config['periodic_axes'])
            or mapping.element_interval_counts != tuple(config['expected_element_counts'])
            or mapping.element_shape != tuple(reversed(config['expected_orders']))):
        raise ValueError('Cached omega_x operator configuration/order differs.')
    nodes_xyz = tuple(mapping.element_shape[a] for a in mapping.physical_to_array_axes)
    expected_shape = structured_global_node_shape(mapping.element_interval_counts, nodes_xyz,
                                                   periodic_axes=config['periodic_axes'])
    if mapping.global_shape_xyz != expected_shape:
        raise ValueError('Cached omega_x global node mapping differs.')
    _require_ram(_mesh_validation_ram_bytes(mapping.element_count, mapping.element_shape) + RAM_SAFETY_MARGIN_BYTES,
                 'Validating cached omega_x mesh')
    mapping.validate_mesh(data)


def resolve_case(case):
    path = Path(case).expanduser()
    if len(path.parts) == 1 and path.name in ('N7_H', 'case_N7_H'):
        path = Path('/data/Nek5000_data/case_N7_H')
    return path.resolve()


def _paths(directory, case, index):
    stem = f'{case}_f{index:05d}_omega_x'
    return {key: Path(directory)/(stem + suffix) for key, suffix in
            (('vtr', '.vtr'), ('pvd', '.pvd'), ('metadata', '.json'))}


def _signature():
    base = Path(__file__).parent
    names = ('gll.py', 'spectral_derivatives.py', 'l2_projection.py', 'vorticity.py',
             'rectilinear_export.py', 'omega_x_workflow.py')
    return {name: _file_hash(base/name) for name in names}


def _approved_ram_workflow(current_digest):
    """Verify the exact reviewed successor source, including this check itself."""
    from hashlib import sha256
    source = Path(__file__).read_bytes()
    if sha256(source).hexdigest() != current_digest:
        return False
    marker = b"_RAM_PREFLIGHT_WORKFLOW_SHA256 = '" + _RAM_PREFLIGHT_WORKFLOW_SHA256.encode() + b"'"
    if source.count(marker) != 1:
        return False
    normalized = source.replace(marker, b"_RAM_PREFLIGHT_WORKFLOW_SHA256 = '" + b'0'*64 + b"'", 1)
    return sha256(normalized).hexdigest() == _RAM_PREFLIGHT_WORKFLOW_SHA256


def _resume_signature_matches(stored):
    current = _signature()
    if stored == current:
        return True
    if not isinstance(stored, dict) or set(stored) != set(current):
        return False
    workflow = 'omega_x_workflow.py'
    return (stored[workflow] in (_RAM_PREFLIGHT_PREDECESSOR_SHA256, _CACHE_PREDECESSOR_SHA256)
            and all(stored[name] == digest for name, digest in current.items() if name != workflow)
            and _approved_ram_workflow(current[workflow]))


def _coordinate_hash(vector):
    from hashlib import sha256
    return sha256(np.asarray(vector, dtype=np.float64).tobytes()).hexdigest()


def validate_omega_x_frame(directory, *, case, index, time_value, config, source_sha256,
                           expected_field=None, retained_operator_bytes=0):
    """Checksum and VTK-reader validation; existence alone never means completed."""
    paths = _paths(directory, case, index)
    if any(not p.is_file() or p.is_symlink() or not p.stat().st_size for p in paths.values()):
        raise ValueError('Incomplete omega_x frame outputs.')
    shape = structured_global_node_shape(config['expected_element_counts'], config['expected_orders'], periodic_axes=())
    retained = _positive_integer(retained_operator_bytes, 'retained_operator_bytes', 0)
    if expected_field is not None:
        shape = tuple(len(v) for v in expected_field.coordinates)
        retained += expected_field.values.nbytes + sum(v.nbytes for v in expected_field.coordinates)
    _require_ram(_validation_ram_bytes(shape, paths['vtr'].stat().st_size, retained_bytes=retained),
                 f'Validating omega_x frame {index}')
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    metadata = json.loads(paths['metadata'].read_text())
    if (metadata.get('schema_version') != 1 or metadata.get('completion_status') != 'completed'
            or metadata.get('field') != 'omega_x' or metadata.get('case') != case
            or metadata.get('frame_index') != index or metadata.get('time') != time_value
            or metadata.get('configuration') != config or metadata.get('source_sha256') != source_sha256
            or not _resume_signature_matches(metadata.get('numerical_software_signature'))):
        raise ValueError('Incompatible omega_x metadata/source/configuration/signature.')
    checksums = metadata.get('output_sha256', {})
    if set(checksums) != {'vtr', 'pvd'} or any(_file_hash(paths[k]) != h for k, h in checksums.items()):
        raise ValueError('omega_x output checksum differs.')
    reader = vtkXMLRectilinearGridReader();reader.SetFileName(str(paths['vtr']));reader.Update()
    grid = reader.GetOutput();info = metadata['visualization']
    if reader.GetErrorCode() or list(grid.GetDimensions()) != info['shape_xyz'] or grid.GetNumberOfCells() != int(np.prod(np.asarray(info['shape_xyz']) - 1)):
        raise ValueError('omega_x VTR dimensions/topology differ.')
    coordinates = tuple(vtk_to_numpy(getter()) for getter in
                        (grid.GetXCoordinates, grid.GetYCoordinates, grid.GetZCoordinates))
    for axis, vector in enumerate(coordinates):
        if vector.dtype != np.float64 or not np.isfinite(vector).all() or not np.all(np.diff(vector) > 0) or _coordinate_hash(vector) != info['coordinate_sha256'][axis]:
            raise ValueError('omega_x VTR coordinates differ.')
        if expected_field is not None and not np.array_equal(vector, expected_field.coordinates[axis]):
            raise ValueError('omega_x VTR changed original physical coordinates.')
    timestamp = grid.GetFieldData().GetArray('TIME_VALUE')
    scalar = grid.GetPointData().GetArray('omega_x')
    if timestamp is None or timestamp.GetNumberOfTuples() != 1 or timestamp.GetValue(0) != time_value:
        raise ValueError('omega_x VTR timestamp differs from actual source.')
    if scalar is None or scalar.GetNumberOfComponents() != 1 or grid.GetPointData().GetNumberOfArrays() != 1:
        raise ValueError('omega_x VTR must contain only scalar PointData omega_x.')
    values = vtk_to_numpy(scalar)
    if values.dtype != np.float64 or values.size != grid.GetNumberOfPoints() or not np.isfinite(values).all():
        raise ValueError('omega_x VTR scalar is invalid.')
    if expected_field is not None and not np.array_equal(values, expected_field.values.ravel()):
        raise ValueError('omega_x VTR changed scalar values or ordering.')
    if metadata['omega_x'] != {'min':float(values.min()), 'max':float(values.max())}:
        raise ValueError('omega_x VTR min/max differ from diagnostics.')
    entries = ET.parse(paths['pvd']).findall('Collection/DataSet')
    if len(entries) != 1 or float(entries[0].get('timestep')) != time_value or entries[0].get('file') != paths['vtr'].name:
        raise ValueError('omega_x PVD frame/timestamp/reference differs.')
    return metadata


def inspect_omega_x_series(case, indices, output_dir=None, *, resume=False, overwrite=False,
                           periodic_axes=('x', 'y'), expected_element_counts=(272,12,8),
                           expected_orders=(8,8,8), chunk_size=256):
    """Read-only header/resource plan; validate any explicitly reused outputs."""
    if resume and overwrite:
        raise ValueError('resume and overwrite are mutually exclusive.')
    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    case_dir = resolve_case(case)
    root = (DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)).expanduser().resolve()/case_dir.name
    repository = Path(__file__).resolve().parents[2]
    for directory in (root, root/'.omega-x-staging', root/'.omega-x-backups'):
        target = directory.resolve()
        if target.is_relative_to(case_dir) or target.is_relative_to(repository) or target.is_relative_to(LAMBDA_ROOT.resolve()):
            raise ValueError('omega_x outputs must be outside original data, repository and lambda_ci outputs.')
        if not target.is_relative_to(root):
            raise ValueError('Operational directories must remain inside the case output root.')
    if not case_dir.is_dir():raise FileNotFoundError(case_dir)
    selection = expand_frame_indices(indices=indices)
    # Validate explicit topology/periodicity even in a header-only dry run.
    structured_global_node_shape(expected_element_counts, expected_orders, periodic_axes=periodic_axes)
    config = {'periodic_axes':list(periodic_axes), 'expected_element_counts':list(expected_element_counts),
              'expected_orders':list(expected_orders)}
    frames = [];times = [];operator_retained = False
    for index in selection:
        source = case_dir/f'GC0.f{index:05d}';header = read_nek_header(source)
        if (header.nb_dims != 3 or header.nb_vars[0] != 3 or header.nb_vars[1] != 3
                or header.nb_files != 1 or header.nb_elems_file != header.nb_elems
                or header.nb_elems != int(np.prod(expected_element_counts))
                or tuple(header.orders) != tuple(expected_orders) or header.wdsz not in (4,8)
                or not np.isfinite(header.time)):
            raise ValueError(f'Invalid/incompatible native 3D snapshot: {source}')
        final = root/f'f{index:05d}';paths = _paths(final, case_dir.name, index)
        if final.is_symlink() or any(not p.resolve().is_relative_to(root) for p in paths.values()):
            raise ValueError('Frame outputs must not redirect outside their case root.')
        action = 'generated'
        if final.exists():
            if resume:
                action = 'reused'
            elif overwrite:
                # Never replace an unrelated directory or silently discard unknown files.
                if not paths['metadata'].is_file() or paths['metadata'].is_symlink():
                    raise FileExistsError('Overwrite requires recognizable omega_x metadata.')
                previous = json.loads(paths['metadata'].read_text())
                if (previous.get('field') != 'omega_x' or previous.get('case') != case_dir.name
                        or previous.get('frame_index') != index or any(p.is_symlink() for p in final.rglob('*'))
                        or set(final.iterdir()) != set(paths.values())):
                    raise FileExistsError('Refusing overwrite of unrelated files or symlinks.')
                action = 'overwritten'
            else:raise FileExistsError(f'Existing omega_x frame: {final}; choose --resume or --overwrite.')
        times.append(float(header.time))
        visual_shape = structured_global_node_shape(expected_element_counts, expected_orders, periodic_axes=())
        local_nodes = int(header.nb_elems*np.prod(header.orders))
        projection_nodes = int(np.prod(structured_global_node_shape(expected_element_counts,expected_orders,periodic_axes=periodic_axes)))
        visual_nodes = int(np.prod(visual_shape))
        computation_ram = local_nodes*(sum(header.nb_vars)*header.wdsz + 3*8) + source.stat().st_size + projection_nodes*8*2 + visual_nodes*8*3 + RAM_SAFETY_MARGIN_BYTES
        cache_bytes = _operator_ram_bytes(local_nodes, projection_nodes, header.nb_elems, visual_shape)
        mesh_scratch = _mesh_validation_ram_bytes(header.nb_elems, header.orders)
        # The existing computation formula already includes weights + mass.
        # Add signature/mapping overhead and explicit validation scratch, not
        # a second full operator. Keep one operator through later read-back.
        computation_ram += cache_bytes - 8*(local_nodes + projection_nodes) + mesh_scratch
        if action != 'reused':operator_retained = True
        retained_operator = cache_bytes if operator_retained else 0
        vtr = paths['vtr']
        if action == 'reused' and (not vtr.is_file() or vtr.is_symlink()):
            raise ValueError('Incomplete omega_x frame outputs.')
        output_bytes = visual_nodes*8 + 2*1024**2
        file_bytes = vtr.stat().st_size if action == 'reused' else output_bytes
        retained = retained_operator + (0 if action == 'reused' else 8 * (visual_nodes + sum(visual_shape)))
        validation_ram = _validation_ram_bytes(visual_shape, file_bytes, retained_bytes=retained)
        estimated_ram = validation_ram if action == 'reused' else max(computation_ram, validation_ram)
        frames.append({'index':index,'time':float(header.time),'istep':int(header.istep),'source_file':str(source),
                       'action':action,'output_paths':{k:str(p) for k,p in paths.items()},
                       'estimated_ram_bytes':estimated_ram,'estimated_computation_ram_bytes':0 if action == 'reused' else computation_ram,
                       'estimated_validation_ram_bytes':validation_ram,'retained_validation_bytes':retained,
                       'estimated_operator_cache_bytes':retained_operator,
                       'estimated_mesh_validation_scratch_bytes':0 if action == 'reused' else mesh_scratch,
                       'resume_validation':'pending' if action == 'reused' else 'not_applicable',
                       'estimated_output_bytes':output_bytes,
                       'estimated_visual_shape_xyz':list(visual_shape)})
    if not np.all(np.diff(times) > 0):raise ValueError('Actual snapshot times must increase with frame index.')
    ancestor = _nearest_directory(root)
    memory, memory_error = _available_ram();disk = shutil.disk_usage(ancestor).free
    pending = [f for f in frames if f['action'] != 'reused']
    required_disk = sum(f['estimated_output_bytes'] for f in pending) + (max(f['estimated_output_bytes'] for f in pending) if pending else 0)
    required_ram = max(f['estimated_ram_bytes'] for f in frames)
    errors = []
    if memory_error:errors.append(memory_error)
    elif required_ram >= memory:errors.append(f'Insufficient RAM: require {required_ram} bytes, available {memory} bytes.')
    if required_disk >= disk:errors.append(f'Insufficient disk: require {required_disk} bytes, available {disk} bytes.')
    report = {'case_dir':str(case_dir),'output_root':str(root),'configuration':config,'frames':frames,
            'available_ram_bytes':memory,'estimated_ram_bytes':required_ram,
            'estimated_computation_ram_bytes':max(f['estimated_computation_ram_bytes'] for f in frames),
            'estimated_validation_ram_bytes':max(f['estimated_validation_ram_bytes'] for f in frames),
            'estimated_operator_cache_bytes':max(f['estimated_operator_cache_bytes'] for f in frames),
            'available_disk_bytes':disk,'required_disk_bytes':required_disk,
            'resource_errors':errors,'resources_acceptable':not errors}
    if errors:
        return report  # Header/stat-only rejection; no hashes, VTK reads or computation.
    for frame in frames:
        if frame['action'] != 'reused':continue
        try:
            _require_ram(frame['estimated_validation_ram_bytes'], f'Resume preflight for frame {frame["index"]}')
            validate_omega_x_frame(Path(frame['output_paths']['vtr']).parent,case=case_dir.name,index=frame['index'],
                time_value=frame['time'],config=config,source_sha256=_file_hash(Path(frame['source_file'])))
        except MemoryError as exc:
            report['available_ram_bytes'], _ = _available_ram()
            report['resource_errors'].append(str(exc))
            report['resources_acceptable'] = False
            return report
        frame['resume_validation'] = 'passed'
    return report


def run_omega_x_series(case='N7_H', output_dir=None, *, indices=None,start_index=None,end_index=None,
                       resume=False,overwrite=False,dry_run=False,chunk_size=256,
                       periodic_axes=('x','y'),expected_element_counts=(272,12,8),
                       expected_orders=(8,8,8),progress=None):
    """Default to frame 79; publish verified native-volume frames sequentially.

    Explicit overwrite preserves the previous recognized frame in a unique
    backup directory. Failed/private staging is retained for review. Resume
    revalidates completed frames without recomputing any velocity derivatives.
    """
    if indices is not None and (start_index is not None or end_index is not None):
        raise ValueError('Choose indices or a start/end range.')
    if (start_index is None) != (end_index is None):raise ValueError('Specify both start-index and end-index.')
    selection = expand_frame_indices(start_index,end_index) if start_index is not None else expand_frame_indices(indices=[79] if indices is None else indices)
    settings = dict(resume=resume,overwrite=overwrite,periodic_axes=periodic_axes,
                    expected_element_counts=expected_element_counts,expected_orders=expected_orders,chunk_size=chunk_size)
    plan = inspect_omega_x_series(case,selection,output_dir,**settings)
    if dry_run:return plan
    if not plan['resources_acceptable']:raise MemoryError('; '.join(plan['resource_errors']))
    root = Path(plan['output_root']);started = time.perf_counter();frames = []
    operator = None;operator_build_count = 0;operator_bytes = 0
    def notify(message):
        if progress:progress(message)
    with _pvd_update_lock(root/'omega_x'):
        # Check again under the writer lock before touching existing results.
        plan = inspect_omega_x_series(case,selection,output_dir,**settings)
        if not plan['resources_acceptable']:raise MemoryError('Resources changed: ' + '; '.join(plan['resource_errors']))
        for item in plan['frames']:
            source = Path(item['source_file']);index = item['index'];final = Path(item['output_paths']['vtr']).parent
            _require_ram(item['estimated_ram_bytes'], f'Processing omega_x frame {index}')
            source_hash = _file_hash(source)
            if item['action'] == 'reused':
                metadata = validate_omega_x_frame(final,case=Path(plan['case_dir']).name,index=index,
                    time_value=item['time'],config=plan['configuration'],source_sha256=source_hash,
                    retained_operator_bytes=operator_bytes)
                frames.append({'index':index,'time':item['time'],'status':'reused','omega_x':metadata['omega_x'],
                               'output_paths':item['output_paths']});continue
            tick = time.perf_counter();frame_started = tick
            header = read_nek_header(source)
            skip = ('ux','pressure','temperature') + tuple(f's{i:02d}' for i in range(1,header.nb_vars[4]+1))
            _require_ram(item['estimated_ram_bytes'], f'Loading native snapshot for frame {index}')
            data = read_nek_file(source,dtype=f'float{header.wdsz*8}',skip_vars=skip)
            if data.time != item['time'] or len(data.elem) != header.nb_elems:raise ValueError('Snapshot reader/header mismatch.')
            timings = {'read':time.perf_counter()-tick};tick = time.perf_counter()
            reused_operator = operator is not None
            if operator is None:
                notify(f'Frame {index}, actual time {item["time"]}: prepare scalar L2 operator')
                operator = build_l2_projection_operator(data,periodic_axes=periodic_axes,expected_element_counts=expected_element_counts,chunk_size=chunk_size)
                operator_build_count += 1
                operator_bytes = operator.storage_bytes + operator.node_map.element_count*256
                if operator_bytes > item['estimated_operator_cache_bytes']:
                    raise MemoryError('Prepared omega_x operator exceeds the preflight cache budget.')
            else:
                notify(f'Frame {index}, actual time {item["time"]}: validate cached scalar L2 operator')
            timings['projection_preparation'] = time.perf_counter()-tick;tick = time.perf_counter()
            _validate_cached_operator(operator, data, plan['configuration'])
            timings['mesh_validation'] = time.perf_counter()-tick;tick = time.perf_counter()
            values = compute_projected_omega_x(data,operator,chunk_size=chunk_size,progress=progress)
            timings['local_curl_and_scalar_projection'] = time.perf_counter()-tick;tick = time.perf_counter()
            field = assemble_rectilinear_scalar_field(data,operator.node_map,values,name='omega_x',chunk_size=chunk_size)
            del values,data
            timings['native_visualization_assembly'] = time.perf_counter()-tick
            staging = root/'.omega-x-staging';staging.mkdir(exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f'f{index:05d}.',dir=staging))
            paths = _paths(stage,Path(plan['case_dir']).name,index);tick = time.perf_counter()
            write_rectilinear_scalar_vtr(paths['vtr'],field,time=item['time'])
            write_single_frame_pvd(paths['pvd'],paths['vtr'],time=item['time'])
            timings['export'] = time.perf_counter()-tick
            if _file_hash(source) != source_hash:raise RuntimeError('Original snapshot changed; publication stopped.')
            metadata = {'schema_version':1,'field':'omega_x','completion_status':'completed',
                'case':Path(plan['case_dir']).name,'frame_index':index,'time':item['time'],'istep':item['istep'],
                'source_file':str(source),'source_sha256':source_hash,'source_unchanged':True,
                'configuration':plan['configuration'],'numerical_software_signature':_signature(),
                'omega_x':{'min':float(field.values.min()),'max':float(field.values.max())},
                'visualization':{'shape_xyz':[len(v) for v in field.coordinates],
                    'bounds_xyz':[[float(v[0]),float(v[-1])] for v in field.coordinates],
                    'coordinate_sha256':[_coordinate_hash(v) for v in field.coordinates],
                    'native_gll':True,'periodic_endpoints_preserved':True,'interpolation':'none'},
                'output_paths':item['output_paths'],'output_sha256':{k:_file_hash(paths[k]) for k in ('vtr','pvd')},
                'stage_seconds':timings,'total_seconds':time.perf_counter()-frame_started,
                'operator_reused':reused_operator,'operator_build_count':operator_build_count,
                'operator_storage_bytes':operator_bytes,
                'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
            with paths['metadata'].open('x') as stream:json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
            validate_omega_x_frame(stage,case=metadata['case'],index=index,time_value=item['time'],
                config=plan['configuration'],source_sha256=source_hash,expected_field=field,
                retained_operator_bytes=operator_bytes)
            del field
            backup = None
            if item['action'] == 'overwritten':
                backups = root/'.omega-x-backups';backups.mkdir(exist_ok=True)
                backup = backups/f'f{index:05d}.{uuid.uuid4().hex}'
                _publish_directory(final,backup)
            try:_publish_directory(stage,final)
            except BaseException:
                if backup is not None:_publish_directory(backup,final)
                raise
            record = {'index':index,'time':item['time'],'status':item['action'],'omega_x':metadata['omega_x'],
                      'runtime_seconds':time.perf_counter()-frame_started,'peak_rss_bytes':metadata['peak_rss_bytes'],
                      'visualization':metadata['visualization'],'output_paths':item['output_paths'],'validation':'passed'}
            record.update(operator_reused=reused_operator,operator_build_count=operator_build_count,
                          operator_storage_bytes=operator_bytes,stage_seconds=timings)
            if backup is not None:record['preserved_previous_frame'] = str(backup)
            frames.append(record);notify(json.dumps(record,allow_nan=False))
    return {'status':'completed','frames':frames,'preflight':plan,'total_seconds':time.perf_counter()-started,
            'operator_build_count':operator_build_count,
            'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
