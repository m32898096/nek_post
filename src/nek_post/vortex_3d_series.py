"""Sequential vortex VTUs with strict resume, stationary meshes and atomic PVDs.

All scientific calculations and frame publication delegate to the Phase 1/2
single-frame workflow. Cropped dry-runs read coordinates only, never velocity
payloads, VTU decoders, gradients, eigenvalues or high-resolution 3D meshes.
"""
from __future__ import annotations

from pathlib import Path
from numbers import Integral
import gc
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

from nek_post import vortex_3d_workflow as single
from nek_post.l2_projection import _positive_integer
from nek_post.lambda_ci_series import _atomic_json, _publish_directory
from nek_post.lambda_ci_slice_export import _collection, _pvd_update_lock
from nek_post.vortex_3d_artifacts import (
    validate_saved_grid, binary_vtu_fingerprint, open_regular_file,
    regular_file_identity, regular_file_hash,
)
from nek_post.vortex_3d_selection import VolumeSelection, series_volume_selection, apply_volume_output_plan


DEFAULT_OUTPUT_ROOT = Path('/data/Nek5000_data/results/h_refinement/vortex_3d_series')
SCHEMA = 1
CONTROL_MARGIN = 64 * 1024**2
MAX_METADATA_BYTES = 64 * 1024**2


def frame_indices(indices=None, *, start_index=None, end_index=None, max_frames=None):
    """Bound expansion before allocating ranges; reject duplicates before capping."""
    if indices is not None and (start_index is not None or end_index is not None):
        raise ValueError('Choose --indices or --start-index/--end-index, not both.')
    if indices is None:
        if (start_index is None) != (end_index is None):
            raise ValueError('Supply both --start-index and --end-index.')
        start, end = (1, 81) if start_index is None else (start_index, end_index)
        indices = f'{start}-{end}'
        # Preserve strict integer typing for the Python API, before formatting.
        if start_index is not None:
            _positive_integer(start_index, 'start_index')
            _positive_integer(end_index, 'end_index')
    if isinstance(indices, str):
        values, seen = [], set()
        for token in indices.split(','):
            match = re.fullmatch(r'\s*(\d+)(?:-(\d+))?\s*', token)
            if not match:
                raise ValueError('Invalid frame index/range.')
            first, last = int(match[1]), int(match[2] or match[1])
            if not 1 <= first <= last <= 99999:
                raise ValueError('Require increasing frame ranges within 1..99999.')
            for value in range(first, last+1):
                if value in seen:
                    raise ValueError('Duplicate frame indices.')
                seen.add(value)
                values.append(value)
    else:
        try:
            values = tuple(indices)
        except TypeError as exc:
            raise ValueError('indices must contain frame integers.') from exc
    if not values:
        raise ValueError('Select at least one frame.')
    values = tuple(_positive_integer(i, 'frame index') for i in values)
    if max(values) > 99999 or len(set(values)) != len(values):
        raise ValueError('Duplicate or out-of-range frame indices.')
    result = tuple(sorted(values))
    if max_frames is not None:
        result = result[:_positive_integer(max_frames, 'max_frames')]
    return result


def _signature():
    values = single._signature()
    base = Path(__file__).parent
    for name in ('vortex_3d_series.py', 'lambda_ci_slice_export.py'):
        values[name] = single._file_hash(base/name)
    return values


def _options(config):
    return dict(periodic_axes=config.periodic_axes,
                expected_element_counts=config.expected_element_counts, crop=config.crop,
                chunk_size=config.chunk_size, sampling_nodes=config.sampling_nodes)


def _configuration(config):
    result = _options(config)
    for name in ('periodic_axes', 'expected_element_counts', 'crop'):
        if result[name] is not None:
            result[name] = list(result[name])
    if result['sampling_nodes'] is None:
        del result['sampling_nodes']
    return result


def _regular(path):
    return regular_file_identity(path)


def _json(path, *, expected=None):
    try:
        with open_regular_file(path, expected=expected) as stream:
            if os.fstat(stream.fileno()).st_size > MAX_METADATA_BYTES:
                raise ValueError(f'Oversized metadata: {path}')
            payload = stream.read(MAX_METADATA_BYTES + 1)
            if len(payload) > MAX_METADATA_BYTES:
                raise ValueError(f'Oversized metadata: {path}')
            value = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid JSON: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Metadata must be an object: {path}')
    return value


def _stat(path):
    return regular_file_identity(path)


def _require_identity(path, expected, message):
    try:
        if _stat(path) == expected:
            return
    except (OSError, ValueError) as exc:
        raise RuntimeError(message) from exc
    raise RuntimeError(message)


def _locations(case, output_dir, config, first):
    case = single._case_directory(case)
    parent = DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)
    effective = parent / f'gll{config.sampling_nodes}' if config.sampling_nodes and config.sampling_nodes > 8 else parent
    root = single._safe_destination(effective, case, first, allow_existing=True).parent
    for path in (root/'series', root/'logs', root/'.vortex-3d-series.lock'):
        if path.is_symlink() or (path.exists() and (not path.is_file() if path.name.endswith('.lock') else not path.is_dir())):
            raise ValueError(f'Unsafe series control path: {path}')
    return case, parent, root


def _filesystem_limits(root):
    ancestor = single._nearest_directory(root)
    maximum, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    process_limit = None if maximum == resource.RLIM_INFINITY else int(maximum)
    filesystem = 'unknown'
    if shutil.which('stat'):
        result = subprocess.run(['stat', '-f', '-c', '%T', str(ancestor)],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            filesystem = result.stdout.strip()
    filesystem_limit = 2**32-1 if filesystem.lower() in ('msdos', 'vfat', 'fat', 'fat32') else None
    known = [v for v in (process_limit, filesystem_limit) if v is not None]
    return dict(filesystem_type=filesystem, process_file_size_limit_bytes=process_limit,
                filesystem_file_size_limit_bytes=filesystem_limit,
                effective_file_size_limit_bytes=min(known) if known else None,
                uncertainty='Other filesystem/quota limits and compressed VTU size are not established.')


def _control_allowance(frame_count, selection_count):
    # Summaries retain metadata, not scientific arrays. Bound simultaneous
    # summary/manifest/temp files and logs when inventories grow beyond 81.
    per_file = 1024**2+16384*frame_count
    return per_file, max(CONTROL_MARGIN, 4*per_file+8192*selection_count)


def _frame_record(directory, header, config, *, deep, source_hash):
    paths = {name: directory/Path(path).name for name, path in header['output_paths'].items()}
    if directory.is_symlink() or not directory.is_dir() or set(directory.iterdir()) != set(paths.values()):
        raise ValueError(f'Incomplete or unexpected frame artifacts: {directory}')
    for path in paths.values():
        _regular(path)
    original_stats = {str(path): _stat(path) for path in paths.values()}
    report = _json(paths['metadata'], expected=original_stats[str(paths['metadata'])])
    if type(report.get('schema_version')) is not int or type(report.get('frame_index')) is not int:
        raise ValueError('Frame schema/index must be integers, not booleans.')
    for key in ('case', 'source_file', 'frame_index', 'time', 'istep', 'input_word_bytes',
                'element_count', 'element_local_shape', 'element_map', 'configuration', 'output_paths',
                'full_local_point_count', 'full_linear_hex_count', 'native_local_point_count', 'output_selection'):
        if report.get(key) != header[key]:
            raise ValueError(f'Frame metadata differs from source/configuration: {key}')
    if report.get('output_plan') != header.get('output_plan'):
        raise ValueError('Frame selected-volume plan/provenance differs from the requested region.')
    if (report.get('schema_version') != 1 or report.get('completion_status') != 'completed' or
            report.get('source_sha256') != source_hash or report.get('source_unchanged') is not True or
            report.get('binary_vtu_roundtrip_verified') is not True or
            report.get('numerical_software_signature') != single._signature()):
        raise ValueError('Incomplete frame, changed source or incompatible implementation signature.')
    if (report.get('software') != single._software_versions() or
            report.get('python') != platform.python_version()):
        raise ValueError('Frame software/Python provenance differs from the current environment.')
    dense = config.sampling_nodes is not None and config.sampling_nodes > 8
    if report.get('numerical_definition') != single._numerical_definition(dense):
        raise ValueError('Frame numerical definition/projection order provenance differs.')
    mesh = report.get('native_mesh_signature')
    arrays = report.get('array_sha256')
    if (not isinstance(mesh, dict) or set(mesh) != {'coordinate_order_sha256', 'element_shape',
            'element_counts', 'physical_to_array_axes', 'cells_sha256', 'orientations_sha256'} or
            not isinstance(arrays, dict) or set(arrays) != {'points', 'connectivity', 'offsets',
            'cell_types', 'time', 'omega_x', 'lambda_ci'} or
            any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in arrays.values())):
        raise ValueError('Frame lacks native geometry/ordering or array fingerprints.')
    if (mesh['element_shape'] != header['element_local_shape'] or
            mesh['element_counts'] != list(config.expected_element_counts) or
            sorted(mesh['physical_to_array_axes']) != [0, 1, 2] or
            any(not isinstance(mesh[k], str) or not re.fullmatch('[0-9a-f]{64}', mesh[k])
                for k in ('coordinate_order_sha256', 'cells_sha256', 'orientations_sha256'))):
        raise ValueError('Native geometry signature is malformed or incompatible.')
    if 'output_plan' in header and mesh != header['output_plan']['native_mesh_signature']:
        raise ValueError('Native geometry differs from the selected-volume preflight plan.')
    if config.sampling_nodes is not None and report.get('reconstruction', {}).get('target_node_count') != config.sampling_nodes:
        raise ValueError('Reconstruction sampling differs.')
    if config.sampling_nodes is not None and report['reconstruction'].get('omega_route_audit_performed') is not dense:
        raise ValueError('Reconstruction-route audit provenance differs from the sampling mode.')
    visual = report.get('visualization', {})
    if (type(visual.get('points')) is not int or not 0 < visual['points'] <= header['full_local_point_count'] or
            type(visual.get('cells')) is not int or not 0 < visual['cells'] <= header['full_linear_hex_count'] or
            visual.get('physical_to_array_axes') != mesh['physical_to_array_axes'] or
            visual.get('shared_points') != 'preserved element-local, never merged' or
            visual.get('cell_type') != 'VTK_HEXAHEDRON' or visual.get('bounds') is None):
        raise ValueError('Frame visualization metadata is missing or incompatible.')
    single.validate_crop(visual['bounds'])
    ranges = report.get('field_ranges')
    if not isinstance(ranges, dict) or set(ranges) != {'omega_x', 'lambda_ci'}:
        raise ValueError('Frame field-range metadata is incomplete.')
    for name, values in ranges.items():
        if (not isinstance(values, dict) or set(values) != {'min', 'max'} or
                any(type(v) not in (int, float) or not float('-inf') < v < float('inf') for v in values.values()) or
                values['min'] > values['max'] or (name == 'lambda_ci' and values['min'] < 0)):
            raise ValueError('Frame field-range metadata is invalid.')
    if binary_vtu_fingerprint(paths['vtu'], expected=original_stats[str(paths['vtu'])]) != report.get('output_sha256', {}).get('vtu'):
        raise ValueError('VTU checksum differs from the completed frame.')
    if deep:
        required = 4*max(header['estimated_output_bytes'], paths['vtu'].stat().st_size)+1024**3
        if single.available_memory_bytes() <= required:
            raise MemoryError('Insufficient RAM for one saved VTU validation.')
        validate_saved_grid(paths['vtu'], report, expected=original_stats[str(paths['vtu'])])
    metadata_hash = regular_file_hash(paths['metadata'], expected=original_stats[str(paths['metadata'])])
    if any(_stat(Path(path)) != before for path, before in original_stats.items()):
        raise ValueError('Frame artifact changed during checksum/payload validation.')
    return dict(index=header['frame_index'], time=header['time'], source_sha256=source_hash,
                artifact_directory=str(directory),
                output_paths=header['output_paths'], native_mesh_signature=mesh,
                mesh_array_sha256={k: arrays[k] for k in ('points', 'connectivity', 'offsets', 'cell_types')},
                metadata_sha256=metadata_hash, vtu_sha256=report['output_sha256']['vtu'],
                file_stats=original_stats,
                payload_validation='passed' if deep else 'pending', validation='passed' if deep else 'metadata_and_checksums_passed')


def _pvd_entries(path, records):
    """Reuse the validated XML parser, then enforce this workflow's exact manifest."""
    try:
        tree, collection = _collection(path)
    except ET.ParseError as exc:
        raise ValueError('Malformed PVD XML.') from exc
    if len(tree) != 1 or len(tree.findall('Collection')) != 1:
        raise ValueError('Malformed PVD collection structure.')
    by_path = {Path(r['output_paths']['vtu']).resolve(): r for r in records.values()}
    previous = None
    refs = set()
    for entry in collection:
        target = (path.parent/entry.get('file', '')).resolve()
        value = float(entry.get('timestep', 'nan'))
        if (target not in by_path or target in refs or value != by_path[target]['time'] or
                (previous is not None and value <= previous) or entry.get('part', '0') != '0' or
                entry.get('group', '') != ''):
            raise ValueError('PVD contains invalid, duplicate or unsorted frame references/timestamps.')
        refs.add(target)
        previous = value
    return len(collection)


def _publish_pvd(path, records):
    """Case lock is held. Preserve every old entry; publish only validated frames."""
    if any(r['validation'] != 'passed' for r in records.values()):
        raise ValueError('PVD publication requires validated completed frames.')
    _unchanged_artifacts(records)
    original = path.read_bytes() if path.exists() else None
    _pvd_entries(path, records)
    tree = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(tree, 'Collection')
    for record in sorted(records.values(), key=lambda r: r['time']):
        ET.SubElement(collection, 'DataSet', timestep=repr(record['time']), group='', part='0',
                      file=Path(os.path.relpath(record['output_paths']['vtu'], path.parent)).as_posix())
    payload = ET.tostring(tree, encoding='utf-8', xml_declaration=True)
    if original == payload:
        return len(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if _pvd_entries(temporary, records) != len(records):
        raise ValueError('Staged PVD validation failed.')
    if path.is_symlink() or (path.read_bytes() if path.exists() else None) != original:
        raise RuntimeError('PVD changed concurrently; staging retained.')
    _unchanged_artifacts(records)
    if original is None:
        _publish_directory(temporary, path)  # renameat2 NOREPLACE also supports files.
    else:
        os.replace(temporary, path)
    return len(records)


def _coherent(records):
    if records:
        first = next(iter(records.values()))
        for record in records.values():
            if (record['native_mesh_signature'] != first['native_mesh_signature'] or
                    record['mesh_array_sha256'] != first['mesh_array_sha256']):
                raise ValueError('Native geometry/element ordering or output mesh changed within the series.')


def _control_files(root):
    paths = dict(pvd=root/'series'/f'{root.name}_vortex_3d_series.pvd',
                 manifest=root/'series'/'manifest.json', summary=root/'series'/'processing_summary.json')
    for path in paths.values():
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f'Unsafe series control file: {path}')
    directory = root/'series'
    if directory.exists():
        for path in directory.iterdir():
            temporary = (path.name.startswith('.'+paths['pvd'].name+'.') or
                         path.name.startswith('manifest.json.') or path.name.startswith('processing_summary.json.'))
            if path.is_symlink() or not path.is_file() or (path not in paths.values() and not (temporary and path.suffix == '.tmp')):
                raise ValueError(f'Foreign/unsafe artifact in series control directory: {path}')
    return paths


def _preflight(case, parent, root, selection, config, *, resume, recover_staging, deep,
               output_selection=None, output_plan=None):
    policy = output_selection or VolumeSelection('full' if config.crop is None else 'crop', config.crop)
    controls = _control_files(root)
    configuration = _configuration(config)
    signature = _signature()
    failures = []
    inventory = {}
    if root.exists():
        for path in root.glob('f*'):
            if not re.fullmatch(r'f\d{5}', path.name) or path.is_symlink() or not path.is_dir():
                raise ValueError(f'Unexpected/unsafe frame location: {path}')
            inventory[int(path.name[1:])] = path
    if _control_allowance(len(set(selection) | set(inventory)), len(selection))[0] > MAX_METADATA_BYTES:
        raise ValueError('Series inventory exceeds the bounded control-metadata budget.')
    manifest = None
    if controls['summary'].exists():
        summary = _json(controls['summary'])
        if (summary.get('workflow') != 'vortex_3d_series' or summary.get('schema_version') != SCHEMA or
                summary.get('configuration') != configuration or summary.get('source_case') != str(case) or
                summary.get('output_selection') != policy.metadata() or
                summary.get('series_signature') != signature):
            raise ValueError('Incompatible/foreign series checkpoint; refusing replacement.')
    if controls['manifest'].exists():
        manifest = _json(controls['manifest'])
        if (manifest.get('workflow') != 'vortex_3d_series' or manifest.get('schema_version') != SCHEMA or
                manifest.get('configuration') != configuration or manifest.get('source_case') != str(case) or
                manifest.get('output_selection') != policy.metadata() or
                manifest.get('series_signature') != signature or not isinstance(manifest.get('frames'), dict)):
            raise ValueError('Incompatible series manifest/configuration/signature.')
        if any(not re.fullmatch(r'[1-9]\d{0,4}', key) or int(key) not in inventory for key in manifest['frames']):
            raise ValueError('Series manifest references missing/invalid frames.')
    headers, input_errors = {}, {}
    for index in sorted(set(selection) | set(inventory)):
        try:
            source = case/f'GC0.f{index:05d}'
            before = _stat(source)
            header = single._inspect_vortex_frame(case, index, parent, config, allow_existing=True)
            header['output_selection'] = policy.metadata()
            if policy.bounds is not None:
                if output_plan is None and not failures:
                    output_plan = single.inspect_volume_output_plan(header, config, policy)
                if output_plan is not None:
                    apply_volume_output_plan(header, output_plan, config, policy)
            if _stat(source) != before:
                raise ValueError('Source changed while inspecting its header/map.')
            header['source_stat'] = before
            headers[index] = header
        except (OSError, ValueError, MemoryError) as exc:
            failures.append(f'Frame {index}: {type(exc).__name__}: {exc}')
            present = source.is_file() and not source.is_symlink()
            input_errors[index] = dict(index=index, status='incompatible_input' if present else 'missing_input',
                                       action='stop', input_available=present, error=str(exc))
    times = [headers[i]['time'] for i in sorted(headers)]
    if any(b <= a for a, b in zip(times, times[1:])):
        failures.append('Physical timestamps must be unique and increasing in frame-index order.')
    states, records = [], {}
    required_ram = 0
    pending_disk = []
    stages_by_index = {}
    for stage in root.glob('.vortex-3d-f*'):
        match = re.fullmatch(r'\.vortex-3d-f(\d{5})-.+', stage.name)
        if not match or stage.is_symlink() or not stage.is_dir():
            raise ValueError(f'Unexpected/unsafe staging location: {stage}')
        stages_by_index.setdefault(int(match[1]), []).append(stage)
    # First classify using bounded metadata/checksum reads, never a VTK decoder.
    for index, header in headers.items():
        source = Path(header['source_file'])
        # Streaming hashes are allowed in dry-run; native/VTU payloads are not decoded.
        header['source_sha256'] = single._file_hash(source)
        if _stat(source) != header['source_stat']:
            failures.append(f'Frame {index}: source changed during preflight hashing.')
        state = dict(index=index, time=header['time'], source_file=header['source_file'],
                     input_available=True, output_paths=header['output_paths'],
                     status='missing', action='generate', payload_validation='pending',
                     estimated_ram_bytes=header['estimated_ram_bytes'],
                     estimated_output_bytes=header['estimated_output_bytes'],
                     full_local_point_count=header['full_local_point_count'],
                     full_linear_hex_count=header['full_linear_hex_count'],
                     selected_local_point_count=header.get('selected_local_point_count', header['full_local_point_count']),
                     selected_linear_hex_count=header.get('selected_linear_hex_count', header['full_linear_hex_count']),
                     staging=[str(p) for p in stages_by_index.get(index, ())])
        try:
            if index in inventory:
                if index in selection and not resume:
                    raise FileExistsError('Requested frame exists; --resume is required for strict validation.')
                record = _frame_record(inventory[index], header, config, deep=False,
                                       source_hash=header['source_sha256'])
                if manifest is not None and str(index) in manifest['frames']:
                    stored = manifest['frames'][str(index)]
                    if stored != {k: record[k] for k in ('index', 'time', 'source_sha256', 'metadata_sha256', 'vtu_sha256')}:
                        raise ValueError('Frame differs from the committed series checkpoint checksums.')
                records[index] = record
                state.update(status='metadata_valid', action='validate_then_skip',
                             payload_validation=record['payload_validation'])
            elif state['staging']:
                if not recover_staging:
                    raise ValueError('Unpublished staging exists; use --resume --recover-staging to validate or preserve/recompute.')
                candidates = []
                for stage in stages_by_index[index]:
                    meta = stage/Path(header['output_paths']['metadata']).name
                    if not meta.is_file() or meta.is_symlink():
                        continue  # Incomplete artifacts are preserved, never deleted.
                    try:
                        staged_metadata = _json(meta)
                    except ValueError:
                        continue
                    if staged_metadata.get('completion_status') != 'completed':
                        continue
                    record = _frame_record(stage, header, config, deep=False,
                                           source_hash=header['source_sha256'])
                    candidates.append((stage, record))
                if len(candidates) > 1:
                    raise ValueError('Multiple completed staging candidates require manual assessment.')
                if candidates:
                    stage, record = candidates[0]
                    state.update(status='staging_metadata_valid', action='validate_then_recover', stage=str(stage))
                    records[index] = record
            if index in selection:
                states.append(state)
        except (ValueError, OSError, KeyError, TypeError, MemoryError) as exc:
            state.update(status='incompatible', action='stop', error=str(exc))
            if index in selection:
                states.append(state)
            failures.append(f'Frame {index}: {type(exc).__name__}: {exc}')
    for index in selection:
        if index not in headers:
            states.append(input_errors[index])
    try:
        _coherent(records)
        if manifest is not None and records and manifest.get('native_mesh_signature') != next(iter(records.values()))['native_mesh_signature']:
            raise ValueError('Series manifest native geometry signature differs from its frames.')
        # Staged frames are never valid existing PVD targets.
        published = {i: r for i, r in records.items() if i in inventory}
        existing_entries = _pvd_entries(controls['pvd'], published)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        failures.append(f'PVD/mesh: {exc}')
        existing_entries = None
    # Only generated frames need new scientific disk space. Recovery renames
    # existing bytes, and validation retains just one decoded grid at a time.
    for index, header in headers.items():
        if index in records:
            actual = Path(records[index]['artifact_directory'])/Path(header['output_paths']['vtu']).name
            ram = 4*max(header['estimated_output_bytes'], actual.stat().st_size)+1024**3
        else:
            ram = header['estimated_ram_bytes']
            if index in selection and index not in inventory:
                pending_disk.append(header['required_disk_bytes'])
        required_ram = max(required_ram, ram)
    available = single.available_memory_bytes()
    if isinstance(available, bool) or not isinstance(available, Integral) or available < 0:
        raise MemoryError('Cannot establish available RAM for series preflight.')
    disk = shutil.disk_usage(single._nearest_directory(root)).free
    limits = _filesystem_limits(root)
    control_file, control_disk = _control_allowance(len(headers), len(selection))
    required_ram = max(required_ram, 4*control_file+1024**3)
    file_allowance = max([control_file, *pending_disk])
    required_disk = sum(pending_disk)+max(pending_disk, default=0)+control_disk
    if required_ram >= available:
        failures.append('Insufficient RAM for sequential calculation or saved VTU validation.')
    if required_disk >= disk:
        failures.append('Insufficient disk for pending frames plus staging/control allowance.')
    if limits['effective_file_size_limit_bytes'] is not None and file_allowance > limits['effective_file_size_limit_bytes']:
        failures.append('Estimated single-file allowance exceeds a known filesystem/process file-size limit.')
    if deep and not failures:
        for index, record in records.items():
            try:
                directory = Path(record['artifact_directory'])
                header = headers[index]
                actual = directory/Path(header['output_paths']['vtu']).name
                required = 4*max(header['estimated_output_bytes'], actual.stat().st_size)+1024**3
                if single.available_memory_bytes() <= required:
                    raise MemoryError('Insufficient RAM before saved VTU payload validation.')
                meta_path = directory/Path(header['output_paths']['metadata']).name
                validate_saved_grid(actual, _json(meta_path, expected=record['file_stats'][str(meta_path)]),
                                    expected=record['file_stats'][str(actual)])
                for name, original_stat in record['file_stats'].items():
                    if _stat(Path(name)) != original_stat:
                        raise RuntimeError('Artifact changed during payload validation.')
                record.update(validation='passed', payload_validation='passed')
                for state in states:
                    if state['index'] == index:
                        state.update(action='recover' if 'stage' in state else 'skip',
                                     status='recoverable' if 'stage' in state else 'valid', payload_validation='passed')
            except (ValueError, OSError, MemoryError, KeyError, RuntimeError, TypeError) as exc:
                failures.append(f'Frame {index} payload validation: {exc}')
    report = dict(status='preflight_passed' if not failures else 'invalid', case=case.name,
                  source_case=str(case), selected_indices=list(selection), configuration=configuration,
                  sampling_nodes=config.sampling_nodes, output_root=str(root),
                  output_selection=policy.metadata(), output_plan=output_plan,
                  frames=sorted(states, key=lambda f: f['index']), failures=failures,
                  estimated_ram_bytes=required_ram, available_ram_bytes=int(available),
                  pending_frame_disk_allowances=pending_disk, required_disk_bytes=required_disk,
                  control_disk_allowance_bytes=control_disk,
                  available_disk_bytes=disk, filesystem=limits,
                  maximum_single_file_allowance_bytes=file_allowance,
                  series_signature=signature, pvd_path=str(controls['pvd']),
                  existing_indices=sorted(inventory),
                  pvd_plan=dict(existing_entries=existing_entries,
                                action='retain validated entries and reconcile every completed frame'),
                  assumptions='Selected output allocations; full-domain native gradients/projection. Geometry plans assume '
                              'stationary meshes and are checked before differentiation. No measured compression/runtime or reserved capacity.')
    return report, headers, records, controls


def _require_plan(report):
    if report['failures']:
        raise ValueError('Series preflight failed: '+'; '.join(report['failures']))


def _unchanged(headers, records, *, hashes=False):
    for header in headers.values():
        source = Path(header['source_file'])
        if _stat(source) != header['source_stat'] or (hashes and single._file_hash(source) != header['source_sha256']):
            raise RuntimeError('Source snapshot changed during series execution.')
    _unchanged_artifacts(records, hashes=hashes)


def _unchanged_artifacts(records, *, hashes=False):
    for record in records.values():
        directory = Path(record['artifact_directory'])
        if directory.is_symlink() or set(directory.iterdir()) != {Path(p) for p in record['file_stats']}:
            raise RuntimeError('Validated frame/staging artifact set changed during series execution.')
        for name, path in record['output_paths'].items():
            path = Path(path)
            identity = record['file_stats'][str(path)]
            _require_identity(path, identity, 'Validated output changed during series execution.')
            expected = record['vtu_sha256'] if name == 'vtu' else record['metadata_sha256']
            if hashes and regular_file_hash(path, expected=identity) != expected:
                raise RuntimeError('Validated output checksum changed during series execution.')


def run_vortex_series(case='N7_H', output_dir=None, *, indices=None, start_index=None,
                      end_index=None, max_frames=None, sampling_nodes=None, crop=None,
                      periodic_axes=('x', 'y'), expected_element_counts=(272, 12, 8),
                      chunk_size=256, dry_run=False, resume=False, recover_staging=False,
                      progress=None, positive_x=False, full_domain=False):
    """Preflight read-only, lock/revalidate, then process one native frame at a time."""
    if recover_staging and not resume:
        raise ValueError('--recover-staging requires --resume.')
    selection = frame_indices(indices, start_index=start_index, end_index=end_index, max_frames=max_frames)
    config = single._configuration(periodic_axes, expected_element_counts, crop, chunk_size, sampling_nodes)
    config, policy = series_volume_selection(single._case_directory(case), config,
                                             positive_x=positive_x, full_domain=full_domain)
    case, parent, root = _locations(case, output_dir, config, selection[0])
    report, headers, _, _ = _preflight(case, parent, root, selection, config,
                                     resume=resume, recover_staging=recover_staging, deep=False,
                                     output_selection=policy)
    report['dry_run'] = dry_run
    if dry_run:
        return report
    _require_plan(report)
    initial = {i: (h['source_stat'], h['source_sha256']) for i, h in headers.items()}
    initial_signature = report['series_signature']
    prepared_output = report['output_plan']
    with _pvd_update_lock(root/'vortex-3d-series'):
        # Never execute stale actions after waiting for another cooperating writer.
        _locations(case, parent, config, selection[0])
        report, headers, records, controls = _preflight(case, parent, root, selection, config,
            resume=resume, recover_staging=recover_staging, deep=True,
            output_selection=policy, output_plan=prepared_output)
        _require_plan(report)
        if report['series_signature'] != initial_signature or any(
                initial[i] != (h['source_stat'], h['source_sha256']) for i, h in headers.items() if i in initial):
            raise RuntimeError('Source or implementation changed while waiting for the series lock.')
        root.joinpath('logs').mkdir(exist_ok=True)
        run_id = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12]
        log_path = root/'logs'/f'{run_id}.jsonl'
        summary_path = root/'logs'/f'{run_id}_summary.json'
        if summary_path.exists() or summary_path.is_symlink():
            raise FileExistsError('Invocation summary already exists.')
        result = dict(workflow='vortex_3d_series', schema_version=SCHEMA, source_case=str(case),
                      status='running', selected_indices=list(selection), configuration=_configuration(config),
                      output_selection=policy.metadata(),
                      frames=[], preflight=report, log_path=str(log_path), summary_path=str(summary_path))
        started = time.perf_counter()
        # Only final directories count as published. Complete staging is separate.
        published = {i: r for i, r in records.items() if Path(headers[i]['output_directory']).exists()}

        def checkpoint():
            result['elapsed_seconds'] = time.perf_counter()-started
            result['series_signature'] = initial_signature
            _atomic_json(summary_path, result)
            _atomic_json(controls['summary'], result)

        def commit_progress():
            _coherent(published)
            _unchanged(headers, published)
            if _signature() != initial_signature:
                raise RuntimeError('Implementation changed during series execution.')
            result['pvd_entries'] = _publish_pvd(controls['pvd'], published)
            _atomic_json(controls['manifest'], dict(workflow='vortex_3d_series', schema_version=SCHEMA,
                source_case=str(case), configuration=_configuration(config), series_signature=initial_signature,
                output_selection=policy.metadata(),
                native_mesh_signature=next(iter(published.values()))['native_mesh_signature'] if published else None,
                frames={str(i): {k: r[k] for k in ('index', 'time', 'source_sha256', 'metadata_sha256', 'vtu_sha256')}
                        for i, r in sorted(published.items())}))
            checkpoint()

        with log_path.open('x', buffering=1) as log:
            def event(message, **values):
                item = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message=message, **values)
                log.write(json.dumps(item, allow_nan=False)+'\n')
                log.flush()
                if progress:
                    progress(json.dumps(item, allow_nan=False))
            try:
                checkpoint()
                for position, state in enumerate(report['frames']):
                    index = state['index']
                    header = headers[index]
                    _unchanged(headers, published)
                    remaining = [headers[s['index']]['required_disk_bytes'] for s in report['frames'][position:]
                                 if s['action'] == 'generate']
                    required_disk = sum(remaining)+max(remaining, default=0)+report['control_disk_allowance_bytes']
                    if shutil.disk_usage(single._nearest_directory(root)).free <= required_disk:
                        raise OSError('Insufficient disk for the remaining frames and staging/control allowance.')
                    event('Frame started', index=index, time=header['time'], action=state['action'])
                    if state['action'] == 'skip':
                        record = published[index]
                        action = 'reused'
                    elif state['action'] == 'recover':
                        _unchanged(headers, published)
                        stage = Path(state['stage'])
                        if stage.is_symlink() or not stage.is_dir() or set(stage.iterdir()) != {Path(p) for p in records[index]['file_stats']}:
                            raise RuntimeError('Validated staging changed before recovery publication.')
                        for path, expected_stat in records[index]['file_stats'].items():
                            _require_identity(Path(path), expected_stat,
                                              'Validated staging changed before recovery publication.')
                        _publish_directory(Path(state['stage']), Path(header['output_directory']))
                        record = _frame_record(Path(header['output_directory']), header, config,
                                               deep=False, source_hash=header['source_sha256'])
                        record.update(validation='passed', payload_validation='passed')
                        action = 'recovered'
                    else:
                        # One producer call owns every frame-local native/VTK allocation.
                        expected = next(iter(records.values()))['native_mesh_signature'] if records else None
                        if prepared_output is not None:
                            expected = prepared_output['native_mesh_signature']
                        single.run_single_frame_vortex(case, index, parent, **_options(config),
                            expected_mesh_signature=expected, expected_source_sha256=header['source_sha256'],
                            output_selection=policy, expected_output_plan=prepared_output,
                            progress=lambda message: event(message, index=index))
                        record = _frame_record(Path(header['output_directory']), header, config,
                                               deep=False, source_hash=header['source_sha256'])
                        # The producer just performed exact binary read-back; do not decode twice.
                        record.update(validation='passed', payload_validation='passed')
                        action = 'generated'
                    records[index] = published[index] = record
                    result['frames'].append(dict(index=index, time=header['time'], action=action, validation='passed'))
                    gc.collect()
                    commit_progress()
                    event('Frame completed', index=index, time=header['time'], action=action)
                _unchanged(headers, published, hashes=True)
                if _pvd_entries(controls['pvd'], published) != len(published):
                    raise ValueError('Final PVD is incomplete.')
                result.update(status='completed', published_frames=len(published), selected_frames=len(selection),
                              protected_existing_unchanged=True)
                event('Series completed', frames=len(published))
                _unchanged(headers, published)
                checkpoint()
            except BaseException as exc:
                result.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                              error=f'{type(exc).__name__}: {exc}')
                event('Series stopped safely', error=result['error'])
                checkpoint()
                raise
        return result
