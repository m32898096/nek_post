"""Validate saved omega_x volumes and publish a no-replace ParaView series.

No native payload reads, derivatives or projection are performed here.
Statistics are arithmetic over exported points, not GLL-weighted integrals.
"""
from pathlib import Path
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.io_nek import read_nek_header
from nek_post.l2_projection import structured_global_node_shape
from nek_post import omega_x_workflow as workflow


def _inventory(root):
    workflow._require_ram(workflow.RAM_SAFETY_MARGIN_BYTES + 16*1024**2,
                          'Production checksum inventory')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink is not a production input: {path}')
        if path.is_file() and path.suffix in ('.vtr', '.pvd', '.json'):
            files[str(path)] = workflow._file_hash(path)
    return files


def _unchanged(root, before):
    if _inventory(root) != before:
        raise RuntimeError('Existing production artifacts changed; validation/publication stopped.')


def _indices(values):
    result = tuple(range(1, 82)) if values is None else tuple(values)
    if not result or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in result):
        raise ValueError('Frame indices must be nonempty positive integers.')
    if len(set(result)) != len(result):
        raise ValueError('Duplicate requested frame indices.')
    return tuple(sorted(result))


def _collection(path):
    root = ET.parse(path).getroot()
    if (root.tag != 'VTKFile' or root.get('type') != 'Collection'
            or len(root) != 1 or root[0].tag != 'Collection'
            or any(child.tag != 'DataSet' for child in root[0])):
        raise ValueError(f'Invalid ParaView Collection XML: {path}')
    return root[0]


def validate_series_pvd(path, frames):
    """Require exact actual times and unique, existing relative VTR references."""
    path = Path(path)
    entries = list(_collection(path))
    if len(entries) != len(frames):
        raise ValueError('Series PVD entry count differs.')
    references = set(); times = []
    for entry, frame in zip(entries, frames):
        ref = Path(entry.get('file', ''))
        target = (path.parent/ref).resolve()
        time = float(entry.get('timestep', 'nan'))
        if (not entry.get('file') or ref.is_absolute() or not target.is_file()
                or target != Path(frame['output_paths']['vtr']).resolve()
                or target in references or not np.isfinite(time) or time != frame['time']
                or entry.get('part', '0') != '0' or entry.get('group', '') != ''):
            raise ValueError('Series PVD timestamp/reference differs or is duplicated.')
        references.add(target); times.append(time)
    if not np.all(np.diff(times) > 0):
        raise ValueError('Series PVD timestamps must increase strictly.')
    return len(entries)


def _diagnostics(path, *, shape, bounds, time, reference):
    """One VTK dataset at a time; retain only tiny coordinate vectors."""
    retained = sum(v.nbytes for v in reference) if reference is not None else 0
    workflow._require_ram(workflow._validation_ram_bytes(shape, path.stat().st_size,
                                                       retained_bytes=retained),
                          f'Series statistics/read-back: {path.name}')
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    reader = vtkXMLRectilinearGridReader(); reader.SetFileName(str(path)); reader.Update()
    grid = reader.GetOutput()
    if reader.GetErrorCode() or grid.GetDimensions() != tuple(shape):
        raise ValueError('VTR dimensions/read-back differ.')
    coordinates = tuple(vtk_to_numpy(getter()) for getter in
                        (grid.GetXCoordinates, grid.GetYCoordinates, grid.GetZCoordinates))
    for vector, expected in zip(coordinates, bounds):
        if (vector.dtype != np.float64 or not np.isfinite(vector).all()
                or not np.all(np.diff(vector) > 0) or [float(vector[0]), float(vector[-1])] != list(expected)):
            raise ValueError('VTR coordinate bounds/type/order differ.')
    if reference is not None and any(not np.array_equal(a, b) for a, b in zip(coordinates, reference)):
        raise ValueError('Coordinate grid differs across frames.')
    timestamp = grid.GetFieldData().GetArray('TIME_VALUE')
    scalar = grid.GetPointData().GetArray('omega_x')
    if (timestamp is None or timestamp.GetNumberOfTuples() != 1 or timestamp.GetValue(0) != time
            or scalar is None or scalar.GetNumberOfComponents() != 1
            or grid.GetPointData().GetNumberOfArrays() != 1):
        raise ValueError('VTR timestamp/scalar field differs.')
    values = vtk_to_numpy(scalar)
    if values.dtype != np.float64 or values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError('Invalid omega_x scalar values/type.')
    # VTK rectilinear storage defines interior x-fast indexing. Verify its
    # zero-based extent and each row's first, adjacent and last point IDs.
    nx, ny, nz = shape
    if grid.GetExtent() != (0,nx-1,0,ny-1,0,nz-1):
        raise ValueError('VTR x-fast extent differs.')
    for z in range(nz):
        for y in range(ny):
            for x in (0,1,nx-1):
                point = (z*ny+y)*nx+x
                xyz = (float(coordinates[0][x]),float(coordinates[1][y]),float(coordinates[2][z]))
                if grid.ComputePointId((x,y,z)) != point or grid.GetPoint(point) != xyz:
                    raise ValueError('VTR x-fast point ordering differs.')
    low, high = float(values.min()), float(values.max())
    scale = max(abs(low), abs(high))
    total = square = 0.0
    if scale:
        for start in range(0, values.size, 65536):
            normalized = values[start:start + 65536]/scale
            total += float(normalized.sum()); square += float(np.sum(normalized*normalized))
    statistics = {'min':low, 'max':high, 'mean':scale*(total/values.size),
                  'rms':scale*float(np.sqrt(square/values.size))}
    if not all(np.isfinite(v) for v in statistics.values()):
        raise ValueError('Nonfinite omega_x statistics.')
    return statistics, tuple(v.copy() for v in coordinates)


def validate_omega_x_production(case='N7_H', input_dir=None, *, indices=None,
                                expected_element_counts=(272,12,8), expected_orders=(8,8,8),
                                expected_bounds=((-17.,17.),(0.,1.5),(0.,1.)), progress=None):
    """Read-only complete audit; report every missing/invalid requested frame."""
    source_root = workflow.resolve_case(case)
    root = (workflow.DEFAULT_OUTPUT_ROOT/source_root.name if input_dir is None else Path(input_dir)).resolve()
    selection = _indices(indices)
    if not source_root.is_dir() or not root.is_dir():
        raise FileNotFoundError('Source and production input directories must exist.')
    before = _inventory(root)  # Baseline precedes all full VTR reads.
    config = {'periodic_axes':['x','y'], 'expected_element_counts':list(expected_element_counts),
              'expected_orders':list(expected_orders)}
    shape = structured_global_node_shape(expected_element_counts, expected_orders, periodic_axes=())
    directories = {}
    for path in root.rglob('*'):
        match = re.fullmatch(r'f(\d+)', path.name)
        if path.is_dir() and match:
            directories.setdefault(int(match[1]), []).append(path)
    errors = []; frames = []; reference = None
    missing = [i for i in selection if i not in directories]
    duplicates = [i for i, paths in directories.items() if len(paths) != 1]
    if missing: errors.append(f'Missing frame indices: {missing}')
    if duplicates: errors.append(f'Duplicate frame indices: {duplicates}')
    expected_files = set()
    for index in selection:
        for directory in directories.get(index, []):
            expected_files.update(str(p) for p in workflow._paths(directory, source_root.name, index).values())
    known_series = root/f'{source_root.name}_omega_x_series.pvd'
    unexpected = sorted(set(before) - expected_files - {str(known_series)})
    if unexpected: errors.append(f'Unexpected scientific output files: {unexpected}')
    sources = list(source_root.glob('GC0.f*'))
    source_indices = [int(p.name[5:]) for p in sources if re.fullmatch(r'GC0\.f\d{5}', p.name)]
    if indices is None and (sorted(source_indices) != list(selection) or len(sources) != len(selection)):
        errors.append('Source inventory must contain exactly frames 1 through 81.')
    times = []; source_stats = {}
    for index in selection:
        try:
            source = source_root/f'GC0.f{index:05d}'
            header = read_nek_header(source)
            if (source.is_symlink() or header.nb_dims != 3 or tuple(header.nb_vars[:2]) != (3,3)
                    or header.nb_files != 1 or header.nb_elems_file != header.nb_elems
                    or header.nb_elems != int(np.prod(expected_element_counts))
                    or tuple(header.orders) != tuple(expected_orders) or header.wdsz not in (4,8)
                    or not np.isfinite(header.time)):
                raise ValueError('Incompatible original snapshot header.')
            times.append((index, float(header.time)))
            if index not in directories or index in duplicates: continue
            directory = directories[index][0]
            if directory.name != f'f{index:05d}': raise ValueError('Incorrect frame directory naming.')
            paths = workflow._paths(directory, source_root.name, index)
            if set(directory.iterdir()) != set(paths.values()):
                raise ValueError('Missing metadata/VTR/PVD or unexpected frame files.')
            metadata = json.loads(paths['metadata'].read_text())
            if not isinstance(metadata, dict): raise ValueError('Metadata must be a JSON object.')
            checks = metadata.get('output_sha256', {})
            if (not re.fullmatch(r'[0-9a-f]{64}', str(metadata.get('source_sha256', '')))
                    or set(checks) != {'vtr','pvd'}
                    or any(not re.fullmatch(r'[0-9a-f]{64}', str(v)) for v in checks.values())):
                raise ValueError('Missing or invalid required source/artifact checksum.')
            if (metadata.get('source_file') != str(source) or metadata.get('istep') != header.istep
                    or metadata.get('source_unchanged') is not True
                    or metadata.get('output_paths') != {k:str(p) for k,p in paths.items()}):
                raise ValueError('Metadata source association/output paths differ.')
            source_stats[str(source)] = (source.stat().st_size, source.stat().st_mtime_ns)
            workflow._require_ram(workflow._validation_ram_bytes(shape, paths['vtr'].stat().st_size),
                                  f'Source/artifact validation for frame {index}')
            source_hash = workflow._file_hash(source)
            _collection(paths['pvd'])
            validated = workflow.validate_omega_x_frame(directory, case=source_root.name, index=index,
                time_value=float(header.time), config=config, source_sha256=source_hash)
            visual = validated['visualization']
            if (visual.get('bounds_xyz') != [list(v) for v in expected_bounds]
                    or visual.get('native_gll') is not True or visual.get('periodic_endpoints_preserved') is not True
                    or visual.get('interpolation') != 'none'):
                raise ValueError('Metadata grid bounds/native-coordinate contract differs.')
            statistics, coordinates = _diagnostics(paths['vtr'], shape=shape, bounds=expected_bounds,
                time=float(header.time), reference=reference)
            if reference is None: reference = coordinates
            frames.append({'index':index,'time':float(header.time),'istep':int(header.istep),
                           'statistics':statistics,'output_paths':{k:str(p) for k,p in paths.items()},
                           'source_sha256':source_hash,'output_sha256':checks})
            if progress: progress(f'Validated frame {index}: time={header.time}')
        except (ValueError, OSError, KeyError, TypeError, RuntimeError, ET.ParseError) as exc:
            errors.append(f'Frame {index}: {type(exc).__name__}: {exc}')
    if not np.all(np.diff([t for _,t in times]) > 0): errors.append('Actual source timestamps are not strictly increasing.')
    for path, stat in source_stats.items():
        if (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != stat:
            raise RuntimeError(f'Original snapshot changed during validation: {path}')
    _unchanged(root, before)
    if known_series.exists() and not errors: validate_series_pvd(known_series, frames)
    return {'status':'invalid' if errors else 'validated','case_dir':str(source_root),'input_dir':str(root),
            'expected_frames':len(selection),'valid_frames':len(frames),'missing_frames':missing,
            'duplicate_frames':duplicates,'unexpected_files':unexpected,'errors':errors,'frames':frames,
            'artifact_sha256':before,'grid_shape_xyz':list(shape),'bounds_xyz':[list(v) for v in expected_bounds],
            'statistics_definition':'arithmetic mean and RMS over exported points; no GLL weighting'}


def build_omega_x_series_pvd(case='N7_H', input_dir=None, output_dir=None, *, dry_run=False, **validation):
    """Publish only after a complete successful audit, atomically without replacement."""
    source_root = workflow.resolve_case(case)
    root = (workflow.DEFAULT_OUTPUT_ROOT/source_root.name if input_dir is None else Path(input_dir)).resolve()
    destination = (root if output_dir is None else Path(output_dir)).resolve()
    target = destination/f'{source_root.name}_omega_x_series.pvd'
    repository = Path(__file__).resolve().parents[2]
    if any(destination.is_relative_to(p.resolve()) for p in (source_root, repository, workflow.LAMBDA_ROOT)):
        raise ValueError('Series output must be outside original snapshots, repository and lambda_ci results.')
    if destination != root and destination.is_relative_to(root):
        raise ValueError('Series output must be at the input root or outside its tree, not inside frame outputs.')
    if not dry_run and (target.exists() or target.is_symlink()):
        raise FileExistsError(f'Refusing to overwrite existing series PVD: {target}')
    report = validate_omega_x_production(case, root, **validation)
    report.update(series_pvd=str(target), dry_run=dry_run)
    if report['errors'] or dry_run: return report
    xml = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(xml, 'Collection')
    for frame in report['frames']:
        relative = Path(os.path.relpath(frame['output_paths']['vtr'], destination)).as_posix()
        ET.SubElement(collection, 'DataSet', timestep=repr(frame['time']), group='', part='0', file=relative)
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=destination)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            ET.ElementTree(xml).write(stream, encoding='utf-8', xml_declaration=True)
            stream.flush(); os.fsync(stream.fileno())
        validate_series_pvd(temporary, report['frames'])
        _unchanged(root, report['artifact_sha256'])
        os.link(temporary, target)  # Atomic no-replace creation, including races.
        directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    report['pvd_entries'] = validate_series_pvd(target, report['frames'])
    report['status'] = 'completed'
    return report
