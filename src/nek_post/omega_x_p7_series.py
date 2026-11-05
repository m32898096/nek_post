"""Sequential saved-scalar P7 slices with strict resume and complete-series publication.

Native velocity payloads, derivatives, curl, projection and eigenvalues are never
used. Subset runs produce validated frames, never a partial production PVD.
"""
from dataclasses import dataclass, field
from pathlib import Path
import gc
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from nek_post import omega_x_p7_slice as single
from nek_post import omega_x_workflow as workflow
from nek_post.io_nek import read_nek_header
from nek_post.lambda_ci_series import _atomic_json, _publish_directory, _image_readable, _pdf_readable
from nek_post.lambda_ci_slice_export import _pvd_update_lock
from nek_post.omega_x_series_pvd import _indices, _collection, validate_series_pvd


SCHEMA_VERSION = 2
FULL_INDICES = tuple(range(1, 82))
# The historical report has no interpolation-code signature. Only these exact
# immutable bytes may enter the legacy assessment; full array equality with the
# current reconstruction is additionally required during every invocation.
APPROVED_LEGACY_REPORTS = {
    ('case_N7_H', 79): '34d0d560cac1930a676892b88efe040e1768d0472ffcf9e8122c272441c07d41',
}
DISK_MARGIN_BYTES = 1024**3


@dataclass
class _Cache:
    plan: object = None
    builds: int = 0
    assessed_legacy: set = field(default_factory=set)

    def prepare(self, coordinates, config):
        args = dict(element_counts=tuple(config['element_counts_xyz']), y=config['plane_y'],
                    x_min=0., samples_per_element=config['samples_per_element'])
        if self.plan is None:
            self.plan = single.build_p7_slice_plan(coordinates, **args)
            self.builds += 1
        self.plan.validate(coordinates, **args)
        return self.plan


def frame_indices(indices=None, *, start_index=None, end_index=None):
    if indices is not None and (start_index is not None or end_index is not None):
        raise ValueError('Choose --indices or a start/end range, not both.')
    if indices is None:
        start = 1 if start_index is None else workflow._positive_integer(start_index, 'start_index')
        end = 81 if end_index is None else workflow._positive_integer(end_index, 'end_index')
        if start > end:
            raise ValueError('Frame range must increase.')
        indices = range(start, end + 1)
    elif isinstance(indices, str):
        expanded = []
        for part in indices.split(','):
            match = re.fullmatch(r'\s*(\d+)(?:-(\d+))?\s*', part)
            if not match:
                raise ValueError('Invalid frame index/range.')
            a = int(match[1]); b = int(match[2] or a)
            if a > b:
                raise ValueError('Frame range must increase.')
            expanded.extend(range(a, b + 1))
        indices = expanded
    selection = _indices(indices)
    if selection[-1] > 81:
        raise ValueError('Only frame indices 1 through 81 are supported.')
    return selection


def _signature():
    base = Path(__file__).parent
    names = ('gll.py', 'gll_directional_integration.py', 'lambda_ci_plotting.py',
             'lambda_ci_series.py', 'omega_x_p7_slice.py', 'omega_x_p7_series.py')
    return {name: workflow._file_hash(base / name) for name in names}


def _configuration(element_counts, bounds, y, samples_per_element):
    if len(element_counts) != 3 or any(type(n) is not int or n < 1 for n in element_counts):
        raise ValueError('Three positive element counts are required.')
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (3, 2) or not np.isfinite(b).all() or not np.all(b[:, 1] > b[:, 0]):
        raise ValueError('Three finite increasing physical bounds are required.')
    if not np.isfinite(y) or not b[1, 0] <= y <= b[1, 1] or b[0, 1] <= 0:
        raise ValueError('Invalid physical slice/crop.')
    count = workflow._positive_integer(samples_per_element, 'samples_per_element', 2)
    return dict(plane_y=float(y), x_condition='0 < x', polynomial_degree=7,
                samples_per_element=count, element_counts_xyz=list(element_counts),
                native_bounds_xyz=b.tolist(), native_orders=[8, 8, 8], smoothing=False, filtering=False)


def _paths(directory, case, index, config):
    plane = f"{config['plane_y']:g}".replace('.', 'p')
    stem = f"{case}_f{index:05d}_omega_x_y{plane}_xgt0_n{config['samples_per_element']}"
    paths = {'vtr': directory / (stem + '_p7.vtr'), 'metadata': directory / (stem + '_validation.json')}
    for kind in ('linear', 'p7', 'comparison'):
        for ext in ('png', 'pdf'):
            paths[kind + '_' + ext] = directory / (stem + '_' + kind + '.' + ext)
    return paths


def _check_frame_ancestry(root):
    """Protect frame manifests even through aliases or nonexistent descendants."""
    resolved = root.resolve()
    for directory in (resolved, *resolved.parents):
        if directory.is_dir() and any(
                re.fullmatch(r'.+_f\d{5}_(?:omega_x|lambda_ci).*\.(?:vtr|json|pvd|png|pdf)', p.name)
                for p in directory.iterdir()):
            raise ValueError(f'Output root is inside an existing scientific frame: {directory}')


def _locations(case, input_dir, output_dir):
    case_dir = workflow.resolve_case(case)
    inputs = (workflow.DEFAULT_OUTPUT_ROOT / case_dir.name if input_dir is None else Path(input_dir)).resolve()
    parent = (single.DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)).resolve()
    root = parent / case_dir.name
    _check_frame_ancestry(root)
    protected = (case_dir, inputs, Path(__file__).resolve().parents[2],
                 workflow.DEFAULT_OUTPUT_ROOT, workflow.LAMBDA_ROOT)
    if any(root.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Outputs must be outside native production, snapshots, repository and lambda_ci.')
    if root.is_symlink() or not case_dir.is_dir() or not inputs.is_dir():
        raise ValueError('Source/input directories must exist and output case root must not be a symlink.')
    for p in (root, root / '.omega-x-p7-staging', root / 'logs'):
        if p.is_symlink() or (p.exists() and not p.is_dir()):
            raise ValueError('Unsafe output directory.')
    return case_dir, inputs, root


def _discover(inputs, case, selection):
    found = {}
    for p in inputs.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'Symlink in native input tree: {p}')
        match = re.fullmatch(re.escape(case) + r'_f(\d{5})_omega_x\.json', p.name)
        if match:
            found.setdefault(int(match[1]), []).append(p)
    duplicates = [i for i, paths in found.items() if len(paths) != 1]
    missing = [i for i in selection if i not in found]
    if duplicates or missing or any(i not in FULL_INDICES for i in found):
        raise ValueError(f'Missing native frames: {missing}; duplicate native frames: {duplicates}')
    result = {}
    for i in selection:
        folder = found[i][0].parent
        paths = workflow._paths(folder, case, i)
        if folder.name != f'f{i:05d}' or set(folder.iterdir()) != set(paths.values()):
            raise ValueError(f'Incorrect native frame numbering/triplet: {i}')
        result[i] = paths
    return result


def _resources(root, frames, config):
    counts = config['element_counts_xyz']; n = config['samples_per_element']
    shape = tuple(7*c + 1 for c in counts)
    upper_points = (counts[0]*(n-1) + 1)*(counts[2]*(n-1) + 1)
    # Retain one native volume, one plan, output/read-back/figure buffers only.
    ram = max(workflow._validation_ram_bytes(shape, p['vtr'].stat().st_size)
              for p in frames.values()) + 64*8*upper_points
    workflow._require_ram(ram, 'Omega_x P7 slice-series preflight')
    # Conservative uncompressed three-field VTR + figure/metadata allowance.
    per_frame = 3*8*upper_points + 16*1024**2
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    return ram, per_frame, free


def _require_disk(root, required):
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free <= required + DISK_MARGIN_BYTES:
        raise OSError('Insufficient disk space for Omega_x P7 outputs and safety margin.')


def _load(case_dir, index, paths, config, ram, cache):
    _collection(paths['pvd'])
    header, metadata, coordinates, values, source_hash, hashes = single.load_projected_omega_frame(
        case_dir, index, paths, expected_element_counts=tuple(config['element_counts_xyz']),
        expected_bounds=config['native_bounds_xyz'], required_ram=ram)
    plan = cache.prepare(coordinates, config)
    item = dict(index=index, time=float(header.time), source_sha256=source_hash,
                source_artifact_sha256=hashes, source_numerical_signature=metadata['numerical_software_signature'],
                source_file=str(case_dir / f'GC0.f{index:05d}'))
    return item, coordinates, values, plan


def _sample(coordinates, values, item, config, plan):
    return single.reconstruct_p7_midplane(coordinates, values, element_counts=tuple(config['element_counts_xyz']),
        y=config['plane_y'], samples_per_element=config['samples_per_element'], time=item['time'], plan=plan)


def _read_slice(path, plan, time_value, ram):
    workflow._require_ram(ram, 'Reading existing P7 slice')
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    reader = vtkXMLRectilinearGridReader(); reader.SetFileName(str(path)); reader.Update(); grid = reader.GetOutput()
    x = plan.x[plan.x > 0]; z = plan.z
    if reader.GetErrorCode() or grid.GetDimensions() != (len(x), 1, len(z)):
        raise ValueError('Saved P7 VTR dimensions differ.')
    coordinates = tuple(vtk_to_numpy(f()) for f in (grid.GetXCoordinates, grid.GetYCoordinates, grid.GetZCoordinates))
    for a, b in zip(coordinates, (x, np.array([plan.y]), z)):
        if a.dtype != np.float64 or not np.array_equal(a, b):
            raise ValueError('Saved P7 physical coordinates differ.')
    expected = {'omega_x', 'omega_x_linear', 'omega_x_p7_minus_linear'}
    if ({grid.GetPointData().GetArrayName(i) for i in range(grid.GetPointData().GetNumberOfArrays())} != expected
            or grid.GetPointData().GetScalars() is None or grid.GetPointData().GetScalars().GetName() != 'omega_x'):
        raise ValueError('Saved P7 PointData arrays differ.')
    arrays = {}
    for name in expected:
        a = grid.GetPointData().GetArray(name); v = vtk_to_numpy(a)
        if a.GetNumberOfComponents() != 1 or v.dtype != np.float64 or v.size != len(x)*len(z) or not np.isfinite(v).all():
            raise ValueError('Saved P7 scalar type/values differ.')
        arrays[name] = v.copy().reshape(len(z), len(x))
    if not np.array_equal(arrays['omega_x']-arrays['omega_x_linear'], arrays['omega_x_p7_minus_linear']):
        raise ValueError('Saved P7 difference array differs.')
    return single.OmegaSlice(x, z, plan.y, time_value, arrays['omega_x'], arrays['omega_x_linear'], {})


def _validate_output(directory, final, case, item, config, cache, ram, *, legacy_samples=None):
    paths = _paths(directory, case, item['index'], config)
    if (directory.is_symlink() or set(directory.iterdir()) != set(paths.values())
            or any(not p.is_file() or p.is_symlink() or not p.stat().st_size for p in paths.values())):
        raise ValueError('Incomplete/unrecognized P7 frame; existing outputs are preserved.')
    digest = workflow._file_hash(paths['metadata'])
    report = json.loads(paths['metadata'].read_text())
    if not isinstance(report, dict):
        raise ValueError('P7 metadata must be a JSON object.')
    signature = _signature()
    legacy = 'schema_version' not in report
    token = (digest, tuple(sorted(signature.items())))
    if legacy:
        if APPROVED_LEGACY_REPORTS.get((case, item['index'])) != digest:
            raise ValueError('Unapproved legacy P7 metadata/signature format.')
        if token not in cache.assessed_legacy and legacy_samples is None:
            raise ValueError('Legacy P7 output requires an explicit numerical compatibility assessment.')
    elif (report.get('schema_version') != SCHEMA_VERSION or report.get('configuration') != config
          or report.get('interpolation_signature') != signature):
        raise ValueError('P7 configuration/implementation signature differs.')
    if (report.get('status') != 'completed' or report.get('case') != case or report.get('frame') != item['index']
            or report.get('time') != item['time'] or report.get('plane_y') != config['plane_y']
            or report.get('source_snapshot_sha256') != item['source_sha256']
            or report.get('source_artifact_sha256') != item['source_artifact_sha256']
            or report.get('source_numerical_signature') != item['source_numerical_signature']
            or report.get('output_paths') != {k:str(p) for k,p in _paths(final, case, item['index'], config).items()}):
        raise ValueError('P7 source provenance/frame/time/output paths differ.')
    hashes = report.get('output_sha256', {})
    if set(hashes) != set(paths)-{'metadata'} or any(workflow._file_hash(paths[k]) != h for k,h in hashes.items()):
        raise ValueError('P7 output checksum differs.')
    d = report.get('diagnostics', {})
    if (d.get('polynomial_degree') != 7 or d.get('samples_per_element') != config['samples_per_element']
            or d.get('element_counts_xyz') != config['element_counts_xyz']
            or d.get('native_nodes_per_element') != 8 or d.get('native_stride') != 7
            or d.get('overshoot_clipped') is not False):
        raise ValueError('P7 interpolation diagnostics/configuration differ.')
    samples = _read_slice(paths['vtr'], cache.plan, item['time'], ram)
    if legacy_samples is not None:
        if not np.array_equal(samples.p7, legacy_samples.p7) or not np.array_equal(samples.linear, legacy_samples.linear):
            raise ValueError('Legacy P7 scalar arrays differ from current reconstruction.')
        cache.assessed_legacy.add(token)
    for name, a in (('p7', samples.p7), ('linear', samples.linear)):
        if d.get(name, {}).get('min') != float(a.min()) or d.get(name, {}).get('max') != float(a.max()):
            raise ValueError('P7 scalar extrema diagnostics differ.')
    single.validate_slice_vtr(paths['vtr'], samples)
    for key, path in paths.items():
        if key.endswith('_png'):
            _image_readable(path)
        elif key.endswith('_pdf'):
            _pdf_readable(path)
    return dict(index=item['index'], time=item['time'], output_paths={k:str(p) for k,p in _paths(final, case, item['index'], config).items()},
                validation='passed', legacy_compatibility='exact scalar/coordinate equality' if legacy else None)


def _stage_candidate(root, case, index, config):
    candidates = []
    for p in sorted((root / '.omega-x-p7-staging').glob(f'f{index:05d}.*')):
        if p.is_symlink() or not p.is_dir():
            raise ValueError('Unsafe P7 staging entry.')
        paths = _paths(p, case, index, config)
        if all(q.is_file() for q in paths.values()):
            candidates.append(p)
    if len(candidates) > 1:
        raise ValueError(f'Duplicate complete staged frame {index}; explicit review required.')
    return candidates[0] if candidates else None


def _assert_unchanged(hashes, snapshots):
    for path, digest in hashes.items():
        if Path(path).is_symlink() or workflow._file_hash(Path(path)) != digest:
            raise RuntimeError(f'Protected source/output checksum changed: {path}')
    for path, stat in snapshots.items():
        s = Path(path).stat()
        if (s.st_size, s.st_mtime_ns) != stat:
            raise RuntimeError(f'Original snapshot changed: {path}')


def _preflight(case_dir, inputs, root, selection, config, resume, recover_staging, progress, cache):
    signature = _signature()
    native = _discover(inputs, case_dir.name, selection)
    if selection == FULL_INDICES and sorted(p.name for p in case_dir.glob('GC0.f*')) != [f'GC0.f{i:05d}' for i in FULL_INDICES]:
        raise ValueError('Original source inventory must contain exactly frames 1 through 81.')
    if recover_staging and not resume:
        raise ValueError('Staging recovery requires --resume.')
    for i in selection:
        final = root / f'f{i:05d}'
        if final.is_symlink() or (final.exists() and not resume):
            raise FileExistsError(f'Existing P7 frame {i}; use --resume for validated reuse.')
    ram, per_frame, free = _resources(root, native, config)
    frames = []; protected = {}; snapshots = {}
    for i in selection:
        item, coordinates, values, plan = _load(case_dir, i, native[i], config, ram, cache)
        final = root / f'f{i:05d}'
        candidate = None if final.exists() or not recover_staging else _stage_candidate(root, case_dir.name, i, config)
        action = 'reused' if final.exists() else 'recovered' if candidate else 'generated'
        if action != 'generated':
            directory = candidate or final
            metadata_path = _paths(directory, case_dir.name, i, config)['metadata']
            existing = json.loads(metadata_path.read_text())
            # An unchanged approved legacy report needs one equality assessment
            # per invocation, not another interpolation during the lock refresh.
            token = (workflow._file_hash(metadata_path), tuple(sorted(signature.items())))
            expected = (_sample(coordinates, values, item, config, plan)
                        if 'schema_version' not in existing and token not in cache.assessed_legacy else None)
            checked = _validate_output(directory, final, case_dir.name, item, config, cache, ram, legacy_samples=expected)
            item['legacy_compatibility'] = checked['legacy_compatibility']
            protected.update({str(p):workflow._file_hash(p) for p in _paths(directory, case_dir.name, i, config).values()})
        item.update(action=action, input_paths={k:str(p) for k,p in native[i].items()}, staging=str(candidate) if candidate else None)
        frames.append(item); protected.update(item['source_artifact_sha256'])
        source = Path(item['source_file']); stat = source.stat(); snapshots[str(source)] = (stat.st_size, stat.st_mtime_ns)
        del values, coordinates
        if progress:
            progress(f'Preflight {len(frames)}/{len(selection)}: frame {i}, t={item["time"]}, {action}')
    if not np.all(np.diff([f['time'] for f in frames]) > 0):
        raise ValueError('Actual timestamps must increase strictly with frame numbering.')
    pending = sum(f['action'] == 'generated' for f in frames)
    _require_disk(root, pending*per_frame)
    pvd = root / f'{case_dir.name}_omega_x_p7_slice_series.pvd'
    if pvd.is_symlink():
        raise ValueError('Series PVD symlink is not allowed.')
    if pvd.exists():
        if selection != FULL_INDICES:
            raise ValueError('Existing complete series must be validated with all 81 frames.')
        validate_series_pvd(pvd, [dict(index=f['index'], time=f['time'], output_paths={k:str(p) for k,p in _paths(root/f"f{f['index']:05d}", case_dir.name, f['index'], config).items()}) for f in frames])
        protected[str(pvd)] = workflow._file_hash(pvd)
    _assert_unchanged(protected, snapshots)
    if _signature() != signature:
        raise RuntimeError('Implementation signature changed during preflight.')
    return dict(status='validated', frames=frames, configuration=config, interpolation_signature=signature,
                estimated_ram_bytes=ram,
                estimated_new_output_bytes=pending*per_frame, disk_safety_margin_bytes=DISK_MARGIN_BYTES,
                available_disk_bytes=free, plan_build_count=cache.builds,
                protected_sha256=protected, snapshot_stats=snapshots, series_pvd=str(pvd),
                requested_frames=len(selection), generate_frames=pending,
                reuse_frames=sum(f['action']=='reused' for f in frames),
                recover_frames=sum(f['action']=='recovered' for f in frames))


def _export(stage, final, case, item, config, cache, samples, ram):
    paths = _paths(stage, case, item['index'], config)
    single.write_omega_slice_vtr(paths['vtr'], samples)
    readback = single.validate_slice_vtr(paths['vtr'], samples)
    rendering = single._figures(samples, paths, case=case, frame=item['index'])
    report = dict(schema_version=SCHEMA_VERSION, status='completed', case=case, frame=item['index'],
                  time=item['time'], plane_y=config['plane_y'], configuration=config,
                  interpolation_signature=_signature(), source_artifact_sha256=item['source_artifact_sha256'],
                  source_snapshot_sha256=item['source_sha256'], source_numerical_signature=item['source_numerical_signature'],
                  estimated_ram_bytes=ram, diagnostics=samples.diagnostics, rendering=rendering,
                  vtr_readback=readback, output_paths={k:str(p) for k,p in _paths(final, case, item['index'], config).items()},
                  output_sha256={k:workflow._file_hash(p) for k,p in paths.items() if k!='metadata'})
    with paths['metadata'].open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    _validate_output(stage, final, case, item, config, cache, ram)


def _publish_pvd(path, frames):
    if tuple(f['index'] for f in frames) != FULL_INDICES or any(f.get('validation') != 'passed' for f in frames):
        raise ValueError('Complete PVD requires validated exact frames 1 through 81.')
    if path.is_symlink():
        raise FileExistsError('Refusing series PVD symlink.')
    if path.exists():
        return validate_series_pvd(path, frames)
    xml = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(xml, 'Collection')
    for f in frames:
        ET.SubElement(collection, 'DataSet', timestep=repr(f['time']), group='', part='0',
                      file=Path(os.path.relpath(f['output_paths']['vtr'], path.parent)).as_posix())
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            ET.ElementTree(xml).write(stream, encoding='utf-8', xml_declaration=True)
            stream.flush(); os.fsync(stream.fileno())
        validate_series_pvd(temporary, frames)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_series_pvd(path, frames)


def run_omega_x_p7_series(case='N7_H', input_dir=None, output_dir=None, *, indices=None,
                         start_index=None, end_index=None, resume=False, recover_staging=False,
                         dry_run=False, y=.75, samples_per_element=29,
                         expected_element_counts=(272,12,8),
                         expected_bounds=((-17.,17.),(0.,1.5),(0.,1.)), progress=None):
    """Validate all requested inputs first; publish a PVD only for the full 81."""
    started = time.perf_counter()
    selection = frame_indices(indices, start_index=start_index, end_index=end_index)
    config = _configuration(expected_element_counts, expected_bounds, y, samples_per_element)
    case_dir, inputs, root = _locations(case, input_dir, output_dir)
    cache = _Cache()
    report = _preflight(case_dir, inputs, root, selection, config, resume, recover_staging, progress, cache)
    report.update(case=case_dir.name, output_root=str(root), dry_run=dry_run)
    if dry_run:
        report['total_seconds'] = time.perf_counter()-started
        return report
    log_path = root / 'logs' / (uuid.uuid4().hex+'.json')
    with _pvd_update_lock(root/'omega_x_p7_slice_series'):
        # A cooperating writer can publish initially complete staging while we
        # wait. Preserve fixed sources/final outputs, then validate staging or
        # its new final location through a fresh authoritative preflight.
        movable = {str(p) for item in report['frames'] if item['action'] == 'recovered'
                   for p in _paths(Path(item['staging']), case_dir.name, item['index'], config).values()}
        _assert_unchanged({p:h for p,h in report['protected_sha256'].items() if p not in movable},
                          report['snapshot_stats'])
        signature = report['interpolation_signature']
        report = _preflight(case_dir, inputs, root, selection, config, resume, recover_staging, progress, cache)
        if report['interpolation_signature'] != signature:
            raise RuntimeError('Implementation signature changed while waiting for the writer lock.')
        report.update(case=case_dir.name, output_root=str(root), dry_run=False)
        result = dict(report, status='running', frames=[], log_path=str(log_path))
        _atomic_json(log_path, result)
        try:
            for item in report['frames']:
                if _signature() != signature:
                    raise RuntimeError('Implementation signature changed during execution.')
                tick = time.perf_counter(); i = item['index']; final = root/f'f{i:05d}'
                _require_disk(root, 0)
                if item['action'] == 'generated':
                    remaining = sum(f['action']=='generated' for f in report['frames'][len(result['frames']):])
                    per_frame = report['estimated_new_output_bytes']//report['generate_frames']
                    _require_disk(root, remaining*per_frame)
                    loaded, coordinates, values, plan = _load(case_dir, i,
                        {k:Path(p) for k,p in item['input_paths'].items()}, config, report['estimated_ram_bytes'], cache)
                    if any(loaded[k] != item[k] for k in ('time','source_sha256','source_artifact_sha256','source_numerical_signature')):
                        raise RuntimeError('Native input changed after preflight.')
                    samples = _sample(coordinates, values, item, config, plan)
                    del values, coordinates
                    staging = root/'.omega-x-p7-staging'; staging.mkdir(exist_ok=True)
                    stage = Path(tempfile.mkdtemp(prefix=f'f{i:05d}.', dir=staging))
                    _export(stage, final, case_dir.name, item, config, cache, samples, report['estimated_ram_bytes'])
                    del samples
                    _assert_unchanged(item['source_artifact_sha256'], report['snapshot_stats'])
                    _publish_directory(stage, final)
                elif item['action'] == 'recovered':
                    stage = Path(item['staging'])
                    _validate_output(stage, final, case_dir.name, item, config, cache, report['estimated_ram_bytes'])
                    _publish_directory(stage, final)
                    # The same protected staged bytes now live at their final paths.
                    for key, old in _paths(stage, case_dir.name, i, config).items():
                        digest = result['protected_sha256'].pop(str(old))
                        result['protected_sha256'][str(_paths(final, case_dir.name, i, config)[key])] = digest
                checked = _validate_output(final, final, case_dir.name, item, config, cache, report['estimated_ram_bytes'])
                checked.update(action=item['action'], seconds=time.perf_counter()-tick)
                result['frames'].append(checked)
                result['plan_build_count'] = cache.builds
                _atomic_json(log_path, result)
                gc.collect()
                if progress:
                    progress(f'Completed {len(result["frames"])}/{len(selection)}: frame {i}, t={item["time"]}, {item["action"]}')
            # Recheck every target and protected source before publishing any collection.
            for item in report['frames']:
                final = root/f"f{item['index']:05d}"
                _validate_output(final, final, case_dir.name, item, config, cache, report['estimated_ram_bytes'])
            _assert_unchanged(result['protected_sha256'], report['snapshot_stats'])
            if _signature() != signature:
                raise RuntimeError('Implementation signature changed before series publication.')
            result['pvd_entries'] = _publish_pvd(Path(report['series_pvd']), result['frames']) if selection == FULL_INDICES else 0
            result['status'] = 'completed' if selection == FULL_INDICES else 'completed_subset'
            result['protected_existing_unchanged'] = True
            result['total_seconds'] = time.perf_counter()-started
            _atomic_json(log_path, result)
        except BaseException as exc:
            result.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt,SystemExit)) else 'failed',
                          error=f'{type(exc).__name__}: {exc}', total_seconds=time.perf_counter()-started)
            _atomic_json(log_path, result)
            raise
    return result
