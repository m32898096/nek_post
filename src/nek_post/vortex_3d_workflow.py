"""Protected single-snapshot native-grid binary VTU workflow (no batch mode)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from math import prod
from pathlib import Path
import json
import os
import platform
import resource
import shutil
import tempfile
import time

import numpy as np

from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.l2_projection import _positive_integer, _periodic_axes, _integer_triple, build_structured_node_map
from nek_post.lambda_ci_series import _publish_directory
from nek_post.lambda_ci_workflow import _file_hash, _nearest_directory, available_memory_bytes
from nek_post.vortex_3d import (
    build_native_vortex_grid, compute_native_vortex_fields, compute_projected_vortex_fields,
    validate_crop, write_binary_vtu,
)
from nek_post.vortex_3d_reconstruction import (
    build_reconstructed_vortex_grid, sampling_metadata, sampling_nodes as _sampling_nodes,
)
from nek_post.vortex_3d_artifacts import (
    native_mesh_signature, grid_fingerprints, binary_vtu_fingerprint, regular_file_identity,
)
from nek_post.vortex_3d_selection import (
    VolumeSelection, build_volume_output_plan, apply_volume_output_plan, validate_volume_grid,
)


DEFAULT_OUTPUT_ROOT = Path('/data/Nek5000_data/results/vortex_3d_native')
DEFAULT_RECONSTRUCTION_ROOT = Path('/data/Nek5000_data/results/vortex_3d_reconstructed')
REPOSITORY = Path(__file__).resolve().parents[2]
SIGNATURE_FILES = (
    'fields.py', 'io_nek.py', 'gll.py', 'spectral_derivatives.py',
    'spectral_interpolation.py', 'gll_directional_integration.py', 'l2_projection.py',
    'swirling_strength.py', 'rectilinear_export.py', 'lambda_ci_workflow.py',
    'lambda_ci_series.py', 'hex_validation.py', 'vortex_3d.py', 'vortex_3d_workflow.py',
    'gll_refinement.py', 'vortex_3d_reconstruction.py',
    'vortex_3d_artifacts.py',
    'vortex_3d_selection.py',
)


@dataclass(frozen=True)
class _Configuration:
    periodic_axes: tuple[str, ...]
    expected_element_counts: tuple[int, int, int]
    crop: tuple[float, ...] | None
    chunk_size: int
    sampling_nodes: int | None


def _configuration(periodic_axes, expected_element_counts, crop, chunk_size, sampling_nodes):
    """Consume user iterables once; use this immutable object at every stage."""
    periodic = _periodic_axes(periodic_axes)
    counts = _integer_triple(expected_element_counts, 'expected_element_counts', 1)
    try:
        crop_values = None if crop is None else tuple(crop)
    except TypeError as exc:
        raise ValueError('crop must contain six finite real bounds.') from exc
    bounds = validate_crop(crop_values)
    return _Configuration(periodic, counts, None if bounds is None else tuple(bounds.ravel()),
                          _positive_integer(chunk_size, 'chunk_size'), _sampling_nodes(sampling_nodes))


def _validate_element_map(values, count):
    """Complete unsplit Nek maps are permutations of the one-based IDs 1..N."""
    values = np.asarray(values)
    if values.shape != (count,) or values.dtype.kind not in 'iu':
        raise ValueError('Element map must contain exactly one integer ID per element.')
    if np.any(values < 1) or np.any(values > count):
        raise ValueError('Element map IDs must be one-based and within 1..element_count.')
    if not np.array_equal(np.sort(values), np.arange(1, count + 1)):
        raise ValueError('Element map must be complete, without duplicate or missing IDs.')


def _read_element_map(source, header):
    """Inspect the original bytes before Pymech can apply its zero-map fallback.

    Pymech 2.0.1 read_header/readnek/writenek use a 132-byte header, a float32
    endian tag, then signed int32 global IDs. readnek places record ID into
    data.elem[ID-1]; it does not preserve file-record order in data.elem.
    """
    with Path(source).open('rb') as stream:
        stream.seek(132)
        tag = stream.read(4)
        if len(tag) != 4:
            raise ValueError('Element map endian tag is truncated.')
        endian = None
        for candidate in ('<', '>'):
            value = float(np.frombuffer(tag, dtype=candidate+'f4')[0])
            if np.isfinite(value) and int(value * 1e5) / 1e5 == 6.54321:
                endian = candidate
                break
        if endian is None:
            raise ValueError('Element map endian tag is invalid.')
        raw = stream.read(4 * header.nb_elems_file)
    if len(raw) != 4 * header.nb_elems_file:
        raise ValueError('Element map is truncated or missing IDs.')
    values = np.frombuffer(raw, dtype=endian+'i4').astype(np.int32)
    _validate_element_map(values, header.nb_elems)
    return values


def _map_provenance(values):
    return dict(indexing='one-based global element IDs, 1..N',
                file_order_ids_sha256=sha256(values.astype('<i4').tobytes()).hexdigest(),
                reader_element_order='ascending global ID; data.elem[ID-1]')


def _signature():
    directory = Path(__file__).parent
    return {name: _file_hash(directory / name) for name in SIGNATURE_FILES}


def _case_directory(case):
    value = Path(case).expanduser()
    if len(value.parts) == 1 and (value.name.startswith('N') or value.name.startswith('case_')):
        value = Path('/data/Nek5000_data') / ('case_' + value.name.removeprefix('case_'))
    return value.resolve()


def _safe_destination(root, case_dir, index, *, allow_existing=False):
    root = Path(root).expanduser().absolute()
    final = root / case_dir.name / f'f{index:05d}'
    # Reject aliases, including dangling symlinks, before resolving ancestry.
    if any(p.is_symlink() for p in (final, *final.parents)):
        raise ValueError('Output paths must not contain symlinks.')
    root, final = root.resolve(), final.resolve()
    if root.is_relative_to(REPOSITORY) or final.is_relative_to(REPOSITORY):
        raise ValueError('Output must be outside the repository.')
    if root.is_relative_to(case_dir) or final.is_relative_to(case_dir):
        raise ValueError('Output must be outside the original case directory.')
    for parent in (root, *final.parents):
        if not parent.exists():
            continue
        if not parent.is_dir():
            raise ValueError('Output ancestor must be a directory.')
        # Do not nest a new workflow under a raw or scientific frame, even
        # when the frame was renamed or the proposed suffix does not exist.
        if any(p.suffix in ('.vtu', '.vtr', '.pvd', '.nek5000') or
               (p.name.startswith('GC') and len(p.suffix) == 7 and p.suffix.startswith('.f'))
               for p in parent.iterdir()):
            raise ValueError(f'Output must be outside existing scientific frame directories: {parent}')
    if final.exists() and (not allow_existing or not final.is_dir()):
        raise FileExistsError(f'Refusing to replace an existing frame directory: {final}')
    return final


def _resources(report):
    try:
        available = available_memory_bytes()
        if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or available < 0:
            raise ValueError('available memory must be a nonnegative integer')
    except Exception as exc:
        raise MemoryError(f'Cannot establish available RAM: {exc}') from exc
    disk = shutil.disk_usage(_nearest_directory(Path(report['output_directory']))).free
    return int(available), int(disk)


def inspect_vortex_frame(case='N7_H', index=79, output_dir=None, *, crop=None,
                         periodic_axes=('x', 'y'), expected_element_counts=(272, 12, 8),
                         chunk_size=256, sampling_nodes=None):
    """Read one header/map and resource state; never decode or write a frame."""
    config = _configuration(periodic_axes, expected_element_counts, crop, chunk_size, sampling_nodes)
    return _inspect_vortex_frame(case, index, output_dir, config)


def _inspect_vortex_frame(case, index, output_dir, config, *, allow_existing=False):
    index = _positive_integer(index, 'index')
    if index > 99999:
        raise ValueError('index must fit the five-digit source naming convention.')
    counts = config.expected_element_counts
    case_dir = _case_directory(case)
    if case_dir.is_relative_to(REPOSITORY):
        raise ValueError('Original simulation data must be outside the repository.')
    source = case_dir / f'GC0.f{index:05d}'
    if not case_dir.is_dir() or not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise ValueError('Snapshot must be a regular file, not a symlink.')
    dense = config.sampling_nodes is not None and config.sampling_nodes > 8
    root = (DEFAULT_RECONSTRUCTION_ROOT if dense else DEFAULT_OUTPUT_ROOT) if output_dir is None else Path(output_dir)
    if dense:
        root = root / f'gll{config.sampling_nodes}'
    final = _safe_destination(root, case_dir, index, allow_existing=allow_existing)
    ancestor = _nearest_directory(final)
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f'Output ancestor is not writable: {ancestor}')
    header = read_nek_header(source)
    if header.nb_dims != 3 or header.nb_vars[:2] != (3, 3):
        raise ValueError('Snapshot must contain 3D coordinates and all three velocities.')
    if (header.nb_files != 1 or header.nb_elems_file != header.nb_elems or
            header.wdsz not in (4, 8) or not np.isfinite(header.time)):
        raise ValueError('Require a complete unsplit snapshot with finite actual time.')
    orders = _integer_triple(header.orders, 'header node counts', 2)
    if config.sampling_nodes is not None and orders != (8, 8, 8):
        raise ValueError('Explicit sampling_nodes requires native P7 (8,8,8) source elements.')
    if header.nb_elems != prod(counts):
        raise ValueError('Header element count differs from expected_element_counts.')
    element_map = _read_element_map(source, header)
    local = header.nb_elems * prod(orders)
    output_orders = (config.sampling_nodes,) * 3 if dense else orders
    output_points = header.nb_elems * prod(output_orders)
    cells = header.nb_elems * prod(n - 1 for n in output_orders)
    if max(output_points, 8*cells) > np.iinfo(np.int64).max:
        raise ValueError('Requested sampling exceeds signed int64 topology capacity.')
    # Full-domain compute even for a small crop. Conservative upper bound
    # includes reader float64 arrays, two tensors, projection/scalar work,
    # element-local point/connectivity lists, VTK copies and binary read-back.
    output_bytes = 40 * output_points + 80 * cells + 1024**2
    # Dense tensors are element-bounded; no full-domain dense tensor or scalar
    # lambda interpolation buffer. 1024 bytes/sample allows tensor/eigenvalue,
    # at most eight cached local topologies, and interpolation work.
    reconstruction_workspace = 1024 * prod(output_orders) if dense else 0
    ram = (8 * sum(header.nb_vars) * local + source.stat().st_size + 240 * local +
           6 * output_bytes + reconstruction_workspace + 1024**3)
    report = dict(
        case=case_dir.name, source_file=str(source), frame_index=index,
        time=float(header.time), istep=int(header.istep), input_word_bytes=int(header.wdsz),
        element_count=int(header.nb_elems), element_local_shape=list(reversed(orders)),
        element_map=_map_provenance(element_map),
        configuration=dict(periodic_axes=list(config.periodic_axes), expected_element_counts=list(counts),
                           chunk_size=config.chunk_size, crop=None if config.crop is None else list(config.crop)),
        full_local_point_count=output_points, full_linear_hex_count=cells,
        native_local_point_count=local, estimated_output_bytes=output_bytes,
        estimated_reconstruction_workspace_bytes=reconstruction_workspace,
        estimated_ram_bytes=ram, required_disk_bytes=2 * output_bytes + 1024**2,
        output_directory=str(final),
        output_paths={k: str(final / (f'GC0_f{index:05d}' +
                                     (f'_gll{config.sampling_nodes}' if dense else '') + suffix))
                      for k, suffix in (('vtu', '.vtu'), ('metadata', '.json'))},
        geometry_validation='pending payload read',
    )
    if config.sampling_nodes is not None:
        report['configuration']['sampling_nodes'] = config.sampling_nodes
        report['reconstruction'] = sampling_metadata(config.sampling_nodes)
        report['reconstruction']['omega_route_audit_performed'] = False
        if config.sampling_nodes == 8:
            report['reconstruction'].update(
                interpolation='native identity; no dense interpolation executed',
                omega_x='native P(A[2,1]-A[1,2]); reconstruction-route audit not performed',
                lambda_ci='native eigenvalues of P(A); no dense interpolation executed')
    available, disk = _resources(report)
    report.update(available_ram_bytes=available, available_disk_bytes=disk,
                  resources_acceptable=ram < available and report['required_disk_bytes'] < disk)
    return report


def _require_resources(report):
    available, disk = _resources(report)
    if available <= report['estimated_ram_bytes']:
        raise MemoryError('Insufficient RAM for full-domain gradients and binary VTU read-back.')
    if disk <= report['required_disk_bytes']:
        raise OSError('Insufficient output disk space.')


def _numerical_definition(dense):
    result = dict(omega_x='P(A[2,1] - A[1,2])', lambda_ci='max(abs(imag(eig(P(A)))))',
                  A='du_i/dx_j from element_velocity_gradient',
                  P='existing mass-lumped GLL L2 projection on full domain')
    if dense:
        result.update(omega_x='I(P(A[2,1] - A[1,2]))', lambda_ci='max(abs(imag(eig(I(P(A))))))',
                      I='existing P7 tensor-product interpolation at target GLL nodes')
    return result


def _software_versions():
    return {name: version(name) for name in ('numpy', 'scipy', 'pymech', 'pyvista', 'vtk')}


def _read_snapshot(source, header, *, coordinates_only=False):
    skip = ('pressure', 'temperature') + tuple(f's{i:02d}' for i in range(1, header.nb_vars[4] + 1))
    if coordinates_only:
        skip += ('ux', 'uy', 'uz')
    return read_nek_file(source, dtype=f'float{header.wdsz * 8}', skip_vars=skip)


def inspect_volume_output_plan(report, config, selection):
    """Read native coordinates only, with a RAM gate; never build a dense 3D mesh."""
    source = Path(report['source_file'])
    before = source.stat()
    header = read_nek_header(source)
    needed = 8*sum(header.nb_vars)*report['native_local_point_count'] + before.st_size
    # Gate the bounded dense workspace before target-GLL node generation too.
    # An enormous target must not reach Legendre root/interpolation planning
    # merely because the coordinate-only native snapshot itself fits in RAM.
    needed += (96*report['native_local_point_count']+
               report['estimated_reconstruction_workspace_bytes']+1024**3)
    available, _ = _resources(report)
    if available <= needed:
        raise MemoryError('Insufficient RAM for coordinate-only crop planning.')
    data = _read_snapshot(source, header, coordinates_only=True)
    if (data.time != report['time'] or len(data.elem) != report['element_count'] or
            any(np.shape(e.pos) != (3, *report['element_local_shape']) for e in data.elem)):
        raise ValueError('Coordinate planning header/payload time, count or ordering differs.')
    _validate_element_map(data.elmap, header.nb_elems)
    if not np.array_equal(data.elmap, _read_element_map(source, header)):
        raise ValueError('Coordinate planning reader element map differs from the original.')
    mapping = build_structured_node_map(data, periodic_axes=config.periodic_axes,
        expected_element_counts=config.expected_element_counts, chunk_size=config.chunk_size)
    plan = build_volume_output_plan(data, mapping, config.sampling_nodes, selection)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during coordinate-only crop planning.')
    return plan


def run_single_frame_vortex(case='N7_H', index=79, output_dir=None, *, crop=None,
                            periodic_axes=('x', 'y'), expected_element_counts=(272, 12, 8),
                            chunk_size=256, sampling_nodes=None, dry_run=False, progress=None,
                            expected_mesh_signature=None, expected_source_sha256=None,
                            output_selection=None, expected_output_plan=None):
    """Compute exactly one frame, read back binary data, publish without replacement.

    Failed staging is retained. No resume/overwrite/batch paths are provided.
    Linux renameat2 publishes the VTU and provenance JSON together atomically.
    """
    started = time.perf_counter()
    config = _configuration(periodic_axes, expected_element_counts, crop, chunk_size, sampling_nodes)
    report = _inspect_vortex_frame(case, index, output_dir, config)
    prepared = expected_output_plan
    if output_selection is not None:
        if not isinstance(output_selection, VolumeSelection) or output_selection.bounds != config.crop:
            raise ValueError('Volume selection and immutable crop configuration disagree.')
        report['output_selection'] = output_selection.metadata()
        if config.crop is not None:
            if prepared is None:
                prepared = inspect_volume_output_plan(report, config, output_selection)
            apply_volume_output_plan(report, prepared, config, output_selection)
            if expected_mesh_signature is not None and expected_mesh_signature != prepared['native_mesh_signature']:
                raise ValueError('Volume plan and series native geometry signatures disagree.')
            expected_mesh_signature = prepared['native_mesh_signature']
    elif prepared is not None:
        raise ValueError('A volume output plan requires an explicit selection policy.')
    index = report['frame_index']
    if dry_run:
        return report
    _require_resources(report)
    notify = progress or (lambda message: None)
    source = Path(report['source_file'])
    signature = _signature()
    source_hash = _file_hash(source)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError('Source snapshot differs from the series preflight checksum.')
    source_stat = source.stat()
    header = read_nek_header(source)
    if (header.time != report['time'] or header.istep != report['istep'] or
            header.wdsz != report['input_word_bytes'] or header.nb_elems != report['element_count'] or
            tuple(reversed(header.orders)) != tuple(report['element_local_shape']) or
            header.nb_files != 1 or header.nb_elems_file != header.nb_elems or
            header.nb_vars[:2] != (3, 3)):
        raise ValueError('Source header changed after preflight.')
    file_map = _read_element_map(source, header)
    if _map_provenance(file_map) != report['element_map']:
        raise ValueError('Element map changed after preflight.')
    notify('Reading one snapshot; computing full-domain native gradients and L2 projection')
    data = _read_snapshot(source, header)
    if (data.time != report['time'] or len(data.elem) != report['element_count'] or
            any(np.shape(e.pos) != (3, *report['element_local_shape']) for e in data.elem)):
        raise ValueError('Header/payload time, count or point ordering shape mismatch.')
    _validate_element_map(data.elmap, header.nb_elems)
    if not np.array_equal(data.elmap, file_map):
        raise ValueError('Element map retained by the reader differs from the original map.')
    dense = config.sampling_nodes is not None and config.sampling_nodes > 8
    def validate_geometry(current, mapping):
        if prepared is not None and build_volume_output_plan(current, mapping, config.sampling_nodes, output_selection) != prepared:
            raise ValueError('Selected volume geometry/counts differ from the preflight plan.')
    compute = compute_projected_vortex_fields if dense else compute_native_vortex_fields
    fields = compute(data, periodic_axes=config.periodic_axes,
                     expected_element_counts=config.expected_element_counts,
                     chunk_size=config.chunk_size, progress=notify,
                     expected_mesh_signature=expected_mesh_signature,
                     geometry_validator=validate_geometry if prepared is not None else None)
    mesh_signature = native_mesh_signature(fields.node_map)
    if dense:
        notify(f'Reconstructing projected P7 tensors on GLL{config.sampling_nodes} nodes, one element at a time')
        grid, diagnostics = build_reconstructed_vortex_grid(
            data, fields, sampling_nodes=config.sampling_nodes, crop=config.crop,
            chunk_size=config.chunk_size, progress=notify,
        )
        report['reconstruction'].update(diagnostics)
        report['reconstruction']['omega_route_audit_performed'] = True
    else:
        notify('Building native linear hex subcells (element-local points, no merging)')
        grid = build_native_vortex_grid(data, fields, crop=config.crop)
    mapping = fields.node_map
    if output_selection is not None:
        validate_volume_grid(grid, output_selection, prepared)
    report['visualization'] = dict(
        points=grid.n_points, cells=grid.n_cells, bounds=list(grid.bounds),
        point_order='element-major C order (t,s,r), r fastest; crop retains used points in order',
        shared_points='preserved element-local, never merged', cell_type='VTK_HEXAHEDRON',
        physical_to_array_axes=list(mapping.physical_to_array_axes),
        paraview_verified=False, tecplot_verified=False,
    )
    report['field_ranges'] = {name: dict(min=float(grid.point_data[name].min()),
                                       max=float(grid.point_data[name].max()))
                              for name in ('omega_x', 'lambda_ci')}
    del data, fields
    _require_resources(report)
    final = Path(report['output_directory'])
    # Recheck paths immediately before any directory creation. A private
    # staging directory avoids exposing incomplete scientific results.
    _safe_destination(final.parent.parent, source.parent, index)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.vortex-3d-f{index:05d}-', dir=final.parent))
    notify('Writing binary VTU and verifying exact coordinates, fields, connectivity and time')
    vtu = stage / Path(report['output_paths']['vtu']).name
    write_binary_vtu(vtu, grid, time=report['time'])
    vtu_identity = regular_file_identity(vtu)
    current = source.stat()
    if ((current.st_size, current.st_mtime_ns, current.st_ino) !=
            (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino) or
            _file_hash(source) != source_hash):
        raise RuntimeError('Source snapshot changed during export; staging preserved.')
    if _signature() != signature:
        raise RuntimeError('Implementation changed during export; staging preserved.')
    report.update(
        schema_version=1, completion_status='completed',
        geometry_validation=('passed exact native geometry and reconstructed profiles' if dense
                             else 'passed exact native geometry'),
        source_sha256=source_hash, source_unchanged=True, numerical_software_signature=signature,
        output_sha256={'vtu': binary_vtu_fingerprint(vtu, expected=vtu_identity)}, binary_vtu_roundtrip_verified=True,
        native_mesh_signature=mesh_signature, array_sha256=grid_fingerprints(grid),
        numerical_definition=_numerical_definition(dense),
        software=_software_versions(),
        python=platform.python_version(), elapsed_seconds=time.perf_counter() - started,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    )
    metadata = stage / Path(report['output_paths']['metadata']).name
    with metadata.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    metadata_identity = regular_file_identity(metadata)
    _safe_destination(final.parent.parent, source.parent, index)
    for path, identity in ((vtu, vtu_identity), (metadata, metadata_identity)):
        if regular_file_identity(path) != identity:
            raise RuntimeError('Staged artifact changed before publication; staging preserved.')
    _publish_directory(stage, final)
    return report
