"""Bounded fingerprints and read-only validation of completed vortex VTUs."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.hex_validation import validate_hexahedral_grid


def _file_identity(value):
    if not stat.S_ISREG(value.st_mode) or value.st_size == 0:
        raise ValueError('Require a nonempty regular artifact, never a symlink or special file.')
    return [value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns]


def regular_file_identity(path):
    """Identify the directory entry itself; never follow a substituted symlink."""
    try:
        return _file_identity(Path(path).lstat())
    except ValueError as exc:
        raise ValueError(f'{exc} Path: {path}') from exc


@contextmanager
def open_regular_file(path, *, expected=None):
    """Pin a no-follow descriptor and check both its inode and the named entry.

    O_NONBLOCK prevents a substituted FIFO from hanging open(). Parent paths
    remain subject to the workflow's directory checks and cooperating lock;
    these checks cannot freeze a hostile writable directory after returning.
    """
    path = Path(path)
    before = regular_file_identity(path)
    if expected is not None and before != expected:
        raise ValueError(f'Artifact identity changed before opening: {path}')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if _file_identity(os.fstat(descriptor)) != before:
            raise ValueError(f'Artifact changed while opening: {path}')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            yield stream
            if (_file_identity(os.fstat(descriptor)) != before or
                    regular_file_identity(path) != before):
                raise ValueError(f'Artifact changed while reading: {path}')
    finally:
        os.close(descriptor)


def regular_file_hash(path, *, expected=None):
    digest = sha256()
    with open_regular_file(path, expected=expected) as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def binary_vtu_fingerprint(path, *, expected=None):
    """Stream SHA-256 and verify inline binary XML tags without decoding payloads.

    The Phase 1/2 PyVista writer uses inline base64 DataArrays. Their text has
    no '<' bytes, allowing bounded tag scanning even for multi-GiB arrays.
    Other serializers, including ASCII/appended formats, are not adopted.
    """
    digest = sha256()
    carry = b''
    arrays = roots = 0
    with open_regular_file(path, expected=expected) as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            scan = carry+block
            for match in re.finditer(rb'<(?:VTKFile|DataArray)\b[^>]*>', scan):
                tag = match[0]
                try:
                    element = ET.fromstring(tag if tag.endswith(b'/>') else tag[:-1]+b'/>')
                except ET.ParseError as exc:
                    raise ValueError('Malformed VTU XML tag.') from exc
                if element.tag == 'VTKFile':
                    roots += 1
                    if element.get('type') != 'UnstructuredGrid':
                        raise ValueError('Expected a VTK XML UnstructuredGrid.')
                else:
                    arrays += 1
                    if element.get('format') != 'binary':
                        raise ValueError('Saved VTU must use the producer inline binary format.')
            start = scan.rfind(b'<')
            carry = scan[start:] if start >= 0 and b'>' not in scan[start:] else b''
            if len(carry) > 65536:
                raise ValueError('Oversized or malformed VTU XML tag.')
    if carry or roots != 1 or arrays < 7:
        raise ValueError('Missing or malformed binary VTU XML headers.')
    return digest.hexdigest()


def array_fingerprint(values):
    """Hash dtype, shape and C-order bytes without a complete array byte copy."""
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        raise ValueError('Artifact fingerprints require contiguous arrays.')
    digest = sha256(repr((array.dtype.str, array.shape)).encode())
    view = memoryview(array).cast('B')
    for start in range(0, len(view), 1024**2):
        digest.update(view[start:start + 1024**2])
    return digest.hexdigest()


def native_mesh_signature(mapping):
    """Full original geometry/reader ordering, including regions outside a crop."""
    digest = sha256()
    for signature in mapping.element_coordinate_signatures:
        digest.update(signature.encode('ascii'))
    return dict(coordinate_order_sha256=digest.hexdigest(),
                element_shape=list(mapping.element_shape),
                element_counts=list(mapping.element_interval_counts),
                physical_to_array_axes=list(mapping.physical_to_array_axes),
                cells_sha256=array_fingerprint(mapping.element_cell_indices),
                orientations_sha256=array_fingerprint(mapping.element_axis_increasing))


def grid_fingerprints(grid):
    arrays = dict(points=grid.points, connectivity=grid.cell_connectivity,
                  offsets=grid.offset, cell_types=grid.celltypes,
                  time=grid.field_data['TIME_VALUE'],
                  **{name: grid.point_data[name] for name in ('omega_x', 'lambda_ci')})
    return {name: array_fingerprint(value) for name, value in arrays.items()}


def validate_saved_grid(path, metadata, *, expected=None):
    """Decode exactly one saved grid; retain no VTK objects or scientific arrays."""
    with open_regular_file(path, expected=expected) as stream:
        # The Linux VTK reader needs a filename. This procfs path refers to the
        # already verified open inode, rather than reopening a mutable name.
        _validate_saved_grid(Path(f'/proc/self/fd/{stream.fileno()}'), metadata)


def _validate_saved_grid(path, metadata):
    import pyvista as pv
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetErrorCode():
        raise ValueError('Saved VTU reader failed.')
    grid = pv.wrap(reader.GetOutput())
    if set(grid.point_data) != {'omega_x', 'lambda_ci'}:
        raise ValueError('Saved VTU must contain exactly omega_x and lambda_ci PointData.')
    if len(grid.cell_data) or set(grid.field_data) != {'TIME_VALUE'}:
        raise ValueError('Saved VTU has unexpected cell/field arrays, including possible ghost-cell masks.')
    validate_hexahedral_grid(grid)
    for name in ('omega_x', 'lambda_ci'):
        array = grid.point_data[name]
        if array.dtype != np.float64 or array.shape != (grid.n_points,) or not np.isfinite(array).all():
            raise ValueError('Saved VTU scalar shape, precision or finiteness differs.')
        if name == 'lambda_ci' and np.any(array < 0):
            raise ValueError('Saved lambda_ci is negative.')
        ranges = dict(min=float(array.min()), max=float(array.max()))
        if metadata['field_ranges'][name] != ranges:
            raise ValueError('Saved VTU field ranges differ from metadata.')
    timestamp = grid.field_data.get('TIME_VALUE')
    if (timestamp is None or timestamp.dtype != np.float64 or timestamp.shape != (1,) or
            timestamp[0] != metadata['time']):
        raise ValueError('Saved VTU physical time differs.')
    visual = metadata['visualization']
    if (grid.n_points != visual['points'] or grid.n_cells != visual['cells'] or
            list(grid.bounds) != visual['bounds'] or grid_fingerprints(grid) != metadata['array_sha256']):
        raise ValueError('Saved VTU geometry, topology or array fingerprints differ.')
    if 'output_selection' in metadata:
        from nek_post.vortex_3d_selection import VolumeSelection, validate_volume_grid
        policy = metadata['output_selection']
        validate_volume_grid(grid, VolumeSelection(policy['mode'], policy['bounds']), metadata.get('output_plan'))
