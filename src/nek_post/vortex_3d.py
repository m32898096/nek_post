"""Native-GLL vortex fields and element-local linear hexahedral visualization.

No numerical kernel is defined here: both fields share one local Cartesian
gradient dataset, then follow their established projection operation order.
The supported global geometry is the existing affine structured box topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nek_post.fields import get_coordinates
from nek_post.hex_validation import validate_hexahedral_grid
from nek_post.l2_projection import (
    StructuredGLLNodeMap, _positive_integer, build_l2_projection_operator,
    project_scalar_field, project_velocity_gradient,
)
from nek_post.rectilinear_export import _rectilinear_geometry
from nek_post.spectral_derivatives import element_velocity_gradient
from nek_post.swirling_strength import compute_swirling_strength


@dataclass(frozen=True)
class NativeVortexFields:
    node_map: StructuredGLLNodeMap
    omega_x: np.ndarray
    lambda_ci: np.ndarray


def compute_native_vortex_fields(data, *, periodic_axes, expected_element_counts=None,
                                chunk_size=256, progress=None):
    """Return float64 (nelem,nt,ns,nr) scalars in original reader order.

    omega_x = P(A[2,1] - A[1,2]); lambda_ci = max(abs(imag(eig(P(A))))).
    P is the existing mass-lumped GLL L2 operator. Crop only after this call.
    The full-tensor solve can round differently from the optimized curl solve.
    """
    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    operator = build_l2_projection_operator(
        data, periodic_axes=periodic_axes, expected_element_counts=expected_element_counts,
        chunk_size=chunk_size,
    )
    # The map accepts storage uncertainty; visualization must retain exact
    # replicas/endpoints without introducing any coordinate averaging.
    _rectilinear_geometry(data, operator.node_map)
    shape = (operator.node_map.element_count, *operator.node_map.element_shape)
    raw = np.empty(shape + (3, 3), dtype=np.float64)
    for ei, element in enumerate(data.elem):
        raw[ei] = element_velocity_gradient(element)
        if progress and (ei + 1) % 4096 == 0:
            progress(f'Native gradients: {ei + 1}/{len(data.elem)} elements')
    omega = project_scalar_field(operator, raw[..., 2, 1] - raw[..., 1, 2],
                                 chunk_size=chunk_size)
    projected = project_velocity_gradient(operator, raw, diagnostics=False,
                                          chunk_size=chunk_size).values
    del raw
    strength = compute_swirling_strength(projected, chunk_size=chunk_size).lambda_ci
    return NativeVortexFields(operator.node_map, omega, strength)


def validate_crop(crop):
    """Six finite physical bounds, xmin,xmax,ymin,ymax,zmin,zmax, or None."""
    if crop is None:
        return None
    array = np.asarray(crop)
    if array.shape != (6,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('crop must contain six finite real bounds.')
    bounds = array.astype(np.float64).reshape(3, 2)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('crop minima must be strictly less than maxima.')
    return bounds


def _physical_hexes(mapping, ei):
    """Local C-order IDs; connectivity winds positively in physical x/y/z."""
    ids = np.arange(np.prod(mapping.element_shape), dtype=np.int64).reshape(mapping.element_shape)
    ids = ids.transpose(tuple(mapping.physical_to_array_axes[i] for i in (2, 1, 0)))
    for axis, physical in enumerate((2, 1, 0)):
        if not mapping.element_axis_increasing[ei, physical]:
            ids = np.flip(ids, axis=axis)
    return np.stack((ids[:-1, :-1, :-1], ids[:-1, :-1, 1:],
                     ids[:-1, 1:, 1:], ids[:-1, 1:, :-1],
                     ids[1:, :-1, :-1], ids[1:, :-1, 1:],
                     ids[1:, 1:, 1:], ids[1:, 1:, :-1]), axis=-1).reshape(-1, 8)


def build_native_vortex_grid(data, fields, *, crop=None):
    """Create one vtkUnstructuredGrid with original coordinates and PointData.

    Keep wholly contained native subcells; retain used points in element-major
    (t,s,r) C order, r fastest. Never merge even agreeing coincident points.
    Physical positive hex ordering is independent of the element array axes.
    """
    import pyvista as pv

    mapping = fields.node_map
    mapping.validate_mesh(data)
    coordinates, _ = _rectilinear_geometry(data, mapping)
    bounds = validate_crop(crop)
    domain = np.asarray([(v[0], v[-1]) for v in coordinates])
    if bounds is not None and (np.any(bounds[:, 0] < domain[:, 0]) or
                               np.any(bounds[:, 1] > domain[:, 1])):
        raise ValueError('crop must lie within the physical domain.')
    shape = (mapping.element_count, *mapping.element_shape)
    for name in ('omega_x', 'lambda_ci'):
        values = np.asarray(getattr(fields, name))
        if values.shape != shape or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f'{name} must be finite float64 with shape {shape}.')
    if np.any(fields.lambda_ci < 0):
        raise ValueError('lambda_ci must be nonnegative.')
    points, cells, omega, strength = [], [], [], []
    offset = 0
    for ei, element in enumerate(data.elem):
        xyz = np.stack(get_coordinates(element), axis=-1).reshape(-1, 3).astype(np.float64)
        local_cells = _physical_hexes(mapping, ei)
        if bounds is not None:
            inside = ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(axis=1)
            local_cells = local_cells[inside[local_cells].all(axis=1)]
        if not len(local_cells):
            continue
        used, inverse = np.unique(local_cells, return_inverse=True)
        points.append(xyz[used])
        cells.append(inverse.reshape(-1, 8) + offset)
        omega.append(fields.omega_x[ei].ravel()[used])
        strength.append(fields.lambda_ci[ei].ravel()[used])
        offset += len(used)
    if not cells:
        raise ValueError('crop contains no complete native 3D subcells.')
    connectivity = np.concatenate(cells)
    packed = np.empty((len(connectivity), 9), dtype=np.int64)
    packed[:, 0], packed[:, 1:] = 8, connectivity
    grid = pv.UnstructuredGrid(packed.ravel(), np.full(len(packed), pv.CellType.HEXAHEDRON,
                                                     dtype=np.uint8), np.concatenate(points))
    grid.point_data['omega_x'] = np.concatenate(omega)
    grid.point_data['lambda_ci'] = np.concatenate(strength)
    grid.set_active_scalars('omega_x')
    return grid


def write_binary_vtu(path, grid, *, time):
    """Save exclusively with PyVista's binary VTU writer, then verify exact data.

    Callers publish the enclosing private staging directory after validation.
    Failed files are preserved for diagnosis; existing files are never opened.
    """
    import pyvista as pv
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    path = Path(path)
    if path.suffix != '.vtu' or not np.isfinite(time):
        raise ValueError('Require a .vtu destination and finite actual simulation time.')
    if path.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Generated VTU files must be outside the repository.')
    if not isinstance(grid, pv.UnstructuredGrid) or grid.n_cells == 0 or grid.n_points == 0:
        raise ValueError('Require a nonempty 3D vtkUnstructuredGrid.')
    if not np.all(grid.celltypes == pv.CellType.HEXAHEDRON):
        raise ValueError('Require linear 3D hexahedral cells.')
    if grid.points.dtype != np.float64 or not np.isfinite(grid.points).all():
        raise ValueError('Require finite float64 physical points.')
    if set(grid.point_data) != {'omega_x', 'lambda_ci'}:
        raise ValueError('Require exactly omega_x and lambda_ci PointData.')
    for name in grid.point_data:
        values = grid.point_data[name]
        if values.shape != (grid.n_points,) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError('Require finite float64 scalar PointData.')
    if np.any(grid.point_data['lambda_ci'] < 0):
        raise ValueError('lambda_ci must be nonnegative.')
    validate_hexahedral_grid(grid)
    grid.field_data['TIME_VALUE'] = np.asarray([time], dtype=np.float64)
    # Reserve only a new path. The public workflow writes inside a private
    # directory and uses atomic no-replace publication for the scientific pair.
    with path.open('xb'):
        pass
    grid.save(path, binary=True)
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetErrorCode():
        raise OSError('VTK could not read the binary VTU.')
    actual = pv.wrap(reader.GetOutput())
    if (set(actual.point_data) != {'omega_x', 'lambda_ci'} or actual.n_cells != grid.n_cells or
            actual.n_points != grid.n_points):
        raise ValueError('VTU round-trip changed topology or PointData names.')
    # Native offsets + IDs contain exactly the same topology as legacy .cells,
    # without materializing two full packed 9-integer-per-hex temporary arrays.
    for name, expected, result in (
        ('points', grid.points, actual.points),
        ('connectivity', grid.cell_connectivity, actual.cell_connectivity),
        ('offsets', grid.offset, actual.offset),
        ('cell types', grid.celltypes, actual.celltypes),
        ('time', grid.field_data['TIME_VALUE'], actual.field_data.get('TIME_VALUE')),
        *((n, grid.point_data[n], actual.point_data.get(n)) for n in grid.point_data),
    ):
        if result is None or expected.dtype != result.dtype or not np.array_equal(expected, result):
            raise ValueError(f'VTU round-trip changed {name}, dtype or ordering.')
    return path
