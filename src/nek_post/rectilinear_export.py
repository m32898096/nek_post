"""Exact native rectilinear geometry and signed float64 scalar VTR export.

Visualization retains both physical periodic endpoints. Internal shared nodes
must agree exactly; no scalar interpolation or coordinate averaging is used.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from nek_post.fields import get_coordinates
from nek_post.l2_projection import structured_global_node_shape, _positive_integer


def _rectilinear_geometry(data, node_map):
    """Original lambda_ci geometry assembly; caller validates mesh identity."""
    elements = tuple(data.elem)
    profiles = [[None] * n for n in node_map.element_interval_counts]
    for ei, element in enumerate(elements):
        for ax, values in enumerate(get_coordinates(element)):
            array_axis = node_map.physical_to_array_axes[ax]
            sl = [0, 0, 0]
            sl[array_axis] = slice(None)
            raw = np.asarray(values)
            line = raw[tuple(sl)].astype(np.float64)
            expanded = [1, 1, 1]
            expanded[array_axis] = len(line)
            if not np.array_equal(raw, np.broadcast_to(line.reshape(expanded), raw.shape)):
                raise ValueError(f"Element {ei} coordinates are not exactly rectilinear.")
            if not node_map.element_axis_increasing[ei, ax]:
                line = line[::-1]
            cell = node_map.element_cell_indices[ei, ax]
            previous = profiles[ax][cell]
            if previous is not None and not np.array_equal(line, previous):
                raise ValueError("Original coordinate replicas differ; rectilinear export would alter geometry.")
            profiles[ax][cell] = line
    coordinates = []
    for axis_profiles in profiles:
        for a, b in zip(axis_profiles[:-1], axis_profiles[1:]):
            if a[-1] != b[0]:
                raise ValueError("Original adjacent coordinate endpoints differ.")
        vector = np.concatenate([axis_profiles[0], *(p[1:] for p in axis_profiles[1:])])
        if not np.isfinite(vector).all() or not np.all(np.diff(vector) > 0):
            raise ValueError("Visualization coordinates must be finite and increasing.")
        coordinates.append(vector)
    global_shape = structured_global_node_shape(
        node_map.element_interval_counts,
        tuple(node_map.element_shape[a] for a in node_map.physical_to_array_axes),
        periodic_axes=(),
    )
    visual_map = replace(node_map, periodic_axes=(), global_shape_xyz=global_shape)
    return tuple(coordinates), visual_map


@dataclass(frozen=True)
class RectilinearScalarField:
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: np.ndarray
    name: str


def assemble_rectilinear_scalar_field(data, node_map, values, *, name, chunk_size=256):
    """Assemble a signed native-node volume in (Nz,Ny,Nx), x-fast order."""
    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    node_map.validate_mesh(data)
    array = np.asarray(values)
    shape = (node_map.element_count, *node_map.element_shape)
    if array.shape != shape or array.dtype.kind not in 'fiu':
        raise ValueError(f'Scalar field must have real shape {shape}.')
    if not isinstance(name, str) or not name:
        raise ValueError('PointData name must be a nonempty string.')
    coordinates, visual_map = _rectilinear_geometry(data, node_map)
    target = np.full(visual_map.global_node_count, np.nan, dtype=np.float64)
    for start in range(0, node_map.element_count, chunk_size):
        stop = min(start + chunk_size, node_map.element_count)
        local = array[start:stop].reshape(-1)
        if not np.isfinite(local).all():
            raise ValueError('Exported scalar field must be finite.')
        target[visual_map.node_ids(start, stop).ravel()] = local
    for start in range(0, node_map.element_count, chunk_size):
        stop = min(start + chunk_size, node_map.element_count)
        if not np.array_equal(target[visual_map.node_ids(start, stop).ravel()], array[start:stop].reshape(-1)):
            raise ValueError('Shared visualization nodes have inconsistent scalar values.')
    if not np.isfinite(target).all():
        raise ValueError('Visualization grid has missing or non-finite scalar nodes.')
    return RectilinearScalarField(coordinates, target.reshape(visual_map.global_shape_xyz[::-1]), name)


def write_rectilinear_scalar_vtr(path, field, *, time):
    """Write a signed 3D PointData volume exclusively; lambda_ci rules are separate."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
    path = Path(path)
    if path.suffix != '.vtr' or not np.isfinite(time):
        raise ValueError('Require a .vtr destination and finite actual timestamp.')
    dimensions = tuple(len(v) for v in field.coordinates)
    if len(dimensions) != 3 or any(n < 2 for n in dimensions):
        raise ValueError('Require a complete 3D volume.')
    for vector in field.coordinates:
        if np.asarray(vector).ndim != 1 or not np.isfinite(vector).all() or not np.all(np.diff(vector) > 0):
            raise ValueError('Coordinate vectors must be finite and strictly increasing.')
    values = field.values
    if not isinstance(field.name, str) or not field.name or values.shape != dimensions[::-1] or values.dtype != np.float64 or not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ValueError('Require a named finite C-contiguous float64 scalar matching the grid.')
    grid = vtkRectilinearGrid();grid.SetDimensions(*dimensions)
    for setter, vector in zip((grid.SetXCoordinates, grid.SetYCoordinates, grid.SetZCoordinates), field.coordinates):
        setter(numpy_to_vtk(np.asarray(vector, dtype=np.float64), deep=False))
    scalar = numpy_to_vtk(values.ravel(), deep=False);scalar.SetName(field.name)
    grid.GetPointData().AddArray(scalar);grid.GetPointData().SetActiveScalars(field.name)
    timestamp = numpy_to_vtk(np.asarray([time], dtype=np.float64), deep=False);timestamp.SetName('TIME_VALUE')
    grid.GetFieldData().AddArray(timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb'):pass
    try:
        writer = vtkXMLRectilinearGridWriter();writer.SetFileName(str(path));writer.SetInputData(grid)
        writer.SetDataModeToAppended();writer.EncodeAppendedDataOff();writer.SetCompressorTypeToNone()
        if writer.Write() != 1 or writer.GetErrorCode():
            raise OSError('VTK scalar volume writer failed.')
    except BaseException:
        path.unlink(missing_ok=True)  # Only the file reserved by this invocation.
        raise
    return path
