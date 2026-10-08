"""Native-GLL scalar export as a linear rectilinear VTK volume (.vtr).

The verified affine box mesh is exactly separable into physical axis vectors.
Adjacent GLL planes bound linear volume cells; interpolation for visualization
is linear, not a high-order SEM representation. VTK Lagrange cells use uniform
parametric nodes, so assigning native nonuniform GLL nodes to those DOFs would
change interpolation. Rectilinear connectivity is implicit (x-fast ordering),
avoiding millions of explicit hexahedral connectivity records.

Only internal duplicate nodes are identified for visualization. Both physical
endpoints remain distinct on periodic axes. Raw coordinates must be exactly
separable and shared scalar values must agree; neither is averaged here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.fields import get_coordinates
from nek_post.l2_projection import StructuredGLLNodeMap, structured_global_node_shape


@dataclass(frozen=True)
class RectilinearLambdaFields:
    """Physical x/y/z vectors and float64 scalar arrays in (Nz,Ny,Nx) order."""

    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    lambda_ci: np.ndarray
    lambda_ci_squared: np.ndarray


def assemble_rectilinear_lambda_fields(
    data: object, node_map: StructuredGLLNodeMap,
    lambda_ci: object, lambda_ci_squared: object, *, chunk_size: int = 256,
) -> RectilinearLambdaFields:
    """Validate original coordinates and assemble scalars without seam collapse."""
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, (bool, np.bool_)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    node_map.validate_mesh(data)
    elements = tuple(data.elem)
    shape = (node_map.element_count, *node_map.element_shape)
    fields = tuple(np.asarray(v) for v in (lambda_ci, lambda_ci_squared))
    if any(v.shape != shape or v.dtype.kind not in "fiu" for v in fields):
        raise ValueError(f"Scalar fields must be real arrays of shape {shape}.")
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
    assembled = [np.full(visual_map.global_node_count, np.nan, dtype=np.float64) for _ in fields]
    for start in range(0, len(elements), chunk_size):
        stop = min(start + chunk_size, len(elements))
        ids = visual_map.node_ids(start, stop).ravel()
        for target, field in zip(assembled, fields):
            values = field[start:stop].reshape(-1)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Exported scalar fields must be finite and nonnegative.")
            target[ids] = values
        with np.errstate(over="ignore", invalid="ignore"):
            square = np.square(fields[0][start:stop])
        if not np.array_equal(square, fields[1][start:stop]):
            raise ValueError("lambda_ci_squared must equal lambda_ci**2.")
    # Independent second pass detects disagreements even within one chunk.
    for start in range(0, len(elements), chunk_size):
        stop = min(start + chunk_size, len(elements))
        ids = visual_map.node_ids(start, stop).ravel()
        for target, field in zip(assembled, fields):
            if not np.array_equal(target[ids], field[start:stop].reshape(-1)):
                raise ValueError("Shared visualization nodes have inconsistent scalar values.")
    for target in assembled:
        if not np.isfinite(target).all():
            raise ValueError("Visualization grid has missing or non-finite scalar nodes.")
    return RectilinearLambdaFields(tuple(coordinates), *(a.reshape(global_shape[::-1]) for a in assembled))


def write_rectilinear_lambda_vtr(
    path: str | Path, fields: RectilinearLambdaFields, *, time: float,
) -> Path:
    """Write exclusive, uncompressed appended-binary VTR using installed VTK.

    Point data are x-fast, with implicit axis-aligned volume-cell connectivity.
    Both fields are float64. TIME_VALUE stores the actual simulation time.
    This routine never overwrites an existing file, including dangling links.
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter

    path = Path(path)
    if path.suffix != ".vtr" or not np.isfinite(time):
        raise ValueError("Output must be a .vtr path with a finite timestamp.")
    dimensions = tuple(len(v) for v in fields.coordinates)
    if len(dimensions) != 3 or any(n < 2 for n in dimensions):
        raise ValueError("A volume grid requires at least two nodes per axis.")
    for vector in fields.coordinates:
        if np.asarray(vector).ndim != 1 or not np.isfinite(vector).all() or not np.all(np.diff(vector) > 0):
            raise ValueError("Coordinate vectors must be finite and strictly increasing.")
    for array in (fields.lambda_ci, fields.lambda_ci_squared):
        if array.shape != dimensions[::-1] or array.dtype != np.float64 or not array.flags.c_contiguous or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Output fields must be finite nonnegative C-contiguous float64 arrays matching the grid.")
    with np.errstate(over="ignore", invalid="ignore"):
        # Validate in z slabs to keep the square check bounded.
        for z in range(dimensions[2]):
            if not np.array_equal(fields.lambda_ci[z]**2, fields.lambda_ci_squared[z]):
                raise ValueError("Squared field is inconsistent.")
    grid = vtkRectilinearGrid()
    grid.SetDimensions(*dimensions)
    for name, setter, vector in zip("XYZ", (grid.SetXCoordinates, grid.SetYCoordinates, grid.SetZCoordinates), fields.coordinates):
        coordinate_array = numpy_to_vtk(np.asarray(vector, dtype=np.float64), deep=False)
        coordinate_array.SetName(name)
        setter(coordinate_array)
    for name, values in (("lambda_ci", fields.lambda_ci), ("lambda_ci_squared", fields.lambda_ci_squared)):
        vtk_array = numpy_to_vtk(values.reshape(-1), deep=False)
        vtk_array.SetName(name)
        grid.GetPointData().AddArray(vtk_array)
    grid.GetPointData().SetActiveScalars("lambda_ci")
    timestamp = numpy_to_vtk(np.asarray([time], dtype=np.float64), deep=False)
    timestamp.SetName("TIME_VALUE")
    grid.GetFieldData().AddArray(timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before VTK opens the new empty file for writing.
    with path.open("xb"):
        pass
    try:
        writer = vtkXMLRectilinearGridWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(grid)
        writer.SetDataModeToAppended()
        writer.EncodeAppendedDataOff()
        writer.SetCompressorTypeToNone()
        if writer.Write() != 1 or writer.GetErrorCode() != 0:
            raise OSError("VTK rectilinear writer failed.")
    except Exception:
        path.unlink(missing_ok=True)  # Only the new file reserved above.
        raise
    return path


def write_single_frame_pvd(path: str | Path, vtr_path: str | Path, *, time: float) -> Path:
    """Write one dataset entry, preserving its physical timestamp for ParaView."""
    path, vtr_path = Path(path), Path(vtr_path)
    if path.suffix != ".pvd" or path.parent.resolve() != vtr_path.parent.resolve() or not vtr_path.is_file() or not np.isfinite(time):
        raise ValueError("PVD and its existing VTR must share one output directory and a finite timestamp.")
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    ET.SubElement(collection, "DataSet", timestep=repr(float(time)), group="", part="0", file=vtr_path.name)
    with path.open("xb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
    return path
