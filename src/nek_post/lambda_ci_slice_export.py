"""Dense SEM slice export and timestamp-preserving ParaView collections."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET

import numpy as np

from nek_post.lambda_ci_export import RectilinearLambdaFields, write_rectilinear_lambda_vtr


def slice_rectilinear_fields(samples):
    """Embed (Nz,Nh) slice scalars into physical (Nz,Ny,Nx) order.

    y-normal planes have dimensions (Nx,1,Nz); x-normal planes (1,Ny,Nz).
    The scalar samples are unchanged, with x-fast VTK point indexing.
    """
    if samples.normal_axis == 'y' and samples.horizontal_axis == 'x':
        coordinates = (samples.horizontal, np.array([samples.plane_value]), samples.z)
        shape = (len(samples.z), 1, len(samples.horizontal))
    elif samples.normal_axis == 'x' and samples.horizontal_axis == 'y':
        coordinates = (np.array([samples.plane_value]), samples.horizontal, samples.z)
        shape = (len(samples.z), len(samples.horizontal), 1)
    else:
        raise ValueError('Only physical y-midplane and x-slice exports are supported.')
    expected = (len(samples.z), len(samples.horizontal))
    if any(np.shape(a) != expected for a in (samples.lambda_ci, samples.lambda_ci_squared)):
        raise ValueError('Sample arrays must have (Nz,Nhorizontal) shape.')
    return RectilinearLambdaFields(coordinates, *(
        np.ascontiguousarray(a, dtype=np.float64).reshape(shape)
        for a in (samples.lambda_ci, samples.lambda_ci_squared)))


def write_lambda_ci_slice_vtr(path, samples, *, time):
    """Write a directly displayable 2D plane without overwriting a file."""
    return write_rectilinear_lambda_vtr(path, slice_rectilinear_fields(samples),
                                       time=time, allow_2d=True)


def write_tecplot_paraview_preset(path):
    """Export the same continuous RGB controls as an importable ParaView JSON."""
    import json
    from nek_post.lambda_ci_plotting import TECPLOT_RGB_POINTS
    controls = [value for position, rgb in TECPLOT_RGB_POINTS for value in (position, *rgb)]
    preset = [{'Name': 'NekPost Tecplot Continuous Blue Red', 'ColorSpace': 'RGB',
               'RGBPoints': controls, 'NanColor': [1.0, 0.0, 1.0]}]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(preset, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def sample_linear_vtr_reference(path, samples, *, time, chunk_rows=16):
    """Diagnostic comparison only: old VTR scalar, trilinear physical sampling.

    This is NOT the high-order reconstruction pipeline. Return (Nz,Nh) values
    at exactly the new slice's physical sample coordinates; no extrapolation.
    """
    from scipy.interpolate import RegularGridInterpolator
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer.')
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    reader = vtkXMLRectilinearGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if reader.GetErrorCode() or grid.GetNumberOfPoints() == 0:
        raise ValueError('Failed to read reference VTR.')
    timestamp = grid.GetFieldData().GetArray('TIME_VALUE')
    if timestamp is None or float(vtk_to_numpy(timestamp)[0]) != float(time):
        raise ValueError('Reference VTR physical timestamp does not match.')
    coords = [vtk_to_numpy(getter()) for getter in
              (grid.GetXCoordinates, grid.GetYCoordinates, grid.GetZCoordinates)]
    if any(not np.isfinite(c).all() or not np.all(np.diff(c) > 0) for c in coords):
        raise ValueError('Invalid reference coordinates.')
    scalar = grid.GetPointData().GetArray('lambda_ci')
    if scalar is None:
        raise ValueError('Reference VTR has no lambda_ci PointData.')
    values = vtk_to_numpy(scalar).reshape(grid.GetDimensions()[::-1])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite reference scalar.')
    interpolate = RegularGridInterpolator(tuple(coords[::-1]), values, method='linear', bounds_error=True)
    result = np.empty_like(samples.lambda_ci)
    for start in range(0, len(samples.z), chunk_rows):
        stop = min(start + chunk_rows, len(samples.z))
        z, h = np.meshgrid(samples.z[start:stop], samples.horizontal, indexing='ij')
        fixed = np.full_like(h, samples.plane_value)
        points = np.stack((z, fixed, h) if samples.normal_axis == 'y' else (z, h, fixed), axis=-1)
        result[start:stop] = interpolate(points)
    return result


def _collection(path):
    if path.is_symlink():
        raise FileExistsError('PVD symlinks are not update targets.')
    if path.exists():
        root = ET.parse(path).getroot()
        collection = root.find('Collection')
        if root.tag != 'VTKFile' or root.get('type') != 'Collection' or collection is None:
            raise ValueError('Invalid PVD collection.')
        for entry in collection:
            if entry.tag != 'DataSet' or not np.isfinite(float(entry.get('timestep', 'nan'))):
                raise ValueError('Invalid PVD entry.')
            ref = entry.get('file', '')
            if not ref or Path(ref).is_absolute() or not (path.parent / ref).is_file():
                raise ValueError('PVD entries must reference existing relative files.')
        return root, collection
    root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    return root, ET.SubElement(root, 'Collection')


def check_slice_series_entry(path, vtr_path, *, time):
    """Preflight: reject duplicate timestamps/paths without modifying the PVD."""
    path, vtr_path = Path(path), Path(vtr_path)
    if path.suffix != '.pvd' or vtr_path.suffix != '.vtr' or not np.isfinite(time):
        raise ValueError('Require PVD/VTR paths and a finite physical timestamp.')
    _, collection = _collection(path)
    target = vtr_path.resolve()
    for entry in collection:
        if float(entry.get('timestep')) == float(time) or (path.parent / entry.get('file')).resolve() == target:
            raise FileExistsError('PVD already contains this timestamp or VTR reference.')


def append_slice_series_pvd(path, vtr_path, *, time):
    """Append one actual frame while retaining every prior entry unchanged.

    No frames or times are inferred. Relative references permit case-level
    collections to point into frame/slices directories. Existing collections
    are replaced atomically after checking they have not changed concurrently.
    """
    path, vtr_path = Path(path), Path(vtr_path)
    check_slice_series_entry(path, vtr_path, time=time)
    if not vtr_path.is_file():
        raise ValueError('The referenced VTR must exist before publishing its entry.')
    original = path.read_bytes() if path.exists() else None
    root, collection = _collection(path)
    reference = Path(os.path.relpath(vtr_path.resolve(), path.parent.resolve())).as_posix()
    ET.SubElement(collection, 'DataSet', timestep=repr(float(time)), group='', part='0', file=reference)
    payload = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if original is None:
        with path.open('xb') as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + '.append')
        with temporary.open('xb') as stream:
            stream.write(payload)
        try:
            if path.is_symlink() or path.read_bytes() != original:
                raise RuntimeError('PVD changed concurrently; no entries were replaced.')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path
