"""Serialization-only VTU diagnostics; independent of the production writer."""
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
from xml.parsers import expat

import numpy as np

from nek_post.hex_validation import validate_hexahedral_grid
from nek_post.lambda_ci_series import _publish_directory
from nek_post.lambda_ci_workflow import available_memory_bytes, _nearest_directory
from nek_post.vortex_3d_artifacts import (
    array_fingerprint, open_regular_file, regular_file_hash, regular_file_identity,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILES = ('reference', 'appended-zlib', 'appended-none', 'appended-int32')
MAX_SUBSET_CELLS = 1000000
_SIZES = {f'{prefix}{bits}': bits//8 for prefix in ('Int', 'UInt') for bits in (8,16,32,64)}
_SIZES.update(Float32=4,Float64=8)


def inspect_vtu(path):
    """Stream XML headers, skipping inline text and stopping before appended data.

    This is header inspection, not proof that the binary payload is valid.
    Raw appended bytes are deliberately never submitted to an XML parser.
    """
    class HeadersComplete(Exception):
        pass
    report = dict(vtk_file={}, pieces=[], arrays=[], appended=None)
    stack = []
    parser = expat.ParserCreate()
    def start(name, attrs):
        if name == 'VTKFile':
            report['vtk_file'] = dict(attrs)
        elif name == 'Piece':
            report['pieces'].append(dict(attrs))
        elif name == 'DataArray':
            report['arrays'].append(dict(association=stack[-1], **attrs))
        elif name == 'AppendedData':
            report['appended'] = dict(attrs)
            raise HeadersComplete
        stack.append(name)
    parser.StartElementHandler = start
    parser.EndElementHandler = lambda name: stack.pop()
    def reject_doctype(*args):
        raise ValueError('VTU diagnostic headers must not contain a DTD.')
    parser.StartDoctypeDeclHandler = reject_doctype
    read_bytes = 0
    with open_regular_file(path) as stream:
        try:
            while block := stream.read(1024**2):
                read_bytes += len(block)
                parser.Parse(block, False)
            parser.Parse(b'', True)
        except HeadersComplete:
            pass
        except expat.ExpatError as exc:
            raise ValueError('Malformed VTU XML header.') from exc
    if report['vtk_file'].get('type') != 'UnstructuredGrid' or len(report['pieces']) != 1:
        raise ValueError('Require a single-piece VTK XML UnstructuredGrid.')
    piece = report['pieces'][0]
    try:
        report.update(points=int(piece['NumberOfPoints']), cells=int(piece['NumberOfCells']))
    except (KeyError, ValueError) as exc:
        raise ValueError('Missing or invalid VTU point/cell counts.') from exc
    if report['points'] < 8 or report['cells'] < 1:
        raise ValueError('Require a nonempty 3D volume.')
    report['inspection_bytes_read'] = read_bytes
    report['payload_validated'] = False
    return report


@dataclass(frozen=True)
class WriterConfiguration:
    profile: str
    encoding: str | None = None
    header_bits: int | None = None

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError('Unknown diagnostic writer profile.')
        encoding = ('base64' if self.profile == 'reference' else 'raw') if self.encoding is None else self.encoding
        header = 32 if self.header_bits is None else self.header_bits
        if encoding not in ('raw', 'base64') or type(header) is not int or header not in (32, 64):
            raise ValueError('Require raw/base64 encoding and 32/64 header bits.')
        if self.profile == 'reference' and (encoding != 'base64' or header != 32):
            raise ValueError('The reference profile reproduces inline/base64, UInt32 headers.')
        if self.profile == 'appended-int32' and header != 32:
            raise ValueError('The conservative Int32 profile requires UInt32 headers.')
        object.__setattr__(self, 'encoding', encoding)
        object.__setattr__(self, 'header_bits', header)

    def metadata(self):
        return dict(profile=self.profile, mode='binary' if self.profile == 'reference' else 'appended',
                    encoding=self.encoding, compressor='zlib' if self.profile in ('reference','appended-zlib') else None,
                    compression_level=5, header_bits=self.header_bits,
                    id_bits=32 if self.profile == 'appended-int32' else 64, byte_order='LittleEndian')


def check_storage_limits(points, cells, blocks, config):
    """Pure integer checks, testable at capacity boundaries without huge arrays.

    blocks are (name, serialized byte count). For UInt32 also bound the total
    uncompressed payload, per-array length words and compression overhead.
    This is deliberately stricter than VTK's per-compression-block limits.
    """
    if type(points) is not int or type(cells) is not int or points < 8 or cells < 1 or max(points-1, 8*cells) > np.iinfo(np.int64).max:
        raise ValueError('Invalid or unrepresentable topology counts.')
    if config.profile == 'appended-int32' and max(points-1, cells, 8*cells) > np.iinfo(np.int32).max:
        raise ValueError('Connectivity/offset values exceed signed Int32 capacity.')
    if any(type(size) is not int or size < 0 for _, size in blocks):
        raise ValueError('Invalid data-block size.')
    if config.header_bits == 32:
        total = sum(size + 4 + ((size+32767)//32768)*16 for _, size in blocks)
        if any(size > np.iinfo(np.uint32).max for _, size in blocks) or total > np.iinfo(np.uint32).max:
            raise ValueError('UInt32 block/aggregate payload limit exceeded; use a UInt64 appended profile.')


def _arrays(grid):
    for association in ('point_data', 'cell_data', 'field_data'):
        for name in getattr(grid, association):
            yield association, name, np.asarray(getattr(grid, association)[name])


def _roles(grid):
    from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes
    return [[data.GetAttribute(i).GetName() if data.GetAttribute(i) is not None else None
             for i in range(vtkDataSetAttributes.NUM_ATTRIBUTES)]
            for data in (grid.GetPointData(),grid.GetCellData())]


def _component_names(grid):
    return {f'{a}/{n}':[getattr(grid,a).GetArray(n).GetComponentName(i) for i in range(v.shape[1] if v.ndim==2 else 1)]
            for a,n,v in _arrays(grid)}


def validate_grid(grid):
    import pyvista as pv
    if not isinstance(grid, pv.UnstructuredGrid):
        raise ValueError('Require a 3D UnstructuredGrid.')
    validate_hexahedral_grid(grid)
    for name in ('omega_x', 'lambda_ci'):
        array = grid.point_data.get(name)
        if array is None or array.dtype != np.float64 or array.shape != (grid.n_points,) or not np.isfinite(array).all():
            raise ValueError('Require finite float64 omega_x and lambda_ci PointData.')
    if np.any(grid.point_data['lambda_ci'] < 0):
        raise ValueError('Negative lambda_ci is not supported.')
    time = grid.field_data.get('TIME_VALUE')
    if time is None or time.dtype != np.float64 or time.shape != (1,) or not np.isfinite(time).all():
        raise ValueError('Require the original float64 scalar TIME_VALUE; never infer time from a filename.')
    for _, name, array in _arrays(grid):
        if array.dtype.kind not in 'fiu' or array.ndim not in (1,2):
            raise ValueError(f'Unsupported nonnumeric or higher-rank metadata/array: {name}')
        if name == 'vtkGhostType':
            raise ValueError('Ghost masks are not supported by this diagnostic.')


def _read_grid(path, expected):
    import pyvista as pv
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    with open_regular_file(path, expected=expected) as stream:
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(f'/proc/self/fd/{stream.fileno()}')
        reader.Update()
        if reader.GetErrorCode():
            raise OSError('VTK reader failed.')
        grid = pv.wrap(reader.GetOutput())
        validate_grid(grid)
    return grid


def select_subset(source, max_cells=4096, seed_cell=None):
    """Copy whole original cells; remap IDs only, never merge or interpolate."""
    import pyvista as pv
    if type(max_cells) is not int or not 1 <= max_cells <= MAX_SUBSET_CELLS:
        raise ValueError(f'max_cells must be an integer in 1..{MAX_SUBSET_CELLS}.')
    ids = source.cell_connectivity.reshape(-1,8)
    strength = source.point_data['lambda_ci']
    if seed_cell is None:
        peak = int(np.argmax(strength))
        seed_cell = None
        for start in range(0, source.n_cells, 4096):
            found = np.flatnonzero(np.any(ids[start:start+4096] == peak, axis=1))
            if len(found):
                seed_cell = start+int(found[0])
                break
        if seed_cell is None:
            raise ValueError('Maximum lambda_ci point is unused by cells; choose --seed-cell explicitly.')
    if type(seed_cell) is not int or not 0 <= seed_cell < source.n_cells:
        raise ValueError('seed_cell is outside the original cell range.')
    count = min(max_cells, source.n_cells)
    start = max(0, min(seed_cell-count//2, source.n_cells-count))
    cell_ids = np.arange(start,start+count,dtype=np.int64)
    point_ids, inverse = np.unique(ids[start:start+count],return_inverse=True)
    packed = np.empty((count,9),dtype=np.int64)
    packed[:,0], packed[:,1:] = 8, inverse.reshape(-1,8)
    # Advanced indexing already owns a new array; do not duplicate it again.
    grid = pv.UnstructuredGrid(packed.ravel(), source.celltypes[cell_ids], source.points[point_ids])
    del packed, inverse
    for association, name, array in _arrays(source):
        index = point_ids if association == 'point_data' else cell_ids if association == 'cell_data' else slice(None)
        getattr(grid,association)[name] = array.copy() if association == 'field_data' else array[index]
        original,target=getattr(source,association).GetArray(name),getattr(grid,association).GetArray(name)
        for component in range(original.GetNumberOfComponents()):
            label=original.GetComponentName(component)
            if label is not None:
                target.SetComponentName(component,label)
    for data,roles in zip((grid.GetPointData(),grid.GetCellData()),_roles(source)):
        for role,name in enumerate(roles):
            data.SetActiveAttribute(-1 if name is None else name,role)
    validate_grid(grid)
    return grid, point_ids, cell_ids


def _crossing_isovalue(grid):
    """Return a float64 level strictly inside at least one retained Hex range.

    Keep the global midpoint only when a cell crosses it. Otherwise use the
    smallest representable cell midpoint (order independent), with nextafter
    handling midpoint rounding at adjacent endpoints. Ignore unused points
    when establishing crossings. Scan in chunks to bound full-grid temporaries.
    The grid and nonnegative, finite lambda_ci must already be validated.
    """
    strength = np.asarray(grid.point_data['lambda_ci'])
    ids = grid.cell_connectivity.reshape(-1,8)
    low, high = float(strength.min()), float(strength.max())
    level = low + (high-low)/2
    crossing_count = 0
    fallback = None
    for start in range(0, grid.n_cells, 4096):
        values = strength[ids[start:start+4096]]
        cell_low, cell_high = values.min(axis=1), values.max(axis=1)
        crossing_count += int(np.count_nonzero((cell_low < level) & (level < cell_high)))
        middle = cell_low + (cell_high-cell_low)/2
        middle = np.where((cell_low < middle) & (middle < cell_high),
                          middle, np.nextafter(cell_low,cell_high))
        usable = middle[(cell_low < middle) & (middle < cell_high)]
        if usable.size:
            candidate = float(usable.min())
            fallback = candidate if fallback is None else min(fallback,candidate)
    if crossing_count:
        return level, crossing_count
    if fallback is None:
        raise ValueError('No usable strict cell-local lambda_ci crossing; increase max_cells or change seed_cell.')
    crossing_count = 0
    for start in range(0, grid.n_cells, 4096):
        values = strength[ids[start:start+4096]]
        crossing_count += int(np.count_nonzero((values.min(axis=1) < fallback) &
                                               (fallback < values.max(axis=1))))
    return fallback, crossing_count


def compare_grids(expected, actual):
    """Exact semantic equality; only connectivity integer storage width may vary."""
    validate_grid(actual)
    if (actual.n_points,actual.n_cells) != (expected.n_points,expected.n_cells):
        raise ValueError('Round-trip point/cell counts changed.')
    for name in ('points','cell_connectivity','offset','celltypes'):
        a,b = np.asarray(getattr(expected,name)),np.asarray(getattr(actual,name))
        if not np.array_equal(a,b) or (name in ('points','celltypes') and a.dtype != b.dtype):
            raise ValueError(f'Round-trip changed {name}.')
    for association in ('point_data','cell_data','field_data'):
        a,b = getattr(expected,association),getattr(actual,association)
        if set(a) != set(b):
            raise ValueError(f'Round-trip changed {association} names.')
        for name in a:
            if a[name].dtype != b[name].dtype or not np.array_equal(a[name],b[name],equal_nan=True):
                raise ValueError(f'Round-trip changed {association}/{name}.')
    if _roles(actual) != _roles(expected):
        raise ValueError('Round-trip changed active attribute roles.')
    if _component_names(actual) != _component_names(expected):
        raise ValueError('Round-trip changed array component names.')


def _blocks(grid, config):
    width = config.metadata()['id_bits']//8
    return [('points',int(grid.points.nbytes)), ('connectivity',8*grid.n_cells*width),
            ('offsets',grid.n_cells*width), ('types',grid.n_cells),
            *[(f'{a}/{n}',int(v.nbytes)) for a,n,v in _arrays(grid)]]


def write_variant(path, grid, config):
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
    path=Path(path)
    if path.suffix != '.vtu' or path.resolve().is_relative_to(REPOSITORY) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Require a new external .vtu without symlink ancestry.')
    validate_grid(grid)
    check_storage_limits(grid.n_points,grid.n_cells,_blocks(grid,config),config)
    # Shallow-copy data arrays, independently copy topology for storage changes.
    working = grid.copy(deep=False)
    cells = grid.GetCells().NewInstance()
    cells.DeepCopy(grid.GetCells())
    convert = cells.ConvertTo32BitStorage if config.profile == 'appended-int32' else cells.ConvertTo64BitStorage
    if not convert():
        raise ValueError('VTK could not represent the requested cell storage.')
    working.SetCells(12,cells)
    writer = vtkXMLUnstructuredGridWriter()
    settings = config.metadata()
    writer.SetInputData(working)
    writer.SetByteOrderToLittleEndian()
    getattr(writer,f'SetHeaderTypeToUInt{config.header_bits}')()
    getattr(writer,f'SetIdTypeToInt{settings["id_bits"]}')()
    writer.SetDataModeToBinary() if config.profile == 'reference' else writer.SetDataModeToAppended()
    writer.SetEncodeAppendedData(config.encoding == 'base64')
    if settings['compressor']:
        writer.SetCompressorTypeToZLib()
        writer.SetCompressionLevel(5)
    else:
        writer.SetCompressorTypeToNone()
    with Path(path).open('xb'):
        pass
    writer.SetFileName(str(path))
    if writer.Write() != 1 or writer.GetErrorCode():
        raise OSError('VTK diagnostic writer failed; staging retained.')
    identity = regular_file_identity(path)
    actual = _read_grid(path,identity)
    compare_grids(grid,actual)
    header = inspect_vtu(path)
    root=header['vtk_file']
    if (root.get('header_type') != f'UInt{config.header_bits}' or root.get('byte_order') != 'LittleEndian' or
            root.get('compressor') != ('vtkZLibDataCompressor' if settings['compressor'] else None)):
        raise ValueError('VTU headers differ from the requested configuration.')
    if config.profile != 'reference' and header['appended'] != {'encoding':config.encoding}:
        raise ValueError('Appended encoding differs from requested configuration.')
    expected_format = settings['mode']
    if any(a.get('format') != expected_format for a in header['arrays']):
        raise ValueError('VTU array format differs from requested configuration.')
    for a in header['arrays']:
        if a.get('Name') in ('connectivity','offsets') and a['type'] != f'Int{settings["id_bits"]}':
            raise ValueError('VTK did not write the requested connectivity integer precision.')
    if regular_file_identity(path) != identity:
        raise RuntimeError('Diagnostic VTU changed after round-trip validation.')
    header['validated_file_identity']=identity
    return header


def _destination(source, output):
    output = Path(output).expanduser().absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Diagnostic output ancestry must not contain symlinks.')
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing diagnostic output: {output}')
    output = output.resolve()
    protected = [REPOSITORY,source.parent,*[p for p in source.parents if p.name.startswith('case_')]]
    if any(output.is_relative_to(p) for p in protected):
        raise ValueError('Diagnostic output must be outside Git and the source scientific directories.')
    for parent in output.parents:
        if not parent.exists():
            continue
        if not parent.is_dir() or any(p.suffix in ('.vtu','.vtr','.pvd','.nek5000') or
            (p.name.startswith('GC') and p.suffix.startswith('.f')) for p in parent.iterdir()):
            raise ValueError('Do not nest diagnostics under an existing scientific or CFD directory.')
    return output


def _estimates(header, file_bytes, output_cells=None, *, add_time_alias=False):
    """Conservative allocation allowances, not measured peaks or reservations.

    Keep the original source-decode allowance and add the larger selected-mesh
    phase. Unique-ID sorting allows eight int64 workspaces per cell vertex;
    writing allows five output-array sets plus private topology and mappings.
    All arithmetic uses Python integers; no mesh is allocated here.
    """
    cells=header['cells'] if output_cells is None else min(header['cells'],output_cells)
    points=header['points'] if output_cells is None else min(header['points'],8*cells)
    sizes=[];output_sizes=[];output_blocks=[]
    for array in header['arrays']:
        if array.get('type') not in _SIZES:
            raise ValueError('This diagnostic supports numeric VTK arrays only; metadata will never be dropped.')
        association=array['association'];name=array.get('Name','')
        tuples = (header['points'] if association in ('Points','PointData') else
                  header['cells']*(8 if name=='connectivity' else 1) if association in ('Cells','CellData') else
                  int(array.get('NumberOfTuples','1')))
        components=int(array.get('NumberOfComponents','1'))
        if tuples < 0 or components < 1:
            raise ValueError('Invalid VTK array tuple/component counts.')
        sizes.append(tuples*components*_SIZES[array['type']])
        target=(points if association in ('Points','PointData') else
                cells*(8 if name=='connectivity' else 1) if association in ('Cells','CellData') else tuples)
        # A narrow source can be rewritten using Int64 topology; allow that growth.
        width=8 if association=='Cells' and name in ('connectivity','offsets') else _SIZES[array['type']]
        size=target*components*width
        output_sizes.append(size)
        output_blocks.append((f'{association}/{name}',size))
    if add_time_alias:
        output_sizes.append(8)
        output_blocks.append(('FieldData/time alias allowance',8))
    raw=sum(sizes)
    output=sum(output_sizes)
    mapping=0 if output_cells is None else 8*(points+cells)
    unique_scratch=0 if output_cells is None else 64*(8*cells)
    extraction=0 if output_cells is None else output+unique_scratch+72*cells+mapping
    roundtrip=5*output+72*cells+mapping
    selected_phase=max(extraction,roundtrip)
    # Twice raw output covers base64, compression expansion and length words;
    # include XML overhead and a separate compressed-NPZ allowance.
    vtu_file=2*output+1024**2+4096*len(output_blocks)
    mapping_file=0 if output_cells is None else 2*mapping+1024**2
    return dict(decoded_array_bytes=raw,source_decode_allowance_bytes=7*raw+2*file_bytes,
                extraction_working_bytes=extraction,unique_id_scratch_bytes=unique_scratch,
                roundtrip_working_bytes=roundtrip,mapping_array_upper_bytes=mapping,
                selected_phase_allowance_bytes=selected_phase,safety_reserve_bytes=1024**3,
                required_ram_bytes=7*raw+2*file_bytes+selected_phase+1024**3,
                remaining_ram_bytes=selected_phase+1024**3,
                output_array_upper_bytes=output,output_points_upper=points,output_cells=cells,
                output_blocks_upper=output_blocks,vtu_file_upper_bytes=vtu_file,
                mapping_file_upper_bytes=mapping_file,
                required_disk_bytes=vtu_file+mapping_file+128*1024**2,
                uncertainty='Unmeasured VTK copies/compression and viewer/isocontour memory; checks are not reservations.')


def _preflight_resources(estimates, output, *, after_decode=False):
    """Check current headroom before decode, extraction and publication staging."""
    available=available_memory_bytes()
    if isinstance(available,(bool,np.bool_)) or not isinstance(available,(int,np.integer)) or available < 0:
        raise MemoryError('Cannot establish available RAM.')
    address_limit,_=resource.getrlimit(resource.RLIMIT_AS)
    if address_limit != resource.RLIM_INFINITY:
        lines=Path('/proc/self/status').read_text().splitlines()
        virtual=next(int(line.split()[1])*1024 for line in lines if line.startswith('VmSize:'))
        available=min(int(available),max(0,address_limit-virtual))
    required=estimates['remaining_ram_bytes' if after_decode else 'required_ram_bytes']
    if required >= available:
        raise MemoryError(f'Insufficient RAM/address-space headroom: need {required} bytes, '
                          f'available {available}; reduce max_cells or use a larger-memory host.')
    disk=shutil.disk_usage(_nearest_directory(output)).free
    if estimates['required_disk_bytes'] >= disk:
        raise OSError(f'Insufficient diagnostic disk space: need {estimates["required_disk_bytes"]} bytes, '
                      f'available {disk}; choose another output filesystem or reduce max_cells.')
    file_limit,_=resource.getrlimit(resource.RLIMIT_FSIZE)
    largest=max(estimates['vtu_file_upper_bytes'],estimates['mapping_file_upper_bytes'])
    if file_limit != resource.RLIM_INFINITY and largest > file_limit:
        raise OSError(f'Diagnostic single-file allowance {largest} bytes exceeds process '
                      f'file-size limit {file_limit}; reduce max_cells or raise that limit.')
    return dict(available_ram_bytes=int(available),available_disk_bytes=int(disk))


def rewrite_vtu(source, output_dir, *, profile='reference', max_cells=4096, full_grid=False,
                seed_cell=None, appended_encoding=None, header_bits=None, add_time_alias=False, dry_run=False):
    """One configuration per invocation. Default: bounded whole-cell subset."""
    config=WriterConfiguration(profile,appended_encoding,header_bits)
    if type(full_grid) is not bool or type(add_time_alias) is not bool or type(dry_run) is not bool:
        raise ValueError('Diagnostic switches must be booleans.')
    if type(max_cells) is not int or not 1 <= max_cells <= MAX_SUBSET_CELLS:
        raise ValueError(f'max_cells must be in 1..{MAX_SUBSET_CELLS}.')
    if seed_cell is not None and (type(seed_cell) is not int or seed_cell < 0 or full_grid):
        raise ValueError('seed_cell must be a nonnegative integer and requires subset mode.')
    source=Path(source).expanduser().absolute()
    if source.suffix != '.vtu' or source.is_relative_to(REPOSITORY):
        raise ValueError('Require an existing external .vtu, never a Nek snapshot.')
    identity=regular_file_identity(source)
    if any(p.is_symlink() for p in source.parents):
        raise ValueError('Source ancestry must not contain symlinks.')
    source=source.resolve()
    if source.is_relative_to(REPOSITORY):
        raise ValueError('Source VTU must be outside Git.')
    output=_destination(source,output_dir)
    header=inspect_vtu(source)
    estimates=_estimates(header,identity[3],None if full_grid else max_cells,add_time_alias=add_time_alias)
    check_storage_limits(estimates['output_points_upper'],estimates['output_cells'],
                         estimates['output_blocks_upper'],config)
    resources=_preflight_resources(estimates,output)
    report=dict(workflow='vtu_serialization_diagnostic',source=str(source),output_directory=str(output),
                source_xml=header,writer=config.metadata(),selection_mode='full' if full_grid else 'whole-cell subset',
                max_cells=max_cells,seed_cell=seed_cell,add_time_alias=add_time_alias,estimates=estimates,
                **resources,dry_run=dry_run,
                production_writer_changed=False,tecplot_verified=False,paraview_verified=False)
    if dry_run:
        return report
    source_hash=regular_file_hash(source,expected=identity)
    grid=_read_grid(source,identity)
    if (grid.n_points,grid.n_cells) != (header['points'],header['cells']):
        raise ValueError('Decoded VTU counts differ from its XML header.')
    report['before_extraction_resources']=_preflight_resources(estimates,output,after_decode=True)
    if full_grid:
        selected=grid
        point_ids=cell_ids=None
    else:
        selected,point_ids,cell_ids=select_subset(grid,max_cells,seed_cell)
    del grid
    if add_time_alias:
        time=selected.field_data.get('time')
        if time is not None and (time.dtype != np.float64 or not np.array_equal(time,selected.field_data['TIME_VALUE'])):
            raise ValueError('Existing FieldData time conflicts with TIME_VALUE; refusing replacement.')
        selected=selected.copy(deep=False)
        selected.field_data['time']=selected.field_data['TIME_VALUE'].copy()
    check_storage_limits(selected.n_points,selected.n_cells,_blocks(selected,config),config)
    isovalue,crossing_cells=_crossing_isovalue(selected)
    report.update(points=selected.n_points,cells=selected.n_cells,bounds=list(selected.bounds),
                  time=float(selected.field_data['TIME_VALUE'][0]),
                  field_ranges={n:dict(min=float(selected[n].min()),max=float(selected[n].max())) for n in ('omega_x','lambda_ci')},
                  suggested_isovalue=isovalue,isovalue_crossing_cells=crossing_cells,
                  arrays={f'{a}/{n}':dict(dtype=v.dtype.str,shape=list(v.shape),sha256=array_fingerprint(np.ascontiguousarray(v)))
                          for a,n,v in _arrays(selected)},source_sha256=source_hash,
                  software={n:version(n) for n in ('numpy','pyvista','vtk')},
                  implementation_sha256=regular_file_hash(Path(__file__)))
    sidecar=source.with_suffix('.json')
    sidecar_identity=None
    if sidecar.exists() or sidecar.is_symlink():
        sidecar_identity=regular_file_identity(sidecar)
        if sidecar_identity[3] > 64*1024**2:
            raise ValueError('Oversized source sidecar.')
        with open_regular_file(sidecar,expected=sidecar_identity) as stream:
            metadata=json.loads(stream.read(64*1024**2+1))
        if not isinstance(metadata,dict):
            raise ValueError('Source sidecar must be a JSON object.')
        checksum=metadata.get('output_sha256')
        checksum=checksum.get('vtu') if isinstance(checksum,dict) else checksum
        if checksum is not None and checksum != source_hash:
            raise ValueError('Source sidecar VTU checksum differs from the actual source.')
        if 'time' in metadata and metadata['time'] != report['time']:
            raise ValueError('Source sidecar time differs from TIME_VALUE.')
        report['source_sidecar']=dict(path=str(sidecar),sha256=regular_file_hash(sidecar,expected=sidecar_identity),metadata=metadata)
    _destination(source,output)
    report['before_writing_resources']=_preflight_resources(estimates,output,after_decode=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.vtu-diagnostic-',dir=output.parent))
    label=f'diagnostic__{profile}__{config.encoding}__h{config.header_bits}'+('__time' if add_time_alias else '')
    vtu=stage/(label+'.vtu')
    report['output_xml']=write_variant(vtu,selected,config)
    validated_identity=report['output_xml']['validated_file_identity']
    report.update(completion_status='completed',roundtrip_exact=True,file_bytes=validated_identity[3],
                  output_filename=vtu.name,output_sha256=regular_file_hash(vtu,expected=validated_identity))
    if point_ids is not None:
        mapping=stage/'selection.npz'
        with mapping.open('xb') as stream:
            np.savez_compressed(stream,source_point_ids=point_ids,source_cell_ids=cell_ids)
        report['selection_mapping_sha256']=regular_file_hash(mapping)
    paths=list(stage.iterdir())
    identities={p:regular_file_identity(p) for p in paths}
    identities[vtu]=validated_identity
    if regular_file_identity(source) != identity or regular_file_hash(source,expected=identity) != source_hash:
        raise RuntimeError('Source changed; diagnostic staging retained.')
    if sidecar_identity is not None and regular_file_identity(sidecar) != sidecar_identity:
        raise RuntimeError('Source sidecar changed; staging retained.')
    if regular_file_hash(Path(__file__)) != report['implementation_sha256']:
        raise RuntimeError('Diagnostic implementation changed; staging retained.')
    manifest=vtu.with_suffix('.json')
    with manifest.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    identities[manifest]=regular_file_identity(manifest)
    _destination(source,output)
    if set(stage.iterdir()) != set(identities) or any(regular_file_identity(p)!=stamp for p,stamp in identities.items()):
        raise RuntimeError('Diagnostic artifacts changed before publication; staging retained.')
    _publish_directory(stage,output)
    return report
