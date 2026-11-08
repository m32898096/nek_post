"""Serialization diagnostics: independent synthetic geometry and field values."""
from hashlib import sha256
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import resource
import shutil

import numpy as np
import pyvista as pv
import pytest

from nek_post import vtu_compat as compat


@pytest.fixture
def source(tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'available_memory_bytes',lambda:10**15)
    parent=tmp_path/'scientific_input';parent.mkdir()
    grid=pv.StructuredGrid(*np.meshgrid([0.,.15,.6,1.],[0.,.4,1.5],[0.,.2,1.],indexing='ij')).cast_to_unstructured_grid()
    x,y,z=grid.points.T
    grid['omega_x']=(3*x-2*y+z).astype(np.float64)
    grid['lambda_ci']=(x*x+y+.1*z).astype(np.float64)
    grid.point_data['velocity_metadata']=np.column_stack((x,y,z))
    for i,label in enumerate(('x-component','y-component','z-component')):
        grid.GetPointData().GetArray('velocity_metadata').SetComponentName(i,label)
    grid.set_active_vectors('velocity_metadata')
    grid.set_active_scalars('omega_x')
    grid.cell_data['original_tag']=np.arange(grid.n_cells,dtype=np.int32)
    grid.field_data['TIME_VALUE']=np.array([19.50240408807],dtype=np.float64)
    grid.field_data['source_order']=np.array([7],dtype=np.int32)
    path=parent/'reference.vtu';grid.save(path,binary=True)
    path.with_suffix('.json').write_text(json.dumps(dict(configuration={'sampling_nodes':10},note='preserve provenance')))
    return path


CASES=[('reference',None,32),
       *[(profile,encoding,bits) for profile in ('appended-zlib','appended-none')
         for encoding in ('raw','base64') for bits in (32,64)],
       *[('appended-int32',encoding,32) for encoding in ('raw','base64')]]


def assert_nonempty_contour(report, grid):
    """Check the actual read-back mesh with VTK's independent contour filter."""
    level=report['suggested_isovalue']
    values=grid.point_data['lambda_ci'][grid.cell_connectivity.reshape(-1,8)]
    crossings=(values.min(axis=1)<level)&(level<values.max(axis=1))
    assert int(crossings.sum())==report['isovalue_crossing_cells']>0
    surface=grid.contour([level],scalars='lambda_ci')
    assert surface.n_points>0 and surface.n_cells>0


def band_grid(bands, positions=None):
    """Independent VTK Hex convention; each cell has a linear-in-x scalar."""
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float64)
    if positions is None:positions=3*np.arange(len(bands))
    points=np.concatenate([corners+[x,0,0] for x in positions])
    cells=np.column_stack((np.full(len(bands),8),np.arange(8*len(bands)).reshape(-1,8)))
    grid=pv.UnstructuredGrid(cells.ravel(),np.full(len(bands),12,dtype=np.uint8),points)
    grid['lambda_ci']=np.concatenate([low+(high-low)*corners[:,0] for low,high in bands])
    grid['omega_x']=2*grid['lambda_ci']
    grid.field_data['TIME_VALUE']=np.array([.5],dtype=np.float64)
    return grid


@pytest.mark.parametrize('profile,encoding,bits',CASES)
def test_all_configurations_preserve_full_grid_exactly(source,tmp_path,profile,encoding,bits):
    before={p:sha256(p.read_bytes()).hexdigest() for p in source.parent.iterdir()}
    report=compat.rewrite_vtu(source,tmp_path/'output',profile=profile,appended_encoding=encoding,
                              header_bits=bits,full_grid=True)
    saved=Path(report['output_directory'])/report['output_filename']
    original,actual=pv.read(source),pv.read(saved)
    assert_nonempty_contour(report,actual)
    compat.compare_grids(original,actual)
    np.testing.assert_array_equal(actual.points,original.points)
    np.testing.assert_array_equal(actual.cell_connectivity,original.cell_connectivity)
    assert actual['omega_x'].dtype==actual['lambda_ci'].dtype==np.float64
    assert list(actual.bounds)==[0.,1.,0.,1.5,0.,1.]
    assert report['roundtrip_exact'] is True
    assert report['time']==19.50240408807
    assert report['source_sidecar']['metadata']['configuration']['sampling_nodes']==10
    assert report['output_sha256']==sha256(saved.read_bytes()).hexdigest()
    assert report['output_xml']['vtk_file']['header_type']==f'UInt{bits}'
    assert report['output_xml']['appended']==(None if profile=='reference' else {'encoding':encoding})
    arrays={a.get('Name'):a for a in report['output_xml']['arrays']}
    assert arrays['connectivity']['type']==arrays['offsets']['type']==('Int32' if profile=='appended-int32' else 'Int64')
    assert report['tecplot_verified'] is report['paraview_verified'] is False
    assert {p:sha256(p.read_bytes()).hexdigest() for p in before}==before
    assert saved.with_suffix('.json').is_file()


@pytest.mark.parametrize('profile',compat.PROFILES)
def test_subset_maps_entire_original_cells_without_interpolation(source,tmp_path,profile):
    report=compat.rewrite_vtu(source,tmp_path/'subset',profile=profile,max_cells=5)
    original=pv.read(source)
    selected=pv.read(Path(report['output_directory'])/report['output_filename'])
    with np.load(Path(report['output_directory'])/'selection.npz',allow_pickle=False) as mapping:
        points,cells=mapping['source_point_ids'],mapping['source_cell_ids']
        assert len(cells)==5
        np.testing.assert_array_equal(selected.points,original.points[points])
        np.testing.assert_array_equal(points[selected.cell_connectivity.reshape(-1,8)],
                                      original.cell_connectivity.reshape(-1,8)[cells])
        for name in original.point_data:
            np.testing.assert_array_equal(selected[name],original[name][points])
        np.testing.assert_array_equal(selected.cell_data['original_tag'],original.cell_data['original_tag'][cells])
    assert selected.n_cells==5 and np.all(selected.celltypes==pv.CellType.HEXAHEDRON)
    assert np.all(np.ptp(selected.points[selected.cell_connectivity.reshape(-1,8)],axis=1)>0)
    assert_nonempty_contour(report,selected)
    assert compat._roles(selected)==compat._roles(original)
    # Rewriting the small reference as a whole grid preserves its provenance chain.
    rewritten=compat.rewrite_vtu(Path(report['output_directory'])/report['output_filename'],
                                 tmp_path/'again',profile='appended-none',full_grid=True)
    assert rewritten['source_sidecar']['metadata']['source_sha256']==report['source_sha256']


@pytest.mark.parametrize('profile',compat.PROFILES)
def test_time_alias_is_opt_in_preserves_time_and_rejects_conflicts(source,tmp_path,profile):
    report=compat.rewrite_vtu(source,tmp_path/'alias',profile=profile,full_grid=True,add_time_alias=True)
    actual=pv.read(Path(report['output_directory'])/report['output_filename'])
    np.testing.assert_array_equal(actual.field_data['time'],actual.field_data['TIME_VALUE'])
    assert actual.field_data['time'].dtype==np.float64
    assert 'time' not in pv.read(source).field_data
    changed=pv.read(source);changed.field_data['time']=np.array([1.]);changed.save(source,binary=True)
    with pytest.raises(ValueError,match='conflicts'):
        compat.rewrite_vtu(source,tmp_path/'conflict',profile=profile,full_grid=True,add_time_alias=True)
    assert not (tmp_path/'conflict').exists()


def test_dry_run_is_header_only_and_does_not_create_files(source,tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'_read_grid',lambda *a:pytest.fail('dry run decoded payload'))
    report=compat.rewrite_vtu(source,tmp_path/'none',dry_run=True)
    assert report['source_xml']['cells']==12
    assert report['source_xml']['points']==36
    assert report['source_xml']['vtk_file']['compressor']=='vtkZLibDataCompressor'
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('profile',compat.PROFILES)
def test_collision_fails_without_changes(source,tmp_path,profile):
    output=tmp_path/'existing';output.mkdir();protected=output/'scientific.vtu';protected.write_bytes(b'keep')
    with pytest.raises(FileExistsError):compat.rewrite_vtu(source,output,profile=profile)
    assert protected.read_bytes()==b'keep'


@pytest.mark.parametrize('kind',['repository','source','raw','symlink','input_symlink'])
def test_path_protection(source,tmp_path,kind):
    output=tmp_path/'unsafe'
    if kind=='repository':output=compat.REPOSITORY/'forbidden-phase4'
    if kind=='source':output=source.parent/'diagnostics'
    if kind=='raw':
        parent=tmp_path/'raw_case';parent.mkdir();(parent/'GC0.f00001').write_bytes(b'not read')
        output=parent/'diagnostics'
    if kind=='symlink':output.symlink_to(source.parent,target_is_directory=True)
    if kind=='input_symlink':
        alias=tmp_path/'alias.vtu';alias.symlink_to(source);source=alias
    with pytest.raises((ValueError,FileExistsError)):compat.rewrite_vtu(source,output)


@pytest.mark.parametrize('option,value',[('max_cells',0),('max_cells',True),('max_cells',1000001),
    ('max_cells',-1),('max_cells',1.5),('max_cells',None),('max_cells',10**100),
    ('seed_cell',-1),('seed_cell',True),('profile','unsupported'),('appended_encoding','ascii'),
    ('header_bits',16),('header_bits',True)])
def test_invalid_configuration_fails_before_input_read(tmp_path,monkeypatch,option,value):
    monkeypatch.setattr(compat,'inspect_vtu',lambda *a:pytest.fail('invalid options reached input'))
    with pytest.raises(ValueError):compat.rewrite_vtu(tmp_path/'missing.vtu',tmp_path/'none',**{option:value})


@pytest.mark.parametrize('points,cells,blocks,message',[
    (2**31+1,1,[('x',8)],'Int32'),(8,2**28,[('x',8)],'Int32'),
    (8,1,[('x',2**32)],'UInt32'),(8,1,[('a',2**31),('b',2**31)],'UInt32'),
    (2**63+1,1,[('x',8)],'topology')])
def test_integer_capacity_checks_without_large_allocations(points,cells,blocks,message):
    with pytest.raises(ValueError,match=message):
        compat.check_storage_limits(points,cells,blocks,compat.WriterConfiguration('appended-int32'))


def test_uint64_headers_permit_large_blocks_but_do_not_narrow_ids():
    config=compat.WriterConfiguration('appended-none',header_bits=64)
    compat.check_storage_limits(2**31+1,2**28,[('large',2**33)],config)
    assert config.metadata()['id_bits']==64


def test_subset_disk_estimates_shrink_but_full_source_decode_allowance_is_retained(source):
    header=compat.inspect_vtu(source)
    full=compat._estimates(header,source.stat().st_size)
    subset=compat._estimates(header,source.stat().st_size,1)
    assert subset['required_ram_bytes']<full['required_ram_bytes']
    assert subset['source_decode_allowance_bytes']==full['source_decode_allowance_bytes']
    assert subset['decoded_array_bytes']==full['decoded_array_bytes']
    # Tiny subsets can have a larger total allowance because they add selection.npz.
    assert subset['vtu_file_upper_bytes']<full['vtu_file_upper_bytes']
    assert subset['required_disk_bytes']-subset['mapping_file_upper_bytes']<full['required_disk_bytes']
    assert subset['output_cells']==1 and subset['output_points_upper']==8


@pytest.mark.parametrize('limit',[100000,100001,1000000])
@pytest.mark.parametrize('profile,encoding,bits',CASES)
def test_large_limits_preserve_small_fixture_exactly(source,tmp_path,limit,profile,encoding,bits):
    # Exercise the actual CLI/API limit and all profiles without allocating a large mesh.
    before=sha256(source.read_bytes()).hexdigest()
    report=compat.rewrite_vtu(source,tmp_path/'large_limit',max_cells=limit,
                              profile=profile,appended_encoding=encoding,header_bits=bits)
    original=pv.read(source)
    actual=pv.read(Path(report['output_directory'])/report['output_filename'])
    assert report['max_cells']==limit and report['cells']==original.n_cells
    assert report['estimates']['output_cells']==original.n_cells
    np.testing.assert_array_equal(actual.points,original.points)
    np.testing.assert_array_equal(actual.cell_connectivity,original.cell_connectivity)
    np.testing.assert_array_equal(actual.celltypes,original.celltypes)
    for name in ('omega_x','lambda_ci'):
        np.testing.assert_array_equal(actual[name],original[name])
        assert actual[name].dtype==np.float64
    np.testing.assert_array_equal(actual.field_data['TIME_VALUE'],original.field_data['TIME_VALUE'])
    assert_nonempty_contour(report,actual)
    assert sha256(source.read_bytes()).hexdigest()==before


def large_header():
    """Pure synthetic header with eight distinct vertices per cell (worst case)."""
    return dict(points=16000000,cells=2000000,arrays=[
        dict(association='Points',type='Float64',NumberOfComponents='3'),
        *[dict(association='PointData',Name=name,type='Float64') for name in ('omega_x','lambda_ci')],
        dict(association='Cells',Name='connectivity',type='Int64'),
        dict(association='Cells',Name='offsets',type='Int64'),
        dict(association='Cells',Name='types',type='UInt8'),
        dict(association='FieldData',Name='TIME_VALUE',type='Float64',NumberOfTuples='1')])


@pytest.mark.parametrize('limit',[100000,1000000])
def test_large_subset_estimates_count_every_allocation_without_a_mesh(limit):
    estimates=compat._estimates(large_header(),1234,limit)
    # 8 vertices*(24 coordinate+16 field bytes) + 64 connectivity+8 offset+1 type.
    expected=393*limit+8
    assert estimates['output_array_upper_bytes']==expected
    assert estimates['output_cells']==limit and estimates['output_points_upper']==8*limit
    assert estimates['unique_id_scratch_bytes']==512*limit
    assert estimates['mapping_array_upper_bytes']==72*limit
    assert estimates['extraction_working_bytes']==expected+656*limit
    assert estimates['roundtrip_working_bytes']==5*expected+144*limit
    assert estimates['required_ram_bytes']==(estimates['source_decode_allowance_bytes']+
        max(estimates['extraction_working_bytes'],estimates['roundtrip_working_bytes'])+1024**3)
    assert estimates['remaining_ram_bytes']==estimates['required_ram_bytes']-estimates['source_decode_allowance_bytes']
    assert estimates['required_disk_bytes']==(estimates['vtu_file_upper_bytes']+
        estimates['mapping_file_upper_bytes']+128*1024**2)
    compat.check_storage_limits(estimates['output_points_upper'],estimates['output_cells'],
                               estimates['output_blocks_upper'],compat.WriterConfiguration('reference'))


def test_large_request_dry_run_is_header_only_and_reports_growing_resources(source,tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'inspect_vtu',lambda p:large_header())
    monkeypatch.setattr(compat,'_read_grid',lambda *a:pytest.fail('dry run read full grid'))
    monkeypatch.setattr(compat.shutil,'disk_usage',lambda p:type('D',(),{'free':10**15})())
    small=compat.rewrite_vtu(source,tmp_path/'100k',max_cells=100000,dry_run=True)
    large=compat.rewrite_vtu(source,tmp_path/'1m',max_cells=1000000,dry_run=True)
    for key in ('required_ram_bytes','required_disk_bytes','output_array_upper_bytes'):
        assert large['estimates'][key]>small['estimates'][key]
    assert large['estimates']['output_cells']==1000000
    assert not (tmp_path/'100k').exists() and not (tmp_path/'1m').exists()


@pytest.mark.parametrize('kind',['ram','disk','file_limit','storage'])
def test_large_request_is_rejected_before_decode_or_extraction(source,tmp_path,monkeypatch,kind):
    header=large_header()
    estimate=compat._estimates(header,source.stat().st_size,1000000)
    monkeypatch.setattr(compat,'inspect_vtu',lambda p:header)
    monkeypatch.setattr(compat,'_read_grid',lambda *a:pytest.fail('unsafe request reached decode'))
    monkeypatch.setattr(compat,'select_subset',lambda *a:pytest.fail('unsafe request reached extraction'))
    monkeypatch.setattr(compat.shutil,'disk_usage',lambda p:type('D',(),{'free':10**15})())
    if kind=='ram':monkeypatch.setattr(compat,'available_memory_bytes',lambda:estimate['required_ram_bytes']-1)
    if kind=='disk':monkeypatch.setattr(compat.shutil,'disk_usage',lambda p:type('D',(),{'free':estimate['required_disk_bytes']-1})())
    if kind=='file_limit':
        getlimit=resource.getrlimit
        monkeypatch.setattr(compat.resource,'getrlimit',lambda r:(1,1) if r==resource.RLIMIT_FSIZE else getlimit(r))
    if kind=='storage':header['arrays'][0]['NumberOfComponents']='1000'
    error,message={'ram':(MemoryError,'headroom'),'disk':(OSError,'disk space'),
                   'file_limit':(OSError,'file-size limit'),'storage':(ValueError,'UInt32')}[kind]
    with pytest.raises(error,match=message):
        compat.rewrite_vtu(source,tmp_path/'unsafe',max_cells=1000000)
    assert not (tmp_path/'unsafe').exists() and not list(tmp_path.glob('.vtu-diagnostic-*'))


@pytest.mark.parametrize('stage',['extraction','writing'])
@pytest.mark.parametrize('kind',['ram','disk'])
def test_resource_headroom_is_rechecked_before_each_large_phase(source,tmp_path,monkeypatch,stage,kind):
    checks=iter([10**15,1] if stage=='extraction' else [10**15,10**15,1])
    if kind=='ram':monkeypatch.setattr(compat,'available_memory_bytes',lambda:next(checks))
    else:monkeypatch.setattr(compat.shutil,'disk_usage',lambda p:type('D',(),{'free':next(checks)})())
    if stage=='extraction':monkeypatch.setattr(compat,'select_subset',lambda *a:pytest.fail('low headroom reached extraction'))
    monkeypatch.setattr(compat,'write_variant',lambda *a:pytest.fail('low headroom reached writing'))
    with pytest.raises((OSError,MemoryError)):
        compat.rewrite_vtu(source,tmp_path/'unsafe',max_cells=1000000)
    assert not (tmp_path/'unsafe').exists() and not list(tmp_path.glob('.vtu-diagnostic-*'))


def test_address_space_limit_is_checked_before_decoding(source,tmp_path,monkeypatch):
    getlimit=resource.getrlimit
    monkeypatch.setattr(compat.resource,'getrlimit',lambda r:(1,1) if r==resource.RLIMIT_AS else getlimit(r))
    monkeypatch.setattr(compat,'_read_grid',lambda *a:pytest.fail('address-space limit reached reader'))
    with pytest.raises(MemoryError,match='address-space'):
        compat.rewrite_vtu(source,tmp_path/'unsafe',max_cells=1000000)
    assert not (tmp_path/'unsafe').exists()


def test_subset_owns_selected_arrays_after_redundant_copies_are_removed(source):
    original=pv.read(source)
    selected,points,_=compat.select_subset(original,5)
    for name in ('omega_x','lambda_ci'):
        assert not np.shares_memory(selected[name],original[name])
        expected=selected[name].copy()
        original[name][points]=-123.
        np.testing.assert_array_equal(selected[name],expected)
    assert not np.shares_memory(selected.points,original.points)
    original.points[:]=-123.
    assert not np.any(selected.points==-123.)
    original.field_data['TIME_VALUE'][0]=-123.
    assert selected.field_data['TIME_VALUE'][0]==19.50240408807


@pytest.mark.parametrize('damage',['float32','nan','negative','missing','time','geometry','ghost','strings','zero','constant'])
def test_invalid_scientific_input_is_rejected(source,tmp_path,damage):
    grid=pv.read(source)
    if damage=='float32':grid['omega_x']=grid['omega_x'].astype(np.float32)
    if damage=='nan':grid['omega_x'][0]=np.nan
    if damage=='negative':grid['lambda_ci'][0]=-1
    if damage=='missing':del grid.point_data['lambda_ci']
    if damage=='time':del grid.field_data['TIME_VALUE']
    if damage=='geometry':grid.points[:,0]*=-1
    if damage=='ghost':grid.cell_data['vtkGhostType']=np.ones(grid.n_cells,dtype=np.uint8)
    if damage=='strings':grid.field_data['label']=np.array(['metadata that must not be dropped'])
    if damage=='zero':grid['lambda_ci'][:]=0
    if damage=='constant':grid['lambda_ci'][:]=1
    grid.save(source,binary=True)
    with pytest.raises(ValueError):compat.rewrite_vtu(source,tmp_path/'none')
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('kind',['ram','unknown_ram','disk','file_limit'])
def test_resource_failures_stop_safely(source,tmp_path,monkeypatch,kind):
    if kind=='ram':monkeypatch.setattr(compat,'available_memory_bytes',lambda:1)
    if kind=='unknown_ram':monkeypatch.setattr(compat,'available_memory_bytes',lambda:None)
    if kind=='disk':monkeypatch.setattr(compat.shutil,'disk_usage',lambda p:type('D',(),{'free':1})())
    if kind=='file_limit':
        getlimit=resource.getrlimit
        monkeypatch.setattr(compat.resource,'getrlimit',lambda r:(1,1) if r==resource.RLIMIT_FSIZE else getlimit(r))
    with pytest.raises((OSError,MemoryError)):compat.rewrite_vtu(source,tmp_path/'none')
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('failure',['writer','readback','source_change','publication_collision'])
def test_faults_preserve_source_and_do_not_publish_partial_outputs(source,tmp_path,monkeypatch,failure):
    output=tmp_path/'output'
    original=compat.write_variant
    before=sha256(source.read_bytes()).hexdigest()
    def inject(path,grid,config):
        if failure=='writer':
            Path(path).write_bytes(b'partial diagnostic');raise OSError('interrupted')
        header=original(path,grid,config)
        if failure=='readback':raise ValueError('failed readback')
        if failure=='source_change':
            with source.open('ab') as stream:stream.write(b'changed synthetic source')
        if failure=='publication_collision':
            output.mkdir();(output/'protected.vtu').write_bytes(b'keep')
        return header
    monkeypatch.setattr(compat,'write_variant',inject)
    with pytest.raises((ValueError,OSError,RuntimeError)):compat.rewrite_vtu(source,output)
    if failure!='source_change':assert sha256(source.read_bytes()).hexdigest()==before
    if failure=='publication_collision':assert (output/'protected.vtu').read_bytes()==b'keep'
    else:assert not output.exists()
    assert list(tmp_path.glob('.vtu-diagnostic-*'))


@pytest.mark.parametrize('kind',['symlink','regular_copy'])
def test_substitution_after_output_validation_cannot_be_published(source,tmp_path,monkeypatch,kind):
    original=compat.write_variant
    def substitute(path,grid,config):
        report=original(path,grid,config)
        preserved=tmp_path/'preserved.vtu'
        path.rename(preserved)
        if kind=='symlink':path.symlink_to(preserved)
        else:shutil.copy2(preserved,path)
        return report
    monkeypatch.setattr(compat,'write_variant',substitute)
    with pytest.raises(ValueError):compat.rewrite_vtu(source,tmp_path/'none')
    assert not (tmp_path/'none').exists()
    assert (tmp_path/'preserved.vtu').is_file()


def test_diagnostic_never_calls_nek_readers_or_numerical_operators(source,tmp_path,monkeypatch):
    from nek_post import io_nek,gll,l2_projection,spectral_derivatives,swirling_strength
    def forbidden(*args,**kwargs):pytest.fail('serialization diagnostic called a numerical operator')
    for module,name in ((io_nek,'read_nek_file'),(gll,'gll_interpolation_matrix'),
                        (l2_projection,'project_velocity_gradient'),(l2_projection,'project_scalar_field'),
                        (spectral_derivatives,'element_velocity_gradient'),(swirling_strength,'compute_swirling_strength')):
        monkeypatch.setattr(module,name,forbidden)
    assert compat.rewrite_vtu(source,tmp_path/'only_serialization')['roundtrip_exact']


@pytest.mark.parametrize('damage',['points','connectivity','scalar','time','metadata','extra','roles'])
def test_comparison_detects_serialization_changes(source,damage):
    original=pv.read(source);changed=original.copy()
    if damage=='points':changed.points[0,0]+=.01
    if damage=='connectivity':changed.cell_connectivity[:2]=changed.cell_connectivity[1::-1]
    if damage=='scalar':changed['omega_x'][0]+=.01
    if damage=='time':changed.field_data['TIME_VALUE'][0]+=.01
    if damage=='metadata':changed.field_data['source_order'][0]=6
    if damage=='extra':changed.cell_data['extra']=np.ones(changed.n_cells)
    if damage=='roles':changed.set_active_scalars('lambda_ci')
    with pytest.raises(ValueError):compat.compare_grids(original,changed)


def test_raw_appended_header_inspection_stops_before_non_xml_payload(tmp_path):
    path=tmp_path/'header.vtu'
    path.write_bytes(b'<VTKFile type="UnstructuredGrid"><UnstructuredGrid><Piece NumberOfPoints="8" NumberOfCells="1"/></UnstructuredGrid><AppendedData encoding="raw">_'+b'\x00\xff'*100000)
    header=compat.inspect_vtu(path)
    assert header['appended']=={'encoding':'raw'} and header['payload_validated'] is False


def test_coincident_element_local_points_and_discontinuous_fields_are_never_merged(source,tmp_path):
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float64)
    points=np.concatenate((corners,corners+[1.,0.,0.]))
    packed=np.column_stack((np.full(2,8),np.arange(16).reshape(2,8)))
    grid=pv.UnstructuredGrid(packed.ravel(),np.full(2,12,dtype=np.uint8),points)
    grid['omega_x']=np.arange(16,dtype=np.float64)-8.
    grid['lambda_ci']=np.arange(16,dtype=np.float64)/4
    grid.field_data['TIME_VALUE']=np.array([.25])
    path=source.parent/'discontinuous.vtu';grid.save(path,binary=True)
    report=compat.rewrite_vtu(path,tmp_path/'discontinuous',max_cells=2)
    actual=pv.read(Path(report['output_directory'])/report['output_filename'])
    assert actual.n_points==16 and len(np.unique(actual.points,axis=0))==12
    np.testing.assert_array_equal(actual['omega_x'],grid['omega_x'])
    np.testing.assert_array_equal(actual['lambda_ci'],grid['lambda_ci'])
    assert_nonempty_contour(report,actual)


@pytest.mark.parametrize('profile',compat.PROFILES)
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('full_grid',[False,True])
def test_disjoint_scalar_bands_choose_a_real_contour_in_either_cell_order(source,tmp_path,profile,reverse,full_grid):
    bands=[(0.,1.),(9.,10.)]
    if reverse:bands.reverse()
    grid=band_grid(bands)
    path=source.parent/'bands.vtu';grid.save(path,binary=True)
    before=sha256(path.read_bytes()).hexdigest()
    # The old global midpoint is strictly between global bounds but misses both cells.
    assert grid.contour([5.],scalars='lambda_ci').n_cells==0
    report=compat.rewrite_vtu(path,tmp_path/'bands',profile=profile,max_cells=2,full_grid=full_grid)
    actual=pv.read(Path(report['output_directory'])/report['output_filename'])
    assert report['suggested_isovalue']==.5
    assert_nonempty_contour(report,actual)
    np.testing.assert_array_equal(actual.points,grid.points)
    np.testing.assert_array_equal(actual['lambda_ci'],grid['lambda_ci'])
    np.testing.assert_array_equal(actual.cell_connectivity,grid.cell_connectivity)
    assert sha256(path.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('order',list(permutations(range(3))))
def test_connected_source_with_permuted_storage_keeps_a_useful_subset(source,tmp_path,order):
    # Continuous piecewise-linear field on x=0..3, including an intermediate band.
    bands=[(0.,1.),(1.,9.),(9.,10.)]
    grid=band_grid([bands[i] for i in order],positions=order)
    path=source.parent/'permuted.vtu';grid.save(path,binary=True)
    assert grid.contour([5.],scalars='lambda_ci').n_cells>0
    report=compat.rewrite_vtu(path,tmp_path/'permuted',max_cells=2)
    actual=pv.read(Path(report['output_directory'])/report['output_filename'])
    assert_nonempty_contour(report,actual)
    with np.load(Path(report['output_directory'])/'selection.npz',allow_pickle=False) as mapping:
        np.testing.assert_array_equal(actual.points,grid.points[mapping['source_point_ids']])
        np.testing.assert_array_equal(actual['lambda_ci'],grid['lambda_ci'][mapping['source_point_ids']])


@pytest.mark.parametrize('full_grid',[False,True])
@pytest.mark.parametrize('kind',['different_constants','unused_point','adjacent_floats'])
def test_no_cell_local_interior_level_fails_before_staging(source,tmp_path,monkeypatch,full_grid,kind):
    if kind=='adjacent_floats':grid=band_grid([(1.,np.nextafter(1.,2.))])
    else:grid=band_grid([(0.,0.),(9.,9.)])
    if kind=='unused_point':
        grid=band_grid([(1.,1.)])
        grid.points=np.vstack((grid.points,[9.,9.,9.]))
        grid['lambda_ci']=np.r_[np.ones(8),10.]
        grid['omega_x']=np.r_[np.ones(8),20.]
    path=source.parent/'no_crossing.vtu';grid.save(path,binary=True)
    before=sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(compat,'write_variant',lambda *a:pytest.fail('unusable selection reached writer'))
    output=tmp_path/'no_crossing'
    with pytest.raises(ValueError,match='No usable strict cell-local'):
        compat.rewrite_vtu(path,output,max_cells=2,full_grid=full_grid,
                           seed_cell=0 if not full_grid else None)
    assert not output.exists() and not list(tmp_path.glob('.vtu-diagnostic-*'))
    assert sha256(path.read_bytes()).hexdigest()==before


def test_global_midpoint_is_retained_only_when_a_cell_crosses_it(source,tmp_path):
    original=pv.read(source)
    expected=float(original['lambda_ci'].max()/2)
    report=compat.rewrite_vtu(source,tmp_path/'midpoint',full_grid=True)
    assert report['suggested_isovalue']==expected
    assert_nonempty_contour(report,pv.read(Path(report['output_directory'])/report['output_filename']))


def test_cell_midpoint_rounding_and_large_finite_values():
    low=1.;high=np.nextafter(np.nextafter(low,2.),2.)
    grid=band_grid([(low,high)])
    level,count=compat._crossing_isovalue(grid)
    assert low<level<high and count==1
    # Midpoint addition must not overflow, even though low+high does.
    grid['lambda_ci']=np.where(grid.points[:,0]==0,1.e308,1.7e308)
    level,count=compat._crossing_isovalue(grid)
    assert np.isfinite(level) and 1.e308<level<1.7e308 and count==1


def test_crossing_scan_handles_chunk_boundaries_without_unused_point_evidence():
    # Only the last cell varies; earlier cells have separated constant bands.
    grid=band_grid([(0.,0.)]*4096+[(9.,10.)])
    level,count=compat._crossing_isovalue(grid)
    assert level==9.5 and count==1
    assert_nonempty_contour({'suggested_isovalue':level,'isovalue_crossing_cells':count},grid)


@pytest.mark.parametrize('damage',['checksum_dict','checksum_string','time'])
def test_source_sidecar_inconsistency_is_rejected_without_outputs(source,tmp_path,damage):
    metadata={'time':0.} if damage=='time' else {'output_sha256':{'vtu':'0'*64} if damage=='checksum_dict' else '0'*64}
    source.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='Source sidecar'):
        compat.rewrite_vtu(source,tmp_path/'inconsistent')
    assert not (tmp_path/'inconsistent').exists()


@pytest.mark.parametrize('contents',[b'broken',b'<VTKFile type="PolyData"/>',
    b'<!DOCTYPE VTKFile [<!ENTITY x "bad">]><VTKFile type="UnstructuredGrid"/>',
    b'<VTKFile type="UnstructuredGrid"><Piece NumberOfPoints="8" NumberOfCells="1"/><Piece NumberOfPoints="8" NumberOfCells="1"/></VTKFile>'])
def test_header_errors_fail_without_payload_decode(tmp_path,contents):
    path=tmp_path/'invalid.vtu';path.write_bytes(contents)
    with pytest.raises(ValueError):compat.inspect_vtu(path)


@pytest.mark.parametrize('limit',[5,100000,1000000])
def test_cli_help_and_small_execution(source,tmp_path,capsys,limit):
    path=compat.REPOSITORY/'scripts/42_rewrite_vtu_diagnostic.py'
    spec=importlib.util.spec_from_file_location('vtu_diag_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as info:cli.parse_args(['--help'])
    assert info.value.code==0
    assert '1..1000000' in capsys.readouterr().out
    assert cli.main([str(source),'--output-dir',str(tmp_path/'cli'),'--profile','appended-int32','--max-cells',str(limit)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['cells']==min(limit,12) and report['roundtrip_exact']
    assert cli.main([str(source),'--output-dir',str(tmp_path/'cli')])==1
