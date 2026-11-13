"""Small synthetic catalogs, source preservation, and publication boundaries."""
import csv
import io
import json
from pathlib import Path

import pytest

from nek_post.results_catalog import build_catalog, write_catalog, MAX_CONTROL_BYTES


def put(root, name, text='artifact'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def collection(root, folder='h_refinement/vortex_3d_series/gll10/case_N7_H'):
    put(root, folder+'/f00001/a.vtu')
    put(root, folder+'/f00079/a.vtu')
    put(root, folder+'/series/run.pvd', '<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="0.25" file="../f00001/a.vtu"/>'
        '<DataSet timestep="19.50240408807" file="../f00079/a.vtu"/>'
        '</Collection></VTKFile>')
    put(root, folder+'/series/processing_summary.json', json.dumps({'status':'completed',
        'configuration':{'sampling_nodes':10}, 'output_selection':{'fields':['omega_x','lambda_ci']}}))
    return folder


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_two_levels_and_actual_times(tmp_path):
    root = tmp_path/'results'
    folder = collection(root)
    put(root, 'poly_order_compare/tables/a.csv', 'name,value\na,1\n')
    before = snapshot(root)
    catalog = build_catalog(root, include_artifacts=True)
    assert len(catalog['datasets']) == 2
    row = next(r for r in catalog['datasets'] if r['analysis']=='vortex_3d_series')
    assert row['relative_path'] == folder
    assert row['frame_count'] == 2
    assert row['actual_time_end'] == 19.50240408807
    assert row['status'] == 'reported_completed'
    assert row['sampling_nodes'] == 10
    assert row['polynomial_degree'] is None
    assert row['field_names'] == ['omega_x','lambda_ci']
    assert len(catalog['artifacts']) == catalog['file_count'] == 5
    assert len({a['relative_path'] for a in catalog['artifacts']}) == 5
    assert catalog['logical_bytes'] == sum(len(v[0]) for v in before.values())
    assert snapshot(root) == before
    assert build_catalog(root, include_artifacts=True) == catalog
    assert build_catalog(root)['artifacts'] == []


@pytest.mark.parametrize('label,expected', [('archive','historical'),('smoke','smoke_test'),
                                           ('diagnostics','diagnostic'),('logs','diagnostic')])
def test_classes_are_navigation_labels(tmp_path,label,expected):
    put(tmp_path,f'poly_order_compare/{label}/N7/file.csv')
    row = build_catalog(tmp_path)['datasets'][0]
    assert row['artifact_class'] == expected
    assert row['status'] == 'unknown'
    assert row['validation_level'] == 'inventory_only'
    assert row['polynomial_degree'] is None


def test_conflicting_metadata_and_unusual_layout(tmp_path):
    put(tmp_path,'odd/place/run/a.json','{"polynomial_degree":7}')
    put(tmp_path,'odd/place/run/b.json','{"polynomial_degree":9}')
    put(tmp_path,'odd/place/run/unexpected.xyz')
    catalog=build_catalog(tmp_path)
    row=catalog['datasets'][0]
    assert row['polynomial_degree'] is None
    assert any('conflicting polynomial_degree' in n for n in catalog['warnings'])
    assert row['file_count']==3


@pytest.mark.parametrize('contents', ['{bad', '[]', '{"configuration":null}', 'x'*(MAX_CONTROL_BYTES+1)])
def test_malformed_bounded_metadata(tmp_path,contents):
    put(tmp_path,'study/analysis/data.json',contents)
    catalog=build_catalog(tmp_path)
    assert catalog['datasets'][0]['status']=='unknown'
    assert catalog['datasets'][0]['sampling_nodes'] is None


@pytest.mark.parametrize('reference', ['../../../outside.vtu','/etc/passwd','missing.vtu','link.vtu'])
def test_unsafe_or_missing_pvd(tmp_path,reference):
    root=tmp_path/'results'
    path=put(root,'study/analysis/run.pvd',f'<VTKFile type="Collection"><Collection><DataSet timestep="1" file="{reference}"/></Collection></VTKFile>')
    outside=put(tmp_path,'outside.vtu')
    (path.parent/'link.vtu').symlink_to(outside)
    catalog=build_catalog(root,include_artifacts=True)
    assert catalog['datasets'][0]['status']=='incomplete_or_ambiguous'
    assert catalog['datasets'][0]['actual_time_start'] is None
    assert catalog['file_count']==1
    assert any('skipped symlink' in w for w in catalog['warnings'])


@pytest.mark.parametrize('content', ['bad XML','<VTKFile type="Collection"><Collection/></VTKFile>',
                                    '<!DOCTYPE x><VTKFile/>'])
def test_malformed_pvd(tmp_path,content):
    put(tmp_path,'study/analysis/run.pvd',content)
    assert build_catalog(tmp_path)['datasets'][0]['status']=='incomplete_or_ambiguous'


def test_symlink_directories_and_root(tmp_path):
    root=tmp_path/'results'; root.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    put(outside,'payload.csv')
    (root/'alias').symlink_to(outside,target_is_directory=True)
    assert build_catalog(root)['file_count']==0
    with pytest.raises(ValueError,match='symlink'):
        build_catalog(root/'alias')


def test_csv_roundtrip_collision_and_protected_output(tmp_path):
    root=tmp_path/'results'
    put(root,'study/analysis/中文,"line\nfolder/a.csv')
    catalog=build_catalog(root,include_artifacts=True)
    output=tmp_path/'catalog'
    write_catalog(catalog,output,results_root=root)
    with (output/'datasets.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert rows[0]['relative_path']==catalog['datasets'][0]['relative_path']
    before=snapshot(output)
    with pytest.raises(FileExistsError):write_catalog(catalog,output,results_root=root)
    assert snapshot(output)==before
    with pytest.raises(ValueError):write_catalog(catalog,root/'catalog',results_root=root)
    with pytest.raises(ValueError):write_catalog(catalog,Path('/data/Nek5000_data/results/catalog'),results_root=root)
    assert not (root/'catalog').exists()


def test_empty_missing_and_file_roots(tmp_path):
    assert build_catalog(tmp_path)['datasets']==[]
    with pytest.raises(FileNotFoundError):build_catalog(tmp_path/'missing')
    file=put(tmp_path,'file')
    with pytest.raises(ValueError):build_catalog(file)


def test_nested_frame_without_collection_and_json_cap(tmp_path):
    for index in range(66):put(tmp_path,f'h_refinement/lambda_ci/case_N7_H/f{index:05}/data.json','{}')
    catalog=build_catalog(tmp_path)
    assert len(catalog['datasets'])==1
    assert catalog['file_count']==66
    assert any('capped' in w for w in catalog['warnings'])


def test_metadata_time_and_types(tmp_path):
    put(tmp_path,'study/analysis/data.json','{"time":1.2345,"polynomial_degree":true}')
    catalog=build_catalog(tmp_path)
    assert catalog['datasets'][0]['polynomial_degree'] is None
    assert any('invalid metadata type' in w for w in catalog['warnings'])
    put(tmp_path,'study/analysis/data.json','{"time":1.2345,"polynomial_degree":7}')
    row=build_catalog(tmp_path)['datasets'][0]
    assert row['actual_time_start']==1.2345
    assert row['polynomial_degree']==7
    assert any('extent unverified' in n for n in row['notes'])


def test_cli_collision_and_internal_symlink(tmp_path):
    import importlib.util
    script=Path(__file__).parents[1]/'scripts/43_build_results_catalog.py'
    spec=importlib.util.spec_from_file_location('catalog_cli',script)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    root=tmp_path/'results'
    source=put(root,'study/analysis/data.json','{"field":"omega_x"}')
    (source.parent/'alias.json').symlink_to(source)
    before=snapshot(root)
    destination=tmp_path/'catalog'
    assert cli.main([str(root),'--output-dir',str(destination)])==0
    assert cli.main([str(root),'--output-dir',str(destination)])==1
    assert snapshot(root)==before
    row=json.loads((destination/'catalog.json').read_text())['datasets'][0]
    assert row['field_names']==['omega_x']


@pytest.mark.parametrize('creation_order', [('run_a', 'run_b'), ('run_b', 'run_a')])
@pytest.mark.parametrize('run_collection', [False, True])
def test_nested_independent_run_controls(tmp_path, creation_order, run_collection):
    parent = 'h_refinement/lambda_ci/case_N7_H'
    for run in creation_order:
        count = 10 if run == 'run_a' else 12
        time = 1.25 if run == 'run_a' else 2.75
        folder = f'{parent}/{run}'
        put(tmp_path, folder+'/f00001/data.vtr')
        # Frame-level collections must not hide their enclosing run manifest.
        put(tmp_path, folder+'/f00001/data.pvd', '<VTKFile type="Collection"><Collection>'
            f'<DataSet timestep="{time}" file="data.vtr"/></Collection></VTKFile>')
        put(tmp_path, folder+'/manifest.json', json.dumps({'status':'completed',
            'sampling_nodes':count, 'field_names':['lambda_ci'],
            'producer_script':f'{run}.py', 'code_revision':run,
            'output_paths':{'volume':'f00001/data.vtr'}}))
        if run_collection:
            put(tmp_path, folder+'/series.pvd', '<VTKFile type="Collection"><Collection>'
                f'<DataSet timestep="{time}" file="f00001/data.vtr"/></Collection></VTKFile>')
    put(tmp_path, parent+'/series.pvd', '<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="1.25" file="run_a/f00001/data.vtr"/>'
        '<DataSet timestep="2.75" file="run_b/f00001/data.vtr"/></Collection></VTKFile>')
    before = snapshot(tmp_path)
    catalog = build_catalog(tmp_path, include_artifacts=True)
    rows = {r['relative_path']:r for r in catalog['datasets']}
    assert set(rows) == {parent, parent+'/run_a', parent+'/run_b'}
    assert rows[parent]['file_count'] == 1
    assert any('aggregate' in note for note in rows[parent]['notes'])
    for run, count, time in [('run_a',10,1.25), ('run_b',12,2.75)]:
        row = rows[parent+'/'+run]
        assert row['sampling_nodes'] == count
        assert row['actual_time_start'] == row['actual_time_end'] == time
        assert row['manifest_path'] == parent+'/'+run+'/manifest.json'
        assert row['producer_script'] == run+'.py'
        assert row['code_revision'] == run
        assert row['field_names'] == ['lambda_ci']
        assert row['status'] == 'reported_completed'
    assert len({r['dataset_id'] for r in rows.values()}) == 3
    expected_files = 9 if run_collection else 7
    assert len(catalog['artifacts']) == catalog['file_count'] == expected_files
    assert len({r['relative_path'] for r in catalog['artifacts']}) == expected_files
    assert build_catalog(tmp_path, include_artifacts=True) == catalog
    assert snapshot(tmp_path) == before
    # IDs and ordering depend on paths, never filesystem creation order.
    import hashlib
    assert [r['dataset_id'] for r in catalog['datasets']] == [
        'ds_'+hashlib.sha256(path.encode()).hexdigest()[:20] for path in sorted(rows)]


def test_nested_native_omega_path_identity(tmp_path):
    case = 'h_refinement/omega_x/case_N7_H'
    put(tmp_path, case+'/case_N7_H/f00001/data.vtr')
    put(tmp_path, case+'/series.pvd', '<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="0.25" file="case_N7_H/f00001/data.vtr"/></Collection></VTKFile>')
    catalog = build_catalog(tmp_path, include_artifacts=True)
    assert len(catalog['datasets']) == 1
    assert catalog['datasets'][0]['relative_path'] == case
    assert catalog['datasets'][0]['actual_time_start'] == .25
    assert catalog['file_count'] == 2


@pytest.mark.parametrize('references', [
    {'output_paths':'present.vtu'},
    {'output_paths':{'vtu':'present.vtu','optional':None}},
    {'outputs':['present.vtu',None]},
    {'output_file':'present.vtu'},
])
def test_supported_output_references(tmp_path, references):
    put(tmp_path, 'study/analysis/present.vtu')
    put(tmp_path, 'study/analysis/manifest.json', json.dumps({'status':'completed', **references,
        'source_file':'/unavailable/source.f00001', 'output_directory':'/unused/non-file-label'}))
    catalog = build_catalog(tmp_path)
    row = catalog['datasets'][0]
    assert row['status'] == 'reported_completed'
    assert not any('declared output' in w or 'unsupported output' in w for w in catalog['warnings'])
    assert row['validation_level'] == 'inventory_only'


@pytest.mark.parametrize('reference', ['missing.vtu', '../../../escape.vtu', '/etc/passwd', 'alias.vtu'])
def test_manifest_output_failure(tmp_path, reference):
    root = tmp_path/'results'
    outside = put(tmp_path, 'outside.vtu')
    put(root, 'study/analysis/present.vtu')
    (root/'study/analysis/alias.vtu').symlink_to(outside)
    put(root, 'study/analysis/manifest.json', json.dumps({'status':'completed',
        'output_paths':{'existing':'present.vtu','vtu':reference}}))
    catalog = build_catalog(root)
    assert catalog['datasets'][0]['status'] == 'incomplete_or_ambiguous'
    assert any('manifest.json' in w and 'output_paths.vtu' in w for w in catalog['warnings'])


@pytest.mark.parametrize('reference', [17, {'vtu':{'unexpected':'path'}}, ['present.vtu']])
def test_unsupported_output_reference_schema(tmp_path, reference):
    put(tmp_path, 'study/analysis/manifest.json', json.dumps({'status':'completed','output_paths':reference}))
    catalog = build_catalog(tmp_path)
    assert catalog['datasets'][0]['status'] == 'incomplete_or_ambiguous'
    assert any('unsupported output reference schema' in w for w in catalog['warnings'])


def test_absolute_in_root_and_historical_references(tmp_path):
    artifact = put(tmp_path, 'study/archive/run/present.vtu')
    put(tmp_path, 'study/archive/run/manifest.json', json.dumps({'status':'completed','output_paths':{'vtu':str(artifact)}}))
    row = build_catalog(tmp_path)['datasets'][0]
    assert row['status'] == 'reported_completed'
    assert row['artifact_class'] == 'historical'
    put(tmp_path, 'study/archive/run/manifest.json', '{"status":"completed","schema_version":99,"output_paths":{"vtu":{"url":"x"}}}')
    assert build_catalog(tmp_path)['datasets'][0]['status'] == 'incomplete_or_ambiguous'


def test_deep_json_continues_and_cli_reports(tmp_path, capsys):
    import importlib.util
    import sys
    root = tmp_path/'results'
    put(root, 'study/bad/deep.json', '['*2000+'0'+']'*2000)
    put(root, 'study/bad/valid.json', '{"polynomial_degree":7}')
    put(root, 'study/good/manifest.json', '{"status":"completed","field_names":["omega_x"]}')
    recursion_limit = sys.getrecursionlimit()
    catalog = build_catalog(root)
    assert len(catalog['datasets']) == 2
    bad = next(r for r in catalog['datasets'] if r['analysis']=='bad')
    assert bad['polynomial_degree'] == 7
    assert 'study/bad/deep.json' not in bad['metadata_sources']
    assert any('deep.json' in w and 'depth' in w for w in catalog['warnings'])
    assert sys.getrecursionlimit() == recursion_limit
    script = Path(__file__).parents[1]/'scripts/43_build_results_catalog.py'
    spec = importlib.util.spec_from_file_location('depth_catalog_cli',script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    assert cli.main([str(root),'--output-dir',str(tmp_path/'catalog')]) == 0
    output = capsys.readouterr()
    assert 'depth' in output.out and 'Traceback' not in output.err


def test_series_manifest_reference_base_and_ambiguous_manifest(tmp_path):
    base = 'h_refinement/vortex_3d_series/gll10/case_N7_H'
    put(tmp_path, base+'/f00001/data.vtu')
    put(tmp_path, base+'/series/manifest.json', json.dumps({'status':'completed',
        'output_paths':{'vtu':'../f00001/data.vtu'}}))
    row = build_catalog(tmp_path)['datasets'][0]
    assert row['relative_path'] == base
    assert row['manifest_path'] == base+'/series/manifest.json'
    assert row['status'] == 'reported_completed'
    put(tmp_path, base+'/manifest.json', '{"status":"completed"}')
    row = build_catalog(tmp_path)['datasets'][0]
    assert row['manifest_path'] is None
    assert any('multiple manifests' in note for note in row['notes'])


@pytest.mark.parametrize('reference', [{'vtu':'present.vtu'}, 'present.vtu'])
def test_unsupported_outputs_container(tmp_path, reference):
    put(tmp_path, 'study/analysis/present.vtu')
    put(tmp_path, 'study/analysis/manifest.json', json.dumps({'status':'completed','outputs':reference}))
    row = build_catalog(tmp_path)['datasets'][0]
    assert row['status'] == 'incomplete_or_ambiguous'
    assert any('outputs: unsupported output reference schema' in note for note in row['notes'])


def test_manifest_symlink_loop_is_diagnostic(tmp_path):
    folder = tmp_path/'study/analysis'
    folder.mkdir(parents=True)
    (folder/'loop.vtu').symlink_to('loop.vtu')
    put(tmp_path, 'study/analysis/manifest.json', '{"status":"completed","output_paths":{"vtu":"loop.vtu"}}')
    catalog = build_catalog(tmp_path)
    assert catalog['datasets'][0]['status'] == 'incomplete_or_ambiguous'
    assert any('output_paths.vtu: unsafe or unresolvable' in note for note in catalog['warnings'])


@pytest.mark.parametrize('reference_field, references', [
    ('output_paths.vtu', {'output_paths': {'vtu': 'bad\x00.vtu'}}),
    ('output_file', {'output_file': 'bad\x00.vtu'}),
    ('outputs[0]', {'outputs': ['bad\x00.vtu']}),
])
def test_malformed_output_cannot_be_overridden_by_completed_summary(tmp_path, reference_field, references):
    root = tmp_path/'results'
    put(root, 'study/broken/manifest.json', json.dumps({'status': 'completed', **references}))
    put(root, 'study/broken/processing_summary.json', json.dumps({
        'status': 'completed', 'polynomial_degree': 7,
        'field_names': ['omega_x'], 'producer_script': 'recorded_producer.py',
    }))
    put(root, 'study/healthy/data.vtu')
    put(root, 'study/healthy/manifest.json', json.dumps({
        'status': 'completed', 'output_paths': {'vtu': 'data.vtu'},
    }))
    before = snapshot(root)
    catalog = build_catalog(root)
    rows = {r['relative_path']: r for r in catalog['datasets']}
    broken = rows['study/broken']
    assert broken['status'] == 'incomplete_or_ambiguous'
    assert broken['polynomial_degree'] == 7
    assert broken['field_names'] == ['omega_x']
    assert broken['producer_script'] == 'recorded_producer.py'
    assert 'study/broken/processing_summary.json' in broken['metadata_sources']
    diagnostics = [note for note in broken['notes'] if 'malformed declared output' in note]
    assert len(diagnostics) == 1
    assert 'study/broken/manifest.json' in diagnostics[0]
    assert reference_field in diagnostics[0]
    assert 'bad\\x00.vtu' in diagnostics[0]
    assert 'embedded null' in diagnostics[0].lower()
    assert rows['study/healthy']['status'] == 'reported_completed'
    assert snapshot(root) == before
    assert catalog['file_count'] == 4
