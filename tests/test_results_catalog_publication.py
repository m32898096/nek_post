"""Lightweight navigation publication with unchanged scientific controls."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

from nek_post.results_catalog import DATASET_COLUMNS, build_catalog
from nek_post.results_catalog_publication import (
    prepare_navigation, publish_navigation, validate_navigation,
)


@pytest.fixture
def navigation(tmp_path):
    root = tmp_path / 'results'
    run = root / 'h_refinement/vortex_3d_series/case_N7_H'
    run.mkdir(parents=True)
    (run / 'data.vtu').write_bytes(b'synthetic scientific payload')
    (run / 'run.pvd').write_text('<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="19.50240408807" file="data.vtu"/></Collection></VTKFile>')
    (run / 'manifest.json').write_text(json.dumps({
        'status': 'completed', 'output_paths': {'volume': 'data.vtu'},
        'field_names': ['omega_x'], 'producer_script': '紊流後處理.py',
    }, ensure_ascii=False), encoding='utf-8')
    guide = tmp_path / 'guide.md'
    guide.write_text('# 紊流研究導覽\n\n[場](h_refinement/vortex_3d_series/case_N7_H/data.vtu)\n'
                     '[系列](h_refinement/vortex_3d_series/case_N7_H/run.pvd)\n'
                     '[CSV](results_index.csv)\n', encoding='utf-8')
    return root, run, guide, tmp_path / 'preview'


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_publish_stable_identity_csv_controls_and_utf8(navigation, tmp_path):
    root, run, guide, preview = navigation
    before = snapshot(root)
    catalog = prepare_navigation(root, guide, preview)
    assert snapshot(root) == before
    assert publish_navigation(root, preview) == catalog
    after = snapshot(root)
    assert {k: after[k] for k in before} == before
    assert set(after) - set(before) == {'README.md', 'results_index.csv'}
    assert (root / 'README.md').read_bytes() == guide.read_bytes()
    assert build_catalog(root, include_artifacts=True) == catalog
    second = tmp_path / 'second-preview'
    prepare_navigation(root, guide, second)
    assert (root / 'results_index.csv').read_bytes() == (second / 'datasets.csv').read_bytes()
    rows = list(csv.DictReader(io.StringIO((root / 'results_index.csv').read_text(encoding='utf-8'))))
    assert tuple(rows[0]) == DATASET_COLUMNS
    assert rows[0]['producer_script'] == '紊流後處理.py'
    assert rows[0]['actual_time_start'] == '19.50240408807'
    assert rows[0]['polynomial_degree'] == ''
    assert rows[0]['validation_level'] == 'inventory_only'
    published = snapshot(root)
    with pytest.raises(FileExistsError):
        publish_navigation(root, preview)
    assert snapshot(root) == published


@pytest.mark.parametrize('name', ['README.md', 'results_index.csv'])
@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_target_collision_never_writes_either_file(navigation, name, kind):
    root, run, guide, preview = navigation
    prepare_navigation(root, guide, preview)
    target = root / name
    if kind == 'file':
        target.write_bytes(b'unknown ownership')
    elif kind == 'directory':
        target.mkdir()
    else:
        target.symlink_to(root / 'missing')
    with pytest.raises(FileExistsError):
        publish_navigation(root, preview)
    other = root / ('results_index.csv' if name == 'README.md' else 'README.md')
    assert not other.exists()
    if kind == 'file':
        assert target.read_bytes() == b'unknown ownership'
    elif kind == 'directory':
        assert target.is_dir()
    else:
        assert target.is_symlink()


@pytest.mark.parametrize('change', ['source', 'csv', 'catalog', 'readme'])
def test_stale_or_invalid_preview_stops_before_publication(navigation, change):
    root, run, guide, preview = navigation
    prepare_navigation(root, guide, preview)
    if change == 'source':
        (run / 'data.vtu').write_bytes(b'changed source')
    elif change == 'csv':
        (preview / 'datasets.csv').write_text('edited CSV')
    elif change == 'catalog':
        data = json.loads((preview / 'catalog.json').read_text())
        data['datasets'][0]['polynomial_degree'] = 9
        (preview / 'catalog.json').write_text(json.dumps(data))
    else:
        (preview / 'README.md').write_text('[missing](absent.vtu)')
    with pytest.raises(ValueError):
        publish_navigation(root, preview)
    assert not (root / 'README.md').exists()
    assert not (root / 'results_index.csv').exists()


@pytest.mark.parametrize('link', ['missing.vtu', '../outside', '/etc/passwd',
                                 'https://example.com', 'bad%00.vtu', 'alias.vtu'])
def test_navigation_rejects_missing_unsafe_and_unsupported_links(navigation, link):
    root, run, guide, preview = navigation
    (root / 'alias.vtu').symlink_to(run / 'data.vtu')
    with pytest.raises(ValueError):
        validate_navigation(f'[target]({link})', root)


def test_preview_protection_and_existing_destination(navigation):
    root, run, guide, preview = navigation
    with pytest.raises(ValueError, match='outside'):
        prepare_navigation(root, guide, root / 'preview')
    prepare_navigation(root, guide, preview)
    before = snapshot(preview)
    with pytest.raises(FileExistsError):
        prepare_navigation(root, guide, preview)
    assert snapshot(preview) == before
    with pytest.raises(ValueError, match='outside'):
        publish_navigation(root, root)


def test_nested_production_publication_forbidden(navigation, monkeypatch):
    from nek_post import results_catalog_publication as publication
    root, run, guide, preview = navigation
    prepare_navigation(root, guide, preview)
    monkeypatch.setattr(publication, 'SCIENTIFIC_ROOT', root)
    with pytest.raises(ValueError, match='nested production'):
        publish_navigation(run, preview)
    assert not (run / 'README.md').exists()


def test_mtime_only_change_invalidates_preview(navigation):
    import os
    root, run, guide, preview = navigation
    prepare_navigation(root, guide, preview)
    path = run / 'data.vtu'
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match='inventory or metadata changed'):
        publish_navigation(root, preview)
    assert not (root / 'README.md').exists()


def test_late_collision_uses_exclusive_creation(navigation, monkeypatch):
    root, run, guide, preview = navigation
    prepare_navigation(root, guide, preview)
    original_open = Path.open
    target = root / 'results_index.csv'

    def competing_open(path, mode='r', *args, **kwargs):
        if path == target and mode == 'xb':
            with original_open(target, 'xb') as stream:
                stream.write(b'competing publisher')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', competing_open)
    with pytest.raises(FileExistsError):
        publish_navigation(root, preview)
    assert target.read_bytes() == b'competing publisher'
    # The documented partial-pair policy preserves the successfully created guide.
    assert (root / 'README.md').read_bytes() == guide.read_bytes()


def test_only_exact_root_catalog_names_excluded(navigation):
    root, run, guide, preview = navigation
    (run / 'README.md').write_text('scientific collection notes')
    (run / 'results_index.csv').write_text('historical table')
    catalog = prepare_navigation(root, guide, preview)
    assert catalog['file_count'] == 5
    publish_navigation(root, preview)
    assert build_catalog(root, include_artifacts=True) == catalog
    assert any(a['relative_path'].endswith('/README.md') for a in catalog['artifacts'])


def test_diagnostics_survive_publication(navigation):
    root, run, guide, preview = navigation
    (run / 'manifest.json').write_text(json.dumps({
        'status': 'completed', 'output_file': 'missing.vtu',
        'producer_script': '紊流.py',
    }), encoding='utf-8')
    (run / 'old.json').write_text('[1,2,3]')
    catalog = prepare_navigation(root, guide, preview)
    assert catalog['datasets'][0]['status'] == 'incomplete_or_ambiguous'
    assert any('missing declared output' in w for w in catalog['warnings'])
    assert any('expected JSON object' in w for w in catalog['warnings'])
    publish_navigation(root, preview)
    assert build_catalog(root, include_artifacts=True) == catalog
    csv_text = (root / 'results_index.csv').read_text(encoding='utf-8')
    assert 'incomplete_or_ambiguous' in csv_text and 'missing declared output' in csv_text


def test_publication_cli_preview_publish_and_collision(navigation, capsys):
    root, run, guide, preview = navigation
    script = Path(__file__).parents[1] / 'scripts/44_publish_results_catalog.py'
    spec = importlib.util.spec_from_file_location('publication_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = [str(root), '--preview-dir', str(preview)]
    assert cli.main(args + ['--readme-source', str(guide)]) == 0
    assert cli.main(args + ['--publish']) == 0
    assert cli.main(args + ['--publish']) == 1
    assert 'already exists' in capsys.readouterr().err


def test_traditional_chinese_guide_links_with_synthetic_tree(tmp_path):
    guide = Path(__file__).parents[1] / 'docs/results_catalog/navigation_zh_tw.md'
    text = guide.read_text(encoding='utf-8')
    import re
    links = re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', text)
    for link in links:
        if link == 'results_index.csv':
            continue
        target = tmp_path / link
        if link.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'synthetic')
    assert validate_navigation(text, tmp_path) == links
    assert len(links) >= 40
    assert '紊流' in text and '湍流' not in text
    assert 'inventory_only' in text and '19.50240408807' in text


def _check_lambda_ci_navigation_metadata(root, document):
    """Bind the documented VTR to its control, rather than trusting filenames."""
    import re
    import xml.etree.ElementTree as ET

    row = next(line for line in document.splitlines() if line.startswith('| 重建 lambda_ci 切片 |'))
    links = set(re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', row))
    run = 'h_refinement/lambda_ci/case_N7_H'
    frame = run + '/f00079'
    stem = 'case_N7_H_f00079_lambda_ci'
    vtr = frame + '/slices/' + stem + '_y_midplane.vtr'
    pv_meta = frame + '/plots/' + stem + '_high_order_slices_paraview.json'
    py_meta = frame + '/plots/' + stem + '_high_order_slices.json'
    summary = run + '/processing_summary.json'
    assert {vtr, pv_meta, py_meta, summary} <= links
    pv = json.loads((root / pv_meta).read_text(encoding='utf-8'))
    py = json.loads((root / py_meta).read_text(encoding='utf-8'))
    series = json.loads((root / summary).read_text(encoding='utf-8'))
    assert Path(pv['output_paths']['y_midplane_vtr']).resolve() == (root / vtr).resolve()
    assert pv['samples_per_element'] == 29
    assert pv['subdivisions_per_element'] == series['configuration']['subdivisions_per_element'] == 28
    assert pv['samples_per_element'] == series['configuration']['subdivisions_per_element'] + 1
    pvd = run + '/case_N7_H_lambda_ci_y_midplane_series.pvd'
    assert pvd in links
    targets = [(root / pvd).parent / record.attrib['file']
               for record in ET.parse(root / pvd).findall('./Collection/DataSet')]
    assert (root / vtr).resolve() in {p.resolve() for p in targets}
    with (root / vtr).open('rb') as stream:
        header = stream.read(2048).split(b'<AppendedData')[0]
    extent = [int(v) for v in re.search(rb'WholeExtent="([0-9 ]+)"', header).group(1).split()]
    dimensions = [extent[i + 1] - extent[i] + 1 for i in (0, 2, 4)]
    assert dimensions == [3809, 1, 225]
    assert pv['slices']['y_midplane']['shape'] == [dimensions[2], dimensions[0]]
    frame_record = next(frame for frame in series['frames'] if frame['index'] == 79)
    assert frame_record['slices']['y_midplane']['dimensions'] == dimensions
    assert py['samples_per_element'] == 32
    assert not any(str(p).endswith('.vtr') for p in py['output_paths'].values())
    python_figure = Path(py['output_paths']['y_midplane_png'])
    assert python_figure.relative_to(root).as_posix() in links
    assert python_figure.is_file()
    assert python_figure != Path(pv['output_paths']['y_midplane_png'])
    assert '29 個均勻取樣點' in row and '28 個 subdivisions' in row
    assert '32 個均勻取樣點' in row and '3809 × 1 × 225' in row


def test_lambda_ci_navigation_metadata_association(tmp_path):
    run = tmp_path / 'h_refinement/lambda_ci/case_N7_H'
    plots = run / 'f00079/plots'
    plots.mkdir(parents=True)
    stem = 'case_N7_H_f00079_lambda_ci'
    vtr = run / 'f00079/slices' / (stem + '_y_midplane.vtr')
    vtr.parent.mkdir()
    vtr.write_bytes(b'<VTKFile><RectilinearGrid WholeExtent="0 3808 0 0 0 224"/></VTKFile>')
    python_png = plots / (stem + '_y_midplane_xge0p0.png')
    python_png.write_bytes(b'fixture')
    pv = {'samples_per_element': 29, 'subdivisions_per_element': 28,
          'output_paths': {'y_midplane_vtr': str(vtr),
                           'y_midplane_png': str(plots / (stem + '_y_midplane.png'))},
          'slices': {'y_midplane': {'shape': [225, 3809]}}}
    py = {'samples_per_element': 32, 'output_paths': {'y_midplane_png': str(python_png)}}
    summary = {'configuration': {'subdivisions_per_element': 28},
               'frames': [{'index': 79, 'slices': {'y_midplane': {'dimensions': [3809, 1, 225]}}}]}
    for name, data in [(plots / (stem + '_high_order_slices_paraview.json'), pv),
                       (plots / (stem + '_high_order_slices.json'), py),
                       (run / 'processing_summary.json', summary)]:
        name.write_text(json.dumps(data), encoding='utf-8')
    (run / 'case_N7_H_lambda_ci_y_midplane_series.pvd').write_text(
        '<VTKFile><Collection><DataSet timestep="19.50240408807" '
        f'file="f00079/slices/{vtr.name}"/></Collection></VTKFile>')
    guide = Path(__file__).parents[1] / 'docs/results_catalog/navigation_zh_tw.md'
    _check_lambda_ci_navigation_metadata(tmp_path, guide.read_text(encoding='utf-8'))


def test_optional_local_lambda_ci_navigation_metadata_association():
    root = Path('/data/Nek5000_data/results')
    control = root / ('h_refinement/lambda_ci/case_N7_H/f00079/plots/'
                      'case_N7_H_f00079_lambda_ci_high_order_slices_paraview.json')
    if not control.is_file() or not (root / 'README.md').is_file():
        pytest.skip('Optional local N7_H slice controls and published navigation unavailable')
    guide = Path(__file__).parents[1] / 'docs/results_catalog/navigation_zh_tw.md'
    for document in (guide, root / 'README.md'):
        _check_lambda_ci_navigation_metadata(root, document.read_text(encoding='utf-8'))
