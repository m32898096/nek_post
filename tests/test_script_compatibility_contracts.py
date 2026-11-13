"""Batch 2: script 08 aggregation and script 13 front-kinematics CLI contracts.

References: a4e671f committed source and documented schemas; x=1+2t is
an independent analytic reference. Only temporary config/data are executed.
No literal help golden: assert flags/defaults and stabilize child width.
"""
import csv
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ('08_collect_multitime_error_summary.py', '13_front_kinematics.py')
AGG_HEADER = ('comparison_set,target_time,field,case,order,reference_case,index,'
              'reference_index,relative_L2,mean_abs_error,max_abs_error,relative_Linf,'
              'valid_point_count,total_grid_point_count').split(',')
FRONT_HEADER = 'time,x_front,v_raw,v_smooth,x_reconstructed,x_reconstruction_error'.split(',')
SUMMARY_HEADER = ('case,n_points,time_start,time_end,x_start,x_end,v_raw_min,v_raw_max,'
                  'v_smooth_min,v_smooth_max,mean_abs_reconstruction_error,'
                  'max_abs_reconstruction_error,rms_reconstruction_error,'
                  'final_reconstruction_error,slumping_velocity_raw_position_fit,'
                  'slumping_velocity_reconstructed_position_fit').split(',')


@pytest.fixture
def checkout(tmp_path):
    """Preserve scripts/config relative layout; never consult production tables."""
    root = tmp_path / 'checkout'
    (root / 'scripts').mkdir(parents=True)
    (root / 'config').mkdir()
    for name in SCRIPTS:
        source = REPO / 'scripts' / name
        (root / 'scripts' / name).write_bytes(source.read_bytes())
    paths = dict(data_root=str(tmp_path / 'data'), case_dirs={},
                 postproc_root=str(tmp_path / 'post'), results_root=str(tmp_path / 'results'),
                 paper_data=dict(cantero_fig5a_re3450_csv=str(tmp_path / 'unused3450.csv'),
                                 cantero_fig5a_re8950_csv=str(tmp_path / 'unused8950.csv')))
    cases = dict(reference_case='N11', multitime_comparison_sets=['late', 'early'],
                 comparison_sets=dict(late=dict(target_time=2), early=dict(target_time=1)))
    (root / 'config/paths.yaml').write_text(yaml.safe_dump(paths))
    (root / 'config/cases.yaml').write_text(yaml.safe_dump(cases))
    return root, paths


def run(script, tmp_path, *args):
    return subprocess.run([sys.executable, '-B', str(script), *map(str, args)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=30,
                          env={**os.environ, 'COLUMNS': '80', 'PYTHONDONTWRITEBYTECODE': '1',
                               'MPLCONFIGDIR': str(tmp_path / 'mpl')})


def load(name):
    spec = importlib.util.spec_from_file_location('contract_' + name[:2], REPO / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', SCRIPTS)
def test_original_entrypoint_help_and_unknown_option(name, tmp_path):
    script = REPO / 'scripts' / name
    parent = dict(os.environ)
    help_result = run(script, tmp_path, '--help')
    assert help_result.returncode == 0 and not help_result.stderr
    required_flags = ('--fields', '--comparison-sets', '--output', '--allow-missing') if name.startswith('08') else (
        '--cases', '--data-root', '--filename', '--output-dir', '--smooth-window',
        '--smooth-method', '--savgol-polyorder', '--overwrite', '--no-plots')
    assert all(flag in help_result.stdout for flag in required_flags)
    bad = run(script, tmp_path, '--unknown-contract-option')
    assert bad.returncode == 2 and 'unrecognized arguments' in bad.stderr
    assert not bad.stdout and dict(os.environ) == parent


def test_defaults_types_choices_and_public_imports(checkout, tmp_path, monkeypatch):
    from nek_post.front_kinematics import compute_kinematics
    from nek_post.front_io import read_front_simple_dat
    from nek_post.paths import ProjectPaths
    root, paths = checkout
    aggregate, front = map(load, SCRIPTS)
    monkeypatch.setattr(sys, 'argv', [SCRIPTS[0]])
    assert vars(aggregate._parse_args()) == dict(comparison_sets=None,
        fields='concentration,velocity,pressure', output=None, allow_missing=False)
    monkeypatch.setattr(sys, 'argv', [SCRIPTS[1]])
    pp = ProjectPaths.from_mapping(paths)
    assert vars(front._parse_args(pp)) == dict(cases='N5,N7,N9', data_root=pp.data_root,
        filename='front_simple.dat', output_dir=pp.front_kinematics_dir, smooth_window=11,
        smooth_method='moving_average', savgol_polyorder=3, overwrite=False, no_plots=False)
    assert front.compute_kinematics is compute_kinematics
    assert front.read_front_simple_dat is read_front_simple_dat
    assert list(inspect.signature(compute_kinematics).parameters) == ['front','method','window','polyorder']
    assert list(inspect.signature(read_front_simple_dat).parameters) == ['path']
    for options, message in [(('--smooth-window','bad'), 'invalid int value'),
                             (('--smooth-method','bad'), 'invalid choice')]:
        result = run(root / 'scripts' / SCRIPTS[1], tmp_path, *options)
        assert result.returncode == 2 and message in result.stderr
    assert not Path(paths['results_root']).exists()


def read_csv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_front_linear_physical_times_outputs_and_collisions(checkout, tmp_path):
    root, paths = checkout
    case = Path(paths['data_root']) / 'case_N7'
    case.mkdir(parents=True)
    # Deliberately unsorted, nonuniform physical times. x(t)=1+2t.
    source = case / 'custom.dat'
    original = b'12 25\n3 7\n7 15\n4 9\n'
    source.write_bytes(original)
    script = root / 'scripts' / SCRIPTS[1]
    args = ('--cases','N7','--filename','custom.dat','--smooth-window','1','--no-plots')
    result = run(script, tmp_path, *args)
    assert result.returncode == 0, result.stderr
    assert not result.stderr and 'Figures: skipped (--no-plots)' in result.stdout
    out = Path(paths['results_root']) / 'front_kinematics'
    assert {p.name for p in out.iterdir()} == {'N7_front_kinematics_timeseries.csv','front_kinematics_summary.csv'}
    header, rows = read_csv(out / 'N7_front_kinematics_timeseries.csv')
    assert header == FRONT_HEADER
    expected = np.array([[3,7,2,2,7,0], [4,9,2,2,9,0], [7,15,2,2,15,0], [12,25,2,2,25,0]])
    # Floating gradient/fit roundoff only; linear solution is exact analytically.
    np.testing.assert_allclose([[float(r[k]) for k in header] for r in rows], expected, rtol=0, atol=1e-12)
    header, summary = read_csv(out / 'front_kinematics_summary.csv')
    assert header == SUMMARY_HEADER and len(summary) == 1
    assert summary[0]['case'] == 'N7' and summary[0]['n_points'] == '4'
    assert float(summary[0]['slumping_velocity_raw_position_fit']) == pytest.approx(2, abs=1e-12)
    saved = {p.name:p.read_bytes() for p in out.iterdir()}
    collision = run(script, tmp_path, *args)
    assert collision.returncode == 1 and 'ERROR: Output exists:' in collision.stderr
    assert saved == {p.name:p.read_bytes() for p in out.iterdir()}
    overwrite = run(script, tmp_path, *args, '--overwrite')
    assert overwrite.returncode == 0
    assert saved == {p.name:p.read_bytes() for p in out.iterdir()}
    assert source.read_bytes() == original


def test_front_explicit_paths_and_missing_input(checkout, tmp_path):
    root, _ = checkout
    script = root / 'scripts' / SCRIPTS[1]
    data = tmp_path / 'explicit/case_N5';data.mkdir(parents=True)
    (data / 'front_simple.dat').write_text('3 7\n4 9\n7 15\n12 25\n')
    out = tmp_path / 'explicit-output'
    result = run(script, tmp_path, '--cases','N5','--data-root',data.parent,
                 '--output-dir',out,'--smooth-window','1','--no-plots')
    assert result.returncode == 0, result.stderr
    assert (out / 'N5_front_kinematics_timeseries.csv').is_file()
    missing = run(script,tmp_path,'--cases','N9','--data-root',data.parent,
                  '--output-dir',tmp_path / 'never-created','--no-plots')
    assert missing.returncode == 1 and 'Front-position file not found:' in missing.stderr
    assert not (tmp_path / 'never-created').exists()


def test_aggregate_schema_mapping_order_and_existing_output(checkout, tmp_path):
    root, paths = checkout
    tables = Path(paths['results_root']) / 'tables';tables.mkdir(parents=True)
    # Literal legacy schemas/values from committed script 08; no metric recomputation.
    for group in ('late','early'):
        for field, suffix in [('pressure','p_prime'),('concentration','C'),('velocity','speed')]:
            header = ['case','order','index','reference_index','valid_point_count','total_grid_point_count',
                      f'relative_L2_{suffix}',f'mean_abs_error_{suffix}',f'absolute_Linf_{suffix}',f'relative_Linf_{suffix}']
            with (tables / f'{field}_error_{group}.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(header)
                for case, order in [('N9',9),('N5',5)]:
                    metrics = {'concentration':['0.125','0.25','0.5','0.75'],
                               'velocity':['1.125','1.25','1.5','1.75'],
                               'pressure':['2.125','2.25','2.5','2.75']}[field]
                    writer.writerow([case,order,79,80,10,12,*metrics])
    before={p.name:p.read_bytes() for p in tables.iterdir()}
    script=root/'scripts'/SCRIPTS[0]
    result=run(script,tmp_path,'--fields','pressure,velocity,concentration')
    assert result.returncode==0 and not result.stderr
    assert 'Rows written: 12' in result.stdout
    output=tables/'multitime_error_summary.csv'
    header,rows=read_csv(output)
    assert header==AGG_HEADER
    assert [(r['comparison_set'],r['field'],r['case']) for r in rows]==[
        (g,f,c) for g in ('early','late') for f in ('concentration','velocity','pressure') for c in ('N5','N9')]
    for r in rows:
        assert [r[k] for k in ('reference_case','index','reference_index','valid_point_count','total_grid_point_count')]==['N11','79','80','10','12']
        expected_metrics = {'concentration':['0.125','0.25','0.5','0.75'],
                            'velocity':['1.125','1.25','1.5','1.75'],
                            'pressure':['2.125','2.25','2.5','2.75']}
        assert [r[k] for k in ('relative_L2','mean_abs_error','max_abs_error','relative_Linf')]==expected_metrics[r['field']]
        assert r['target_time']==('1' if r['comparison_set']=='early' else '2')
    expected=output.read_bytes()
    output.write_text('existing output is replaced by the legacy collector\n')
    again=run(script,tmp_path)
    assert again.returncode==0 and output.read_bytes()==expected
    assert all((tables/n).read_bytes()==b for n,b in before.items())


def test_aggregate_missing_validation_and_explicit_output(checkout, tmp_path):
    root,paths=checkout;script=root/'scripts'/SCRIPTS[0];out=tmp_path/'custom.csv'
    missing=run(script,tmp_path,'--output',out)
    assert missing.returncode==1 and 'ERROR: Missing input CSV:' in missing.stderr
    assert not out.exists()
    allowed=run(script,tmp_path,'--comparison-sets','early','--fields','velocity','--output',out,'--allow-missing')
    assert allowed.returncode==0 and 'WARNING: Missing input CSV:' in allowed.stdout
    assert read_csv(out)==(AGG_HEADER,[])
    for options,text in [(('--fields','invalid'),'Unknown field(s)'),
                         (('--comparison-sets','unknown'),'Unknown comparison set(s)')]:
        result=run(script,tmp_path,*options,'--output',tmp_path/'invalid.csv')
        assert result.returncode==1 and text in result.stderr
    assert not (tmp_path/'invalid.csv').exists()
    table=Path(paths['results_root'])/'tables';table.mkdir(parents=True)
    (table/'velocity_error_early.csv').write_text('case\nN5\n')
    malformed=run(script,tmp_path,'--comparison-sets','early','--fields','velocity','--output',tmp_path/'malformed.csv')
    assert malformed.returncode==1 and 'missing required column(s)' in malformed.stderr
    assert not (tmp_path/'malformed.csv').exists()


def test_protected_source_maps_match_phase3a_baseline():
    """Exact content/path guard, not proof of executable or resume equivalence."""
    import hashlib
    import importlib
    import json
    captured = json.loads((REPO / 'docs/refactoring/source_fingerprints.json').read_text())
    for key, module in [('vortex_frame','vortex_3d_workflow'), ('vortex_series','vortex_3d_series'),
                        ('lambda_ci_slices','lambda_ci_series'), ('omega_x_native','omega_x_workflow'),
                        ('omega_x_p7','omega_x_p7_series')]:
        expected = captured['signatures'][key]
        actual = {name:hashlib.sha256((REPO / 'src/nek_post' / name).read_bytes()).hexdigest()
                  for name in expected}
        assert actual == expected
        assert importlib.import_module('nek_post.' + module)._signature() == expected
