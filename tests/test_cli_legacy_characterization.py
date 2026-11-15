"""Execute unchanged legacy orchestration with synthetic inputs and recording fakes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def legacy(name):
    spec = importlib.util.spec_from_file_location('characterize_'+name[:2], REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('field', ['concentration', 'velocity', 'pressure'])
@pytest.mark.parametrize('overwrite', [False, True])
def test_03_dispatch_skip_tables_and_globals(monkeypatch, tmp_path, field, overwrite):
    script = legacy('03_compare_poly_orders.py')
    assert script.compare_concentration_slices.__module__ == 'nek_post.comparison'
    args = SimpleNamespace(field=field, overwrite=overwrite, comparison_set='a', case_indices=None,
                           index=None, method='linear', time_tolerance=.05, strict_time=False, front_threshold_ratio=.01)
    monkeypatch.setattr(script, '_parse_args', lambda: args)
    config = {'paths': {}, 'cases': {'orders': {'N5': 5, 'N11': 11}, 'grid': {'nx': 3, 'nz': 2}}}
    loaded = []
    monkeypatch.setattr(script, 'load_project_config', lambda *paths: loaded.extend(paths) or config)
    monkeypatch.setattr(script, 'ProjectPaths', SimpleNamespace(from_mapping=lambda _: object()))
    selection = SimpleNamespace(case_indices={'N5': 79, 'N11': 40}, reference_case='N11', comparison_set_name='a', target_time=19.5, output_label='a')
    monkeypatch.setattr(script, 'resolve_comparison_selection', lambda **kwargs: selection)
    monkeypatch.setattr(script, 'check_slice_files', lambda *a: {'N5': tmp_path/'one', 'N11': tmp_path/'two'})
    monkeypatch.setattr(script, 'load_slice_file', lambda path: {})
    monkeypatch.setattr(script, 'evaluate_time_alignment', lambda *a: SimpleNamespace(warnings=(), case_times={}, time_differences={}))
    monkeypatch.setattr(script, 'create_common_xz_grid', lambda *a: (1, 2, None, None, {}))
    calls = []
    grids = {'N5': dict(u=1,v=2,w=3,speed=4,p=5,p_prime=6), 'N11': dict(u=1,v=2,w=3,speed=4,p=5,p_prime=6)}
    def kernel(**kwargs):
        calls.append(('kernel', kwargs))
        return SimpleNamespace(grids=grids, error_rows=[], front_rows=[], valid_point_count=6, total_grid_point_count=6)
    monkeypatch.setattr(script, 'compare_'+field+'_slices', kernel)
    existing = tmp_path/'existing.npz'
    existing.write_bytes(b'preserved')
    monkeypatch.setattr(script, field+'_interpolated_path', lambda *a: existing)
    monkeypatch.setattr(script, 'save_'+field+'_interpolated', lambda *a: calls.append(('save', a)))
    for name in (field+'_error_table_path', 'front_position_table_path', 'comparison_metadata_path'):
        monkeypatch.setattr(script, name, lambda *a: tmp_path/'table')
    for name in ('write_'+field+'_error_table', 'write_front_position_table', 'write_comparison_set_metadata', 'append_comparison_log'):
        monkeypatch.setattr(script, name, lambda *a, name=name: calls.append((name, a)))
    monkeypatch.setattr(script, 'build_'+field+'_summary', lambda *a: 'summary')
    script.main()
    assert loaded == [script.REPO_ROOT/'config/paths.yaml', script.REPO_ROOT/'config/cases.yaml']
    assert sum(name == 'kernel' for name, _ in calls) == 1
    assert calls[0][1]['case_indices'] == {'N5': 79, 'N11': 40}
    assert sum(name == 'save' for name, _ in calls) == (2 if overwrite else 0)
    assert any(name == 'write_'+field+'_error_table' for name, _ in calls)
    assert calls[-1][0] == 'append_comparison_log'
    assert any(name == 'write_comparison_set_metadata' for name, _ in calls) == (field == 'concentration')
    assert existing.read_bytes() == b'preserved'


def test_07_dedup_order_and_failure(monkeypatch):
    script = legacy('07_run_multitime_pipeline.py')
    config = {'paths': {}, 'cases': {'comparison_sets': {'a': {'case_indices': {'N5': 1, 'N11': 2}}, 'b': {'case_indices': {'N5': 1, 'N11': 3}}}, 'multitime_comparison_sets': ['a', 'b']}}
    commands = script._slice_commands(config, ['a', 'b'], True)
    assert [(cmd[3], cmd[5]) for cmd in commands] == [('N5', '1'), ('N11', '2'), ('N11', '3')]
    assert all(cmd[0] == sys.executable and cmd[-1] == '--overwrite' for cmd in commands)
    assert script._default_comparison_sets({'cases': {}}) == ['t19p5']
    monkeypatch.setattr(script, 'load_project_config', lambda *a: config)
    monkeypatch.setattr(script, 'ProjectPaths', SimpleNamespace(from_mapping=lambda _: object()))
    monkeypatch.setattr(script, '_parse_args', lambda: SimpleNamespace(comparison_sets=None, fields='velocity,pressure', overwrite=False, dry_run=True, skip_slices=False, skip_plots=False, strict=False))
    monkeypatch.setattr(script, '_append_log', lambda *a: None)
    monkeypatch.setattr(script, '_log_path', lambda *a: 'synthetic-log')
    recorded = []
    monkeypatch.setattr(script, '_run_command', lambda command, paths, **kw: recorded.append(command))
    script.main()
    assert [Path(cmd[1]).name[:2] for cmd in recorded] == ['02']*3 + ['03','04']*4
    assert [(cmd[3], cmd[5]) for cmd in recorded[3:]] == [('a','velocity')]*2 + [('a','pressure')]*2 + [('b','velocity')]*2 + [('b','pressure')]*2
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(7, ['synthetic'])
    monkeypatch.setattr(script, '_run_command', fail)
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 7


def test_07_dry_run_logs_and_subprocess_contract(monkeypatch):
    script = legacy('07_run_multitime_pipeline.py')
    logs, processes = [], []
    monkeypatch.setattr(script, '_append_log', lambda paths, text: logs.append(text))
    monkeypatch.setattr(script.subprocess, 'run', lambda *a, **kw: processes.append((a, kw)))
    script._run_command(['synthetic'], object(), dry_run=True)
    assert len(logs) == 2 and not processes
    script._run_command(['synthetic'], object(), dry_run=False)
    assert processes == [((['synthetic'],), {'cwd': script.REPO_ROOT, 'check': True})]
    assert logs[-1].startswith('Success:')
