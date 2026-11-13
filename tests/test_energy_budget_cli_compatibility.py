"""Stable script-12 interface and analytical temporary-DAT regression fixtures.

No production results are inputs. Expectations are fixed independently of the
energy implementation; Phase 3B changes navigation only.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/12_check_energy_budget_closure.py'


def invoke(tmp_path, *args):
    return subprocess.run(
        [sys.executable, '-B', str(SCRIPT), *map(str, args)], cwd=REPO,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'COLUMNS': '80',
             'MPLCONFIGDIR': str(tmp_path / 'mpl')},
        capture_output=True, text=True, check=False,
    )


def test_legacy_help_matches_pre_batch_capture(tmp_path):
    captured = json.loads((REPO / 'docs/refactoring/cli_contracts.json').read_text())
    expected = next(c for c in captured['scripts'] if c['path'] == str(SCRIPT.relative_to(REPO)))
    result = invoke(tmp_path, '--help')
    assert result.returncode == expected['returncode'] == 0
    assert result.stdout == expected['stdout']
    assert not result.stderr


def test_file_import_and_default_parser_contract(monkeypatch):
    spec = importlib.util.spec_from_file_location('energy_legacy_cli', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from nek_post.energy_budget import compute_energy_diagnostics
    from nek_post.energy_budget_io import load_energy_budget
    from nek_post.paths import load_project_paths
    assert module.compute_energy_diagnostics is compute_energy_diagnostics
    assert module.load_energy_budget is load_energy_budget
    assert callable(module.main)
    assert module.REPO_ROOT == REPO
    paths = load_project_paths(REPO / 'config/paths.yaml')
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT)])
    args = module._parse_args(paths)
    assert vars(args) == dict(cases='N5,N7,N9', data_root=paths.data_root,
                             filename='energy_budget.dat', target=12.0,
                             output_dir=paths.energy_budget_closure_dir,
                             overwrite=False, no_plots=False)
    assert module._parse_cases('n7, N5') == ['N7', 'N5']
    with pytest.raises(ValueError, match='Invalid case'):
        module._parse_cases('bad')


def test_synthetic_cli_outputs_and_collision_preservation(tmp_path):
    data = tmp_path / 'input'
    case = data / 'case_N7'
    case.mkdir(parents=True)
    source = case / 'custom.dat'
    original = b'0 2 10 12 1\n1 2 9 11 1\n2 2 8 10 1\n'
    source.write_bytes(original)
    output = tmp_path / 'output'
    options = ['--cases', 'N7', '--data-root', data, '--filename', 'custom.dat',
               '--target', '12', '--output-dir', output, '--no-plots']
    result = invoke(tmp_path, *options)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert 'Figures: skipped (--no-plots)' in result.stdout
    assert 'Energy budget closure summary:' in result.stdout
    assert {p.name for p in output.iterdir()} == {
        'N7_energy_budget_closure_timeseries.csv', 'energy_budget_closure_summary.csv'}
    with (output / 'N7_energy_budget_closure_timeseries.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [r['time'] for r in rows] == ['0', '1', '2']
    assert [r['cumulative_epsilon'] for r in rows] == ['0', '1', '2']
    for r in rows:
        assert r['energy_closure'] == '12'
        assert r['closure_residual_from_target'] == '0'
        assert r['dEtotal_dt'] == '-1'
        assert r['differential_closure_residual'] == '0'
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    rejected = invoke(tmp_path, *options)
    assert rejected.returncode == 1
    assert 'ERROR: Output exists:' in rejected.stderr
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    replaced = invoke(tmp_path, *options, '--overwrite')
    assert replaced.returncode == 0, replaced.stderr
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    assert source.read_bytes() == original


def test_missing_input_keeps_output_absent(tmp_path):
    output = tmp_path / 'absent-output'
    result = invoke(tmp_path, '--cases', 'N7', '--data-root', tmp_path / 'absent-input',
                    '--output-dir', output, '--no-plots')
    assert result.returncode == 1
    assert 'ERROR: Energy budget file not found:' in result.stderr
    assert not output.exists()
