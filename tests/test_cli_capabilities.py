"""B3 passive discovery, environment planning and installed operation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nek_post.cli import capabilities as cap
from nek_post.cli.configuration import Refusal, resolve_configuration
from nek_post.cli.environment import plan_environment
from nek_post.cli.safety import build_context

REPO = Path(__file__).resolve().parents[1]


def test_profiles_match_architecture():
    rows = [line.split('|') for line in (REPO/'docs/refactoring/phase4/cli_migration_matrix.md').read_text().splitlines()
            if line.startswith('| ') and len(line.split('|')) == 12 and line.split('|')[1].strip() in {*(f'{i:02d}' for i in range(45)), 'Extra'}]
    assert dict(cap.ROUTE_PROFILES) == {row[3].strip().strip('`'): row[-2].strip().split('; ')[-1] for row in rows}
    assert len(cap.ROUTE_PROFILES) == 49
    assert 'matplotlib' not in {r.identity for r in cap.requirements_for('inspect files')}
    assert 'pdftoppm' in {r.identity for r in cap.requirements_for('omega-x saved-scalar-slice')}
    native = {r.identity: r for r in cap.requirements_for('vortex export')}
    assert native['renameat2(RENAME_NOREPLACE)'].platforms == ('linux',)
    assert {'fcntl', 'resource'} <= native.keys()
    assert '/usr/bin/time' in {r.identity for r in cap.requirements_for('leading-edge benchmark')}
    assert 'vtk' in {r.identity for r in cap.requirements_for('lambda-ci tensor-slices')}
    with pytest.raises(KeyError):
        cap.requirements_for('unknown')


@pytest.mark.parametrize('present,required,status', [(True, True, 'available'), (True, False, 'available'),
                                                    (False, True, 'missing_required'), (False, False, 'missing_optional')])
def test_module_presence(monkeypatch, present, required, status):
    monkeypatch.setattr(cap, 'find_spec', lambda name: object() if present else None)
    result = cap.check_requirement(cap.Requirement('numpy', 'optional_scientific', 'module', required))
    assert result.status == status and not result.execution_authorized
    assert 'unverified' in result.diagnostic


@pytest.mark.parametrize('command,identity', [
    ('lambda-ci native', 'resource'),
    ('lambda-ci tensor-slices', 'resource'),
    ('lambda-ci tensor-slice-series', 'resource'),
    ('omega-x native', 'resource'),
    ('omega-x validate-series', 'resource'),
    ('omega-x publish-series-pvd', 'resource'),
    ('omega-x saved-scalar-slice', 'resource'),
    ('omega-x saved-scalar-series', 'resource'),
    ('lambda-ci tensor-slices', 'fcntl'),
    ('omega-x validate-series', 'fcntl'),
    ('omega-x publish-series-pvd', 'fcntl'),
])
def test_route_platform_requirements(monkeypatch, command, identity):
    """Exercise native imports, including the starred tensor-slice export branch."""
    monkeypatch.setattr(cap.shutil, 'which', lambda name, path: '/synthetic/'+name.lstrip('/'))
    unrelated = ['inspect files', 'p-refinement plot-summary', 'leading-edge benchmark']
    if identity == 'fcntl':
        unrelated.append('lambda-ci native')
    for present in (True, False):
        module_spec = object() if present else None
        monkeypatch.setattr(cap, 'find_spec', lambda name: module_spec if name == identity else object())
        results = cap.check_prerequisites(command)
        result = next(item for item in results if item.requirement.identity == identity)
        assert result.requirement.required
        assert result.requirement.category == 'platform' and result.requirement.probe == 'module'
        assert result.status == ('available' if present else 'missing_required')
        assert result.diagnostic.startswith(f'{command}: {identity}: ')
        assert [item.requirement.identity for item in results if item.status == 'missing_required'] == ([] if present else [identity])
        assert all(not item.execution_authorized for item in results)
        assert results == cap.check_prerequisites(command)
        for other in unrelated:
            other_results = cap.check_prerequisites(other)
            assert identity not in {item.requirement.identity for item in other_results}
            assert all(not item.execution_authorized for item in other_results)


def test_diagnostics(monkeypatch):
    assert cap.check_requirement(cap.Requirement('fcntl', 'platform', 'module', platforms=('linux',)), platform='win32').status == 'unsupported_platform'
    monkeypatch.setattr(cap.sys, 'version_info', (3, 11))
    assert cap.check_requirement(cap.Requirement('python>=3.12', 'stdlib', 'python')).status == 'unsupported_version'
    for probe in ('deferred', 'unknown'):
        result = cap.check_requirement(cap.Requirement('native-filesystem', 'workflow', probe))
        assert result.status == ('deferred' if probe == 'deferred' else 'indeterminate')
    assert cap.check_requirement(cap.Requirement('parent.child', 'scientific', 'module')).status == 'indeterminate'
    def broken(name):
        raise ValueError('broken spec')
    monkeypatch.setattr(cap, 'find_spec', broken)
    assert cap.check_requirement(cap.Requirement('numpy', 'scientific', 'module')).status == 'indeterminate'


def test_tools_are_never_executed(tmp_path):
    tool = tmp_path/'pdftoppm'
    tool.write_text('#!/bin/sh\ntouch "'+str(tmp_path/'executed')+'"\n')
    tool.chmod(0o755)
    requirement = cap.Requirement('pdftoppm', 'external_tool', 'executable')
    assert cap.check_requirement(requirement, search_path=str(tmp_path)).status == 'indeterminate'
    assert cap.check_requirement(requirement, search_path=str(tmp_path/'absent')).status == 'missing_required'
    assert not (tmp_path/'executed').exists()


@pytest.fixture
def safety(tmp_path):
    source = tmp_path/'paths.yaml'
    source.write_text('''data_root: data
case_dirs: {N7: data/N7, unselected: data/unselected}
postproc_root: old/postproc
results_root: old/results
paper_data: {cantero_fig5a_re3450_csv: refs/3450.csv, cantero_fig5a_re8950_csv: refs/8950.csv}
''')
    return build_context(resolve_configuration('paths', source, cwd=tmp_path))


@pytest.mark.parametrize('supplied', [False, True])
def test_environment_is_pure_authorized_and_immutable(safety, tmp_path, supplied):
    inherited = {'KEEP': 'yes', 'PYTHONPATH': '/external'}
    if supplied:
        inherited.update(MPLCONFIGDIR='/external/mpl', TMPDIR='/external/tmp', MPLBACKEND='svg')
    original = dict(inherited)
    parent = dict(os.environ)
    root = tmp_path/'run'
    spec = plan_environment(safety, root, cwd=tmp_path, inherited=inherited, unset=('PYTHONPATH',), benchmark=True)
    assert dict(os.environ) == parent and inherited == original
    assert dict(spec.inherited) == original and spec.environment['KEEP'] == 'yes'
    assert 'PYTHONPATH' not in spec.environment and spec.environment['LC_ALL'] == 'C'
    assert spec.environment['MPLBACKEND'] == 'Agg'
    assert spec.environment['MPLCONFIGDIR'] == str(root/'.cli/matplotlib')
    assert all(path.is_relative_to(root) for path in spec.writable_paths)
    assert spec.cwd == root/'.cli/cwd' and not root.exists()
    assert not spec.execution_authorized
    inherited['KEEP'] = 'changed'
    assert spec.inherited['KEEP'] == 'yes'
    with pytest.raises(TypeError):
        spec.environment['KEEP'] = 'bad'
    with pytest.raises(Refusal):
        plan_environment(safety, root, cwd=tmp_path, unset=('TMPDIR',))


@pytest.mark.parametrize('unsafe', ['data/N7/out', 'data/unselected/out', 'old/results/out', 'data', 'x/../escape'])
def test_environment_refuses_unsafe_roots(safety, tmp_path, unsafe):
    with pytest.raises(Refusal):
        plan_environment(safety, tmp_path/unsafe, cwd=tmp_path)


def test_environment_protects_destination_inputs(safety, tmp_path):
    root = tmp_path/'run'
    effective = root/'.cli/config/effective.yaml'
    effective.parent.mkdir(parents=True)
    effective.write_text('immutable')
    context = build_context(safety.source, run_owned_configurations=[effective])
    assert plan_environment(context, root, cwd=tmp_path).cwd == root/'.cli/cwd'
    (root/'.cli/tmp').symlink_to(tmp_path/'elsewhere')
    with pytest.raises(Refusal):
        plan_environment(context, root, cwd=tmp_path)


def test_fresh_passive_discovery_has_no_side_effects(tmp_path):
    probe = r'''
import sys, os, importlib.abc
sys.path.insert(0, sys.argv[1])
def audit(event, args):
    if event == 'import':
        name = args[0]
        assert name.split('.')[0] not in {'numpy','scipy','yaml','pymech','matplotlib','PIL','vtk','pyvista'}
        assert not name.startswith('nek_post.') or name.startswith('nek_post.cli')
    if event == 'open':
        assert not args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
    assert event not in {'os.mkdir','os.remove','os.rename','os.putenv','os.unsetenv','subprocess.Popen'}
from nek_post.cli import capabilities


before = dict(os.environ)
sys.addaudithook(audit)
for route in capabilities.ROUTE_PROFILES:
    results = capabilities.check_prerequisites(route)
    assert results and all(not result.execution_authorized for result in results)
assert dict(os.environ) == before
'''
    result = subprocess.run([sys.executable, '-B', '-I', '-c', probe, str(REPO/'src')], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


from test_cli_foundation import installed_wheel


def test_installed_b3_modules(installed_wheel, tmp_path):
    python, console, package = installed_wheel
    code = "from nek_post.cli.capabilities import check_prerequisites; from nek_post.cli.environment import EnvironmentSpec; r=check_prerequisites('inspect files'); assert any(x.status == 'missing_required' for x in r); assert all(not x.execution_authorized for x in r); print('installed B3 verified')"
    result = subprocess.run([str(python), '-B', '-I', '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'installed B3 verified'
