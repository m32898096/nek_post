"""Direct synthetic Script 00 behavior and allocation tests; no legacy execution."""

import errno
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nek_post.cli import main
from nek_post.cli import script00_adapter as adapter
from nek_post.cli import capabilities
from nek_post.cli.configuration import Refusal
from nek_post.cli.contracts import CONTRACTS


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path/'data/N7'
    data.mkdir(parents=True)
    for index in (1, 2):
        (data/f'GC0.f{index:05d}').touch()
    paths = {'data_root': 'data', 'case_dirs': {'N7': 'data/N7'},
             'postproc_root': 'old/post', 'results_root': 'old/results',
             'paper_data': {'cantero_fig5a_re3450_csv': 'refs/3450.csv',
                            'cantero_fig5a_re8950_csv': 'refs/8950.csv'}}
    cases = {'reference_case': 'N7', 'orders': {'N7': 7},
             'file_prefix': 'GC0', 'file_indices': [1, 2]}
    return {'base': tmp_path, 'data': data, 'paths': paths, 'cases': cases,
            'paths_config': tmp_path/'paths.yaml', 'cases_config': tmp_path/'cases.yaml',
            'root': tmp_path/'run'}


def configure(f):
    f['paths_config'].write_text(json.dumps(f['paths']), encoding='utf-8')
    f['cases_config'].write_text(json.dumps(f['cases']), encoding='utf-8')


def arguments(f):
    return {'paths_config': f['paths_config'], 'cases_config': f['cases_config'],
            'output_root': f['root'], 'cwd': f['base'], 'environ': {}}


def invoke(f, **options):
    stdout, stderr = io.StringIO(), io.StringIO()
    status = adapter.inspect_files(**(arguments(f) | options), stdout=stdout, stderr=stderr)
    return status, stdout.getvalue(), stderr.getvalue()


def log_path(f):
    return f['root']/'inspect/logs/check_files.log'


def snapshot(base):
    return {str(p.relative_to(base)): ('link', os.readlink(p)) if p.is_symlink()
            else ('file', p.read_bytes()) if p.is_file() else ('directory',)
            for p in base.rglob('*')}


def test_exact_report_log_status_and_determinism(fixture):
    f = fixture
    configure(f)
    status, stdout, stderr = invoke(f)
    expected = (
        'Nek5000 expected file check\n\n'
        'Case   Order  Index   File path                                                        Status\n'
        '------ ------ ------- ---------                                                        ------\n'
        'N7     7      1       '+str(f['data']/'GC0.f00001').ljust(64)+' OK\n'
        'N7     7      2       '+str(f['data']/'GC0.f00002').ljust(64)+' OK\n'
        '\nSummary\nTotal expected files: 2\nExisting files: 2\nMissing files: 0\n')
    assert (status, stdout, stderr) == (0, expected, '')
    assert log_path(f).read_bytes() == expected.encode('utf-8')
    assert log_path(f).stat().st_mode & 0o777 == 0o600
    f['root'] = f['base']/'second-run'
    assert invoke(f) == (status, stdout, stderr)
    assert log_path(f).read_bytes() == expected.encode('utf-8')


@pytest.mark.parametrize('kind,expected_status,existing,missing', [
    ('missing_one', 1, 1, 1), ('missing_directory', 1, 0, 2),
    ('directory_present', 0, 2, 0), ('symlink_present', 0, 2, 0),
    ('dangling_symlink', 1, 1, 1), ('empty_indices', 0, 0, 0),
])
def test_input_existence_cases(fixture, kind, expected_status, existing, missing):
    f = fixture
    first = f['data']/'GC0.f00001'
    if kind in {'missing_one', 'directory_present', 'symlink_present', 'dangling_symlink'}:
        first.unlink()
    if kind == 'directory_present':
        first.mkdir()
    if kind in {'symlink_present', 'dangling_symlink'}:
        target = f['data']/'target'
        if kind == 'symlink_present':
            target.write_bytes(b'snapshot payload must never be read')
        first.symlink_to(target)
    if kind == 'missing_directory':
        f['paths']['case_dirs']['N7'] = 'data/absent'
    if kind == 'empty_indices':
        f['cases']['file_indices'] = []
    configure(f)
    before = snapshot(f['base'])
    status, stdout, stderr = invoke(f)
    assert status == expected_status and stderr == ''
    assert f'Existing files: {existing}\nMissing files: {missing}\n' in stdout
    assert log_path(f).read_text() == stdout
    after_inputs = {p:v for p,v in snapshot(f['base']).items() if not p.startswith('run')}
    assert after_inputs == before


def test_unknown_order_case_insertion_order_and_duplicate_indices(fixture):
    f = fixture
    z = f['base']/'data/Z'
    z.mkdir()
    (z/'GC0.f00002').touch()
    f['paths']['case_dirs'] = {'Z': 'data/Z', 'N7': 'data/N7'}
    f['cases']['file_indices'] = [2, 1, 2]
    configure(f)
    status, text, error = invoke(f)
    assert status == 1 and error == ''
    rows = [line.split() for line in text.splitlines()[4:10]]
    assert [(r[0], r[1], r[2], r[-1]) for r in rows] == [
        ('Z', 'UNKNOWN', '2', 'OK'), ('Z', 'UNKNOWN', '1', 'MISSING'),
        ('Z', 'UNKNOWN', '2', 'OK'), ('N7', '7', '2', 'OK'),
        ('N7', '7', '1', 'OK'), ('N7', '7', '2', 'OK')]
    assert 'Total expected files: 6\nExisting files: 5\nMissing files: 1' in text


def test_relative_yaml_uses_canonical_config_parent_from_unrelated_cwd(fixture):
    f = fixture
    configure(f)
    elsewhere = f['base']/'elsewhere'
    elsewhere.mkdir()
    link = elsewhere/'machine.yaml'
    link.symlink_to(f['paths_config'])
    assert invoke(f, cwd=elsewhere, paths_config='machine.yaml')[0] == 0
    assert str(f['data']/'GC0.f00001') in log_path(f).read_text()


def test_explicit_and_environment_configuration_precedence(fixture):
    f = fixture
    configure(f)
    env = {'NEK_POST_PATHS_CONFIG': str(f['paths_config']),
           'NEK_POST_CASES_CONFIG': str(f['cases_config'])}
    assert invoke(f, paths_config=None, cases_config=None, environ=env, dry_run=True)[0] == 0
    assert invoke(f, environ={k:'missing.yaml' for k in env}, dry_run=True)[0] == 0
    assert not f['root'].exists()


@pytest.mark.parametrize('change,code', [
    ('empty_inventory', 'invalid_inventory'), ('missing_paper', 'invalid_inventory'),
    ('missing_cases_key', 'invalid_configuration'), ('empty_orders', 'invalid_configuration'),
    ('bad_prefix', 'invalid_configuration'), ('bad_indices', 'invalid_configuration'),
    ('bool_index', 'invalid_configuration'), ('string_indices', 'invalid_configuration'),
    ('missing_label', 'invalid_inventory'), ('case_directory_is_file', 'invalid_path'),
])
def test_invalid_configuration_refuses_before_outputs(fixture, change, code):
    f = fixture
    if change == 'empty_inventory': f['paths']['case_dirs'] = {}
    if change == 'missing_paper': f['paths'].pop('paper_data')
    if change == 'missing_cases_key': f['cases'].pop('file_indices')
    if change == 'empty_orders': f['cases']['orders'] = {}
    if change == 'bad_prefix': f['cases']['file_prefix'] = '../escape'
    if change == 'bad_indices': f['cases']['file_indices'] = ['1']
    if change == 'bool_index': f['cases']['file_indices'] = [True]
    if change == 'string_indices': f['cases']['file_indices'] = '12'
    if change == 'missing_label': f['cases']['orders']['absent'] = 3
    if change == 'case_directory_is_file': f['paths']['case_dirs']['N7'] = str(f['data']/'GC0.f00001')
    configure(f)
    before = snapshot(f['base'])
    status, stdout, stderr = invoke(f)
    assert status == 1 and stdout == '' and code in stderr
    assert snapshot(f['base']) == before


@pytest.mark.parametrize('kind', ['malformed', 'missing_paths', 'missing_cases', 'directory_config'])
def test_bad_yaml_sources(fixture, kind):
    f = fixture
    configure(f)
    if kind == 'malformed': f['paths_config'].write_text('broken: [')
    if kind == 'missing_paths': f['paths_config'].unlink()
    if kind == 'missing_cases': f['cases_config'].unlink()
    if kind == 'directory_config': f['paths_config'].unlink(); f['paths_config'].mkdir()
    status, text, error = invoke(f)
    assert status == 1 and text == '' and 'invalid_configuration' in error
    assert not f['root'].exists()


@pytest.mark.parametrize('root', ['data/N7/new', 'old/post/new', 'old/results/new',
                                 'data/results/h_refinement/new', 'data', '.',
                                 'run/../escape', 'missing-parent/run'])
def test_unauthorized_or_invalid_roots(fixture, root):
    f = fixture
    configure(f)
    before = snapshot(f['base'])
    status, text, error = invoke(f, output_root=f['base']/root)
    assert status == 1 and text == '' and error.startswith('ERROR: inspect files:')
    assert snapshot(f['base']) == before


def test_all_configured_cases_remain_protected(fixture):
    f = fixture
    f['paths']['case_dirs']['unselected'] = 'data/unselected'
    configure(f)
    assert invoke(f, output_root=f['base']/'data/unselected/out')[0] == 1
    assert not (f['base']/'data/unselected').exists()


@pytest.mark.parametrize('kind', ['file', 'directory', 'symlink', 'dangling', 'hardlink'])
def test_existing_log_is_never_replaced(fixture, kind):
    f = fixture
    configure(f)
    log = log_path(f)
    log.parent.mkdir(parents=True)
    target = f['base']/'sentinel'
    target.write_bytes(b'PRESERVE')
    if kind == 'file': log.write_bytes(b'OLD LONG LOG'*100)
    if kind == 'directory': log.mkdir()
    if kind in {'symlink', 'dangling'}: log.symlink_to(target if kind == 'symlink' else f['base']/'absent')
    if kind == 'hardlink': os.link(target, log)
    before = snapshot(f['base'])
    status, text, error = invoke(f)
    assert status == 1 and text == '' and error
    assert snapshot(f['base']) == before
    assert target.read_bytes() == b'PRESERVE'


@pytest.mark.parametrize('component', ['run', 'inspect', 'logs'])
def test_preexisting_output_symlink_ancestors(fixture, component):
    f = fixture
    configure(f)
    outside = f['base']/'outside'
    outside.mkdir()
    target = f['root'] if component == 'run' else f['root']/component if component == 'inspect' else f['root']/'inspect/logs'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    before = snapshot(f['base'])
    assert invoke(f)[0] == 1
    assert snapshot(f['base']) == before


@pytest.mark.parametrize('component', ['inspect', 'logs'])
def test_file_in_output_directory_position_is_preserved(fixture, component):
    f = fixture
    configure(f)
    target = f['root']/'inspect' if component == 'inspect' else f['root']/'inspect/logs'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'BLOCK')
    before = snapshot(f['base'])
    status, text, error = invoke(f)
    assert status == 1 and text == '' and 'invalid_path' in error
    assert snapshot(f['base']) == before


def test_failed_directory_creation_after_report_is_nonzero(fixture, monkeypatch):
    f = fixture
    configure(f)
    real_report = adapter.build_report
    def report(plan):
        def denied(*args, **kwargs):
            raise PermissionError(errno.EACCES, 'injected mkdir permission failure')
        monkeypatch.setattr(os, 'mkdir', denied)
        return real_report(plan)
    monkeypatch.setattr(adapter, 'build_report', report)
    status, text, error = invoke(f)
    assert status == 1 and 'Summary' in text and 'io_error' in error
    assert not f['root'].exists()


@pytest.mark.parametrize('component', ['run', 'inspect', 'logs'])
def test_allocation_time_directory_symlink_swap_is_not_followed(fixture, monkeypatch, component):
    f = fixture
    configure(f)
    outside = f['base']/'outside'
    outside.mkdir()
    (outside/'sentinel').write_bytes(b'PRESERVE')
    real_open, real_report = os.open, adapter.build_report
    swapped = []
    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == component and dir_fd is not None and not swapped:
            os.rmdir(path, dir_fd=dir_fd)
            os.symlink(str(outside), path, dir_fd=dir_fd)
            swapped.append(True)
        return real_open(path, flags, mode, dir_fd=dir_fd)
    def report(plan):
        monkeypatch.setattr(os, 'open', racing_open)
        return real_report(plan)
    monkeypatch.setattr(adapter, 'build_report', report)
    status, text, error = invoke(f)
    assert swapped and status == 1 and 'Summary' in text and 'io_error' in error
    assert snapshot(outside) == {'sentinel': ('file', b'PRESERVE')}


@pytest.mark.parametrize('kind', ['file', 'symlink', 'dangling'])
def test_exclusive_open_rejects_final_name_race(fixture, monkeypatch, kind):
    f = fixture
    configure(f)
    target = f['base']/'sentinel'
    target.write_bytes(b'PRESERVE')
    real_open, real_report = os.open, adapter.build_report
    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == 'check_files.log':
            if kind == 'file': log_path(f).write_bytes(b'RACED FILE')
            else: log_path(f).symlink_to(target if kind == 'symlink' else f['base']/'absent')
            assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
        return real_open(path, flags, mode, dir_fd=dir_fd)
    def report(plan):
        monkeypatch.setattr(os, 'open', racing_open)
        return real_report(plan)
    monkeypatch.setattr(adapter, 'build_report', report)
    status, text, error = invoke(f)
    assert status == 1 and 'Summary' in text and 'io_error' in error
    assert target.read_bytes() == b'PRESERVE'
    if kind == 'file': assert log_path(f).read_bytes() == b'RACED FILE'
    else: assert log_path(f).is_symlink()


@pytest.mark.parametrize('which', ['paths_config', 'cases_config'])
def test_configuration_change_after_report_refuses_allocation(fixture, monkeypatch, which):
    f = fixture
    configure(f)
    real_report = adapter.build_report
    def report(plan):
        value = real_report(plan)
        f[which].write_text('changed: true')
        return value
    monkeypatch.setattr(adapter, 'build_report', report)
    status, text, error = invoke(f)
    assert status == 1 and 'Summary' in text and 'configuration_changed' in error
    assert not f['root'].exists()


@pytest.mark.parametrize('failure', ['permission', 'partial', 'zero', 'short'])
def test_write_failures_and_short_write_completion(fixture, monkeypatch, failure):
    f = fixture
    configure(f)
    real_write = os.write
    calls = []
    def write(fd, payload):
        calls.append(len(payload))
        if failure == 'short': return real_write(fd, payload[:7])
        if failure == 'zero': return 0
        if failure == 'partial' and len(calls) == 1: return real_write(fd, payload[:7])
        if failure == 'partial': raise OSError(errno.ENOSPC, 'injected disk full after partial write')
        raise PermissionError(errno.EACCES, 'injected write permission failure')
    monkeypatch.setattr(os, 'write', write)
    status, text, error = invoke(f)
    if failure == 'short':
        assert status == 0 and error == '' and len(calls) > 1
        assert log_path(f).read_bytes() == text.encode()
    else:
        assert status == 1 and 'Summary' in text and 'io_error' in error
        assert log_path(f).read_bytes() == (text.encode()[:7] if failure == 'partial' else b'')
    # Both successful and partial products block a new attempt.
    saved = log_path(f).read_bytes()
    assert invoke(f)[0] == 1
    assert log_path(f).read_bytes() == saved


def test_real_creation_permission_error_preserves_existing_paths(fixture):
    f = fixture
    configure(f)
    folder = log_path(f).parent
    folder.mkdir(parents=True)
    folder.chmod(0o500)
    try:
        if os.geteuid() == 0: pytest.skip('Root bypasses ordinary DAC permissions')
        status, text, error = invoke(f)
        assert status == 1 and 'Summary' in text and 'Permission denied' in error
        assert not log_path(f).exists()
    finally:
        folder.chmod(0o700)


@pytest.mark.parametrize('missing', [False, True])
def test_dry_run_is_pure_validates_safety_and_has_no_environment_mutation(fixture, missing):
    f = fixture
    if missing: (f['data']/'GC0.f00002').unlink()
    configure(f)
    before, env = snapshot(f['base']), dict(os.environ)
    status, text, error = invoke(f, dry_run=True)
    assert status == int(missing) and error == '' and 'Summary' in text
    assert snapshot(f['base']) == before and dict(os.environ) == env
    assert invoke(f, dry_run=True, output_root=f['data'])[0] == 1
    assert snapshot(f['base']) == before


def test_missing_configuration_and_root_have_no_allocation(fixture):
    f = fixture
    configure(f)
    for options, code in [({'paths_config': None}, 'missing_safety_source'),
                          ({'output_root': None}, 'missing_output_root')]:
        status, text, error = invoke(f, **options)
        assert status == 1 and text == '' and code in error
        assert not f['root'].exists()


def test_safety_precedes_capabilities_and_capability_failure_precedes_allocation(fixture, monkeypatch):
    f = fixture
    configure(f)
    called = []
    def missing(command):
        called.append(command)
        req = capabilities.Requirement('yaml', 'package_dependency', 'module')
        return (capabilities.CapabilityResult(req, 'missing_required', 'synthetic missing yaml'),)
    monkeypatch.setattr(adapter, 'check_prerequisites', missing)
    assert invoke(f, output_root=f['data'])[0] == 1
    assert called == []
    status, text, error = invoke(f)
    assert status == 1 and text == '' and 'missing_dependency' in error
    assert called == ['inspect files'] and not f['root'].exists()


def test_unsupported_safe_allocation_platform_refuses(fixture, monkeypatch):
    f = fixture
    configure(f)
    monkeypatch.setattr(os, 'supports_dir_fd', set())
    assert invoke(f)[2].startswith('ERROR: inspect files: unsupported_platform:')
    assert not f['root'].exists()


def test_help_parser_public_route_and_unrelated_routes_stay_gated(fixture, capsys):
    f = fixture
    configure(f)
    with pytest.raises(SystemExit) as help_exit:
        main(['inspect', 'files', '--help'])
    assert help_exit.value.code == 0
    assert 'not_implemented' in capsys.readouterr().out
    with pytest.raises(SystemExit) as bad:
        main(['inspect', 'files', '--invented'])
    assert bad.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert main(['inspect', 'files']) == 3
    assert 'missing_safety_source' in capsys.readouterr().err
    common = ['--paths-config', str(f['paths_config']), '--cases-config', str(f['cases_config']),
              '--output-root', str(f['root'])]
    for route in [['inspect', 'files'], ['inspect', 'nek-file'], ['p-refinement', 'compare-fields']]:
        assert main([*common, *route]) == 3
        assert 'not_implemented' in capsys.readouterr().err
    assert main([*common, 'inspect', 'files', '--plan']) == 3
    plan = json.loads(capsys.readouterr().out)
    assert plan['unknown_write_set'] and not plan['execution_authorized']
    assert main([*common, 'inspect', 'files', '--dry-run']) == 3
    capsys.readouterr()
    assert CONTRACTS['inspect files'].unknown and not f['root'].exists()


def test_help_does_not_import_the_internal_adapter(tmp_path):
    probe = """
import sys
sys.path.insert(0, sys.argv[1])
from nek_post.cli import main
try: main(['inspect','files','--help'])
except SystemExit as e: assert e.code == 0
assert 'nek_post.cli.script00_adapter' not in sys.modules
assert 'yaml' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-B', '-I', '-c', probe,
                             str(Path(__file__).resolve().parents[1]/'src')],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
