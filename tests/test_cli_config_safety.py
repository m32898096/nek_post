"""Synthetic B2 fixtures exercise authorization without scientific execution."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from nek_post.cli import main
from nek_post.cli.configuration import Refusal, resolve_configuration
from nek_post.cli.contracts import CONTRACTS, READ_ONLY
from nek_post.cli.parser import build_parser
from nek_post.cli.registry import COMMANDS
from nek_post.cli.safety import build_context

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def machine(tmp_path):
    values = {'data_root': 'data', 'case_dirs': {k: 'data/'+k for k in
              ('N5', 'N7', 'N9', 'N11', 'GC8950_N7', 'N7_H', 'N7_VH', 'N7_VVH')},
              'postproc_root': 'old/postproc', 'results_root': 'old/results',
              'paper_data': {'cantero_fig5a_re3450_csv': 'paper/3450.csv',
                             'cantero_fig5a_re8950_csv': 'paper/8950.csv'}}
    path = tmp_path/'paths.yaml'
    path.write_text(yaml.safe_dump(values))
    return path, values


def context(machine):
    return build_context(resolve_configuration('paths', machine[0], cwd=machine[0].parent))


def test_precedence_and_canonical_parent(machine, tmp_path):
    path, values = machine
    other = tmp_path/'bad.yaml'
    other.write_text('[')
    source = resolve_configuration('paths', path, cwd=tmp_path,
                                   environ={'NEK_POST_PATHS_CONFIG': str(other)})
    link = tmp_path/'link.yaml'
    link.symlink_to(path)
    linked = resolve_configuration('paths', link.name, cwd=tmp_path)
    assert linked.canonical == path
    assert context(machine).data_root == tmp_path/'data'
    assert source.values['case_dirs']['GC8950_N7'] == 'data/GC8950_N7'
    assert resolve_configuration('paths', cwd=tmp_path,
                                environ={'NEK_POST_PATHS_CONFIG': str(path)}).digest == source.digest
    with pytest.raises(Refusal):
        resolve_configuration('paths', other, cwd=tmp_path,
                              environ={'NEK_POST_PATHS_CONFIG': str(path)})
    path.write_text('changed: true')
    with pytest.raises(Refusal, match='changed'):
        source.verify_unchanged()


@pytest.mark.parametrize('raw', ['', '[]', 'false', 'bad: [', 'null', '1: value'])
def test_invalid_configurations(tmp_path, raw):
    path = tmp_path/'bad.yaml'
    path.write_text(raw)
    with pytest.raises(Refusal):
        resolve_configuration('paths', path, cwd=tmp_path)


def test_missing_source_and_resources(tmp_path):
    with pytest.raises(Refusal, match='--paths-config'):
        resolve_configuration('paths', cwd=tmp_path, environ={})
    with pytest.raises(Refusal):
        resolve_configuration('paths', 'missing.yaml', cwd=tmp_path)
    for role, filename in [('cases', 'cases.yaml'), ('study', 'h_refinement.yaml')]:
        source = resolve_configuration(role, cwd=tmp_path, environ={})
        assert source.canonical is None
        assert dict(source.values)
        assert yaml.safe_load((REPO/'src/nek_post/resources'/filename).read_text()) == yaml.safe_load((REPO/'config'/filename).read_text())


def test_complete_inventory_and_views(machine, tmp_path):
    c = context(machine)
    assert len([p for p in c.protected if p.role == 'raw_case']) == 8
    assert any(p.path == tmp_path/'data/results/h_refinement' for p in c.protected)
    assert len([p for p in c.protected if p.role == 'reference']) == 2
    root = c.check_root(tmp_path/'data/new-run', cwd=tmp_path)
    assert c.output_view(root)['results_root'] == root/'p-refinement/results'
    effective = c.effective_paths()
    assert effective['results_root'] == str(tmp_path/'old/results')
    assert effective['h_refinement_results_root'] == str(tmp_path/'data/results/h_refinement')
    with pytest.raises(TypeError):
        c.input_values['results_root'] = 'changed'
    assert c.check_root(tmp_path/'paper/sibling', cwd=tmp_path)


@pytest.mark.parametrize('key,value', [('case_dirs', {}), ('case_dirs', []), ('data_root', None),
                                      ('results_root', ''), ('paper_data', {}), ('unknown', 'root')])
def test_incomplete_inventory(machine, key, value):
    path, values = machine
    values[key] = value
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(Refusal):
        context(machine)


@pytest.mark.parametrize('suffix', ['data/N7/output', 'data/GC8950_N7/out', 'old/results/out',
                                   'old/postproc/out', 'data/results/h_refinement/new', 'data', '.'])
def test_protected_roots(machine, tmp_path, suffix):
    with pytest.raises(Refusal):
        context(machine).check_root(tmp_path/suffix, cwd=tmp_path)


def test_explicit_h_root_and_required_labels(machine, tmp_path):
    path, values = machine
    values['h_refinement_results_root'] = 'explicit-h'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(Refusal):
        context(machine).check_root(tmp_path/'explicit-h/out', cwd=tmp_path)
    with pytest.raises(Refusal, match='missing required'):
        build_context(resolve_configuration('paths', path, cwd=tmp_path), required_labels=['ABSENT'])
    for item in context(machine).protected:
        if item.role in {'package', 'native_ancestor'}:
            with pytest.raises(Refusal):
                context(machine).check_root(item.path/'out', cwd=tmp_path)


def test_symlinks_types_traversal_and_exact_directory(machine, tmp_path):
    c = context(machine)
    root = c.check_root(tmp_path/'run', cwd=tmp_path)
    root.mkdir()
    d = root/'runs/workers_2/run_001'
    d.mkdir(parents=True)
    assert c.destination(root, d, cwd=tmp_path, kind='directory', fresh=False) == d
    assert c.destination(root, d/'N7_leading_edge_timeseries.csv', cwd=tmp_path).parent == d
    with pytest.raises(Refusal):
        c.destination(root, tmp_path/'outside', cwd=tmp_path)
    with pytest.raises(Refusal):
        c.destination(root, root/'x/../escape', cwd=tmp_path)
    (root/'alias').symlink_to(tmp_path/'outside')
    with pytest.raises(Refusal):
        c.destination(root, root/'alias/file', cwd=tmp_path)
    (root/'file').write_text('existing')
    with pytest.raises(Refusal):
        c.destination(root, root/'file', cwd=tmp_path)
    with pytest.raises(Refusal):
        c.destination(root, root/'file/child', cwd=tmp_path)
    (root/'dangling').symlink_to(tmp_path/'missing')
    with pytest.raises(Refusal):
        c.destination(root, root/'dangling', cwd=tmp_path)


def test_publication_exception_and_partial_pair(machine, tmp_path):
    c = context(machine)
    preview = tmp_path/'preview'
    preview.mkdir()
    p = tmp_path/'old/results'
    targets = c.publication_targets(p, preview, cwd=tmp_path)
    assert targets == (p/'README.md', p/'results_index.csv')
    p.mkdir(parents=True)
    targets[0].write_text('preserve partial publication')
    with pytest.raises(Refusal, match='preserve partial'):
        c.publication_targets(p, preview, cwd=tmp_path)
    with pytest.raises(Refusal):
        c.publication_targets(tmp_path/'data/N7', preview, cwd=tmp_path)
    with pytest.raises(Refusal):
        c.publication_targets(preview, preview, cwd=tmp_path)


def test_contract_authority():
    rows = [line.split('|') for line in (REPO/'docs/refactoring/phase4/destination_contracts.md').read_text().splitlines()
            if line.startswith('| `') and len(line.split('|')) == 9]
    assert set(CONTRACTS) == {c.command_id for c in COMMANDS}
    for row in rows:
        c = CONTRACTS[row[1].strip().strip('`')]
        assert (c.primary, c.products, c.companions, c.unknown) == (row[2].strip(), row[3].strip(), row[4].strip(), 'Unknown' in row[6])
    assert sum(c.unknown for c in CONTRACTS.values()) == 31
    assert len(READ_ONLY) == 3


@pytest.mark.parametrize('command', [c.command_id for c in COMMANDS if c.command_id not in READ_ONLY])
@pytest.mark.parametrize('mode', [[], ['--plan'], ['--dry-run']])
def test_all_writing_routes_require_safety(command, mode, monkeypatch, capsys):
    monkeypatch.delenv('NEK_POST_PATHS_CONFIG', raising=False)
    assert main([*command.split(), *mode]) == 3
    assert 'missing_safety_source' in capsys.readouterr().err


def test_cli_plans_are_read_only_and_incomplete(machine, tmp_path, capsys):
    before = dict(os.environ)
    root = tmp_path/'output'
    args = ['--paths-config', str(machine[0]), '--output-root', str(root), 'results', 'catalog', 'build', '--plan']
    assert main(args) == 3
    plan = json.loads(capsys.readouterr().out)
    assert plan['destinations'] == [str(root/'catalog')]
    assert not plan['execution_authorized'] and plan['status'] == 'incomplete'
    assert not root.exists() and dict(os.environ) == before
    assert main(['results', 'navigation', 'preview', '--paths-config', str(machine[0]), '--output-root', str(root), '--preview-dir', str(tmp_path/'outside'), '--plan']) == 3
    assert 'outside_authorization' in capsys.readouterr().err
    assert main(['results', 'inventory', '--input-root', str(tmp_path), '--plan']) == 0
    assert not json.loads(capsys.readouterr().out)['execution_authorized']
    with pytest.raises(SystemExit) as error:
        main(['--plan', 'results', 'inventory', '--dry-run'])
    assert error.value.code == 2


def test_parser_preserves_root_options():
    args = build_parser().parse_args(['--paths-config', 'one', 'results', 'catalog', 'build', '--plan'])
    assert args.paths_config == 'one' and args.plan


def test_fresh_failure_has_no_writes_or_handler_imports(machine, tmp_path):
    probe = '''
import importlib.abc, os, sys
sys.path.insert(0, sys.argv[1])
before = dict(os.environ)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('numpy', 'matplotlib', 'pymech', 'vtk', 'nek_post.vortex', 'nek_post.results_catalog')):
            raise AssertionError(fullname)
def audit(event, args):
    if event == 'open':
        assert not args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND), args
    assert event not in {'os.mkdir','os.remove','os.rename','os.putenv','os.unsetenv','subprocess.Popen'}, event
sys.meta_path.insert(0, Guard())
sys.addaudithook(audit)
from nek_post.cli import main
assert main(['results','catalog','build','--paths-config',sys.argv[2],'--output-root',sys.argv[3],'--plan']) == 3
assert dict(os.environ) == before
'''
    result = subprocess.run([sys.executable, '-B', '-I', '-c', probe, str(REPO/'src'), str(machine[0]), str(tmp_path/'data/GC8950_N7/out')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_configuration_symlink_uses_target_directory(machine, tmp_path):
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    link = elsewhere/'paths.yaml'
    link.symlink_to(machine[0])
    source = resolve_configuration('paths', link, cwd=elsewhere)
    assert build_context(source).data_root == tmp_path/'data'
    assert build_context(source).data_root != elsewhere/'data'


@pytest.mark.parametrize('role,raw', [('cases', 'orders: {N7: 7}\nreference_case: []'),
                                     ('study', 'cases: [N7]\nprovisional_reference_case: []')])
def test_invalid_scientific_schema(role, raw, tmp_path):
    file = tmp_path/'invalid.yaml'
    file.write_text(raw)
    with pytest.raises(Refusal):
        resolve_configuration(role, file, cwd=tmp_path)


@pytest.mark.parametrize('command', [c.command_id for c in COMMANDS if c.command_id not in READ_ONLY])
def test_valid_safety_never_enables_workflows(machine, tmp_path, command, capsys):
    root = tmp_path/'run'
    assert main([*command.split(), '--paths-config', str(machine[0]), '--output-root', str(root), '--dry-run']) == 3
    assert not root.exists()
    assert 'registered but not implemented' in capsys.readouterr().err


def test_publication_v_is_read_only(machine, tmp_path, capsys):
    root = tmp_path/'run'
    preview = root/'navigation-preview'
    preview.mkdir(parents=True)
    (preview/'README.md').write_text('reviewed')
    assert main(['results','navigation','publish','--paths-config',str(machine[0]),
                 '--output-root',str(root),'--publication-root',str(tmp_path/'old/results'),
                 '--preview-dir',str(preview),'--plan']) == 3
    plan = json.loads(capsys.readouterr().out)
    assert plan['destinations'] == [str(tmp_path/'old/results/README.md'), str(tmp_path/'old/results/results_index.csv')]
    assert (preview/'README.md').read_text() == 'reviewed'
    assert not (root/'.cli').exists()
    assert main(['results','navigation','publish','--paths-config',str(machine[0]),
                 '--output-root',str(root),'--publication-root',str(tmp_path/'old/results'),
                 '--preview-dir',str(root/'.cli'),'--plan']) == 3
    assert not (root/'.cli').exists()


def test_declared_inputs_and_equivalent_paths(machine, tmp_path):
    source = resolve_configuration('paths', machine[0], cwd=tmp_path)
    saved = tmp_path/'saved'
    saved.mkdir()
    c = build_context(source, inputs=[saved])
    for path in [saved, saved/'out', tmp_path]:
        with pytest.raises(Refusal):
            c.check_root(path, cwd=tmp_path)
    r = c.check_root(tmp_path/'safe', cwd=tmp_path)
    assert c.destination(r, './safe/output', cwd=tmp_path) == r/'output'
    with pytest.raises(Refusal):
        c.destination(r, tmp_path/'safe-other/output', cwd=tmp_path)


@pytest.mark.parametrize('failure', ['symlink', 'ancestor', 'resolution'])
def test_structured_path_refusals(machine, tmp_path, capsys, failure):
    from nek_post.cli.safety import normalize_path
    if failure == 'symlink':
        path = tmp_path/'alias'
        path.symlink_to(tmp_path/'elsewhere')
        expected = ('unsafe_path', 'Symlink in output:')
    elif failure == 'ancestor':
        parent = tmp_path/'file'
        parent.write_text('immutable')
        path = parent/'child'
        expected = ('invalid_path', 'Non-directory ancestor:')
    else:
        path = str(tmp_path) + '/nul\0path'
        expected = ('invalid_path', 'Cannot resolve path:')
    with pytest.raises(Refusal) as error:
        normalize_path(path, anchor=tmp_path, output=True)
    assert error.value.code == expected[0]
    assert expected[1] in str(error.value)
    assert main(['results', 'catalog', 'build', '--paths-config', str(machine[0]),
                 '--output-root', str(path), '--plan']) == 3
    output = capsys.readouterr()
    assert not output.out
    assert expected[0] + ':' in output.err and expected[1] in output.err


@pytest.fixture
def shared_r(machine, tmp_path):
    root = tmp_path/'shared'
    effective = root/'.cli/config/effective.yaml'
    effective.parent.mkdir(parents=True)
    effective.write_text('effective: immutable\n')
    source = resolve_configuration('paths', machine[0], cwd=tmp_path)
    safety = build_context(source, run_owned_configurations=[effective])
    return safety, root, effective


def test_explicit_shared_r_and_exact_benchmark_destination(shared_r, tmp_path):
    safety, root, effective = shared_r
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert safety.check_root(root, cwd=tmp_path) == root
    assert any(p.path == effective and p.role == 'run_owned_configuration' for p in safety.protected)
    for phase in ('warmups', 'runs'):
        directory = root/'leading-edge-benchmarks/N7'/phase/'workers_4/run_001'
        assert safety.destination(root, directory, cwd=tmp_path, kind='directory') == directory
        for name in ('N7_leading_edge_timeseries.csv', 'N7_leading_edge_metadata.csv'):
            assert safety.destination(root, directory/name, cwd=tmp_path).parent == directory
        assert not directory.exists()
    assert {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    assert not (root/'leading-edge-benchmarks').exists()


@pytest.mark.parametrize('operation', ['write', 'replace', 'delete'])
@pytest.mark.parametrize('target', ['file', 'config-directory', 'cli-directory'])
def test_shared_r_immutable_targets(shared_r, tmp_path, operation, target):
    safety, root, effective = shared_r
    path = {'file': effective, 'config-directory': effective.parent,
            'cli-directory': effective.parent.parent}[target]
    with pytest.raises(Refusal) as error:
        safety.destination(root, path, cwd=tmp_path,
                           kind='file' if target == 'file' else 'directory', fresh=operation == 'write')
    assert error.value.code == 'protected_overlap'
    assert 'run_owned_configuration' in str(error.value)
    assert effective.read_text() == 'effective: immutable\n'


def test_shared_r_alias_and_non_inferred_roles(shared_r, tmp_path):
    safety, root, effective = shared_r
    alias = root/'alias.yaml'
    alias.symlink_to(effective)
    with pytest.raises(Refusal) as error:
        safety.destination(root, alias, cwd=tmp_path, fresh=False)
    assert error.value.code == 'unsafe_path'
    with pytest.raises(Refusal) as error:
        safety.destination(root, str(root)+'/./.cli/config/effective.yaml', cwd=tmp_path, fresh=False)
    assert error.value.code == 'protected_overlap'
    # Prefix and filename alone confer no special role.
    ordinary = build_context(safety.source, configurations=[effective])
    with pytest.raises(Refusal):
        ordinary.check_root(root, cwd=tmp_path)
    # Explicit ordinary protection wins even if the same file is also declared run-owned.
    both = build_context(safety.source, inputs=[effective], run_owned_configurations=[effective])
    with pytest.raises(Refusal):
        both.check_root(root, cwd=tmp_path)


@pytest.mark.parametrize('protected', ['raw_case', 'historical', 'reference', 'package', 'native_ancestor', 'input'])
def test_run_owned_role_cannot_exempt_other_protection(shared_r, tmp_path, protected):
    safety, root, effective = shared_r
    item = next(p for p in safety.protected if p.role == protected)
    with pytest.raises(Refusal):
        safety.check_root(item.path, cwd=tmp_path)
    with pytest.raises(Refusal):
        safety.destination(root, item.path, cwd=tmp_path, kind='directory' if item.tree else 'file', fresh=False)
    with pytest.raises(Refusal):
        safety.check_root(item.path.parent, cwd=tmp_path)
    # The new role cannot exempt a protected path by duplicate declaration.
    if not item.tree and item.path.is_file():
        duplicate = build_context(safety.source, run_owned_configurations=[item.path])
        with pytest.raises(Refusal):
            duplicate.check_root(item.path.parent, cwd=tmp_path)


@pytest.mark.parametrize('invalid', ['relative', 'absent', 'directory', 'symlink'])
def test_run_owned_requires_explicit_existing_file(shared_r, tmp_path, invalid):
    safety, root, effective = shared_r
    alias = root/'effective-link.yaml'
    if invalid == 'symlink':
        alias.symlink_to(effective)
    path = {'relative': Path('effective.yaml'), 'absent': root/'absent.yaml',
            'directory': effective.parent, 'symlink': alias}[invalid]
    with pytest.raises(Refusal):
        build_context(safety.source, run_owned_configurations=[path])


def test_shared_r_publication_inputs_remain_protected(shared_r, tmp_path):
    safety, root, effective = shared_r
    preview = tmp_path/'reviewed-v'
    preview.mkdir()
    with pytest.raises(Refusal):
        safety.publication_targets(effective.parent, preview, cwd=tmp_path)
    with pytest.raises(Refusal):
        safety.publication_targets(root, root, cwd=tmp_path)
    with pytest.raises(Refusal):
        safety.destination(root, root/'escape/../outside', cwd=tmp_path)


def test_shared_r_fresh_interpreter_zero_write(shared_r, tmp_path):
    safety, root, effective = shared_r
    probe = '''
import importlib.abc, os, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
before = dict(os.environ)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','pymech','matplotlib','vtk','pyvista'}:
            raise AssertionError(fullname)
        if fullname.startswith('nek_post.') and not fullname.startswith(('nek_post.cli','nek_post.resources')):
            raise AssertionError(fullname)
def audit(event, args):
    if event == 'open':
        assert not args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND), args
    assert event not in {'os.mkdir','os.remove','os.rmdir','os.rename','os.putenv','os.unsetenv','subprocess.Popen'}, event
sys.meta_path.insert(0, Guard())
sys.addaudithook(audit)
from nek_post.cli.configuration import resolve_configuration, Refusal
from nek_post.cli.safety import build_context
cwd, root, effective = map(Path, sys.argv[2:5])
c = build_context(resolve_configuration('paths', sys.argv[5], cwd=cwd), run_owned_configurations=[effective])
assert c.check_root(root, cwd=cwd) == root
assert c.destination(root, root/'runs/workers_4/run_001', cwd=cwd, kind='directory') == root/'runs/workers_4/run_001'
for target, kind in [(effective,'file'), (effective.parent,'directory')]:
    try: c.destination(root, target, cwd=cwd, kind=kind, fresh=False)
    except Refusal as error: assert error.code == 'protected_overlap'
    else: raise AssertionError(target)
assert dict(os.environ) == before
'''
    result = subprocess.run([sys.executable, '-B', '-I', '-c', probe, str(REPO/'src'),
                             str(tmp_path), str(root), str(effective), str(safety.source.canonical)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (root/'runs').exists()


def test_shared_r_effective_paths_source_requires_explicit_role(machine, tmp_path):
    original = resolve_configuration('paths', machine[0], cwd=tmp_path)
    original_context = build_context(original)
    root = tmp_path/'child-run'
    effective = root/'.cli/config/paths.yaml'
    effective.parent.mkdir(parents=True)
    effective.write_text(yaml.safe_dump(original_context.effective_paths()))
    source = resolve_configuration('paths', effective, cwd=tmp_path)
    with pytest.raises(Refusal):
        build_context(source).check_root(root, cwd=tmp_path)
    child = build_context(source, configurations=[original.canonical],
                          run_owned_configurations=[effective])
    assert child.check_root(root, cwd=tmp_path) == root
    assert child.effective_paths() == original_context.effective_paths()
    assert child.destination(root, root/'runs/run_001', cwd=tmp_path, kind='directory') == root/'runs/run_001'
    for path in (original.canonical, effective):
        assert any(item.path == path for item in child.protected)
        with pytest.raises(Refusal):
            child.destination(root, path, cwd=tmp_path, fresh=False)
    # An ordinary original-input declaration cannot be overridden by another role.
    both = build_context(source, configurations=[effective], run_owned_configurations=[effective])
    with pytest.raises(Refusal):
        both.check_root(root, cwd=tmp_path)
