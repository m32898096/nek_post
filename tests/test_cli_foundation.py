"""Phase 4B-1 command coverage, purity and offline installed-wheel checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from nek_post.cli import UNAVAILABLE_EXIT_STATUS, main
from nek_post.cli.parser import build_parser
from nek_post.cli.registry import COMMANDS, GROUP_HELP


REPO = Path(__file__).resolve().parents[1]
ROUTES = [tuple(command.command_id.split()) for command in COMMANDS]
GROUPS = sorted({route[:depth] for route in ROUTES for depth in range(1, len(route))})
HELP_ROUTES = [(), *GROUPS, *ROUTES]


def test_registry_matches_committed_architecture():
    matrix = REPO / "docs/refactoring/phase4/cli_migration_matrix.md"
    rows = [line.split("|") for line in matrix.read_text().splitlines()
            if re.match(r"\| (?:\d\d|Extra) \|", line)]
    expected = {row[3].strip().strip("`"): row[4].strip() for row in rows}
    identities = [command.command_id for command in COMMANDS]
    assert len(rows) == len(expected) == len(identities) == len(set(identities)) == 49
    assert {command.command_id: command.help for command in COMMANDS} == expected
    assert {command.status for command in COMMANDS} == {"not_implemented"}
    assert all(command.group and command.leaf for command in COMMANDS)
    assert set(GROUP_HELP) == set(GROUPS)


def test_parser_hierarchy_matches_registry():
    def leaves(parser, prefix=()):
        children = [action for action in parser._actions
                    if isinstance(action, argparse._SubParsersAction)]
        if not children:
            return {prefix}
        assert len(children) == 1 and children[0].required
        found = set()
        for name, child in children[0].choices.items():
            found.update(leaves(child, (*prefix, name)))
        return found

    parser = build_parser()
    assert leaves(parser) == set(ROUTES)
    for command in COMMANDS:
        assert parser.parse_args(command.command_id.split()).command == command


@pytest.mark.parametrize("route", HELP_ROUTES, ids=lambda route: " ".join(route) or "root")
def test_help(route, capsys):
    with pytest.raises(SystemExit) as error:
        main([*route, "--help"])
    assert error.value.code == 0
    output = capsys.readouterr()
    assert "usage: nek-post" in output.out and not output.err
    assert "Execution exits with status 3" in " ".join(output.out.split())


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.command_id)
def test_unavailable_execution(command, capsys):
    assert main(command.command_id.split()) == UNAVAILABLE_EXIT_STATUS == 3
    output = capsys.readouterr()
    assert not output.out
    assert command.command_id in output.err
    assert "registered but not implemented" in output.err
    assert "not_implemented" in output.err


@pytest.mark.parametrize("route", [(), *GROUPS], ids=lambda route: " ".join(route) or "root")
def test_missing_command(route, capsys):
    with pytest.raises(SystemExit) as error:
        main(list(route))
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out and "required" in output.err


@pytest.mark.parametrize("route", [(), *GROUPS], ids=lambda route: " ".join(route) or "root")
def test_unknown_command(route, capsys):
    with pytest.raises(SystemExit) as error:
        main([*route, "unknown-command"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--overwrite", "--future-option"])
def test_future_execution_options_are_unrecognized(flag, capsys):
    with pytest.raises(SystemExit) as error:
        main(["leading-edge", "extract", flag])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# A fresh interpreter blocks scientific imports, configuration/production access,
# all filesystem writes and child execution, including transient create/delete.
PURITY_PROBE = r'''
from contextlib import redirect_stdout, redirect_stderr
import importlib.abc
import io
import json
import os
import runpy
import sys

source, mode, encoded_calls = sys.argv[1:]
if source:
    sys.path.insert(0, source)
calls = json.loads(encoded_calls)
environment = dict(os.environ)
allowed_package_modules = {
    "nek_post", "nek_post.cli", "nek_post.cli.parser",
    "nek_post.cli.registry", "nek_post.cli.__main__",
    "nek_post.cli.configuration", "nek_post.cli.contracts", "nek_post.cli.safety", "nek_post.cli.planning",
}
scientific_roots = {"numpy", "scipy", "pymech", "matplotlib", "vtk",
                    "vtkmodules", "pyvista", "PIL", "yaml"}

class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in scientific_roots:
            raise AssertionError("Scientific import: " + fullname)
        if fullname.startswith("nek_post") and fullname not in allowed_package_modules:
            raise AssertionError("Handler import: " + fullname)

def audit(event, args):
    if event == "open":
        filename, mode, flags = args
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        assert not flags & write_flags, (event, args)
        if isinstance(filename, (str, bytes)):
            filename = os.fsdecode(filename)
            assert not filename.endswith((".yaml", ".yml")), filename
            assert not filename.startswith(("/data/", os.environ["B1_PROTECTED_ROOT"])), filename
    if event in {"os.listdir", "os.scandir"} and isinstance(args[0], (str, bytes)):
        filename = os.fsdecode(args[0])
        assert not filename.startswith(("/data/", os.environ["B1_PROTECTED_ROOT"])), filename
    assert event not in {
        "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink",
        "os.truncate", "os.chmod", "os.chown", "os.utime", "os.chdir",
        "os.putenv", "os.unsetenv", "subprocess.Popen", "os.system", "os.fork",
    }, (event, args)

sys.meta_path.insert(0, ImportGuard())
sys.addaudithook(audit)
import nek_post
import nek_post.cli
if mode == "import":
    assert "nek_post.cli.parser" not in sys.modules
    assert "nek_post.cli.registry" not in sys.modules
for arguments, expected_status in calls:
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            if mode == "module":
                sys.argv = ["nek_post.cli", *arguments]
                runpy.run_module("nek_post.cli", run_name="__main__")
            else:
                status = nek_post.cli.main(arguments)
        except SystemExit as error:
            status = error.code
    assert status == expected_status, (arguments, status, stdout.getvalue(), stderr.getvalue())
    if expected_status == 3:
        assert "registered but not implemented" in stderr.getvalue()
        assert not stdout.getvalue()
assert dict(os.environ) == environment
assert not scientific_roots.intersection(name.split(".")[0] for name in sys.modules)
assert {name for name in sys.modules if name.startswith("nek_post")} <= allowed_package_modules
print("purity verified")
'''


def isolated_environment(root, matplotlib_state):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE", "MPLCONFIGDIR", "MPLBACKEND"):
        env.pop(name, None)
    env.update({
        "HOME": str(root / "home"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "TMPDIR": str(root / "tmp"),
        "B1_PROTECTED_ROOT": str(root / "research"),
        "NEK_POST_PATHS_CONFIG": str(root / "research" / "missing-paths.yaml"),
        "NEK_POST_CASES_CONFIG": str(root / "research" / "missing-cases.yaml"),
        "NEK_POST_STUDY_CONFIG": str(root / "research" / "missing-study.yaml"),
        "COLUMNS": "100",
    })
    if matplotlib_state == "inherited":
        env["MPLCONFIGDIR"] = str(root / "inherited-mpl")
        env["MPLBACKEND"] = "svg"
    return env


def snapshot(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_mode, path.stat().st_size, path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in [root, *root.rglob("*")]
    }


@pytest.mark.parametrize("mode", ["import", "main", "module"])
@pytest.mark.parametrize("matplotlib_state", ["absent", "inherited"])
def test_source_import_and_invocations_are_pure(tmp_path, mode, matplotlib_state):
    calls = []
    if mode != "import":
        calls = [([*route, "--help"], 0) for route in HELP_ROUTES]
        calls += [(list(route), 3) for route in ROUTES]
    env = isolated_environment(tmp_path, matplotlib_state)
    before = snapshot(tmp_path)
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", PURITY_PROBE, str(REPO / "src"), mode, json.dumps(calls)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "purity verified"
    assert snapshot(tmp_path) == before


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-wheel")
    source = root / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(REPO / name, source / name)
    shutil.copytree(REPO / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", "from setuptools.build_meta import build_wheel, build_sdist; build_wheel('dist'); build_sdist('dist')"],
        cwd=source, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = (source / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        entrypoints = archive.read("nek_post-0.1.0.dist-info/entry_points.txt").decode()
        assert "nek-post = nek_post.cli:main" in entrypoints
        resource_names = {name.rsplit("/", 1)[-1] for name in archive.namelist()
                          if name.startswith("nek_post/resources/") and name.endswith(".yaml")}
        assert resource_names == {"cases.yaml", "h_refinement.yaml", "paths.example.yaml"}
        for name in ("__init__.py", "__main__.py", "parser.py", "registry.py"):
            assert archive.read("nek_post/cli/" + name) == (REPO / "src/nek_post/cli" / name).read_bytes()
        assert not any(name.startswith(("scripts/", "config/")) for name in archive.namelist())
    sdist, = (source / "dist").glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        resources = {Path(name).name for name in archive.getnames()
                     if "/src/nek_post/resources/" in name and name.endswith(".yaml")}
        assert resources == {"cases.yaml", "h_refinement.yaml", "paths.example.yaml"}
    venv = root / "venv"
    subprocess.run([sys.executable, "-B", "-m", "venv", str(venv)], env=env, check=True, timeout=60)
    python = venv / "bin/python"
    result = subprocess.run(
        [str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", str(wheel)],
        cwd=root, env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.util, nek_post; "
         "assert all(importlib.util.find_spec(name) is None "
         "for name in ('numpy', 'scipy', 'pymech', 'matplotlib', 'vtk', 'pyvista', 'yaml')); "
         "print(nek_post.__file__)"],
        cwd=root, env=env, text=True, capture_output=True, check=True, timeout=15,
    )
    package = Path(result.stdout.strip()).parent
    assert package.is_relative_to(venv)
    # Remove the build source before running installed commands.
    shutil.rmtree(source)
    return python, venv / "bin/nek-post", package


@pytest.mark.parametrize("matplotlib_state", ["absent", "inherited"])
def test_installed_console_and_module_without_checkout(installed_wheel, tmp_path, matplotlib_state):
    python, console, package = installed_wheel
    env = isolated_environment(tmp_path, matplotlib_state)
    # Plain invocation exercises normal pip-compiled installation without -B.
    before = snapshot(tmp_path)
    package_before = snapshot(package)
    for route in HELP_ROUTES:
        outputs = []
        for entrypoint in ([str(console)], [str(python), "-m", "nek_post.cli"]):
            result = subprocess.run(
                [*entrypoint, *route, "--help"], cwd=tmp_path, env=env,
                text=True, capture_output=True, timeout=15,
            )
            assert result.returncode == 0, result.stderr
            assert not result.stderr
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1]
    for route in ROUTES:
        for entrypoint in ([str(console)], [str(python), "-m", "nek_post.cli"]):
            result = subprocess.run(
                [*entrypoint, *route], cwd=tmp_path, env=env,
                text=True, capture_output=True, timeout=15,
            )
            assert result.returncode == 3 and not result.stdout
            assert "registered but not implemented" in result.stderr
    assert snapshot(tmp_path) == before
    assert snapshot(package) == package_before
    calls = [([*route, "--help"], 0) for route in HELP_ROUTES]
    calls += [(list(route), 3) for route in ROUTES]
    result = subprocess.run(
        [str(python), "-B", "-I", "-c", PURITY_PROBE, "", "module", json.dumps(calls)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "purity verified"
    assert snapshot(tmp_path) == before
    assert snapshot(package) == package_before


def test_installed_scientific_resources(installed_wheel, tmp_path):
    python, console, package = installed_wheel
    result = subprocess.run(
        [str(python), '-B', '-I', '-c',
         'from importlib.resources import files; import json; '
         'r=files("nek_post.resources"); '
         'print(json.dumps({n:r.joinpath(n).read_text() for n in '
         '("cases.yaml","h_refinement.yaml","paths.example.yaml")}))'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    values = json.loads(result.stdout)
    for name in values:
        assert values[name] == (REPO/'src/nek_post/resources'/name).read_text()
    assert '/data/' not in values['paths.example.yaml']
