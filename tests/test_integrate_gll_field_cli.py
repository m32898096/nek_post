from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nek_post.gll_directional_workflow import load_gll_directional_integral_npz
from nek_post.paths import ProjectPaths
from test_gll_directional_workflow import _data


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "22_integrate_gll_field.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("integrate_gll_field_script", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
integrate_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(integrate_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    case_dir = tmp_path / "data" / "case_N7"
    case_dir.mkdir(parents=True)
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": case_dir},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def _config() -> dict[str, object]:
    return {
        "paths": {},
        "cases": {
            "reference_case": "N7",
            "file_prefix": "GC0",
            "file_indices": [79],
        },
    }


def _configure_script(
    paths: ProjectPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integrate_script, "load_project_config", lambda *_: _config())
    monkeypatch.setattr(
        integrate_script.ProjectPaths,
        "from_mapping",
        classmethod(lambda _cls, _: paths),
    )


def test_direction_is_required_at_argparse_level(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        integrate_script._parse_args(_paths(tmp_path), _config()["cases"], [])

    assert error.value.code == 2


def test_cli_writes_one_directional_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _configure_script(paths, monkeypatch)
    monkeypatch.setattr(integrate_script, "read_nek_file", lambda _: _data())
    monkeypatch.setattr(integrate_script, "get_nek_time", lambda _: 19.5)

    output_root = tmp_path / "artifacts"
    integrate_script.main(
        [
            "--case",
            "N7",
            "--index",
            "79",
            "--field",
            "concentration",
            "--direction",
            "y",
            "--output-dir",
            str(output_root),
        ]
    )

    artifact = output_root / "N7" / "concentration" / "y" / "N7_f00079_concentration_integrate_y.npz"
    loaded = load_gll_directional_integral_npz(artifact)
    assert loaded["source_file"] == str(paths.case_dir("N7") / "GC0.f00079")
    np.testing.assert_allclose(loaded["values"], 2.0)


def test_existing_output_skips_without_reading_or_integrating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    _configure_script(paths, monkeypatch)
    output_root = tmp_path / "artifacts"
    artifact = (
        output_root
        / "N7"
        / "concentration"
        / "y"
        / "N7_f00079_concentration_integrate_y.npz"
    )
    artifact.parent.mkdir(parents=True)
    original = b"existing directional-GLL artifact"
    artifact.write_bytes(original)

    def fail_read(_: object) -> object:
        pytest.fail("read_nek_file must not be called when skipping")

    def fail_compute(*_: object, **__: object) -> object:
        pytest.fail("compute_gll_directional_integral must not be called when skipping")

    monkeypatch.setattr(integrate_script, "read_nek_file", fail_read)
    monkeypatch.setattr(
        integrate_script,
        "compute_gll_directional_integral",
        fail_compute,
    )

    assert integrate_script.main(
        [
            "--case",
            "N7",
            "--index",
            "79",
            "--field",
            "concentration",
            "--direction",
            "y",
            "--output-dir",
            str(output_root),
        ]
    ) is None

    assert "Output file already exists, skipping:" in capsys.readouterr().out
    assert artifact.read_bytes() == original
