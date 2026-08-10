from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nek_post.cantero_equivalent_height import load_cantero_equivalent_height_npz
from nek_post.paths import ProjectPaths
from test_cantero_equivalent_height import _data


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "23_compute_cantero_equivalent_height.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_cantero_equivalent_height_script", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
compute_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(compute_script)


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


def test_cli_writes_one_equivalent_height_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    config = {
        "paths": {},
        "cases": {
            "reference_case": "N7",
            "file_prefix": "GC0",
            "file_indices": [79],
        },
    }
    monkeypatch.setattr(compute_script, "load_project_config", lambda *_: config)
    monkeypatch.setattr(
        compute_script.ProjectPaths,
        "from_mapping",
        classmethod(lambda _cls, _: paths),
    )
    monkeypatch.setattr(
        compute_script,
        "read_nek_file",
        lambda _: _data(lambda x, _y, _z: np.ones_like(x)),
    )
    monkeypatch.setattr(compute_script, "get_nek_time", lambda _: 19.5)

    output_root = tmp_path / "artifacts"
    compute_script.main(
        [
            "--case",
            "N7",
            "--index",
            "79",
            "--output-dir",
            str(output_root),
        ]
    )

    artifact = (
        output_root
        / "N7"
        / "N7_f00079_cantero_equivalent_height.npz"
    )
    loaded = load_cantero_equivalent_height_npz(artifact)
    assert loaded["source_file"] == str(paths.case_dir("N7") / "GC0.f00079")
    np.testing.assert_allclose(loaded["local_equivalent_height"], 1.0)
    np.testing.assert_allclose(loaded["span_averaged_height"], 1.0)
