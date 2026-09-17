"""Configured h-study defaults must not fall back to repository-local results."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from nek_post.paths import ProjectPaths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script(number: int, name: str):
    path = REPO_ROOT / "scripts" / f"{number}_{name}.py"
    spec = importlib.util.spec_from_file_location(f"h_results_script_{number}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={case: tmp_path / "data" / case for case in ("N7_H", "N7_VH", "N7_VVH")},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "p_results",
        h_refinement_results_root=tmp_path / "h_refinement",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def test_field_comparison_default_uses_configured_h_root(tmp_path, monkeypatch, capsys):
    script = _script(29, "compare_h_refinement_fields")
    paths = _paths(tmp_path)
    paths.h_refinement_field_comparison_dir.mkdir(parents=True)
    monkeypatch.setattr(script.ProjectPaths, "from_yaml", lambda *_: paths)
    monkeypatch.setattr(sys, "argv", [str(script.__file__), "--nx", "2", "--nz", "2", "--times", "5"])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 1
    assert str(paths.h_refinement_field_comparison_dir) in capsys.readouterr().err


def test_convergence_default_uses_configured_h_root(tmp_path, monkeypatch, capsys):
    script = _script(30, "analyze_h_refinement_convergence")
    paths = _paths(tmp_path)
    paths.h_refinement_convergence_analysis_dir.mkdir(parents=True)
    monkeypatch.setattr(script.ProjectPaths, "from_yaml", lambda *_: paths)
    monkeypatch.setattr(sys, "argv", [str(script.__file__)])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 1
    assert "Output must be new" in capsys.readouterr().err


def test_leading_edge_analysis_default_uses_configured_h_root(tmp_path, monkeypatch):
    script = _script(31, "compare_h_refinement_leading_edges")
    paths = _paths(tmp_path)
    monkeypatch.setattr(script.ProjectPaths, "from_yaml", lambda *_: paths)
    import nek_post.leading_edge_comparison_io as comparison_io

    calls = []
    monkeypatch.setattr(comparison_io, "analyze_leading_edge_histories",
                        lambda *args: calls.append(args))
    script.main(["--phase", "analyze"])
    root = paths.h_refinement_leading_edge_dir / "step6b"
    assert len(calls) == 1
    assert calls[0][0:2] == (root / "raw", root / "comparison")
