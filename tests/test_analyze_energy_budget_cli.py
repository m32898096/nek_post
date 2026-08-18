from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from nek_post.paths import ProjectPaths


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "26_analyze_energy_budget.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("analyze_energy_budget_script", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
analyze_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(analyze_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": tmp_path / "case_N7"},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def test_case_resolution_uses_configured_case_directory(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    args = analyze_script._parse_args(paths, ["--case", "n7"])

    input_path, case = analyze_script._resolve_input(paths, args)

    assert input_path == paths.case_dir("N7") / "energy_budget.dat"
    assert case == "N7"


def test_cli_writes_required_outputs_from_synthetic_input(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    input_path = paths.case_dir("N7") / "energy_budget.dat"
    input_path.parent.mkdir(parents=True)
    time = np.array([0.0, 0.4, 1.1, 2.0])
    E_k = time**2
    E_p = 10.0 - time**2 - 2.0 * time
    np.savetxt(input_path, np.column_stack((time, E_k, E_p, E_k + E_p, np.full(4, 2.0))))
    output_dir = tmp_path / "output"
    monkeypatch.setattr(analyze_script, "load_project_paths", lambda *_args: paths)

    analyze_script.main(["--case", "N7", "--output-dir", str(output_dir)])

    assert {path.name for path in output_dir.iterdir()} == {
        "energy_timeseries.csv",
        "summary.csv",
        "energy_evolution.png",
        "energy_derivatives.png",
        "energy_closure.png",
        "energy_closure_residual.png",
    }
