from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import nek_post.cantero_re3450_multicase as multicase_module
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "26_overlay_cantero_re3450_multicase.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "overlay_cantero_re3450_multicase_script", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
overlay_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(overlay_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={case: tmp_path / case for case in ("N5", "N7", "N9")},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def test_cli_defaults_are_formal_re3450_configuration(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    args = overlay_script._parse_args(paths, [])

    assert args.cases == ["N5", "N7", "N9"]
    assert args.paper_csv == paths.cantero_fig5a_re3450_csv
    assert args.output_dir == paths.cantero_re3450_multicase_dir
    assert args.smooth_method == "moving_average"
    assert args.smooth_window == 11
    assert args.savgol_polyorder == 3
    assert args.slump_tmin == 3.0
    assert args.slump_tmax == 12.0


def test_cli_successfully_skips_complete_output_set_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(overlay_script, "load_project_paths", lambda *_args: paths)
    outputs = multicase_module.cantero_re3450_multicase_output_paths(
        paths.cantero_re3450_multicase_dir
    )
    for path in outputs.all_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing", encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("successful skip must occur before any input or computation")

    monkeypatch.setattr(overlay_script, "run_cantero_re3450_multicase", fail_if_called)
    monkeypatch.setattr(multicase_module, "read_digitized_paper_csv", fail_if_called)
    monkeypatch.setattr(multicase_module, "read_cantero_mean_front_timeseries_csv", fail_if_called)
    monkeypatch.setattr(multicase_module, "reconstruct_cantero_mean_front", fail_if_called)

    assert overlay_script.main([]) is None
    assert "Output artifacts already exist, skipping:" in capsys.readouterr().out
    assert all(path.read_text(encoding="utf-8") == "existing" for path in outputs.all_paths())


def test_cli_forwards_no_plots_without_requesting_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(overlay_script, "load_project_paths", lambda *_args: paths)
    calls: list[dict[str, object]] = []
    outputs = multicase_module.cantero_re3450_multicase_output_paths(
        paths.cantero_re3450_multicase_dir, include_plots=False
    )

    def run(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            cases=("N5", "N7", "N9"),
            front_csvs=multicase_module.cantero_re3450_front_csvs(
                paths.cantero_mean_front_dir
            ),
            paper_csv=paths.cantero_fig5a_re3450_csv,
            outputs=outputs,
        )

    monkeypatch.setattr(overlay_script, "run_cantero_re3450_multicase", run)
    overlay_script.main(["--no-plots"])

    assert calls[0]["no_plots"] is True
    assert outputs.linear_overlay is None
    assert outputs.loglog_overlay is None
    assert len(outputs.all_paths()) == 7
