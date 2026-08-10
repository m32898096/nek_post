from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nek_post.cantero_front_reconstruction as reconstruction_module
from nek_post.paths import ProjectPaths


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "25_reconstruct_cantero_front.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("reconstruct_cantero_front_script", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
reconstruct_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(reconstruct_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": tmp_path / "data" / "case_N7"},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper-re3450.csv",
        cantero_fig5a_re8950_csv=tmp_path / "paper-re8950.csv",
    )


def test_cli_defaults_use_phase_two_n7_and_re3450_paths(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    args = reconstruct_script._parse_args(paths, [])

    assert args.case == "N7"
    assert args.front_csv == (
        paths.cantero_mean_front_dir / "N7" / "N7_cantero_mean_front_timeseries.csv"
    )
    assert args.paper_csv == paths.cantero_fig5a_re3450_csv
    assert args.output_dir == paths.cantero_front_reconstruction_dir
    assert args.smooth_method == "moving_average"
    assert args.smooth_window == 11
    assert args.savgol_polyorder == 3


def test_cli_rejects_non_n7_before_reading_or_reconstructing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(reconstruct_script, "load_project_paths", lambda *_args: paths)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("argparse must reject non-N7 before any workflow action")

    monkeypatch.setattr(reconstruct_script, "run_cantero_front_reconstruction", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "read_cantero_mean_front_timeseries_csv", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "read_digitized_paper_csv", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "compute_kinematics", fail_if_called)

    with pytest.raises(SystemExit) as error:
        reconstruct_script.main(["--case", "N5"])

    assert error.value.code == 2
    assert not paths.cantero_front_reconstruction_dir.exists()


def test_cli_existing_outputs_skip_before_read_compute_or_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(reconstruct_script, "load_project_paths", lambda *_args: paths)
    outputs = reconstruction_module.cantero_front_reconstruction_output_paths(
        paths.cantero_front_reconstruction_dir, "N7", include_plots=True
    )
    for path in outputs.all_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing", encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("successful skip must not read, reconstruct, or plot")

    monkeypatch.setattr(reconstruct_script, "run_cantero_front_reconstruction", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "read_cantero_mean_front_timeseries_csv", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "read_digitized_paper_csv", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "compute_kinematics", fail_if_called)
    monkeypatch.setattr(reconstruction_module, "write_cantero_front_reconstruction_overlay", fail_if_called)

    assert reconstruct_script.main([]) is None
    assert "Output artifacts already exist, skipping:" in capsys.readouterr().out
    assert all(path.read_text(encoding="utf-8") == "existing" for path in outputs.all_paths())


def test_cli_forwards_no_plots_and_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(reconstruct_script, "load_project_paths", lambda *_args: paths)
    calls: list[dict[str, object]] = []
    outputs = reconstruction_module.cantero_front_reconstruction_output_paths(
        paths.cantero_front_reconstruction_dir, "N7", include_plots=False
    )

    def fake_run(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            reconstruction=SimpleNamespace(
                time=np.array([0.0, 1.0]),
                x_reconstructed_relative=np.array([0.0, 1.0]),
            ),
            comparison=SimpleNamespace(time=np.array([0.0, 1.0])),
            outputs=outputs,
        )

    monkeypatch.setattr(reconstruct_script, "run_cantero_front_reconstruction", fake_run)
    reconstruct_script.main(["--no-plots", "--overwrite"])

    assert calls[0]["no_plots"] is True
    assert calls[0]["overwrite"] is True
