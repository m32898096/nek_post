from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.config import load_yaml
from nek_post.front_detection_io import NekFramePath
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "19_compute_leading_edge_evolution.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_leading_edge_evolution_script",
    SCRIPT_PATH,
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


def _cases_config() -> dict[str, object]:
    return {
        "file_prefix": "GC0",
        "leading_edge": {
            "case": "N7",
            "reynolds_number": 3450,
            "nx": 1000,
            "z_target": 0.04,
            "threshold": 0.1,
            "y_upsample_factor": 2,
            "contour_time_spacing": 0.25,
        },
    }


def _results() -> tuple[SimpleNamespace, SimpleNamespace]:
    evolution = SimpleNamespace(
        time=np.asarray([0.5, 0.75, 1.0]),
        nx=1000,
        native_ny=3,
        dense_ny=6,
        y_upsample_factor=2,
        z_target=0.04,
        threshold=0.1,
        periodic_endpoint_included=False,
    )
    selection = SimpleNamespace(actual_time=np.asarray([0.5, 0.75]))
    return evolution, selection


def _install_success_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace, ProjectPaths]:
    paths = _paths(tmp_path)
    evolution, selection = _results()
    frames = (
        NekFramePath(4, paths.case_dir("N7") / "GC0.f00004"),
        NekFramePath(8, paths.case_dir("N7") / "GC0.f00008"),
    )
    calls: dict[str, object] = {
        "events": [],
        "discover": [],
        "preflight": [],
        "build": [],
        "select": [],
        "write": [],
    }
    monkeypatch.setattr(compute_script, "load_project_paths", lambda _p: paths)
    monkeypatch.setattr(compute_script, "load_yaml", lambda _p: _cases_config())

    def discover(*args: object, **kwargs: object):
        calls["events"].append("discover")  # type: ignore[union-attr]
        calls["discover"].append((args, kwargs))  # type: ignore[union-attr]
        return frames

    def preflight(output_paths: object, overwrite: bool):
        calls["events"].append("preflight")  # type: ignore[union-attr]
        calls["preflight"].append(  # type: ignore[union-attr]
            (tuple(output_paths), overwrite)  # type: ignore[arg-type]
        )
        return tuple(output_paths)  # type: ignore[arg-type]

    def build(supplied_frames: object, **kwargs: object):
        calls["events"].append("build")  # type: ignore[union-attr]
        calls["build"].append((supplied_frames, kwargs))  # type: ignore[union-attr]
        return evolution

    def select(supplied_evolution: object, *, spacing: object):
        calls["select"].append(  # type: ignore[union-attr]
            (supplied_evolution, spacing)
        )
        return selection

    def write(*args: object, **kwargs: object):
        calls["write"].append((args, kwargs))  # type: ignore[union-attr]
        output_dir, case = args[:2]
        return [
            Path(output_dir) / f"{case}_leading_edge_timeseries.csv",
            Path(output_dir) / f"{case}_leading_edge_metadata.csv",
        ]

    monkeypatch.setattr(compute_script, "discover_nek_frame_paths", discover)
    monkeypatch.setattr(compute_script, "preflight_output_paths", preflight)
    monkeypatch.setattr(compute_script, "build_leading_edge_evolution", build)
    monkeypatch.setattr(compute_script, "select_leading_edge_times", select)
    monkeypatch.setattr(compute_script, "write_leading_edge_csvs", write)
    return calls, evolution, selection, paths


def test_help_succeeds_and_contains_only_compute_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        compute_script._parse_args(_paths(tmp_path), _cases_config(), ["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--case",
        "--file-prefix",
        "--start-index",
        "--end-index",
        "--nx",
        "--z-target",
        "--threshold",
        "--y-upsample-factor",
        "--contour-time-spacing",
        "--all-frames",
        "--output-dir",
        "--overwrite",
    ):
        assert flag in help_text
    assert "--no-plots" not in help_text
    assert "--reynolds-number" not in help_text


def test_configured_defaults_include_quarter_time_spacing(tmp_path: Path) -> None:
    cases_config = load_yaml(
        Path(__file__).resolve().parents[1] / "config" / "cases.yaml"
    )

    args = compute_script._parse_args(_paths(tmp_path), cases_config, [])

    assert args.case == "N7"
    assert args.file_prefix == "GC0"
    assert args.nx == 1000
    assert args.z_target == 0.04
    assert args.threshold == 0.1
    assert args.y_upsample_factor == 2
    assert args.contour_time_spacing == 0.25


def test_dynamic_output_directory_uses_final_case(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    args = compute_script._parse_args(
        paths,
        _cases_config(),
        ["--case", "custom"],
    )
    assert args.output_dir == paths.results_root / "leading_edge" / "CUSTOM"


def test_compute_preflights_and_writes_only_two_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, evolution, selection, paths = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )
    output_dir = tmp_path / "artifacts"

    compute_script.main(
        [
            "--start-index",
            "4",
            "--end-index",
            "8",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert calls["events"] == ["discover", "preflight", "build"]
    discover_args, discover_kwargs = calls["discover"][0]  # type: ignore[index]
    assert discover_args == (paths.case_dir("N7"),)
    assert discover_kwargs == {
        "file_prefix": "GC0",
        "start_index": 4,
        "end_index": 8,
    }
    preflight_paths, overwrite = calls["preflight"][0]  # type: ignore[index]
    assert preflight_paths == (
        output_dir / "N7_leading_edge_timeseries.csv",
        output_dir / "N7_leading_edge_metadata.csv",
    )
    assert not overwrite
    assert calls["select"] == [(evolution, 0.25)]
    write_args, write_kwargs = calls["write"][0]  # type: ignore[index]
    assert write_args[:4] == (output_dir, "N7", evolution, selection)
    assert write_kwargs == {"overwrite": False}
    assert not hasattr(compute_script, "write_leading_edge_evolution_plots")
    assert "matplotlib" not in compute_script.__dict__


def test_all_frames_selects_spacing_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, evolution, _selection, _paths_value = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )

    compute_script.main(["--all-frames"])

    assert calls["select"] == [(evolution, None)]


def test_existing_output_fails_before_expensive_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _evolution, _selection, _paths_value = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )

    def reject(_paths: object, _overwrite: bool) -> None:
        raise FileExistsError("Output exists: metadata.csv")

    monkeypatch.setattr(compute_script, "preflight_output_paths", reject)
    with pytest.raises(SystemExit) as error:
        compute_script.main([])

    assert error.value.code == 1
    assert calls["build"] == []
    assert "metadata.csv" in capsys.readouterr().err


def test_importing_compute_script_does_not_execute_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.paths as paths_module

    called: list[Path] = []

    def forbidden(path: Path) -> None:
        called.append(path)
        raise AssertionError("main executed during import")

    monkeypatch.setattr(paths_module, "load_project_paths", forbidden)
    spec = importlib.util.spec_from_file_location(
        "compute_leading_edge_import_only",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert called == []
    assert callable(module.main)
