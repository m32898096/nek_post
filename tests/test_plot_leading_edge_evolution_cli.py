from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from nek_post.config import load_yaml
from nek_post.front_detection_io import NekFramePath
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "19_plot_leading_edge_evolution.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "plot_leading_edge_evolution_script",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
leading_edge_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(leading_edge_script)


def _paths(tmp_path: Path, *, create_case_dir: bool = True) -> ProjectPaths:
    case_dir = tmp_path / "data" / "case_N7"
    if create_case_dir:
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
            "contour_time_spacing": 0.28,
        },
    }


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _results() -> tuple[SimpleNamespace, SimpleNamespace]:
    evolution = SimpleNamespace(
        time=_readonly([0.5, 0.9, 1.3], np.float64),
        x_front=_readonly(
            [[0.2, 0.3], [0.4, np.nan], [0.6, 0.7]],
            np.float64,
        ),
        nx=1000,
        native_ny=3,
        dense_ny=6,
        y_upsample_factor=2,
        z_target=0.04,
        threshold=0.1,
        periodic_endpoint_included=False,
    )
    selection = SimpleNamespace(
        actual_time=_readonly([0.5, 0.9], np.float64),
        x_front=_readonly([[0.2, 0.3], [0.4, np.nan]], np.float64),
    )
    return evolution, selection


def _install_success_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace, ProjectPaths]:
    paths = _paths(tmp_path)
    evolution, selection = _results()
    calls: dict[str, object] = {
        "events": [],
        "discover": [],
        "preflight": [],
        "build": [],
        "select": [],
        "csv": [],
        "plot": [],
    }
    frames = (
        NekFramePath(4, paths.case_dir("N7") / "GC0.f00004"),
        NekFramePath(8, paths.case_dir("N7") / "GC0.f00008"),
        NekFramePath(12, paths.case_dir("N7") / "GC0.f00012"),
    )
    monkeypatch.setattr(
        leading_edge_script,
        "load_project_paths",
        lambda _path: paths,
    )
    monkeypatch.setattr(
        leading_edge_script,
        "load_yaml",
        lambda _path: _cases_config(),
    )

    def discover(*args: object, **kwargs: object):
        calls["discover"].append((args, kwargs))  # type: ignore[union-attr]
        calls["events"].append("discover")  # type: ignore[union-attr]
        return frames

    def preflight(output_paths: object, overwrite: bool):
        calls["preflight"].append(  # type: ignore[union-attr]
            (tuple(output_paths), overwrite)  # type: ignore[arg-type]
        )
        calls["events"].append("preflight")  # type: ignore[union-attr]
        return tuple(output_paths)  # type: ignore[arg-type]

    def build(supplied_frames: object, **kwargs: object):
        calls["build"].append((supplied_frames, kwargs))  # type: ignore[union-attr]
        calls["events"].append("build")  # type: ignore[union-attr]
        return evolution

    def select(supplied_evolution: object, *, spacing: object):
        calls["select"].append(  # type: ignore[union-attr]
            (supplied_evolution, spacing)
        )
        return selection

    def write_csvs(*args: object, **kwargs: object):
        calls["csv"].append((args, kwargs))  # type: ignore[union-attr]
        output_dir, case = args[:2]
        return [
            Path(output_dir) / f"{case}_leading_edge_timeseries.csv",
            Path(output_dir) / f"{case}_leading_edge_metadata.csv",
        ]

    def write_plots(*args: object, **kwargs: object):
        calls["plot"].append((args, kwargs))  # type: ignore[union-attr]
        output_dir, case = args[:2]
        return [
            Path(output_dir) / f"{case}_leading_edge_evolution.png",
            Path(output_dir) / f"{case}_leading_edge_evolution.pdf",
        ]

    monkeypatch.setattr(leading_edge_script, "discover_nek_frame_paths", discover)
    monkeypatch.setattr(leading_edge_script, "preflight_output_paths", preflight)
    monkeypatch.setattr(leading_edge_script, "build_leading_edge_evolution", build)
    monkeypatch.setattr(leading_edge_script, "select_leading_edge_times", select)
    monkeypatch.setattr(leading_edge_script, "write_leading_edge_csvs", write_csvs)
    monkeypatch.setattr(
        leading_edge_script,
        "write_leading_edge_evolution_plots",
        write_plots,
    )
    return calls, evolution, selection, paths


def test_help_succeeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        leading_edge_script._parse_args(
            _paths(tmp_path),
            _cases_config(),
            ["--help"],
        )

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
        "--reynolds-number",
        "--output-dir",
        "--overwrite",
        "--no-plots",
    ):
        assert flag in help_text


def test_configured_defaults_match_production_definition(tmp_path: Path) -> None:
    cases_config = load_yaml(
        Path(__file__).resolve().parents[1] / "config" / "cases.yaml"
    )

    args = leading_edge_script._parse_args(_paths(tmp_path), cases_config, [])

    assert args.case == "N7"
    assert args.reynolds_number == 3450
    assert args.nx == 1000
    assert args.z_target == 0.04
    assert args.threshold == 0.1
    assert args.y_upsample_factor == 2
    assert args.contour_time_spacing == 0.28
    assert args.file_prefix == "GC0"


def test_dynamic_output_uses_final_case_and_explicit_override(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    dynamic = leading_edge_script._parse_args(
        paths,
        _cases_config(),
        ["--case", "custom"],
    )
    explicit = leading_edge_script._parse_args(
        paths,
        _cases_config(),
        ["--case", "custom", "--output-dir", str(tmp_path / "chosen")],
    )

    assert dynamic.output_dir == paths.results_root / "leading_edge" / "CUSTOM"
    assert explicit.output_dir == tmp_path / "chosen"


def test_all_frames_conflicts_with_explicit_spacing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        leading_edge_script._parse_args(
            _paths(tmp_path),
            _cases_config(),
            ["--all-frames", "--contour-time-spacing", "0.5"],
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ["--nx", "1"],
        ["--z-target", "nan"],
        ["--threshold", "inf"],
        ["--y-upsample-factor", "0"],
        ["--contour-time-spacing", "0"],
        ["--reynolds-number", "-1"],
        ["--start-index", "-1"],
    ),
)
def test_invalid_numeric_arguments_are_rejected(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        leading_edge_script._parse_args(
            _paths(tmp_path),
            _cases_config(),
            arguments,
        )
    assert error.value.code == 2


def test_normal_orchestration_arguments_global_preflight_and_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, evolution, selection, paths = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )
    output_dir = tmp_path / "artifacts"

    leading_edge_script.main(
        [
            "--start-index",
            "4",
            "--end-index",
            "12",
            "--nx",
            "1200",
            "--z-target",
            "0.05",
            "--threshold",
            "0.2",
            "--y-upsample-factor",
            "3",
            "--contour-time-spacing",
            "0.4",
            "--reynolds-number",
            "3500",
            "--output-dir",
            str(output_dir),
        ]
    )

    discover_args, discover_kwargs = calls["discover"][0]  # type: ignore[index]
    assert discover_args == (paths.case_dir("N7"),)
    assert discover_kwargs == {
        "file_prefix": "GC0",
        "start_index": 4,
        "end_index": 12,
    }
    assert calls["events"] == ["discover", "preflight", "build"]
    preflight_paths, preflight_overwrite = calls["preflight"][0]  # type: ignore[index]
    assert preflight_paths == (
        output_dir / "N7_leading_edge_timeseries.csv",
        output_dir / "N7_leading_edge_metadata.csv",
        output_dir / "N7_leading_edge_evolution.png",
        output_dir / "N7_leading_edge_evolution.pdf",
    )
    assert not preflight_overwrite
    supplied_frames, build_kwargs = calls["build"][0]  # type: ignore[index]
    assert len(supplied_frames) == 3
    assert build_kwargs == {
        "nx": 1200,
        "z_target": 0.05,
        "threshold": 0.2,
        "y_upsample_factor": 3,
    }
    assert calls["select"] == [(evolution, 0.4)]
    csv_args, csv_kwargs = calls["csv"][0]  # type: ignore[index]
    assert csv_args[:4] == (output_dir, "N7", evolution, selection)
    assert csv_kwargs == {"overwrite": False}
    plot_args, plot_kwargs = calls["plot"][0]  # type: ignore[index]
    assert plot_args[:4] == (output_dir, "N7", evolution, selection)
    assert plot_kwargs == {"overwrite": False, "reynolds_number": 3500.0}

    output = capsys.readouterr().out
    for text in (
        "Case: N7",
        "Selected frame count: 2",
        "native_ny: 3",
        "dense_ny: 6",
        "y_upsample_factor: 2",
        "z_target: 0.04",
        "threshold: 0.1",
        "dense_ny (6) == y_upsample_factor (2) * native_ny (3)",
        str(output_dir / "N7_leading_edge_timeseries.csv"),
        str(output_dir / "N7_leading_edge_evolution.pdf"),
    ):
        assert text in output


def test_all_frames_and_no_plots_select_none_and_write_only_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, evolution, _selection, _paths_value = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )
    output_dir = tmp_path / "csv-only"

    leading_edge_script.main(
        ["--all-frames", "--no-plots", "--output-dir", str(output_dir)]
    )

    assert calls["select"] == [(evolution, None)]
    preflight_paths, _overwrite = calls["preflight"][0]  # type: ignore[index]
    assert preflight_paths == (
        output_dir / "N7_leading_edge_timeseries.csv",
        output_dir / "N7_leading_edge_metadata.csv",
    )
    assert len(calls["csv"]) == 1  # type: ignore[arg-type]
    assert calls["plot"] == []
    output = capsys.readouterr().out
    assert "all processed frames (spacing=None)" in output
    assert "Figures: skipped (--no-plots)" in output


def test_preflight_failure_exits_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, _evolution, _selection, _paths_value = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )

    def reject(_paths: object, _overwrite: bool) -> None:
        raise FileExistsError("Output exists: existing.pdf")

    monkeypatch.setattr(leading_edge_script, "preflight_output_paths", reject)
    with pytest.raises(SystemExit) as error:
        leading_edge_script.main([])

    assert error.value.code == 1
    assert calls["build"] == []
    assert "existing.pdf" in capsys.readouterr().err


def test_unknown_case_and_missing_case_directory_exit_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(leading_edge_script, "load_project_paths", lambda _p: paths)
    monkeypatch.setattr(
        leading_edge_script,
        "load_yaml",
        lambda _p: _cases_config(),
    )
    with pytest.raises(SystemExit) as unknown_error:
        leading_edge_script.main(["--case", "unknown"])
    assert unknown_error.value.code == 1
    assert "Unknown case" in capsys.readouterr().err

    missing_paths = _paths(tmp_path / "missing-root", create_case_dir=False)
    monkeypatch.setattr(
        leading_edge_script,
        "load_project_paths",
        lambda _p: missing_paths,
    )
    with pytest.raises(SystemExit) as missing_error:
        leading_edge_script.main([])
    assert missing_error.value.code == 1
    assert "case directory not found" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    ("failure_stage", "message"),
    (
        ("discover_nek_frame_paths", "no matching files"),
        ("build_leading_edge_evolution", "geometry mismatch"),
        ("write_leading_edge_evolution_plots", "no finite leading-edge point"),
    ),
)
def test_discovery_workflow_and_plot_failures_exit_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
    message: str,
) -> None:
    _install_success_fakes(tmp_path, monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(leading_edge_script, failure_stage, fail)
    with pytest.raises(SystemExit) as error:
        leading_edge_script.main([])

    assert error.value.code == 1
    assert message in capsys.readouterr().err


def test_main_does_not_mutate_package_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, evolution, selection, _paths_value = _install_success_fakes(
        tmp_path,
        monkeypatch,
    )
    original_time = evolution.time.copy()
    original_evolution_front = evolution.x_front.copy()
    original_selection_time = selection.actual_time.copy()
    original_selection_front = selection.x_front.copy()

    leading_edge_script.main(["--no-plots"])

    assert_array_equal(evolution.time, original_time)
    assert_array_equal(evolution.x_front, original_evolution_front)
    assert_array_equal(selection.actual_time, original_selection_time)
    assert_array_equal(selection.x_front, original_selection_front)
    assert not evolution.time.flags.writeable
    assert not selection.x_front.flags.writeable
    assert calls["select"] == [(evolution, 0.28)]


def test_importing_script_does_not_execute_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.paths as paths_module

    called: list[Path] = []

    def forbidden(path: Path) -> None:
        called.append(path)
        raise AssertionError("main executed during import")

    monkeypatch.setattr(paths_module, "load_project_paths", forbidden)
    spec = importlib.util.spec_from_file_location(
        "plot_leading_edge_evolution_import_only",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert called == []
    assert callable(module.main)
