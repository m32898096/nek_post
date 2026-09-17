from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import nek_post.cantero_mean_front as mean_front
from nek_post.cantero_mean_front import (
    STATUS_SUCCESS,
    CanteroMeanFrontTimeseries,
    read_cantero_mean_front_timeseries_csv,
)
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "24_compute_cantero_mean_front.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_cantero_mean_front_script", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
compute_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(compute_script)


def test_phase_two_script_import_does_not_load_legacy_front_detection() -> None:
    source_root = SCRIPT_PATH.parents[1] / "src"
    code = (
        "import importlib.util, sys\n"
        f"path = {str(SCRIPT_PATH)!r}\n"
        "spec = importlib.util.spec_from_file_location('phase_two_import_check', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec is not None and spec.loader is not None\n"
        "spec.loader.exec_module(module)\n"
        "assert 'nek_post.front_detection' not in sys.modules\n"
        "assert 'scipy.ndimage' not in sys.modules\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=SCRIPT_PATH.parents[1],
        env=environment,
    )


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


def _series(paths: ProjectPaths) -> CanteroMeanFrontTimeseries:
    return CanteroMeanFrontTimeseries(
        case="N7",
        file_index=np.array([1, 2]),
        source_file=(
            str(paths.case_dir("N7") / "GC0.f00001"),
            str(paths.case_dir("N7") / "GC0.f00002"),
        ),
        time=np.array([0.5, 1.0]),
        x_front=np.array([1.5, 2.0]),
        x_front_minus_initial=np.array([0.0, 0.5]),
        threshold=0.01,
        reference_x=0.0,
        left_index=np.array([1, 1]),
        right_index=np.array([2, 2]),
        x_left=np.array([1.0, 1.0]),
        x_right=np.array([2.0, 2.0]),
        h_left=np.array([0.02, 0.02]),
        h_right=np.array([0.0, 0.0]),
        crossing_count_in_search_region=np.array([1, 1]),
        status=(STATUS_SUCCESS, STATUS_SUCCESS),
    )


def test_cli_routes_h_case_to_configured_root_by_default(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cases = {"reference_case": "N7", "file_prefix": "GC0"}
    h_cases = ("N7_H", "N7_VH", "N7_VVH")
    h_args = compute_script._parse_args(paths, cases, ["--case", "N7_H"], h_cases=h_cases)
    p_args = compute_script._parse_args(paths, cases, ["--case", "N7"], h_cases=h_cases)
    custom = tmp_path / "custom"
    custom_args = compute_script._parse_args(paths, cases, [
        "--case", "N7_H", "--output-dir", str(custom),
    ], h_cases=h_cases)

    assert h_args.output_dir == paths.h_refinement_cantero_mean_front_dir
    assert p_args.output_dir == paths.cantero_mean_front_dir
    assert custom_args.output_dir == custom


def _configure(paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "paths": {},
        "cases": {"reference_case": "N7", "file_prefix": "GC0"},
    }
    monkeypatch.setattr(compute_script, "load_project_config", lambda *_: config)
    monkeypatch.setattr(
        compute_script.ProjectPaths,
        "from_mapping",
        classmethod(lambda _cls, _: paths),
    )


def test_cli_writes_timeseries_to_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _configure(paths, monkeypatch)
    frames = (object(), object())
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(compute_script, "discover_nek_frame_paths", lambda *_args, **_kwargs: frames)
    def compute(supplied_frames: object, **kwargs: object) -> CanteroMeanFrontTimeseries:
        calls.append(kwargs)
        return _series(paths)
    monkeypatch.setattr(
        compute_script,
        "compute_cantero_mean_front_timeseries",
        compute,
    )

    output_root = tmp_path / "artifacts"
    compute_script.main(["--case", "N7", "--output-dir", str(output_root)])

    path = output_root / "N7" / "N7_cantero_mean_front_timeseries.csv"
    loaded = read_cantero_mean_front_timeseries_csv(path)
    np.testing.assert_array_equal(loaded["file_index"], [1, 2])
    np.testing.assert_allclose(loaded["x_front"], [1.5, 2.0])
    reader = calls[0]["reader"]
    assert getattr(reader, "keywords")["skip_vars"] == ("ux", "uy", "uz", "pressure")
    assert calls[0]["subsequent_reader"] is None
    assert calls[0]["stationary_geometry_check_reader"] is None
    assert calls[0]["validate_geometry_each_frame"] is True


def test_stationary_fast_path_uses_field_only_subsequent_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _configure(paths, monkeypatch)
    monkeypatch.setattr(compute_script, "discover_nek_frame_paths",
                        lambda *_args, **_kwargs: (object(), object()))
    calls: list[dict[str, object]] = []
    def compute(_frames: object, **kwargs: object) -> CanteroMeanFrontTimeseries:
        calls.append(kwargs)
        return _series(paths)
    monkeypatch.setattr(compute_script, "compute_cantero_mean_front_timeseries", compute)

    compute_script.main(["--case", "N7", "--output-dir", str(tmp_path / "out"),
                         "--stationary-geometry-fast-path"])

    assert calls[0]["validate_geometry_each_frame"] is False
    first = calls[0]["reader"]
    later = calls[0]["subsequent_reader"]
    check = calls[0]["stationary_geometry_check_reader"]
    assert getattr(first, "keywords")["dtype"] == "float32"
    assert getattr(later, "keywords")["skip_vars"] == (
        "x", "y", "z", "ux", "uy", "uz", "pressure")
    assert getattr(check, "keywords")["skip_vars"] == (
        "ux", "uy", "uz", "pressure", "temperature")


def test_cli_existing_output_skips_before_expensive_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    _configure(paths, monkeypatch)
    output_root = tmp_path / "artifacts"
    path = output_root / "N7" / "N7_cantero_mean_front_timeseries.csv"
    path.parent.mkdir(parents=True)
    original = b"existing Cantero mean-front CSV"
    path.write_bytes(original)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        pytest.fail("successful skip must not start expensive processing")

    monkeypatch.setattr(compute_script, "discover_nek_frame_paths", fail_if_called)
    monkeypatch.setattr(compute_script, "compute_cantero_mean_front_timeseries", fail_if_called)
    monkeypatch.setattr(mean_front, "read_nek_file", fail_if_called)
    monkeypatch.setattr(mean_front, "build_cantero_equivalent_height_plan", fail_if_called)
    monkeypatch.setattr(mean_front, "apply_cantero_equivalent_height_plan", fail_if_called)
    assert compute_script.main(["--case", "N7", "--output-dir", str(output_root)]) is None

    assert "Output file already exists, skipping:" in capsys.readouterr().out
    assert path.read_bytes() == original
