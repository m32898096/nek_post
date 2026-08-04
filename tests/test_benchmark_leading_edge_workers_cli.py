from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from nek_post.config import load_yaml
from nek_post.leading_edge_benchmark import (
    LeadingEdgeBenchmarkError,
    LeadingEdgeBenchmarkResult,
    LeadingEdgeBenchmarkSummary,
)
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "21_benchmark_leading_edge_workers.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "benchmark_leading_edge_workers_script", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
benchmark_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(benchmark_script)


def _paths(tmp_path: Path) -> ProjectPaths:
    results = tmp_path / "results"
    results.mkdir(parents=True)
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": tmp_path / "data" / "N7"},
        postproc_root=tmp_path / "postproc",
        results_root=results,
        cantero_fig5a_re3450_csv=tmp_path / "paper.csv",
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
            "workers": 2,
        },
    }


def _result(output_dir: Path) -> LeadingEdgeBenchmarkResult:
    summaries = tuple(
        LeadingEdgeBenchmarkSummary(
            case="N7",
            worker_count=workers,
            successful_run_count=2,
            median_wall_seconds={1: 10.0, 2: 6.0, 4: 5.0}[workers],
            minimum_wall_seconds={1: 9.5, 2: 5.5, 4: 4.5}[workers],
            maximum_wall_seconds={1: 10.5, 2: 6.5, 4: 5.5}[workers],
            median_user_cpu_seconds=12.0,
            median_system_cpu_seconds=1.0,
            median_maximum_rss_kb=200000.0 * workers,
            speedup_relative_to_workers_1={1: 1.0, 2: 10 / 6, 4: 2.0}[workers],
            efficiency={1: 1.0, 2: 5 / 6, 4: 0.5}[workers],
            all_artifacts_equivalent=True,
        )
        for workers in (1, 2, 4)
    )
    return LeadingEdgeBenchmarkResult(
        runs=(),
        summaries=summaries,
        benchmark_csv=output_dir / "leading_edge_workers_benchmark.csv",
        summary_csv=output_dir / "leading_edge_workers_summary.csv",
    )


def test_help_succeeds_and_lists_benchmark_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        benchmark_script._parse_args(_paths(tmp_path), _cases_config(), ["--help"])

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
        "--worker-counts",
        "--repeats",
        "--warmup-runs",
        "--output-dir",
        "--overwrite",
    ):
        assert flag in help_text


def test_defaults_load_production_science_and_benchmark_controls(tmp_path: Path) -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "config" / "cases.yaml")

    args = benchmark_script._parse_args(_paths(tmp_path), config, [])

    assert args.case == "N7"
    assert args.file_prefix == "GC0"
    assert args.nx == 1000
    assert args.z_target == 0.04
    assert args.threshold == 0.1
    assert args.y_upsample_factor == 2
    assert args.contour_time_spacing == 0.25
    assert args.worker_counts == (1, 2, 4)
    assert args.repeats == 3
    assert args.warmup_runs == 0


def test_default_output_is_separate_from_production_leading_edge(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    args = benchmark_script._parse_args(paths, _cases_config(), [])

    assert args.output_dir == paths.results_root / "leading_edge_benchmarks" / "N7"
    assert args.output_dir != paths.results_root / "leading_edge" / "N7"


def test_all_frames_conflicts_with_explicit_spacing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        benchmark_script._parse_args(
            _paths(tmp_path),
            _cases_config(),
            ["--all-frames", "--contour-time-spacing", "0.25"],
        )

    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_normal_mode_retains_explicit_quarter_spacing(tmp_path: Path) -> None:
    args = benchmark_script._parse_args(
        _paths(tmp_path), _cases_config(), ["--contour-time-spacing", "0.25"]
    )

    assert not args.all_frames
    assert args.contour_time_spacing == 0.25


def test_user_overrides_are_passed_to_package_benchmark_and_success_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    output = tmp_path / "custom-benchmark"
    captured: dict[str, object] = {}
    monkeypatch.setattr(benchmark_script, "load_project_paths", lambda _path: paths)
    monkeypatch.setattr(benchmark_script, "load_yaml", lambda _path: _cases_config())

    def run(**kwargs: object) -> LeadingEdgeBenchmarkResult:
        captured.update(kwargs)
        return _result(output)

    monkeypatch.setattr(benchmark_script, "run_leading_edge_worker_benchmark", run)
    benchmark_script.main(
        [
            "--case",
            "n7",
            "--file-prefix",
            "ALT",
            "--start-index",
            "37",
            "--end-index",
            "52",
            "--nx",
            "500",
            "--z-target",
            "0.05",
            "--threshold",
            "0.2",
            "--y-upsample-factor",
            "3",
            "--all-frames",
            "--worker-counts",
            "1,4",
            "--repeats",
            "2",
            "--warmup-runs",
            "1",
            "--output-dir",
            str(output),
            "--overwrite",
        ]
    )

    assert captured["case"] == "N7"
    assert captured["file_prefix"] == "ALT"
    assert captured["start_index"] == 37
    assert captured["end_index"] == 52
    assert captured["nx"] == 500
    assert captured["z_target"] == 0.05
    assert captured["threshold"] == 0.2
    assert captured["y_upsample_factor"] == 3
    assert captured["contour_time_spacing"] is None
    assert captured["all_frames"] is True
    assert captured["worker_counts"] == (1, 4)
    assert captured["repeats"] == 2
    assert captured["warmup_runs"] == 1
    assert captured["output_dir"] == output
    assert captured["overwrite"] is True
    assert captured["production_output_dir"] == paths.results_root / "leading_edge" / "N7"
    assert callable(captured["reporter"])
    output_text = capsys.readouterr().out
    assert "median wall=10s" in output_text
    assert "equivalent=True" in output_text


def test_expected_benchmark_failure_exits_one_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(benchmark_script, "load_project_paths", lambda _path: paths)
    monkeypatch.setattr(benchmark_script, "load_yaml", lambda _path: _cases_config())
    monkeypatch.setattr(
        benchmark_script,
        "run_leading_edge_worker_benchmark",
        lambda **_kwargs: (_ for _ in ()).throw(
            LeadingEdgeBenchmarkError("synthetic benchmark failure")
        ),
    )

    with pytest.raises(SystemExit) as error:
        benchmark_script.main([])

    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert "synthetic benchmark failure" in stderr
    assert "Traceback" not in stderr


def test_importing_script_does_not_execute_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.paths as paths_module

    calls: list[Path] = []

    def forbidden(path: Path) -> None:
        calls.append(path)
        raise AssertionError("main executed during import")

    monkeypatch.setattr(paths_module, "load_project_paths", forbidden)
    spec = importlib.util.spec_from_file_location(
        "benchmark_leading_edge_workers_import_only", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == []
    assert callable(module.main)
