from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nek_post.leading_edge_benchmark as benchmark
from nek_post.leading_edge_benchmark import (
    LEADING_EDGE_BENCHMARK_COLUMNS,
    GnuTimeMetrics,
    LeadingEdgeBenchmarkError,
    LeadingEdgeBenchmarkRun,
    ScientificArtifactMismatchError,
    assert_leading_edge_artifacts_equivalent,
    build_leading_edge_compute_command,
    parse_gnu_time_metrics,
    parse_worker_counts,
    prepare_benchmark_output,
    run_leading_edge_worker_benchmark,
    sha256_file,
    summarize_leading_edge_benchmark_runs,
    validate_benchmark_output_path,
    validate_worker_counts,
    write_leading_edge_benchmark_runs_csv,
)
from nek_post.leading_edge_io import (
    LEADING_EDGE_METADATA_COLUMNS,
    LEADING_EDGE_TIMESERIES_COLUMNS,
)


def _write_artifacts(
    directory: Path,
    *,
    x_front: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, np.nan),
        (1.2, 1.3),
    ),
    threshold: float = 0.1,
    maximum_iteration_count: int = 4,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    timeseries = directory / "N7_leading_edge_timeseries.csv"
    metadata = directory / "N7_leading_edge_metadata.csv"
    file_indices = (10, 20)
    target_times = (0.5, 0.75)
    actual_times = (0.5, 0.75)
    y_values = (0.0, 0.5)
    with timeseries.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADING_EDGE_TIMESERIES_COLUMNS)
        writer.writeheader()
        for frame_position, file_index in enumerate(file_indices):
            for y_position, y_value in enumerate(y_values):
                front = x_front[frame_position][y_position]
                success = bool(np.isfinite(front))
                writer.writerow(
                    {
                        "case": "N7",
                        "file_index": file_index,
                        "source_file": f"/data/N7/GC0.f{file_index:05d}",
                        "target_time": target_times[frame_position],
                        "actual_time": actual_times[frame_position],
                        "time_error": 0.0,
                        "y": y_value,
                        "x_front": "nan" if np.isnan(front) else front,
                        "success": success,
                        "crossing_count": 1 if success else 0,
                        "threshold": threshold,
                        "z_target": 0.04,
                        "nx": 3,
                        "native_ny": 1,
                        "dense_ny": 2,
                        "y_upsample_factor": 2,
                    }
                )
    metadata_row = {
        "case": "N7",
        "n_input_frames": 2,
        "n_selected_frames": 2,
        "actual_time_start": 0.5,
        "actual_time_end": 0.75,
        "target_time_spacing": 0.25,
        "threshold": threshold,
        "z_target": 0.04,
        "nx": 3,
        "native_ny": 1,
        "dense_ny": 2,
        "y_upsample_factor": 2,
        "y_min": 0.0,
        "y_max_periodic_endpoint": 1.0,
        "periodic_endpoint_included": False,
        "algorithm_version": 1,
        "inverse_mapping_target_count": 6,
        "inverse_mapping_success_count": 6,
        "inverse_mapping_failure_count": 0,
        "ambiguous_boundary_point_count": 0,
        "maximum_successful_residual": 1.0e-14,
        "maximum_iteration_count": maximum_iteration_count,
    }
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADING_EDGE_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(metadata_row)
    return timeseries, metadata


def _run(
    worker_count: int,
    repetition: int,
    wall: float,
    *,
    user: float | None = None,
    system: float = 0.5,
    rss: int = 1000,
    equivalent: bool = True,
) -> LeadingEdgeBenchmarkRun:
    directory = Path(f"runs/workers_{worker_count}/run_{repetition:03d}")
    return LeadingEdgeBenchmarkRun(
        case="N7",
        file_prefix="GC0",
        worker_count=worker_count,
        repetition=repetition,
        start_index=37,
        end_index=52,
        nx=500,
        z_target=0.04,
        threshold=0.1,
        y_upsample_factor=2,
        contour_time_spacing=0.25,
        all_frames=False,
        wall_seconds=wall,
        user_cpu_seconds=wall if user is None else user,
        system_cpu_seconds=system,
        maximum_rss_kb=rss,
        return_code=0,
        run_directory=directory,
        timeseries_csv=directory / "N7_leading_edge_timeseries.csv",
        metadata_csv=directory / "N7_leading_edge_metadata.csv",
        timeseries_sha256="a" * 64,
        metadata_sha256="b" * 64,
        equivalent_to_baseline=equivalent,
    )


def test_worker_count_parsing_preserves_one_two_four() -> None:
    assert parse_worker_counts("1,2,4") == (1, 2, 4)
    assert parse_worker_counts("4, 1, 2") == (4, 1, 2)
    assert validate_worker_counts((1, 2, 4)) == (1, 2, 4)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "at least one"),
        ("1,", "empty"),
        ("1,2,2", "duplicate"),
        ("1,0,2", "greater than or equal|>= 1"),
        ("1,-2", "greater than or equal|>= 1"),
        ("1,2.0", "integer"),
        ("2,4", "workers=1"),
    ],
)
def test_invalid_worker_count_text_is_rejected(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_worker_counts(text)


def test_package_worker_validation_rejects_bool() -> None:
    with pytest.raises(ValueError, match="integer"):
        validate_worker_counts((1, True))


def test_regular_compute_command_is_deterministic_and_explicit(tmp_path: Path) -> None:
    command = build_leading_edge_compute_command(
        repo_root=tmp_path / "repo",
        case="N7",
        file_prefix="GC0",
        start_index=37,
        end_index=52,
        nx=500,
        z_target=0.04,
        threshold=0.1,
        y_upsample_factor=2,
        contour_time_spacing=0.25,
        all_frames=False,
        worker_count=2,
        output_dir=tmp_path / "run",
        python_executable="/test/python",
    )

    assert command == (
        "/test/python",
        str(tmp_path / "repo" / "scripts" / "19_compute_leading_edge_evolution.py"),
        "--case",
        "N7",
        "--file-prefix",
        "GC0",
        "--start-index",
        "37",
        "--end-index",
        "52",
        "--nx",
        "500",
        "--z-target",
        "0.04",
        "--threshold",
        "0.1",
        "--y-upsample-factor",
        "2",
        "--workers",
        "2",
        "--contour-time-spacing",
        "0.25",
        "--output-dir",
        str(tmp_path / "run"),
        "--overwrite",
    )


def test_all_frames_command_omits_contour_spacing(tmp_path: Path) -> None:
    command = build_leading_edge_compute_command(
        repo_root=tmp_path,
        case="N7",
        file_prefix="GC0",
        start_index=None,
        end_index=None,
        nx=1000,
        z_target=0.04,
        threshold=0.1,
        y_upsample_factor=2,
        contour_time_spacing=None,
        all_frames=True,
        worker_count=4,
        output_dir=tmp_path / "run",
    )

    assert command[0] == benchmark.sys.executable
    assert "--all-frames" in command
    assert "--contour-time-spacing" not in command


def test_gnu_time_metrics_parsing_and_malformed_inputs(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(
        "wall_seconds=12.5\nuser_cpu_seconds=18.25\n"
        "system_cpu_seconds=1.75\nmaximum_rss_kb=456789\n",
        encoding="utf-8",
    )
    assert parse_gnu_time_metrics(metrics) == GnuTimeMetrics(
        12.5, 18.25, 1.75, 456789
    )

    metrics.write_text("wall_seconds=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        parse_gnu_time_metrics(metrics)
    metrics.write_text("not-machine-readable\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        parse_gnu_time_metrics(metrics)
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_gnu_time_metrics(tmp_path / "absent.txt")


def test_sha256_file_uses_standard_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_exact_artifact_equivalence_succeeds(tmp_path: Path) -> None:
    baseline = _write_artifacts(tmp_path / "baseline")
    candidate = _write_artifacts(tmp_path / "candidate")

    assert_leading_edge_artifacts_equivalent(*baseline, *candidate)


def test_mismatched_x_front_is_rejected(tmp_path: Path) -> None:
    baseline = _write_artifacts(tmp_path / "baseline")
    candidate = _write_artifacts(
        tmp_path / "candidate",
        x_front=((1.000000000000001, np.nan), (1.2, 1.3)),
    )

    with pytest.raises(ScientificArtifactMismatchError, match="x_front"):
        assert_leading_edge_artifacts_equivalent(*baseline, *candidate)


def test_mismatched_nan_location_is_rejected(tmp_path: Path) -> None:
    baseline = _write_artifacts(tmp_path / "baseline")
    candidate = _write_artifacts(
        tmp_path / "candidate",
        x_front=((np.nan, 1.0), (1.2, 1.3)),
    )

    with pytest.raises(ScientificArtifactMismatchError, match="x_front"):
        assert_leading_edge_artifacts_equivalent(*baseline, *candidate)


def test_mismatched_metadata_is_rejected(tmp_path: Path) -> None:
    baseline = _write_artifacts(tmp_path / "baseline")
    candidate = _write_artifacts(
        tmp_path / "candidate", maximum_iteration_count=5
    )

    with pytest.raises(
        ScientificArtifactMismatchError,
        match="metadata.maximum_iteration_count",
    ):
        assert_leading_edge_artifacts_equivalent(*baseline, *candidate)


def test_summary_medians_speedup_efficiency_and_order() -> None:
    runs = (
        _run(4, 2, 5.0, user=18.0, rss=4400),
        _run(1, 1, 10.0, user=9.0, rss=1000),
        _run(2, 2, 8.0, user=14.0, rss=2300),
        _run(1, 2, 14.0, user=13.0, rss=1200),
        _run(4, 1, 4.0, user=16.0, rss=4000),
        _run(2, 1, 6.0, user=12.0, rss=2100),
    )

    summaries = summarize_leading_edge_benchmark_runs(runs, (1, 2, 4))

    assert tuple(summary.worker_count for summary in summaries) == (1, 2, 4)
    assert summaries[0].median_wall_seconds == 12.0
    assert summaries[0].minimum_wall_seconds == 10.0
    assert summaries[0].maximum_wall_seconds == 14.0
    assert summaries[1].median_wall_seconds == 7.0
    assert summaries[1].speedup_relative_to_workers_1 == pytest.approx(12 / 7)
    assert summaries[1].efficiency == pytest.approx(6 / 7)
    assert summaries[2].median_maximum_rss_kb == 4200.0
    assert summaries[2].speedup_relative_to_workers_1 == pytest.approx(12 / 4.5)
    assert summaries[2].efficiency == pytest.approx((12 / 4.5) / 4)
    assert all(summary.all_artifacts_equivalent for summary in summaries)


def test_benchmark_csv_has_stable_schema_and_deterministic_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "benchmark.csv"
    runs = (_run(4, 1, 4.0), _run(1, 2, 12.0), _run(1, 1, 10.0))

    write_leading_edge_benchmark_runs_csv(path, runs, (1, 4))

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == LEADING_EDGE_BENCHMARK_COLUMNS
    assert [(row["worker_count"], row["repetition"]) for row in rows] == [
        ("1", "1"),
        ("1", "2"),
        ("4", "1"),
    ]


def _safety_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    results = tmp_path / "results"
    production = results / "leading_edge" / "N7"
    repo.mkdir()
    results.mkdir()
    production.mkdir(parents=True)
    return repo, results, production


def test_unsafe_overwrite_paths_are_rejected(tmp_path: Path) -> None:
    repo, results, production = _safety_paths(tmp_path)
    for unsafe in (Path("/"), Path.home(), repo, results, production):
        with pytest.raises(ValueError, match="Unsafe|separate"):
            validate_benchmark_output_path(
                unsafe,
                repo_root=repo,
                results_root=results,
                production_output_dir=production,
            )


def test_existing_output_preflight_and_safe_known_cleanup(tmp_path: Path) -> None:
    repo, results, production = _safety_paths(tmp_path)
    output = results / "leading_edge_benchmarks" / "N7"
    output.mkdir(parents=True)
    report = output / "leading_edge_workers_benchmark.csv"
    report.write_text("old", encoding="utf-8")
    run_dir = output / "runs" / "workers_1" / "run_001"
    run_dir.mkdir(parents=True)
    marker = output / "unrelated.keep"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        prepare_benchmark_output(
            output,
            overwrite=False,
            repo_root=repo,
            results_root=results,
            production_output_dir=production,
        )

    resolved = prepare_benchmark_output(
        output,
        overwrite=True,
        repo_root=repo,
        results_root=results,
        production_output_dir=production,
    )
    assert resolved == output.resolve()
    assert not report.exists()
    assert not (output / "runs").exists()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_failed_subprocess_is_recorded_and_reports_stderr_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, results, production = _safety_paths(tmp_path)
    output = results / "leading_edge_benchmarks" / "N7"

    def fail_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(benchmark.subprocess, "run", fail_run)
    with pytest.raises(
        LeadingEdgeBenchmarkError,
        match=r"workers=1, repetition=1.*stderr\.log",
    ):
        run_leading_edge_worker_benchmark(
            repo_root=repo,
            results_root=results,
            production_output_dir=production,
            case="N7",
            file_prefix="GC0",
            start_index=37,
            end_index=52,
            nx=500,
            z_target=0.04,
            threshold=0.1,
            y_upsample_factor=2,
            contour_time_spacing=0.25,
            all_frames=False,
            worker_counts=(1, 2, 4),
            repeats=1,
            warmup_runs=0,
            output_dir=output,
            overwrite=False,
        )

    report = output / "leading_edge_workers_benchmark.csv"
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["worker_count"] == "1"
    assert rows[0]["return_code"] == "9"
    assert Path(rows[0]["run_directory"]).joinpath("stderr.log").is_file()


def test_successful_orchestration_isolates_runs_and_excludes_warmups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, results, production = _safety_paths(tmp_path)
    output = results / "leading_edge_benchmarks" / "N7"
    commands: list[tuple[str, ...]] = []
    messages: list[str] = []

    def successful_run(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        metrics = Path(command[command.index("--output") + 1])
        run_directory = Path(command[command.index("--output-dir") + 1])
        workers = int(command[command.index("--workers") + 1])
        metrics.write_text(
            f"wall_seconds={12.0 / workers}\n"
            f"user_cpu_seconds={10.0 + workers}\n"
            "system_cpu_seconds=1.0\n"
            f"maximum_rss_kb={100000 * workers}\n",
            encoding="utf-8",
        )
        _write_artifacts(run_directory)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", successful_run)
    result = run_leading_edge_worker_benchmark(
        repo_root=repo,
        results_root=results,
        production_output_dir=production,
        case="N7",
        file_prefix="GC0",
        start_index=37,
        end_index=52,
        nx=500,
        z_target=0.04,
        threshold=0.1,
        y_upsample_factor=2,
        contour_time_spacing=0.25,
        all_frames=False,
        worker_counts=(1, 2, 4),
        repeats=2,
        warmup_runs=1,
        output_dir=output,
        overwrite=False,
        reporter=messages.append,
    )

    assert len(commands) == 9
    assert len(result.runs) == 6
    assert tuple(run.worker_count for run in result.runs) == (1, 1, 2, 2, 4, 4)
    assert all(run.equivalent_to_baseline is True for run in result.runs)
    assert tuple(summary.worker_count for summary in result.summaries) == (1, 2, 4)
    assert result.benchmark_csv.is_file()
    assert result.summary_csv.is_file()
    assert not (output / "warmups").exists()
    assert sum(message.startswith("workers=") for message in messages) == 6
    for record in result.runs:
        assert record.run_directory.is_dir()
        assert (record.run_directory / "stdout.log").is_file()
        assert (record.run_directory / "stderr.log").is_file()
        assert (record.run_directory / "metrics.txt").is_file()
    for command in commands:
        assert command[0] == "/usr/bin/time"
        assert benchmark.sys.executable in command
        assert "--contour-time-spacing" in command
        assert command[command.index("--contour-time-spacing") + 1] == "0.25"
