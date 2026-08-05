"""Reproducible subprocess benchmarks for leading-edge worker counts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import csv
from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import numpy as np

from nek_post.front_detection_io import format_csv_value
from nek_post.leading_edge_artifacts import (
    LeadingEdgePlotData,
    read_leading_edge_artifacts,
)
from nek_post.leading_edge_io import (
    LEADING_EDGE_METADATA_COLUMNS,
    LEADING_EDGE_TIMESERIES_COLUMNS,
    leading_edge_metadata_path,
    leading_edge_timeseries_path,
)


GNU_TIME_EXECUTABLE = Path("/usr/bin/time")
GNU_TIME_FORMAT = (
    "wall_seconds=%e\n"
    "user_cpu_seconds=%U\n"
    "system_cpu_seconds=%S\n"
    "maximum_rss_kb=%M"
)

LEADING_EDGE_BENCHMARK_COLUMNS = (
    "case",
    "file_prefix",
    "worker_count",
    "repetition",
    "start_index",
    "end_index",
    "nx",
    "z_target",
    "threshold",
    "y_upsample_factor",
    "time_selection_mode",
    "contour_time_spacing",
    "wall_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "maximum_rss_kb",
    "return_code",
    "run_directory",
    "timeseries_csv",
    "metadata_csv",
    "timeseries_sha256",
    "metadata_sha256",
    "equivalent_to_baseline",
)

LEADING_EDGE_BENCHMARK_SUMMARY_COLUMNS = (
    "case",
    "worker_count",
    "successful_run_count",
    "median_wall_seconds",
    "minimum_wall_seconds",
    "maximum_wall_seconds",
    "median_user_cpu_seconds",
    "median_system_cpu_seconds",
    "median_maximum_rss_kb",
    "speedup_relative_to_workers_1",
    "efficiency",
    "all_artifacts_equivalent",
)


class LeadingEdgeBenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a valid comparison."""


class ScientificArtifactMismatchError(LeadingEdgeBenchmarkError):
    """Raised when two leading-edge artifact pairs differ scientifically."""


@dataclass(frozen=True)
class GnuTimeMetrics:
    """Machine-readable metrics emitted by GNU time."""

    wall_seconds: float
    user_cpu_seconds: float
    system_cpu_seconds: float
    maximum_rss_kb: int


@dataclass(frozen=True)
class LeadingEdgeBenchmarkRun:
    """One measured subprocess execution and its artifact identity."""

    case: str
    file_prefix: str
    worker_count: int
    repetition: int
    start_index: int | None
    end_index: int | None
    nx: int
    z_target: float
    threshold: float
    y_upsample_factor: int
    contour_time_spacing: float | None
    all_frames: bool
    wall_seconds: float
    user_cpu_seconds: float
    system_cpu_seconds: float
    maximum_rss_kb: int
    return_code: int
    run_directory: Path
    timeseries_csv: Path
    metadata_csv: Path
    timeseries_sha256: str
    metadata_sha256: str
    equivalent_to_baseline: bool | None


@dataclass(frozen=True)
class LeadingEdgeBenchmarkSummary:
    """Median timing, memory, speedup, and equivalence for one worker count."""

    case: str
    worker_count: int
    successful_run_count: int
    median_wall_seconds: float
    minimum_wall_seconds: float
    maximum_wall_seconds: float
    median_user_cpu_seconds: float
    median_system_cpu_seconds: float
    median_maximum_rss_kb: float
    speedup_relative_to_workers_1: float
    efficiency: float
    all_artifacts_equivalent: bool


@dataclass(frozen=True)
class LeadingEdgeBenchmarkResult:
    """Completed benchmark records and their two report paths."""

    runs: tuple[LeadingEdgeBenchmarkRun, ...]
    summaries: tuple[LeadingEdgeBenchmarkSummary, ...]
    benchmark_csv: Path
    summary_csv: Path


@dataclass(frozen=True)
class _ExecutedCommand:
    return_code: int
    command: tuple[str, ...]
    run_directory: Path
    stdout_log: Path
    stderr_log: Path
    metrics_file: Path


@dataclass(frozen=True)
class _ArtifactSnapshot:
    plot_data: LeadingEdgePlotData
    source_files: tuple[str, ...]
    metadata_values: tuple[tuple[str, object], ...]


def _integer_at_least(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return parsed


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def validate_worker_counts(values: Iterable[int]) -> tuple[int, ...]:
    """Validate worker counts while preserving caller-supplied order."""
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise ValueError("worker_counts must contain at least one value.") from exc
    if not supplied:
        raise ValueError("worker_counts must contain at least one value.")
    parsed = tuple(
        _integer_at_least(value, "worker count", 1) for value in supplied
    )
    if len(set(parsed)) != len(parsed):
        raise ValueError("worker_counts must not contain duplicate values.")
    if 1 not in parsed:
        raise ValueError("worker_counts must include workers=1 as the baseline.")
    return parsed


def parse_worker_counts(text: str) -> tuple[int, ...]:
    """Parse a comma-separated worker list such as ``1,2,4``."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("worker_counts must contain at least one value.")
    raw_values = text.split(",")
    if any(not value.strip() for value in raw_values):
        raise ValueError("worker_counts contains an empty value.")
    parsed: list[int] = []
    for raw_value in raw_values:
        stripped = raw_value.strip()
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"worker count {stripped!r} must be an integer >= 1."
            ) from exc
        if str(value) != stripped and stripped not in {f"+{value}", f"0{value}"}:
            # int() already rejects decimal text; this additionally keeps the
            # accepted representation intentionally narrow and reproducible.
            if stripped.lstrip("+").lstrip("0") != str(value):
                raise ValueError(
                    f"worker count {stripped!r} must be an integer >= 1."
                )
        parsed.append(value)
    return validate_worker_counts(parsed)


def _numeric_text(value: Real) -> str:
    return f"{float(value):.16g}"


def build_leading_edge_compute_command(
    *,
    repo_root: str | Path,
    case: str,
    file_prefix: str,
    start_index: int | None,
    end_index: int | None,
    nx: int,
    z_target: float,
    threshold: float,
    y_upsample_factor: int,
    contour_time_spacing: float | None,
    all_frames: bool,
    worker_count: int,
    output_dir: str | Path,
    python_executable: str | Path | None = None,
) -> tuple[str, ...]:
    """Build the deterministic script-19 command for one isolated run."""
    interpreter = sys.executable if python_executable is None else str(python_executable)
    command = [
        interpreter,
        str(Path(repo_root) / "scripts" / "19_compute_leading_edge_evolution.py"),
        "--case",
        str(case),
        "--file-prefix",
        str(file_prefix),
    ]
    if start_index is not None:
        command.extend(("--start-index", str(start_index)))
    if end_index is not None:
        command.extend(("--end-index", str(end_index)))
    command.extend(
        (
            "--nx",
            str(nx),
            "--z-target",
            _numeric_text(z_target),
            "--threshold",
            _numeric_text(threshold),
            "--y-upsample-factor",
            str(y_upsample_factor),
            "--workers",
            str(worker_count),
        )
    )
    if all_frames:
        command.append("--all-frames")
    else:
        if contour_time_spacing is None:
            raise ValueError(
                "contour_time_spacing is required unless all_frames is enabled."
            )
        command.extend(
            ("--contour-time-spacing", _numeric_text(contour_time_spacing))
        )
    command.extend(("--output-dir", str(Path(output_dir)), "--overwrite"))
    return tuple(command)


def build_leading_edge_benchmark_command(
    *,
    metrics_file: str | Path,
    time_executable: str | Path = GNU_TIME_EXECUTABLE,
    **compute_options: Any,
) -> tuple[str, ...]:
    """Wrap a deterministic script-19 command with machine-readable GNU time."""
    compute_command = build_leading_edge_compute_command(**compute_options)
    return (
        str(time_executable),
        "--format",
        GNU_TIME_FORMAT,
        "--output",
        str(Path(metrics_file)),
        *compute_command,
    )


def require_gnu_time(path: str | Path = GNU_TIME_EXECUTABLE) -> Path:
    """Return the GNU-time executable or fail before benchmark work begins."""
    executable = Path(path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"GNU time executable is unavailable or not executable: {executable}"
        )
    return executable


def parse_gnu_time_metrics(path: str | Path) -> GnuTimeMetrics:
    """Parse the dedicated key-value metrics file emitted by GNU time."""
    metrics_path = Path(path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"GNU time metrics file not found: {metrics_path}")
    expected = {
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "maximum_rss_kb",
    }
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(
                f"Malformed GNU time metrics line {line_number} in {metrics_path}."
            )
        name, raw_value = line.split("=", 1)
        if name not in expected or name in values or not raw_value:
            raise ValueError(
                f"Malformed GNU time metric {name!r} in {metrics_path}."
            )
        values[name] = raw_value
    missing = sorted(expected - values.keys())
    if missing:
        raise ValueError(
            f"GNU time metrics file {metrics_path} is missing: "
            + ", ".join(missing)
            + "."
        )
    wall = _finite_float(values["wall_seconds"], "wall_seconds")
    user = _finite_float(values["user_cpu_seconds"], "user_cpu_seconds")
    system = _finite_float(values["system_cpu_seconds"], "system_cpu_seconds")
    try:
        maximum_rss = int(values["maximum_rss_kb"])
    except ValueError as exc:
        raise ValueError(
            f"maximum_rss_kb in {metrics_path} must be an integer."
        ) from exc
    if wall < 0.0 or user < 0.0 or system < 0.0 or maximum_rss < 0:
        raise ValueError(f"GNU time metrics in {metrics_path} must be non-negative.")
    return GnuTimeMetrics(wall, user, system, maximum_rss)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_METADATA_INTEGER_COLUMNS = {
    "n_input_frames",
    "n_selected_frames",
    "nx",
    "native_ny",
    "dense_ny",
    "y_upsample_factor",
    "algorithm_version",
    "inverse_mapping_target_count",
    "inverse_mapping_success_count",
    "inverse_mapping_failure_count",
    "ambiguous_boundary_point_count",
    "maximum_iteration_count",
}
_METADATA_BOOLEAN_COLUMNS = {"periodic_endpoint_included"}


def _metadata_snapshot(path: Path) -> tuple[tuple[str, object], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEADING_EDGE_METADATA_COLUMNS:
            raise ValueError(f"Invalid leading-edge metadata schema in {path}.")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Leading-edge metadata must contain one row: {path}")
    row = rows[0]
    result: list[tuple[str, object]] = []
    for column in LEADING_EDGE_METADATA_COLUMNS:
        raw = row[column]
        if column in {"case", "extraction_method"}:
            value: object = raw
        elif column in _METADATA_INTEGER_COLUMNS:
            value = int(raw)
        elif column in _METADATA_BOOLEAN_COLUMNS:
            if raw not in {"True", "False"}:
                raise ValueError(f"Malformed boolean {column}={raw!r} in {path}.")
            value = raw == "True"
        else:
            parsed = float(raw)
            value = None if column == "target_time_spacing" and math.isnan(parsed) else parsed
        result.append((column, value))
    return tuple(result)


def _source_files_snapshot(
    path: Path,
    file_indices: np.ndarray,
) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEADING_EDGE_TIMESERIES_COLUMNS:
            raise ValueError(f"Invalid leading-edge timeseries schema in {path}.")
        sources: dict[int, str] = {}
        for row in reader:
            index = int(row["file_index"])
            source = row["source_file"]
            previous = sources.setdefault(index, source)
            if previous != source:
                raise ValueError(
                    f"File index {index} has inconsistent source files in {path}."
                )
    return tuple(sources[int(index)] for index in file_indices)


def _artifact_snapshot(
    timeseries_csv: str | Path,
    metadata_csv: str | Path,
) -> _ArtifactSnapshot:
    timeseries_path = Path(timeseries_csv)
    metadata_path = Path(metadata_csv)
    plot_data = read_leading_edge_artifacts(timeseries_path, metadata_path)
    return _ArtifactSnapshot(
        plot_data=plot_data,
        source_files=_source_files_snapshot(timeseries_path, plot_data.file_indices),
        metadata_values=_metadata_snapshot(metadata_path),
    )


def _require_equal(name: str, baseline: object, candidate: object) -> None:
    if baseline != candidate:
        raise ScientificArtifactMismatchError(
            f"Leading-edge artifact mismatch for {name}: "
            f"baseline={baseline!r}, candidate={candidate!r}."
        )


def _require_exact_array(
    name: str,
    baseline: object,
    candidate: object,
    *,
    floating: bool,
) -> None:
    baseline_array = np.asarray(baseline)
    candidate_array = np.asarray(candidate)
    if baseline_array.shape != candidate_array.shape:
        raise ScientificArtifactMismatchError(
            f"Leading-edge artifact mismatch for {name} shape: "
            f"{baseline_array.shape} != {candidate_array.shape}."
        )
    try:
        if floating:
            np.testing.assert_allclose(
                candidate_array,
                baseline_array,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(candidate_array, baseline_array)
    except AssertionError as exc:
        raise ScientificArtifactMismatchError(
            f"Leading-edge artifact mismatch for {name}: {exc}"
        ) from exc


def assert_leading_edge_artifacts_equivalent(
    baseline_timeseries_csv: str | Path,
    baseline_metadata_csv: str | Path,
    candidate_timeseries_csv: str | Path,
    candidate_metadata_csv: str | Path,
) -> None:
    """Require exact scientific equality between two validated artifact pairs."""
    try:
        baseline = _artifact_snapshot(
            baseline_timeseries_csv, baseline_metadata_csv
        )
        candidate = _artifact_snapshot(
            candidate_timeseries_csv, candidate_metadata_csv
        )
    except ScientificArtifactMismatchError:
        raise
    except Exception as exc:
        raise ScientificArtifactMismatchError(
            f"Could not validate leading-edge artifacts exactly: {exc}"
        ) from exc
    left = baseline.plot_data
    right = candidate.plot_data
    for name in (
        "case",
        "extraction_method",
        "nx",
        "native_ny",
        "dense_ny",
        "y_upsample_factor",
        "periodic_endpoint_included",
    ):
        _require_equal(name, getattr(left, name), getattr(right, name))
    _require_equal("source_files", baseline.source_files, candidate.source_files)
    _require_exact_array(
        "file_indices", left.file_indices, right.file_indices, floating=False
    )
    for name in (
        "target_time",
        "actual_time",
        "time_error",
        "y",
        "x_front",
    ):
        _require_exact_array(
            name, getattr(left, name), getattr(right, name), floating=True
        )
    for name in ("success_mask", "crossing_count"):
        _require_exact_array(
            name, getattr(left, name), getattr(right, name), floating=False
        )
    for name in (
        "threshold",
        "z_target",
        "y_min",
        "y_max_periodic_endpoint",
    ):
        _require_exact_array(
            name, [getattr(left, name)], [getattr(right, name)], floating=True
        )
    if left.target_time_spacing is None or right.target_time_spacing is None:
        _require_equal(
            "target_time_spacing",
            left.target_time_spacing,
            right.target_time_spacing,
        )
    else:
        _require_exact_array(
            "target_time_spacing",
            [left.target_time_spacing],
            [right.target_time_spacing],
            floating=True,
        )
    _require_equal(
        "metadata column names",
        tuple(name for name, _ in baseline.metadata_values),
        tuple(name for name, _ in candidate.metadata_values),
    )
    for (name, baseline_value), (_, candidate_value) in zip(
        baseline.metadata_values, candidate.metadata_values, strict=True
    ):
        if isinstance(baseline_value, float) and isinstance(candidate_value, float):
            _require_exact_array(
                f"metadata.{name}",
                [baseline_value],
                [candidate_value],
                floating=True,
            )
        else:
            _require_equal(f"metadata.{name}", baseline_value, candidate_value)


def _benchmark_report_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "leading_edge_workers_benchmark.csv",
        output_dir / "leading_edge_workers_summary.csv",
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_benchmark_output_path(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    results_root: str | Path,
    production_output_dir: str | Path,
) -> Path:
    """Resolve a benchmark directory and reject destructive target locations."""
    if str(output_dir).strip() == "":
        raise ValueError("Benchmark output directory must not be empty.")
    candidate = Path(output_dir).expanduser().resolve(strict=False)
    protected = (
        Path("/").resolve(),
        Path.home().resolve(strict=False),
        Path(repo_root).resolve(strict=False),
        Path(results_root).resolve(strict=False),
    )
    for unsafe in protected:
        if candidate == unsafe or candidate in unsafe.parents:
            raise ValueError(
                f"Unsafe benchmark output directory {candidate}; it overlaps "
                f"protected path {unsafe}."
            )
    production = Path(production_output_dir).resolve(strict=False)
    if _paths_overlap(candidate, production):
        raise ValueError(
            f"Benchmark output directory {candidate} must be separate from "
            f"production output {production}."
        )
    return candidate


def _remove_known_benchmark_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_benchmark_output(
    output_dir: str | Path,
    *,
    overwrite: bool,
    repo_root: str | Path,
    results_root: str | Path,
    production_output_dir: str | Path,
) -> Path:
    """Preflight reports/runs and remove only known benchmark artifacts."""
    resolved = validate_benchmark_output_path(
        output_dir,
        repo_root=repo_root,
        results_root=results_root,
        production_output_dir=production_output_dir,
    )
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(
            f"Benchmark output path exists and is not a directory: {resolved}"
        )
    reports = _benchmark_report_paths(resolved)
    owned_directories = (resolved / "runs", resolved / "warmups")
    conflicts = tuple(
        path for path in (*reports, *owned_directories) if path.exists() or path.is_symlink()
    )
    if conflicts and not overwrite:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Benchmark output already exists: {joined}. Pass --overwrite to replace it."
        )
    if overwrite:
        for path in (*reports, *owned_directories):
            _remove_known_benchmark_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_row(run: LeadingEdgeBenchmarkRun) -> dict[str, object]:
    return {
        "case": run.case,
        "file_prefix": run.file_prefix,
        "worker_count": run.worker_count,
        "repetition": run.repetition,
        "start_index": run.start_index,
        "end_index": run.end_index,
        "nx": run.nx,
        "z_target": run.z_target,
        "threshold": run.threshold,
        "y_upsample_factor": run.y_upsample_factor,
        "time_selection_mode": "all_frames" if run.all_frames else "regular_spacing",
        "contour_time_spacing": run.contour_time_spacing,
        "wall_seconds": run.wall_seconds,
        "user_cpu_seconds": run.user_cpu_seconds,
        "system_cpu_seconds": run.system_cpu_seconds,
        "maximum_rss_kb": run.maximum_rss_kb,
        "return_code": run.return_code,
        "run_directory": str(run.run_directory),
        "timeseries_csv": str(run.timeseries_csv),
        "metadata_csv": str(run.metadata_csv),
        "timeseries_sha256": run.timeseries_sha256,
        "metadata_sha256": run.metadata_sha256,
        "equivalent_to_baseline": run.equivalent_to_baseline,
    }


def _summary_row(summary: LeadingEdgeBenchmarkSummary) -> dict[str, object]:
    return {
        "case": summary.case,
        "worker_count": summary.worker_count,
        "successful_run_count": summary.successful_run_count,
        "median_wall_seconds": summary.median_wall_seconds,
        "minimum_wall_seconds": summary.minimum_wall_seconds,
        "maximum_wall_seconds": summary.maximum_wall_seconds,
        "median_user_cpu_seconds": summary.median_user_cpu_seconds,
        "median_system_cpu_seconds": summary.median_system_cpu_seconds,
        "median_maximum_rss_kb": summary.median_maximum_rss_kb,
        "speedup_relative_to_workers_1": summary.speedup_relative_to_workers_1,
        "efficiency": summary.efficiency,
        "all_artifacts_equivalent": summary.all_artifacts_equivalent,
    }


def _csv_value(value: object) -> str:
    return "" if value is None else format_csv_value(value)


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row[column]) for column in columns})


def _ordered_runs(
    runs: Iterable[LeadingEdgeBenchmarkRun],
    worker_counts: Sequence[int],
) -> tuple[LeadingEdgeBenchmarkRun, ...]:
    order = {worker_count: position for position, worker_count in enumerate(worker_counts)}
    return tuple(
        sorted(runs, key=lambda run: (order[run.worker_count], run.repetition))
    )


def write_leading_edge_benchmark_runs_csv(
    path: str | Path,
    runs: Iterable[LeadingEdgeBenchmarkRun],
    worker_counts: Sequence[int],
) -> None:
    """Write measured-run records in supplied-worker then repetition order."""
    ordered = _ordered_runs(runs, worker_counts)
    _write_csv(
        Path(path),
        LEADING_EDGE_BENCHMARK_COLUMNS,
        (_run_row(run) for run in ordered),
    )


def write_leading_edge_benchmark_summary_csv(
    path: str | Path,
    summaries: Iterable[LeadingEdgeBenchmarkSummary],
) -> None:
    """Write summaries in the already-determined worker order."""
    _write_csv(
        Path(path),
        LEADING_EDGE_BENCHMARK_SUMMARY_COLUMNS,
        (_summary_row(summary) for summary in summaries),
    )


def summarize_leading_edge_benchmark_runs(
    runs: Iterable[LeadingEdgeBenchmarkRun],
    worker_counts: Sequence[int],
) -> tuple[LeadingEdgeBenchmarkSummary, ...]:
    """Calculate medians, speedup, efficiency, and equivalence by worker count."""
    counts = validate_worker_counts(worker_counts)
    supplied_runs = tuple(runs)
    grouped = {
        worker_count: tuple(
            run
            for run in supplied_runs
            if run.worker_count == worker_count and run.return_code == 0
        )
        for worker_count in counts
    }
    if not grouped[1]:
        raise LeadingEdgeBenchmarkError(
            "Cannot summarize benchmark without a successful workers=1 baseline."
        )
    baseline_median = float(
        statistics.median(run.wall_seconds for run in grouped[1])
    )
    if baseline_median <= 0.0:
        raise LeadingEdgeBenchmarkError(
            "Workers=1 median wall time must be greater than zero."
        )
    summaries: list[LeadingEdgeBenchmarkSummary] = []
    case = grouped[1][0].case
    for worker_count in counts:
        successful = grouped[worker_count]
        if not successful:
            raise LeadingEdgeBenchmarkError(
                f"No successful benchmark runs for workers={worker_count}."
            )
        wall_values = [run.wall_seconds for run in successful]
        median_wall = float(statistics.median(wall_values))
        if median_wall <= 0.0:
            raise LeadingEdgeBenchmarkError(
                f"Median wall time for workers={worker_count} must be positive."
            )
        speedup = baseline_median / median_wall
        summaries.append(
            LeadingEdgeBenchmarkSummary(
                case=case,
                worker_count=worker_count,
                successful_run_count=len(successful),
                median_wall_seconds=median_wall,
                minimum_wall_seconds=float(min(wall_values)),
                maximum_wall_seconds=float(max(wall_values)),
                median_user_cpu_seconds=float(
                    statistics.median(run.user_cpu_seconds for run in successful)
                ),
                median_system_cpu_seconds=float(
                    statistics.median(run.system_cpu_seconds for run in successful)
                ),
                median_maximum_rss_kb=float(
                    statistics.median(run.maximum_rss_kb for run in successful)
                ),
                speedup_relative_to_workers_1=float(speedup),
                efficiency=float(speedup / worker_count),
                all_artifacts_equivalent=all(
                    run.equivalent_to_baseline is True for run in successful
                ),
            )
        )
    return tuple(summaries)


def _execute_command(
    command: tuple[str, ...],
    *,
    repo_root: Path,
    run_directory: Path,
) -> _ExecutedCommand:
    run_directory.mkdir(parents=True, exist_ok=False)
    stdout_log = run_directory / "stdout.log"
    stderr_log = run_directory / "stderr.log"
    metrics_file = run_directory / "metrics.txt"
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=environment,
        )
    return _ExecutedCommand(
        return_code=int(completed.returncode),
        command=command,
        run_directory=run_directory,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        metrics_file=metrics_file,
    )


def _failed_metrics(path: Path) -> GnuTimeMetrics:
    try:
        return parse_gnu_time_metrics(path)
    except Exception:
        return GnuTimeMetrics(float("nan"), float("nan"), float("nan"), 0)


def _make_run_record(
    execution: _ExecutedCommand,
    metrics: GnuTimeMetrics,
    *,
    case: str,
    file_prefix: str,
    worker_count: int,
    repetition: int,
    start_index: int | None,
    end_index: int | None,
    nx: int,
    z_target: float,
    threshold: float,
    y_upsample_factor: int,
    contour_time_spacing: float | None,
    all_frames: bool,
    equivalent_to_baseline: bool | None,
) -> LeadingEdgeBenchmarkRun:
    timeseries_csv = leading_edge_timeseries_path(execution.run_directory, case)
    metadata_csv = leading_edge_metadata_path(execution.run_directory, case)
    return LeadingEdgeBenchmarkRun(
        case=case,
        file_prefix=file_prefix,
        worker_count=worker_count,
        repetition=repetition,
        start_index=start_index,
        end_index=end_index,
        nx=nx,
        z_target=z_target,
        threshold=threshold,
        y_upsample_factor=y_upsample_factor,
        contour_time_spacing=contour_time_spacing,
        all_frames=all_frames,
        wall_seconds=metrics.wall_seconds,
        user_cpu_seconds=metrics.user_cpu_seconds,
        system_cpu_seconds=metrics.system_cpu_seconds,
        maximum_rss_kb=metrics.maximum_rss_kb,
        return_code=execution.return_code,
        run_directory=execution.run_directory,
        timeseries_csv=timeseries_csv,
        metadata_csv=metadata_csv,
        timeseries_sha256=(sha256_file(timeseries_csv) if timeseries_csv.is_file() else ""),
        metadata_sha256=(sha256_file(metadata_csv) if metadata_csv.is_file() else ""),
        equivalent_to_baseline=equivalent_to_baseline,
    )


def _artifacts_equal_to_baseline(
    baseline: LeadingEdgeBenchmarkRun,
    candidate: LeadingEdgeBenchmarkRun,
) -> bool:
    if (
        candidate.timeseries_sha256 == baseline.timeseries_sha256
        and candidate.metadata_sha256 == baseline.metadata_sha256
    ):
        return True
    assert_leading_edge_artifacts_equivalent(
        baseline.timeseries_csv,
        baseline.metadata_csv,
        candidate.timeseries_csv,
        candidate.metadata_csv,
    )
    return True


def run_leading_edge_worker_benchmark(
    *,
    repo_root: str | Path,
    results_root: str | Path,
    production_output_dir: str | Path,
    case: str,
    file_prefix: str,
    start_index: int | None,
    end_index: int | None,
    nx: int,
    z_target: float,
    threshold: float,
    y_upsample_factor: int,
    contour_time_spacing: float | None,
    all_frames: bool,
    worker_counts: Sequence[int] = (1, 2, 4),
    repeats: int = 3,
    warmup_runs: int = 0,
    output_dir: str | Path,
    overwrite: bool,
    reporter: Callable[[str], None] | None = None,
    time_executable: str | Path = GNU_TIME_EXECUTABLE,
) -> LeadingEdgeBenchmarkResult:
    """Run isolated script-19 subprocesses and require exact artifact equality."""
    root = Path(repo_root).resolve(strict=False)
    case_value = str(case).strip().upper()
    prefix_value = str(file_prefix).strip()
    if not case_value:
        raise ValueError("case must not be empty.")
    if not prefix_value:
        raise ValueError("file_prefix must not be empty.")
    counts = validate_worker_counts(worker_counts)
    repeat_count = _integer_at_least(repeats, "repeats", 1)
    warmup_count = _integer_at_least(warmup_runs, "warmup_runs", 0)
    nx_value = _integer_at_least(nx, "nx", 2)
    upsample_value = _integer_at_least(
        y_upsample_factor, "y_upsample_factor", 1
    )
    start_value = (
        None if start_index is None else _integer_at_least(start_index, "start_index", 0)
    )
    end_value = None if end_index is None else _integer_at_least(end_index, "end_index", 0)
    if start_value is not None and end_value is not None and start_value > end_value:
        raise ValueError("start_index must be less than or equal to end_index.")
    z_value = _finite_float(z_target, "z_target")
    threshold_value = _finite_float(threshold, "threshold")
    spacing_value = (
        None
        if all_frames
        else _finite_float(contour_time_spacing, "contour_time_spacing", positive=True)
    )
    time_path = require_gnu_time(time_executable)
    benchmark_dir = prepare_benchmark_output(
        output_dir,
        overwrite=overwrite,
        repo_root=root,
        results_root=results_root,
        production_output_dir=production_output_dir,
    )
    benchmark_csv, summary_csv = _benchmark_report_paths(benchmark_dir)

    common_command_options = {
        "repo_root": root,
        "case": case_value,
        "file_prefix": prefix_value,
        "start_index": start_value,
        "end_index": end_value,
        "nx": nx_value,
        "z_target": z_value,
        "threshold": threshold_value,
        "y_upsample_factor": upsample_value,
        "contour_time_spacing": spacing_value,
        "all_frames": bool(all_frames),
    }

    warmup_root = benchmark_dir / "warmups"
    for worker_count in counts:
        for warmup_number in range(1, warmup_count + 1):
            run_directory = (
                warmup_root
                / f"workers_{worker_count}"
                / f"run_{warmup_number:03d}"
            )
            metrics_file = run_directory / "metrics.txt"
            command = build_leading_edge_benchmark_command(
                metrics_file=metrics_file,
                time_executable=time_path,
                worker_count=worker_count,
                output_dir=run_directory,
                **common_command_options,
            )
            execution = _execute_command(command, repo_root=root, run_directory=run_directory)
            if execution.return_code != 0:
                raise LeadingEdgeBenchmarkError(
                    "Leading-edge warmup failed for workers="
                    f"{worker_count}, warmup={warmup_number}; stderr log: "
                    f"{execution.stderr_log}"
                )
            parse_gnu_time_metrics(execution.metrics_file)
            if reporter is not None:
                reporter(
                    f"Warmup workers={worker_count} run={warmup_number} completed"
                )
    if warmup_root.exists():
        shutil.rmtree(warmup_root)

    runs: list[LeadingEdgeBenchmarkRun] = []
    baseline: LeadingEdgeBenchmarkRun | None = None
    for worker_count in counts:
        for repetition in range(1, repeat_count + 1):
            run_directory = (
                benchmark_dir
                / "runs"
                / f"workers_{worker_count}"
                / f"run_{repetition:03d}"
            )
            metrics_file = run_directory / "metrics.txt"
            command = build_leading_edge_benchmark_command(
                metrics_file=metrics_file,
                time_executable=time_path,
                worker_count=worker_count,
                output_dir=run_directory,
                **common_command_options,
            )
            execution = _execute_command(command, repo_root=root, run_directory=run_directory)
            if execution.return_code != 0:
                failed = _make_run_record(
                    execution,
                    _failed_metrics(execution.metrics_file),
                    case=case_value,
                    file_prefix=prefix_value,
                    worker_count=worker_count,
                    repetition=repetition,
                    start_index=start_value,
                    end_index=end_value,
                    nx=nx_value,
                    z_target=z_value,
                    threshold=threshold_value,
                    y_upsample_factor=upsample_value,
                    contour_time_spacing=spacing_value,
                    all_frames=bool(all_frames),
                    equivalent_to_baseline=False,
                )
                runs.append(failed)
                write_leading_edge_benchmark_runs_csv(benchmark_csv, runs, counts)
                raise LeadingEdgeBenchmarkError(
                    "Leading-edge benchmark failed for workers="
                    f"{worker_count}, repetition={repetition}; stderr log: "
                    f"{execution.stderr_log}"
                )
            metrics = parse_gnu_time_metrics(execution.metrics_file)
            record = _make_run_record(
                execution,
                metrics,
                case=case_value,
                file_prefix=prefix_value,
                worker_count=worker_count,
                repetition=repetition,
                start_index=start_value,
                end_index=end_value,
                nx=nx_value,
                z_target=z_value,
                threshold=threshold_value,
                y_upsample_factor=upsample_value,
                contour_time_spacing=spacing_value,
                all_frames=bool(all_frames),
                equivalent_to_baseline=None,
            )
            if not record.timeseries_sha256 or not record.metadata_sha256:
                runs.append(record)
                write_leading_edge_benchmark_runs_csv(benchmark_csv, runs, counts)
                raise LeadingEdgeBenchmarkError(
                    "Leading-edge benchmark produced missing CSV artifacts for "
                    f"workers={worker_count}, repetition={repetition}; run directory: "
                    f"{run_directory}"
                )
            if worker_count == 1 and baseline is None:
                baseline = replace(record, equivalent_to_baseline=True)
                record = baseline
            elif baseline is not None:
                try:
                    _artifacts_equal_to_baseline(baseline, record)
                except ScientificArtifactMismatchError as exc:
                    record = replace(record, equivalent_to_baseline=False)
                    runs.append(record)
                    write_leading_edge_benchmark_runs_csv(benchmark_csv, runs, counts)
                    raise LeadingEdgeBenchmarkError(
                        "Scientific artifact mismatch for workers="
                        f"{worker_count}, repetition={repetition}: {exc}"
                    ) from exc
                record = replace(record, equivalent_to_baseline=True)
            runs.append(record)

            if baseline is not None:
                for position, pending in enumerate(runs):
                    if pending.equivalent_to_baseline is not None:
                        continue
                    try:
                        _artifacts_equal_to_baseline(baseline, pending)
                    except ScientificArtifactMismatchError as exc:
                        runs[position] = replace(
                            pending, equivalent_to_baseline=False
                        )
                        write_leading_edge_benchmark_runs_csv(
                            benchmark_csv, runs, counts
                        )
                        raise LeadingEdgeBenchmarkError(
                            "Scientific artifact mismatch for workers="
                            f"{pending.worker_count}, repetition={pending.repetition}: "
                            f"{exc}"
                        ) from exc
                    runs[position] = replace(pending, equivalent_to_baseline=True)
            if reporter is not None:
                reporter(
                    f"workers={worker_count} repetition={repetition} "
                    f"wall={metrics.wall_seconds:.6g}s "
                    f"user={metrics.user_cpu_seconds:.6g}s "
                    f"system={metrics.system_cpu_seconds:.6g}s "
                    f"max_rss={metrics.maximum_rss_kb} KiB "
                    f"equivalent={record.equivalent_to_baseline}"
                )

    if baseline is None:
        write_leading_edge_benchmark_runs_csv(benchmark_csv, runs, counts)
        raise LeadingEdgeBenchmarkError(
            "Benchmark completed without a successful workers=1 baseline."
        )
    if any(run.equivalent_to_baseline is not True for run in runs):
        write_leading_edge_benchmark_runs_csv(benchmark_csv, runs, counts)
        raise LeadingEdgeBenchmarkError(
            "Benchmark contains runs that were not verified against workers=1."
        )
    ordered_runs = _ordered_runs(runs, counts)
    summaries = summarize_leading_edge_benchmark_runs(ordered_runs, counts)
    write_leading_edge_benchmark_runs_csv(benchmark_csv, ordered_runs, counts)
    write_leading_edge_benchmark_summary_csv(summary_csv, summaries)
    return LeadingEdgeBenchmarkResult(
        runs=ordered_runs,
        summaries=summaries,
        benchmark_csv=benchmark_csv,
        summary_csv=summary_csv,
    )


__all__ = (
    "GNU_TIME_EXECUTABLE",
    "GNU_TIME_FORMAT",
    "GnuTimeMetrics",
    "LEADING_EDGE_BENCHMARK_COLUMNS",
    "LEADING_EDGE_BENCHMARK_SUMMARY_COLUMNS",
    "LeadingEdgeBenchmarkError",
    "LeadingEdgeBenchmarkResult",
    "LeadingEdgeBenchmarkRun",
    "LeadingEdgeBenchmarkSummary",
    "ScientificArtifactMismatchError",
    "assert_leading_edge_artifacts_equivalent",
    "build_leading_edge_benchmark_command",
    "build_leading_edge_compute_command",
    "parse_gnu_time_metrics",
    "parse_worker_counts",
    "prepare_benchmark_output",
    "require_gnu_time",
    "run_leading_edge_worker_benchmark",
    "sha256_file",
    "summarize_leading_edge_benchmark_runs",
    "validate_benchmark_output_path",
    "validate_worker_counts",
    "write_leading_edge_benchmark_runs_csv",
    "write_leading_edge_benchmark_summary_csv",
)
