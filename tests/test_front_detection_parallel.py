from __future__ import annotations

import argparse
from concurrent.futures import Future
import csv
from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np
import pytest

import nek_post.front_detection_cache as cache_module
import nek_post.front_detection_parallel as parallel_module
from nek_post.front_detection_cache import (
    FrontDetectionCacheSpec,
    acquire_concentration_sequence,
)
from nek_post.front_detection_io import NekFramePath
from nek_post.front_detection_parallel import (
    FrontDetectionFrameResult,
    ParallelFramePreprocessingError,
    preprocess_front_detection_frame,
    preprocess_front_detection_frames_parallel,
    validate_preprocessing_workers,
)
from nek_post.front_detection_workflow import ConcentrationSequence
from nek_post.paths import ProjectPaths


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "16_detect_front_from_concentration.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "detect_front_from_concentration_script",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
front_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(front_script)


def _frame(index: int) -> NekFramePath:
    return NekFramePath(index=index, path=Path(f"GC0.f{index:05d}"))


def _result(index: int) -> FrontDetectionFrameResult:
    concentration = np.full((2, 3), float(index), dtype=np.float64)
    concentration.setflags(write=False)
    return FrontDetectionFrameResult(
        file_index=index,
        source_path=_frame(index).path,
        time=index / 4.0,
        concentration=concentration,
        finite_fraction=1.0,
        concentration_min=float(index),
        concentration_max=float(index),
        selected_y=index / 10.0,
    )


def _script_paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        case_dirs={"N7": tmp_path / "data" / "N7"},
        postproc_root=tmp_path / "postproc",
        results_root=tmp_path / "results",
        cantero_fig5a_re3450_csv=tmp_path / "paper.csv",
    )


def _cases_config() -> dict[str, object]:
    return {
        "file_prefix": "GC0",
        "grid": {"nx": 3, "nz": 2},
        "slice": {
            "mode": "nearest_plane",
            "slab_ratio": 0.01,
            "y_round_decimals": 10,
        },
    }


@pytest.mark.parametrize("text", ["1", "2", "4"])
def test_argparse_workers_adapter_accepts_positive_integer_strings(
    text: str,
) -> None:
    result = front_script._parse_preprocessing_workers(text)

    assert result == int(text)
    assert type(result) is int


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "1.0", "abc", ""])
def test_argparse_workers_adapter_rejects_invalid_strings(text: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="greater than or equal to 1",
    ):
        front_script._parse_preprocessing_workers(text)


def test_script_argparse_accepts_workers_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["script", "--workers", "2"])

    args = front_script._parse_args(_script_paths(tmp_path), _cases_config())

    assert args.workers == 2
    assert type(args.workers) is int


def test_script_argparse_workers_default_remains_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["script"])

    args = front_script._parse_args(_script_paths(tmp_path), _cases_config())

    assert args.workers == 1
    assert type(args.workers) is int
    assert args.interpolation_engine == "spectral_element"
    assert args.slice_y is None
    assert not args.no_reference_comparison


def test_script_argparse_accepts_no_reference_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["script", "--no-reference-comparison"]
    )

    args = front_script._parse_args(_script_paths(tmp_path), _cases_config())

    assert args.no_reference_comparison


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "abc", ""])
def test_script_argparse_rejects_invalid_workers(
    text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["script", "--workers", text])

    with pytest.raises(SystemExit) as error:
        front_script._parse_args(_script_paths(tmp_path), _cases_config())

    assert error.value.code == 2
    assert "greater than or equal to 1" in capsys.readouterr().err


def test_script_passes_integer_workers_to_sequence_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _script_paths(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["script", "--workers", "2", "--overwrite"])
    monkeypatch.setattr(front_script, "load_project_paths", lambda path: paths)
    monkeypatch.setattr(
        front_script,
        "load_yaml",
        lambda path: _cases_config(),
    )
    monkeypatch.setattr(
        front_script,
        "preflight_output_paths",
        lambda output_paths, overwrite: tuple(output_paths),
    )
    monkeypatch.setattr(
        front_script,
        "discover_nek_frame_paths",
        lambda *args, **kwargs: (_frame(1),),
    )
    monkeypatch.setattr(
        front_script,
        "default_front_detection_cache_path",
        lambda *args, **kwargs: tmp_path / "cache",
    )
    monkeypatch.setattr(
        front_script,
        "build_front_detection_cache_spec",
        lambda **kwargs: object(),
    )

    def build_sequence(*args: object, **kwargs: object) -> object:
        captured["workers"] = kwargs["workers"]
        return object()

    def acquire_sequence(**kwargs: object) -> None:
        kwargs["builder"]()
        raise SystemExit(0)

    monkeypatch.setattr(
        front_script,
        "build_concentration_sequence",
        build_sequence,
    )
    monkeypatch.setattr(
        front_script,
        "acquire_concentration_sequence",
        acquire_sequence,
    )

    with pytest.raises(SystemExit) as error:
        front_script.main()

    assert error.value.code == 0
    assert captured["workers"] == 2
    assert type(captured["workers"]) is int


def _script_sequence() -> ConcentrationSequence:
    Xi, Zi = np.meshgrid(
        np.linspace(0.0, 2.0, 3),
        np.linspace(0.0, 1.0, 2),
    )
    concentration = np.zeros((1, 2, 3), dtype=np.float64)
    concentration[0, 0, :2] = 1.0
    return ConcentrationSequence(
        time=np.array([0.0], dtype=np.float64),
        file_indices=np.array([1], dtype=np.int64),
        source_files=(Path("GC0.f00001"),),
        Xi=Xi,
        Zi=Zi,
        C_frames=concentration,
        finite_fraction=np.array([1.0], dtype=np.float64),
        concentration_min=np.array([0.0], dtype=np.float64),
        concentration_max=np.array([1.0], dtype=np.float64),
        grid_metadata=MappingProxyType(
            {
                "xmin": 0.0,
                "xmax": 2.0,
                "zmin": 0.0,
                "zmax": 1.0,
                "nx": 3,
                "nz": 2,
            }
        ),
        selected_y=np.array([0.75], dtype=np.float64),
        interpolation_method="spectral",
        interpolation_engine="spectral_element",
        spectral_element_shape=(4, 4, 4),
        spectral_polynomial_order=(3, 3, 3),
        spectral_slice_y=0.75,
    )


def _install_script_run_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _script_paths(tmp_path)
    sequence = _script_sequence()
    monkeypatch.setattr(front_script, "load_project_paths", lambda path: paths)
    monkeypatch.setattr(front_script, "load_yaml", lambda path: _cases_config())
    monkeypatch.setattr(
        front_script,
        "discover_nek_frame_paths",
        lambda *args, **kwargs: (_frame(1),),
    )
    monkeypatch.setattr(
        front_script,
        "default_front_detection_cache_path",
        lambda *args, **kwargs: tmp_path / "cache",
    )
    monkeypatch.setattr(
        front_script,
        "build_front_detection_cache_spec",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        front_script,
        "acquire_concentration_sequence",
        lambda **kwargs: SimpleNamespace(
            sequence=sequence,
            mode="cache_disabled",
            cache_dir=None,
        ),
    )


def test_single_frame_no_overlap_writes_automatic_outputs_and_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "no-overlap"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--output-dir",
            str(output_dir),
            "--min-component-pixels",
            "1",
            "--bottom-rows",
            "1",
            "--no-cache",
            "--diagnostic-indices",
            "1",
            "--overwrite",
        ],
    )
    _install_script_run_fakes(tmp_path, monkeypatch)
    monkeypatch.setattr(
        front_script,
        "read_front_simple_dat",
        lambda path: {
            "time": np.array([2.0, 3.0]),
            "x_front": np.array([1.0, 2.0]),
        },
    )

    front_script.main()

    assert (output_dir / "N7_detected_front_timeseries.csv").is_file()
    assert (output_dir / "N7_front_detection_summary.csv").is_file()
    comparison_path = output_dir / "N7_front_detection_comparison.csv"
    assert len(list(csv.reader(comparison_path.open(encoding="utf-8")))) == 1
    assert (
        output_dir / "diagnostics/N7_front_diagnostic_f00001.png"
    ).is_file()
    assert not (output_dir / "N7_front_detection_overlay.png").exists()
    assert not (output_dir / "N7_front_detection_difference.png").exists()
    summary_rows = list(
        csv.DictReader(
            (output_dir / "N7_front_detection_summary.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert summary_rows[0]["comparison_status"] == "no_time_overlap"
    assert summary_rows[0]["n_comparison_points"] == "0"
    assert "no overlapping reference times" in capsys.readouterr().out


def test_disabled_reference_ignores_missing_path_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "disabled"
    missing_reference = tmp_path / "does-not-exist.dat"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--output-dir",
            str(output_dir),
            "--reference-file",
            str(missing_reference),
            "--no-reference-comparison",
            "--min-component-pixels",
            "1",
            "--bottom-rows",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )
    _install_script_run_fakes(tmp_path, monkeypatch)
    monkeypatch.setattr(
        front_script,
        "read_front_simple_dat",
        lambda path: pytest.fail("disabled comparison read the reference"),
    )

    front_script.main()

    summary = next(
        csv.DictReader(
            (output_dir / "N7_front_detection_summary.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert summary["comparison_status"] == "disabled"
    assert summary["reference_role"] == "not_requested"
    assert summary["reference_file"] == ""
    comparison_path = output_dir / "N7_front_detection_comparison.csv"
    assert len(list(csv.reader(comparison_path.open(encoding="utf-8")))) == 1
    assert not (output_dir / "N7_front_detection_overlay.png").exists()
    assert not (output_dir / "N7_front_detection_difference.png").exists()


def test_enabled_missing_reference_still_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "missing-enabled"
    missing_reference = tmp_path / "missing.dat"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--output-dir",
            str(output_dir),
            "--reference-file",
            str(missing_reference),
            "--min-component-pixels",
            "1",
            "--bottom-rows",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )
    _install_script_run_fakes(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as error:
        front_script.main()

    assert error.value.code == 1
    assert str(missing_reference) in capsys.readouterr().err


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_workers_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="workers"):
        validate_preprocessing_workers(value)


@pytest.mark.parametrize("value", [1, 2, 4, np.int64(3)])
def test_workers_validation_accepts_positive_integers(value: object) -> None:
    assert validate_preprocessing_workers(value) == int(value)


def test_worker_returns_only_reduced_immutable_frame_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(target_shape=(2, 3))
    concentration = np.asarray(
        [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        parallel_module,
        "read_nek_file",
        lambda path: {"time": 0.5},
    )
    monkeypatch.setattr(
        parallel_module,
        "get_nek_time",
        lambda data: data["time"],
    )
    monkeypatch.setattr(
        parallel_module,
        "extract_y_slice",
        lambda data, **kwargs: {
            "x": np.asarray([0.0, 1.0, 0.0]),
            "z": np.asarray([0.0, 0.0, 1.0]),
            "C": np.asarray([1.0, 2.0, 3.0]),
            "selected_y": 0.25,
        },
    )
    monkeypatch.setattr(
        parallel_module,
        "apply_fixed_grid_interpolation_plan",
        lambda received_plan, x, z, values: concentration.copy(),
    )

    result = preprocess_front_detection_frame(
        _frame(2),
        plan,
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
    )

    assert {field.name for field in fields(result)} == {
        "file_index",
        "source_path",
        "time",
        "concentration",
        "finite_fraction",
        "concentration_min",
        "concentration_max",
        "selected_y",
    }
    assert result.file_index == 2
    assert result.source_path == Path("GC0.f00002")
    assert result.time == 0.5
    assert result.finite_fraction == 5.0 / 6.0
    assert result.concentration_min == 1.0
    assert result.concentration_max == 6.0
    assert result.selected_y == 0.25
    assert result.concentration.dtype == np.dtype("float64")
    assert not result.concentration.flags.writeable


def test_spectral_worker_never_extracts_a_scattered_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(target_shape=(2, 3), y_target=0.75)
    concentration = np.arange(6, dtype=np.float64).reshape(2, 3)
    monkeypatch.setattr(
        parallel_module,
        "read_nek_file",
        lambda path: {"time": 0.5},
    )
    monkeypatch.setattr(
        parallel_module,
        "get_nek_time",
        lambda data: data["time"],
    )
    monkeypatch.setattr(
        parallel_module,
        "extract_y_slice",
        lambda *args, **kwargs: pytest.fail("spectral worker extracted a slice"),
    )
    monkeypatch.setattr(
        parallel_module,
        "apply_spectral_slice_interpolation_plan",
        lambda received_plan, data, **kwargs: concentration.copy(),
    )

    result = preprocess_front_detection_frame(
        _frame(2),
        plan,
        interpolation_engine="spectral_element",
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
    )

    np.testing.assert_array_equal(result.concentration, concentration)
    assert result.selected_y == 0.75


class _FakeExecutor:
    instances: list[_FakeExecutor] = []
    results: dict[int, FrontDetectionFrameResult] = {}
    errors: dict[int, Exception] = {}

    def __init__(
        self,
        *,
        max_workers: int,
        initializer: object,
        initargs: tuple[object, ...],
    ) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        self.submitted: list[int] = []
        type(self).instances.append(self)

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: object, frame: NekFramePath) -> Future:
        self.submitted.append(frame.index)
        future: Future[FrontDetectionFrameResult] = Future()
        error = type(self).errors.get(frame.index)
        if error is None:
            future.set_result(type(self).results[frame.index])
        else:
            future.set_exception(error)
        return future


def _install_fake_executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: dict[int, FrontDetectionFrameResult],
    errors: dict[int, Exception] | None = None,
) -> None:
    _FakeExecutor.instances = []
    _FakeExecutor.results = results
    _FakeExecutor.errors = errors or {}
    monkeypatch.setattr(
        parallel_module,
        "ProcessPoolExecutor",
        _FakeExecutor,
    )


def test_parallel_executor_submits_later_frames_once_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_executor(
        monkeypatch,
        results={index: _result(index) for index in (2, 3, 4, 5, 6)},
    )
    plan = object()
    requested_completion_order = [2, 6, 4, 3, 5]
    observed_completion_order: list[int] = []

    def wait_out_of_order(futures, *, return_when):
        by_index = {
            future.result().file_index: future
            for future in futures
        }
        index = next(
            candidate
            for candidate in requested_completion_order
            if candidate in by_index
            and candidate not in observed_completion_order
        )
        observed_completion_order.append(index)
        future = by_index[index]
        return {future}, set(futures) - {future}

    monkeypatch.setattr(parallel_module, "wait", wait_out_of_order)

    results = preprocess_front_detection_frames_parallel(
        [_frame(index) for index in (6, 3, 5, 2, 4)],
        plan,
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
        workers=2,
    )

    executor = _FakeExecutor.instances[0]
    assert executor.max_workers == 2
    assert executor.initializer is parallel_module._initialize_worker
    assert executor.initargs[0] is plan
    assert sorted(executor.submitted) == [2, 3, 4, 5, 6]
    assert len(executor.submitted) == len(set(executor.submitted))
    assert observed_completion_order == requested_completion_order
    assert tuple(result.file_index for result in results) == (2, 3, 4, 5, 6)


def test_parallel_worker_exception_includes_index_path_and_original_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_executor(
        monkeypatch,
        results={},
        errors={2: ValueError("synthetic geometry failure")},
    )

    with pytest.raises(
        ParallelFramePreprocessingError,
        match=r"index 2.*GC0\.f00002.*synthetic geometry failure",
    ):
        preprocess_front_detection_frames_parallel(
            [_frame(2)],
            object(),
            slice_mode="nearest_plane",
            slab_ratio=0.01,
            y_round_decimals=10,
            workers=2,
        )


def test_failed_parallel_builder_does_not_create_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    def builder() -> object:
        raise ParallelFramePreprocessingError("frame failed")

    with pytest.raises(ParallelFramePreprocessingError, match="frame failed"):
        acquire_concentration_sequence(
            cache_dir=cache_dir,
            spec=object(),
            builder=builder,
            use_cache=True,
            rebuild_cache=False,
        )

    assert not cache_dir.exists()


def test_cache_hit_never_calls_builder_or_creates_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sentinel = object()
    monkeypatch.setattr(
        cache_module,
        "load_concentration_sequence_cache",
        lambda cache_path, spec: sentinel,
    )

    def fail_executor(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache hit created a process executor")

    monkeypatch.setattr(
        parallel_module,
        "ProcessPoolExecutor",
        fail_executor,
    )
    builder_calls = 0

    def builder() -> object:
        nonlocal builder_calls
        builder_calls += 1
        return object()

    acquisition = acquire_concentration_sequence(
        cache_dir=cache_dir,
        spec=object(),
        builder=builder,
        use_cache=True,
        rebuild_cache=False,
    )

    assert acquisition.mode == "cache_hit"
    assert acquisition.sequence is sentinel
    assert builder_calls == 0


def test_worker_count_is_absent_from_cache_specification() -> None:
    assert "workers" not in {field.name for field in fields(FrontDetectionCacheSpec)}
