from __future__ import annotations

import gc
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.front_detection_io import NekFramePath
from nek_post.leading_edge_parallel import LeadingEdgeFrameResult
from nek_post.leading_edge_workflow import (
    LeadingEdgeEvolution,
    build_leading_edge_evolution,
    select_leading_edge_times,
)
from nek_post.spectral_interpolation import (
    SpectralGeometryMismatchError,
    SpectralInverseMappingDiagnostics,
)


def _fake_plan() -> SimpleNamespace:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.25, 0.5, 0.75])
    Xi, Yi = np.meshgrid(x, y)
    diagnostics = SpectralInverseMappingDiagnostics(
        target_point_count=12,
        candidate_pair_count=14,
        inverse_attempt_count=14,
        inverse_success_count=12,
        inverse_failure_count=2,
        ambiguous_boundary_point_count=1,
        maximum_successful_residual=2.0e-13,
        maximum_iteration_count=5,
        spatial_bin_shape=(3, 4),
    )
    return SimpleNamespace(
        target_shape=(4, 3),
        Xi=Xi,
        Yi=Yi,
        z_target=0.04,
        native_ny=2,
        dense_ny=4,
        y_upsample_factor=2,
        inverse_mapping_diagnostics=diagnostics,
    )


def _plan_metadata(_plan: object) -> MappingProxyType:
    return MappingProxyType(
        {
            "xmin": 0.0,
            "xmax": 2.0,
            "ymin": 0.0,
            "ymax_periodic_endpoint": 1.0,
            "nx": 3,
            "native_ny": 2,
            "dense_ny": 4,
            "y_upsample_factor": 2,
            "z_target": 0.04,
            "spectral_horizontal_algorithm_version": 1,
        }
    )


def _run_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: tuple[NekFramePath, ...] | None = None,
    times: dict[str, float] | None = None,
) -> tuple[LeadingEdgeEvolution, dict[str, object]]:
    import nek_post.leading_edge_parallel as parallel
    import nek_post.leading_edge_workflow as workflow

    if frames is None:
        frames = (
            NekFramePath(30, Path("GC0.f00030")),
            NekFramePath(10, Path("GC0.f00010")),
            NekFramePath(20, Path("GC0.f00020")),
        )
    if times is None:
        times = {
            "GC0.f00010": 1.0,
            "GC0.f00020": 1.4,
            "GC0.f00030": 1.9,
        }
    fronts = {
        "GC0.f00010": np.array([0.25, 0.50, 0.75, np.nan]),
        "GC0.f00020": np.array([0.50, 0.75, 1.00, 1.25]),
        "GC0.f00030": np.array([0.75, 1.00, 1.25, 1.50]),
    }
    plan = _fake_plan()
    calls: dict[str, object] = {
        "build": [],
        "apply_plan_ids": [],
        "apply_paths": [],
        "read": [],
        "plane_refs": [],
    }

    def reader(path: Path) -> SimpleNamespace:
        cast_reads = calls["read"]
        assert isinstance(cast_reads, list)
        cast_reads.append(path)
        return SimpleNamespace(time=times[path.name], source=path.name)

    def build(data: object, **kwargs: object) -> SimpleNamespace:
        cast_build = calls["build"]
        assert isinstance(cast_build, list)
        cast_build.append((data, kwargs))
        return plan

    def apply(
        supplied_plan: object,
        data: SimpleNamespace,
        *,
        source_file: Path,
    ) -> np.ndarray:
        cast_ids = calls["apply_plan_ids"]
        cast_paths = calls["apply_paths"]
        assert isinstance(cast_ids, list) and isinstance(cast_paths, list)
        cast_ids.append(id(supplied_plan))
        cast_paths.append(source_file)
        cast_refs = calls["plane_refs"]
        assert isinstance(cast_refs, list)
        if cast_refs:
            gc.collect()
            assert cast_refs[-1]() is None
        x = plan.Xi[0, :]
        front = fronts[data.source]
        plane = x[None, :] - front[:, None] + 0.1
        plane[np.isnan(front), :] = -1.0
        cast_refs.append(weakref.ref(plane))
        return plane

    monkeypatch.setattr(workflow, "build_spectral_horizontal_slice_plan", build)
    monkeypatch.setattr(parallel, "apply_spectral_horizontal_slice_plan", apply)
    monkeypatch.setattr(workflow, "spectral_horizontal_plan_metadata", _plan_metadata)
    evolution = build_leading_edge_evolution(
        frames,
        nx=3,
        z_target=0.04,
        threshold=0.1,
        y_upsample_factor=2,
        _frame_reader=reader,
    )
    calls["plan"] = plan
    return evolution, calls


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _evolution_for_times(
    times: list[float],
    *,
    file_indices: list[int] | None = None,
) -> LeadingEdgeEvolution:
    n_frames = len(times)
    if file_indices is None:
        file_indices = list(range(10, 10 + n_frames))
    y = _readonly([0.0, 0.5], np.float64)
    x_front = np.arange(n_frames * 2, dtype=float).reshape(n_frames, 2)
    metadata = MappingProxyType(
        {
            "ymin": 0.0,
            "ymax_periodic_endpoint": 1.0,
            "spectral_horizontal_algorithm_version": 1,
            "inverse_mapping_target_count": 6,
            "inverse_mapping_success_count": 6,
            "inverse_mapping_failure_count": 0,
            "ambiguous_boundary_point_count": 0,
            "maximum_successful_residual": 1.0e-14,
            "maximum_iteration_count": 4,
        }
    )
    return LeadingEdgeEvolution(
        file_indices=_readonly(file_indices, np.int64),
        source_files=tuple(Path(f"GC0.f{index:05d}") for index in file_indices),
        time=_readonly(times, np.float64),
        x=_readonly([0.0, 1.0, 2.0], np.float64),
        y=y,
        x_front=_readonly(x_front, np.float64),
        success_mask=_readonly(np.ones_like(x_front, dtype=bool), np.bool_),
        crossing_count=_readonly(np.ones_like(x_front, dtype=int), np.int64),
        finite_leading_edge_fraction=_readonly(np.ones(n_frames), np.float64),
        successful_y_count=_readonly(np.full(n_frames, 2), np.int64),
        threshold=0.1,
        z_target=0.04,
        nx=3,
        native_ny=1,
        dense_ny=2,
        y_upsample_factor=2,
        horizontal_plan_metadata=metadata,
        periodic_endpoint_included=False,
    )


def test_empty_duplicate_and_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        build_leading_edge_evolution([], nx=3, z_target=0.04)
    with pytest.raises(ValueError, match="unique file indices"):
        build_leading_edge_evolution(
            [NekFramePath(1, Path("a")), NekFramePath(1, Path("b"))],
            nx=3,
            z_target=0.04,
        )
    with pytest.raises(ValueError, match="unique source paths"):
        build_leading_edge_evolution(
            [NekFramePath(1, Path("a")), NekFramePath(2, Path("a"))],
            nx=3,
            z_target=0.04,
        )
    for kwargs in (
        {"nx": 1, "z_target": 0.04},
        {"nx": 3, "z_target": np.nan},
        {"nx": 3, "z_target": 0.04, "threshold": np.inf},
        {"nx": 3, "z_target": 0.04, "y_upsample_factor": 0},
    ):
        with pytest.raises(ValueError):
            build_leading_edge_evolution(
                [NekFramePath(1, Path("a"))],
                **kwargs,
            )


def test_frames_sort_plan_builds_once_and_exact_plan_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, calls = _run_synthetic(monkeypatch)

    assert_array_equal(evolution.file_indices, [10, 20, 30])
    assert calls["read"] == [
        Path("GC0.f00010"),
        Path("GC0.f00020"),
        Path("GC0.f00030"),
    ]
    assert len(calls["build"]) == 1  # type: ignore[arg-type]
    assert calls["apply_plan_ids"] == [id(calls["plan"])] * 3
    assert calls["apply_paths"] == calls["read"]
    build_kwargs = calls["build"][0][1]  # type: ignore[index]
    assert build_kwargs == {"nx": 3, "z_target": 0.04, "y_upsample_factor": 2}


def test_extraction_shapes_values_shared_coordinates_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, _calls = _run_synthetic(monkeypatch)

    assert evolution.x_front.shape == (3, 4)
    assert evolution.success_mask.shape == (3, 4)
    assert evolution.crossing_count.shape == (3, 4)
    assert_allclose(evolution.x_front[0, :3], [0.25, 0.5, 0.75])
    assert np.isnan(evolution.x_front[0, 3])
    assert_array_equal(evolution.success_mask[0], [True, True, True, False])
    assert_array_equal(evolution.crossing_count[0], [1, 1, 1, 0])
    assert_allclose(evolution.finite_leading_edge_fraction, [0.75, 1.0, 1.0])
    assert_array_equal(evolution.successful_y_count, [3, 4, 4])
    assert_array_equal(evolution.x, [0.0, 1.0, 2.0])
    assert_array_equal(evolution.y, [0.0, 0.25, 0.5, 0.75])
    assert evolution.threshold == 0.1
    assert evolution.z_target == 0.04
    assert evolution.nx == 3
    assert evolution.native_ny == 2
    assert evolution.dense_ny == 4
    assert evolution.y_upsample_factor == 2
    assert not evolution.periodic_endpoint_included
    assert evolution.horizontal_plan_metadata["inverse_mapping_failure_count"] == 2


def test_all_result_arrays_are_copies_with_expected_dtypes_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, calls = _run_synthetic(monkeypatch)
    expected = {
        "file_indices": np.dtype(np.int64),
        "time": np.dtype(np.float64),
        "x": np.dtype(np.float64),
        "y": np.dtype(np.float64),
        "x_front": np.dtype(np.float64),
        "success_mask": np.dtype(np.bool_),
        "crossing_count": np.dtype(np.int64),
        "finite_leading_edge_fraction": np.dtype(np.float64),
        "successful_y_count": np.dtype(np.int64),
    }
    for name, dtype in expected.items():
        array = getattr(evolution, name)
        assert array.dtype == dtype, name
        assert not array.flags.writeable, name

    plan = calls["plan"]
    assert isinstance(plan, SimpleNamespace)
    plan.Xi[0, 0] = -99.0
    assert evolution.x[0] == 0.0


def test_full_concentration_planes_are_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, calls = _run_synthetic(monkeypatch)
    gc.collect()

    assert evolution.x_front.ndim == 2
    assert not hasattr(evolution, "concentration")
    plane_refs = calls["plane_refs"]
    assert isinstance(plane_refs, list)
    assert all(reference() is None for reference in plane_refs)


def test_stage_one_plan_and_stage_two_results_are_not_modified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.leading_edge_parallel as parallel
    import nek_post.leading_edge_workflow as workflow
    from nek_post.leading_edge_extraction import extract_spanwise_leading_edge

    curves: list[tuple[object, tuple[np.ndarray, ...]]] = []

    def recording_extract(*args: object, **kwargs: object):
        curve = extract_spanwise_leading_edge(*args, **kwargs)
        snapshot = tuple(
            getattr(curve, name).copy()
            for name in ("y", "x_front", "success_mask", "crossing_count")
        )
        curves.append((curve, snapshot))
        return curve

    monkeypatch.setattr(parallel, "extract_spanwise_leading_edge", recording_extract)
    evolution, calls = _run_synthetic(monkeypatch)
    plan = calls["plan"]
    assert isinstance(plan, SimpleNamespace)

    assert_array_equal(plan.Xi[0], [0.0, 1.0, 2.0])
    assert_array_equal(plan.Yi[:, 0], [0.0, 0.25, 0.5, 0.75])
    for curve, snapshot in curves:
        for name, expected in zip(
            ("y", "x_front", "success_mask", "crossing_count"),
            snapshot,
            strict=True,
        ):
            assert_array_equal(getattr(curve, name), expected)
    assert evolution.x_front is not curves[0][0].x_front  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "times",
    (
        {"GC0.f00010": np.nan, "GC0.f00020": 1.4, "GC0.f00030": 1.9},
        {"GC0.f00010": 1.0, "GC0.f00020": 1.0, "GC0.f00030": 1.9},
        {"GC0.f00010": 1.0, "GC0.f00020": 0.9, "GC0.f00030": 1.9},
    ),
)
def test_actual_times_must_be_finite_unique_and_increasing(
    monkeypatch: pytest.MonkeyPatch,
    times: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="non-finite|strictly increasing"):
        _run_synthetic(monkeypatch, times=times)


def test_read_failure_has_source_path_context() -> None:
    path = Path("broken/GC0.f00001")

    def reader(_path: Path) -> object:
        raise OSError("synthetic read error")

    with pytest.raises(RuntimeError, match=r"broken/GC0\.f00001.*synthetic read"):
        build_leading_edge_evolution(
            [NekFramePath(1, path)],
            nx=3,
            z_target=0.04,
            _frame_reader=reader,
        )


def test_geometry_mismatch_preserves_type_and_has_source_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nek_post.leading_edge_parallel as parallel
    import nek_post.leading_edge_workflow as workflow

    plan = _fake_plan()
    monkeypatch.setattr(
        workflow, "build_spectral_horizontal_slice_plan", lambda *_a, **_k: plan
    )
    monkeypatch.setattr(workflow, "spectral_horizontal_plan_metadata", _plan_metadata)

    def apply(_plan: object, _data: object, *, source_file: Path) -> np.ndarray:
        raise SpectralGeometryMismatchError("coordinate signature changed")

    monkeypatch.setattr(parallel, "apply_spectral_horizontal_slice_plan", apply)
    path = Path("N7/GC0.f00001")
    with pytest.raises(
        SpectralGeometryMismatchError,
        match=r"N7/GC0\.f00001.*coordinate signature changed",
    ):
        build_leading_edge_evolution(
            [NekFramePath(1, path)],
            nx=3,
            z_target=0.04,
            _frame_reader=lambda _path: SimpleNamespace(time=1.0),
        )


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_workers_argument_validation(workers: object) -> None:
    with pytest.raises(ValueError, match="workers"):
        build_leading_edge_evolution(
            [NekFramePath(1, Path("GC0.f00001"))],
            nx=3,
            z_target=0.04,
            workers=workers,  # type: ignore[arg-type]
        )


def _workflow_frame_result(
    frame: NekFramePath,
    *,
    time: float,
    offset: float = 0.0,
) -> LeadingEdgeFrameResult:
    x_front = _readonly(
        [0.25 + offset, 0.5 + offset, np.nan, 1.0 + offset],
        np.float64,
    )
    success = _readonly([True, True, False, True], np.bool_)
    return LeadingEdgeFrameResult(
        file_index=frame.index,
        source_path=frame.path,
        time=time,
        x_front=x_front,
        success_mask=success,
        crossing_count=_readonly([1, 1, 0, 2], np.int64),
        finite_leading_edge_fraction=0.75,
        successful_y_count=3,
    )


def _install_parallel_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[NekFramePath, ...], dict[int, LeadingEdgeFrameResult], dict[str, object]]:
    import nek_post.leading_edge_workflow as workflow

    frames = tuple(
        NekFramePath(index, Path(f"GC0.f{index:05d}"))
        for index in (10, 20, 30)
    )
    results = {
        frame.index: _workflow_frame_result(
            frame,
            time={10: 1.0, 20: 1.25, 30: 1.5}[frame.index],
            offset=frame.index / 100.0,
        )
        for frame in frames
    }
    plan = _fake_plan()
    calls: dict[str, object] = {
        "read": [],
        "build": [],
        "serial": [],
        "parallel": [],
        "plan": plan,
    }

    def read(path: Path) -> SimpleNamespace:
        calls["read"].append(path)  # type: ignore[union-attr]
        return SimpleNamespace(path=path, time=results[int(path.name[-5:])].time)

    def build(data: object, **kwargs: object) -> SimpleNamespace:
        calls["build"].append((data, kwargs))  # type: ignore[union-attr]
        return plan

    def serial(
        frame: NekFramePath,
        supplied_plan: object,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float,
        *,
        frame_reader: object,
    ) -> LeadingEdgeFrameResult:
        calls["serial"].append(  # type: ignore[union-attr]
            (frame, supplied_plan, x, y, threshold, frame_reader)
        )
        return results[frame.index]

    def parallel(
        later_frames: tuple[NekFramePath, ...],
        supplied_plan: object,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float,
        *,
        workers: int,
    ) -> tuple[LeadingEdgeFrameResult, ...]:
        calls["parallel"].append(  # type: ignore[union-attr]
            (later_frames, supplied_plan, x, y, threshold, workers)
        )
        return tuple(results[frame.index] for frame in reversed(later_frames))

    monkeypatch.setattr(workflow, "read_nek_file", read)
    monkeypatch.setattr(workflow, "build_spectral_horizontal_slice_plan", build)
    monkeypatch.setattr(workflow, "spectral_horizontal_plan_metadata", _plan_metadata)
    monkeypatch.setattr(workflow, "process_leading_edge_frame", serial)
    monkeypatch.setattr(workflow, "process_leading_edge_frames_parallel", parallel)
    return frames, results, calls


def test_workers_two_builds_plan_once_keeps_first_serial_and_only_dispatches_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _results, calls = _install_parallel_workflow_fakes(monkeypatch)

    evolution = build_leading_edge_evolution(
        tuple(reversed(frames)),
        nx=3,
        z_target=0.04,
        workers=2,
    )

    assert len(calls["build"]) == 1  # type: ignore[arg-type]
    serial_calls = calls["serial"]
    assert isinstance(serial_calls, list)
    assert [call[0] for call in serial_calls] == [frames[0]]
    parallel_calls = calls["parallel"]
    assert isinstance(parallel_calls, list) and len(parallel_calls) == 1
    assert parallel_calls[0][0] == frames[1:]
    assert parallel_calls[0][1] is calls["plan"]
    assert parallel_calls[0][-1] == 2
    assert_array_equal(evolution.file_indices, [10, 20, 30])
    assert_allclose(evolution.time, [1.0, 1.25, 1.5])


def test_workers_one_builds_plan_once_and_never_uses_parallel_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _results, calls = _install_parallel_workflow_fakes(monkeypatch)

    evolution = build_leading_edge_evolution(
        frames,
        nx=3,
        z_target=0.04,
        workers=1,
    )

    assert len(calls["build"]) == 1  # type: ignore[arg-type]
    assert [call[0] for call in calls["serial"]] == list(frames)  # type: ignore[index]
    assert calls["parallel"] == []
    assert_array_equal(evolution.file_indices, [10, 20, 30])


def test_one_frame_never_uses_parallel_helper_even_with_workers_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _results, calls = _install_parallel_workflow_fakes(monkeypatch)

    evolution = build_leading_edge_evolution(
        frames[:1],
        nx=3,
        z_target=0.04,
        workers=2,
    )

    assert evolution.file_indices.tolist() == [10]
    assert calls["parallel"] == []
    assert len(calls["build"]) == 1  # type: ignore[arg-type]


def test_custom_reader_is_supported_serially_and_rejected_for_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _results, calls = _install_parallel_workflow_fakes(monkeypatch)

    custom_reads: list[Path] = []

    def custom_reader(path: Path) -> SimpleNamespace:
        custom_reads.append(path)
        return SimpleNamespace(time=1.0)

    build_leading_edge_evolution(
        frames[:1],
        nx=3,
        z_target=0.04,
        workers=1,
        _frame_reader=custom_reader,
    )
    assert custom_reads == [frames[0].path]
    assert calls["parallel"] == []

    with pytest.raises(ValueError, match=r"custom _frame_reader.*workers=1"):
        build_leading_edge_evolution(
            frames,
            nx=3,
            z_target=0.04,
            workers=2,
            _frame_reader=custom_reader,
        )


@pytest.mark.parametrize("second_time", [1.0, 0.9])
def test_parallel_results_are_ordered_before_time_validation(
    monkeypatch: pytest.MonkeyPatch,
    second_time: float,
) -> None:
    frames, results, _calls = _install_parallel_workflow_fakes(monkeypatch)
    results[20] = _workflow_frame_result(frames[1], time=second_time)

    with pytest.raises(ValueError, match=r"strictly increasing.*f00010.*f00020"):
        build_leading_edge_evolution(
            frames,
            nx=3,
            z_target=0.04,
            workers=2,
        )


def test_workers_one_and_two_produce_identical_evolution_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _results, _calls = _install_parallel_workflow_fakes(monkeypatch)
    serial = build_leading_edge_evolution(
        frames,
        nx=3,
        z_target=0.04,
        workers=1,
    )
    parallel = build_leading_edge_evolution(
        tuple(reversed(frames)),
        nx=3,
        z_target=0.04,
        workers=2,
    )

    for name in (
        "file_indices",
        "time",
        "x",
        "y",
        "x_front",
        "success_mask",
        "crossing_count",
        "finite_leading_edge_fraction",
        "successful_y_count",
    ):
        serial_array = getattr(serial, name)
        parallel_array = getattr(parallel, name)
        assert_array_equal(serial_array, parallel_array)
        assert not serial_array.flags.writeable
        assert not parallel_array.flags.writeable
    assert serial.source_files == parallel.source_files
    assert serial.horizontal_plan_metadata == parallel.horizontal_plan_metadata
    for name in (
        "threshold",
        "z_target",
        "nx",
        "native_ny",
        "dense_ny",
        "y_upsample_factor",
        "periodic_endpoint_included",
    ):
        assert getattr(serial, name) == getattr(parallel, name)


def test_default_time_selection_targets_nearest_frames_and_preserves_values() -> None:
    evolution = _evolution_for_times([1.0, 1.1, 1.5, 1.9])
    original_x_front = evolution.x_front.copy()

    selection = select_leading_edge_times(evolution)

    assert_array_equal(selection.selected_positions, [0, 1, 2, 3])
    assert_allclose(selection.target_time, [1.0, 1.25, 1.5, 1.75])
    assert_allclose(selection.actual_time, [1.0, 1.1, 1.5, 1.9])
    assert_allclose(selection.time_error, [0.0, -0.15, 0.0, 0.15])
    assert_array_equal(selection.x_front, evolution.x_front)
    assert_array_equal(evolution.x_front, original_x_front)
    assert selection.evolution is evolution
    for name in (
        "selected_positions",
        "file_indices",
        "target_time",
        "actual_time",
        "time_error",
        "x_front",
        "success_mask",
        "crossing_count",
        "finite_leading_edge_fraction",
        "successful_y_count",
    ):
        assert not getattr(selection, name).flags.writeable, name


def test_time_selection_tie_uses_earlier_actual_time_then_lower_index() -> None:
    evolution = _evolution_for_times([0.0, 1.0, 2.0], file_indices=[8, 4, 9])

    selection = select_leading_edge_times(evolution, spacing=1.5)

    assert_array_equal(selection.selected_positions, [0, 1])
    assert_allclose(selection.target_time, [0.0, 1.5])
    assert_allclose(selection.actual_time, [0.0, 1.0])
    assert_array_equal(selection.file_indices, [8, 4])


def test_time_selection_deduplicates_snapshots_keeps_first_target_and_orders() -> None:
    evolution = _evolution_for_times([0.0, 0.05, 0.5])

    selection = select_leading_edge_times(evolution, spacing=0.1)

    assert_array_equal(selection.selected_positions, [0, 1, 2])
    assert_allclose(selection.target_time, [0.0, 0.1, 0.3])
    assert np.all(np.diff(selection.actual_time) > 0.0)
    assert np.all(np.diff(selection.file_indices) > 0)


def test_all_frame_selection_and_invalid_spacing() -> None:
    evolution = _evolution_for_times([0.2, 0.7, 1.4])

    selection = select_leading_edge_times(evolution, spacing=None)

    assert_array_equal(selection.selected_positions, [0, 1, 2])
    assert_array_equal(selection.target_time, evolution.time)
    assert_array_equal(selection.actual_time, evolution.time)
    assert_array_equal(selection.time_error, [0.0, 0.0, 0.0])
    assert selection.target_time_spacing is None
    for spacing in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="spacing"):
            select_leading_edge_times(evolution, spacing=spacing)
