from __future__ import annotations

from concurrent.futures import Future
from dataclasses import fields, replace
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import nek_post.leading_edge_parallel as parallel_module
from nek_post.front_detection_io import NekFramePath
from nek_post.leading_edge_parallel import (
    LeadingEdgeFrameResult,
    ParallelLeadingEdgeFrameError,
    process_leading_edge_frame,
    process_leading_edge_frames_parallel,
    validate_leading_edge_workers,
)
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


_METHOD_CONTEXT = {
    "extraction_method": "rightmost-crossing",
    "periodic_y": True,
    "y_period": 1.0,
}


def _frame(index: int) -> NekFramePath:
    return NekFramePath(index=index, path=Path(f"GC0.f{index:05d}"))


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _result(
    index: int,
    *,
    returned_index: int | None = None,
    returned_path: Path | None = None,
) -> LeadingEdgeFrameResult:
    result_index = index if returned_index is None else returned_index
    return LeadingEdgeFrameResult(
        file_index=result_index,
        source_path=_frame(index).path if returned_path is None else returned_path,
        time=index / 4.0,
        x_front=_readonly([0.25, np.nan, 0.75], np.float64),
        success_mask=_readonly([True, False, True], np.bool_),
        crossing_count=_readonly([1, 0, 2], np.int64),
        finite_leading_edge_fraction=2.0 / 3.0,
        successful_y_count=2,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_workers_validation_accepts_one_and_two(workers: int) -> None:
    assert validate_leading_edge_workers(workers) == workers


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, np.bool_(False)])
def test_workers_validation_rejects_invalid_values(workers: object) -> None:
    with pytest.raises(ValueError, match="workers"):
        validate_leading_edge_workers(workers)  # type: ignore[arg-type]


def test_empty_parallel_sequence_never_creates_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parallel_module,
        "ProcessPoolExecutor",
        lambda **_kwargs: pytest.fail("empty sequence created an executor"),
    )

    assert process_leading_edge_frames_parallel(
        [],
        object(),
        np.array([0.0, 1.0]),
        np.array([0.0]),
        0.1,
        **_METHOD_CONTEXT,
        workers=2,
    ) == ()


def test_single_frame_helper_returns_only_reduced_read_only_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.5, 0.75])
    plane = np.array(
        [
            [-0.1, 0.3, -0.1],
            [-1.0, -1.0, -1.0],
            [-0.1, 0.1, 0.3],
        ]
    )
    plane_reference = weakref.ref(plane)
    monkeypatch.setattr(
        parallel_module,
        "apply_spectral_horizontal_slice_plan",
        lambda *_args, **_kwargs: plane,
    )

    result = process_leading_edge_frame(
        _frame(3),
        object(),  # type: ignore[arg-type]
        x,
        y,
        0.1,
        **_METHOD_CONTEXT,
        frame_reader=lambda _path: SimpleNamespace(time=0.75),
    )
    del plane
    gc.collect()

    assert result.file_index == 3
    assert result.source_path == Path("GC0.f00003")
    assert result.time == 0.75
    assert result.x_front.shape == (3,)
    assert result.success_mask.shape == (3,)
    assert result.crossing_count.shape == (3,)
    assert_allclose(result.x_front[[0, 2]], [1.5, 1.0])
    assert np.isnan(result.x_front[1])
    assert_array_equal(result.success_mask, [True, False, True])
    assert_array_equal(result.crossing_count, [2, 0, 1])
    assert result.finite_leading_edge_fraction == pytest.approx(2.0 / 3.0)
    assert result.successful_y_count == 2
    assert not result.x_front.flags.writeable
    assert not result.success_mask.flags.writeable
    assert not result.crossing_count.flags.writeable
    assert "concentration" not in {field.name for field in fields(result)}
    assert not hasattr(result, "interpolation_plan")
    assert not hasattr(result, "Xi")
    assert not hasattr(result, "Yi")
    assert plane_reference() is None


def test_single_frame_helper_dispatches_moore_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.arange(5, dtype=float)
    y = np.arange(4, dtype=float) / 4.0
    plane = np.broadcast_to(
        np.where(x[None, :] < 2.0, 1.0, 0.0),
        (y.size, x.size),
    ).copy()
    monkeypatch.setattr(
        parallel_module,
        "apply_spectral_horizontal_slice_plan",
        lambda *_args, **_kwargs: plane,
    )

    result = process_leading_edge_frame(
        _frame(3),
        object(),  # type: ignore[arg-type]
        x,
        y,
        0.5,
        extraction_method="moore-boundary",
        periodic_y=True,
        y_period=1.0,
        frame_reader=lambda _path: SimpleNamespace(time=0.75),
    )

    assert result.extraction_method == "moore-boundary"
    assert_array_equal(result.x_front, np.full(y.size, 2.0))
    assert_array_equal(result.success_mask, np.ones(y.size, dtype=bool))
    assert_array_equal(result.crossing_count, np.ones(y.size, dtype=np.int64))


def test_single_frame_read_failure_has_source_path_context() -> None:
    with pytest.raises(RuntimeError, match=r"GC0\.f00004.*reader failed"):
        process_leading_edge_frame(
            _frame(4),
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            frame_reader=lambda _path: (_ for _ in ()).throw(OSError("reader failed")),
        )


def test_single_frame_geometry_mismatch_preserves_type_and_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> np.ndarray:
        raise SpectralGeometryMismatchError("coordinates changed")

    monkeypatch.setattr(
        parallel_module,
        "apply_spectral_horizontal_slice_plan",
        fail,
    )
    with pytest.raises(
        SpectralGeometryMismatchError,
        match=r"GC0\.f00005.*coordinates changed",
    ):
        process_leading_edge_frame(
            _frame(5),
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            frame_reader=lambda _path: SimpleNamespace(time=1.0),
        )


def test_single_frame_extraction_failure_has_source_path_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parallel_module,
        "apply_spectral_horizontal_slice_plan",
        lambda *_args, **_kwargs: np.zeros((1, 2)),
    )
    monkeypatch.setattr(
        parallel_module,
        "extract_leading_edge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad contour")),
    )
    with pytest.raises(RuntimeError, match=r"GC0\.f00006.*bad contour"):
        process_leading_edge_frame(
            _frame(6),
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            frame_reader=lambda _path: SimpleNamespace(time=1.25),
        )


class _TrackingFuture(Future[LeadingEdgeFrameResult]):
    def __init__(self, frame_index: int) -> None:
        super().__init__()
        self.frame_index = frame_index
        self.cancel_called = False

    def cancel(self) -> bool:
        self.cancel_called = True
        return super().cancel()


class _FakeExecutor:
    instances: list[_FakeExecutor] = []
    results: dict[int, LeadingEdgeFrameResult] = {}
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
        self.submissions: list[tuple[object, tuple[object, ...]]] = []
        self.futures: dict[int, _TrackingFuture] = {}
        self.initializer_calls = 0
        type(self).instances.append(self)
        assert callable(initializer)
        for _ in range(max_workers):
            initializer(*initargs)
            self.initializer_calls += 1

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function: object, *args: object) -> _TrackingFuture:
        assert len(args) == 1
        frame = args[0]
        assert isinstance(frame, NekFramePath)
        self.submissions.append((function, args))
        future = _TrackingFuture(frame.index)
        error = type(self).errors.get(frame.index)
        if error is None:
            future.set_result(type(self).results[frame.index])
        else:
            future.set_exception(error)
        self.futures[frame.index] = future
        return future


def _install_executor(
    monkeypatch: pytest.MonkeyPatch,
    results: dict[int, LeadingEdgeFrameResult],
    *,
    errors: dict[int, Exception] | None = None,
    completion_order: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    _FakeExecutor.instances = []
    _FakeExecutor.results = results
    _FakeExecutor.errors = {} if errors is None else errors
    observed_pending: list[int] = []
    observed_completion: list[int] = []
    requested = list(results) if completion_order is None else completion_order.copy()
    if errors:
        requested = list(errors) + [index for index in requested if index not in errors]

    def fake_wait(futures: tuple[_TrackingFuture, ...], *, return_when: object):
        assert return_when is parallel_module.FIRST_COMPLETED
        observed_pending.append(len(futures))
        selected_index = next(
            index
            for index in requested
            if index not in observed_completion
            and any(future.frame_index == index for future in futures)
        )
        observed_completion.append(selected_index)
        selected = next(
            future for future in futures if future.frame_index == selected_index
        )
        return {selected}, set(futures) - {selected}

    monkeypatch.setattr(parallel_module, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(parallel_module, "wait", fake_wait)
    return observed_pending, observed_completion


def test_initializer_context_bounded_submissions_and_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tuple(_frame(index) for index in (6, 3, 5, 2, 4, 7))
    results = {frame.index: _result(frame.index) for frame in frames}
    pending_counts, completion = _install_executor(
        monkeypatch,
        results,
        completion_order=[5, 6, 2, 7, 3, 4],
    )
    plan = object()
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.5])

    returned = process_leading_edge_frames_parallel(
        frames,
        plan,  # type: ignore[arg-type]
        x,
        y,
        0.1,
        **_METHOD_CONTEXT,
        workers=2,
    )

    executor = _FakeExecutor.instances[0]
    assert executor.max_workers == 2
    assert executor.initializer is parallel_module._initialize_worker
    assert executor.initargs == (
        plan,
        x,
        y,
        0.1,
        "rightmost-crossing",
        True,
        1.0,
    )
    assert executor.initializer_calls == 2
    assert all(
        function is parallel_module._process_worker_frame and args == (frame,)
        for (function, args), frame in zip(executor.submissions, frames, strict=True)
    )
    assert max(pending_counts) == 4
    assert all(count <= 2 * executor.max_workers for count in pending_counts)
    assert completion == [5, 6, 2, 7, 3, 4]
    assert tuple(result.file_index for result in returned) == (2, 3, 4, 5, 6, 7)
    for result in returned:
        assert not result.x_front.flags.writeable
        assert not result.success_mask.flags.writeable
        assert not result.crossing_count.flags.writeable


def test_worker_failure_cancels_pending_and_includes_frame_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_executor(
        monkeypatch,
        {index: _result(index) for index in (2, 3, 4)},
        errors={2: ValueError("synthetic worker failure")},
    )

    with pytest.raises(
        ParallelLeadingEdgeFrameError,
        match=r"index 2.*GC0\.f00002.*synthetic worker failure",
    ):
        process_leading_edge_frames_parallel(
            [_frame(2), _frame(3), _frame(4)],
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            workers=2,
        )

    futures = _FakeExecutor.instances[0].futures
    assert futures[3].cancel_called
    assert futures[4].cancel_called


def test_duplicate_returned_indices_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_executor(
        monkeypatch,
        {2: _result(2), 3: _result(3, returned_index=2)},
        completion_order=[2, 3],
    )

    with pytest.raises(ParallelLeadingEdgeFrameError, match="duplicate file index 2"):
        process_leading_edge_frames_parallel(
            [_frame(2), _frame(3)],
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            workers=2,
        )


def test_returned_frame_identity_must_match_submitted_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_path = Path("wrong/GC0.f00002")
    _install_executor(
        monkeypatch,
        {2: _result(2, returned_path=wrong_path)},
    )

    with pytest.raises(
        ParallelLeadingEdgeFrameError,
        match=r"index 2.*GC0\.f00002.*identity mismatch.*wrong/GC0\.f00002",
    ):
        process_leading_edge_frames_parallel(
            [_frame(2)],
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            workers=1,
        )


def test_returned_extraction_method_must_match_requested_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = replace(_result(2), extraction_method="future-method")
    _install_executor(monkeypatch, {2: mismatched})

    with pytest.raises(
        ParallelLeadingEdgeFrameError,
        match=r"index 2.*extraction-method mismatch.*future-method",
    ):
        process_leading_edge_frames_parallel(
            [_frame(2)],
            object(),  # type: ignore[arg-type]
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.1,
            **_METHOD_CONTEXT,
            workers=1,
        )
