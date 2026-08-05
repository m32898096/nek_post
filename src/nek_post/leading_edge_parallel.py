"""Process-based frame reduction for spanwise leading-edge evolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.front_detection_io import NekFramePath
from nek_post.io_nek import get_nek_time, read_nek_file
from nek_post.leading_edge_methods import (
    DEFAULT_LEADING_EDGE_METHOD,
    extract_leading_edge,
)
from nek_post.spectral_horizontal_slice import (
    SpectralHorizontalSlicePlan,
    apply_spectral_horizontal_slice_plan,
)
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


class ParallelLeadingEdgeFrameError(RuntimeError):
    """Raised when a leading-edge process-pool frame task fails."""


@dataclass(frozen=True)
class LeadingEdgeFrameResult:
    """Reduced one-dimensional result for one Nek snapshot."""

    file_index: int
    source_path: Path
    time: float
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    finite_leading_edge_fraction: float
    successful_y_count: int
    extraction_method: str = DEFAULT_LEADING_EDGE_METHOD


@dataclass(frozen=True)
class _WorkerContext:
    interpolation_plan: SpectralHorizontalSlicePlan
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    threshold: float
    extraction_method: str
    periodic_y: bool
    y_period: float


_WORKER_CONTEXT: _WorkerContext | None = None


def _readonly_copy(array: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def validate_leading_edge_workers(value: int) -> int:
    """Validate and normalize the requested leading-edge process count."""
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError("workers must be an integer greater than or equal to 1.")
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be greater than or equal to 1.")
    return workers


def _finite_frame_time(data: object, source_path: Path) -> float:
    try:
        raw_time = get_nek_time(data)
    except Exception as exc:
        raise ValueError(
            f"Failed to read Nek time from {source_path}: {exc}"
        ) from exc
    try:
        frame_time = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path} has non-finite Nek time {raw_time!r}."
        ) from exc
    if not np.isfinite(frame_time):
        raise ValueError(
            f"{source_path} has non-finite Nek time {raw_time!r}."
        )
    return frame_time


def _reduce_leading_edge_frame(
    frame: NekFramePath,
    data: object,
    interpolation_plan: SpectralHorizontalSlicePlan,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    threshold: float,
    extraction_method: str,
    periodic_y: bool,
    y_period: float,
) -> LeadingEdgeFrameResult:
    """Interpolate and reduce already-read data using the common frame path."""
    source_path = Path(frame.path)
    frame_time = _finite_frame_time(data, source_path)
    try:
        concentration = apply_spectral_horizontal_slice_plan(
            interpolation_plan,
            data,
            source_file=source_path,
        )
    except SpectralGeometryMismatchError as exc:
        raise SpectralGeometryMismatchError(
            f"Source geometry mismatch for {source_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to interpolate horizontal concentration from "
            f"{source_path}: {exc}"
        ) from exc

    try:
        curve = extract_leading_edge(
            x,
            y,
            concentration,
            threshold=threshold,
            method=extraction_method,
            periodic_y=periodic_y,
            y_period=y_period,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract leading edge from {source_path}: {exc}"
        ) from exc

    if not np.array_equal(curve.y, y):
        raise ValueError(
            f"Leading-edge y coordinates from {source_path} do not match the "
            "horizontal interpolation plan."
        )
    x_front = _readonly_copy(curve.x_front, np.float64)
    success_mask = _readonly_copy(curve.success_mask, np.bool_)
    crossing_count = _readonly_copy(curve.crossing_count, np.int64)
    finite_count = int(np.count_nonzero(np.isfinite(x_front)))
    finite_fraction = float(finite_count / y.size)
    successful_y_count = int(np.count_nonzero(success_mask))
    result_method = curve.method

    # No raw data, two-dimensional plane, or extraction object survives return.
    del concentration, curve, data
    return LeadingEdgeFrameResult(
        file_index=int(frame.index),
        source_path=source_path,
        time=frame_time,
        extraction_method=result_method,
        x_front=x_front,
        success_mask=success_mask,
        crossing_count=crossing_count,
        finite_leading_edge_fraction=finite_fraction,
        successful_y_count=successful_y_count,
    )


def process_leading_edge_frame(
    frame: NekFramePath,
    interpolation_plan: SpectralHorizontalSlicePlan,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    threshold: float,
    *,
    extraction_method: str,
    periodic_y: bool,
    y_period: float,
    frame_reader: Callable[[Path], Any] = read_nek_file,
) -> LeadingEdgeFrameResult:
    """Read and reduce one frame without retaining its two-dimensional plane."""
    source_path = Path(frame.path)
    try:
        data = frame_reader(source_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Nek frame {source_path}: {exc}"
        ) from exc
    return _reduce_leading_edge_frame(
        frame,
        data,
        interpolation_plan,
        x,
        y,
        threshold,
        extraction_method,
        periodic_y,
        y_period,
    )


def _initialize_worker(
    interpolation_plan: SpectralHorizontalSlicePlan,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    threshold: float,
    extraction_method: str,
    periodic_y: bool,
    y_period: float,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(
        interpolation_plan=interpolation_plan,
        x=_readonly_copy(x, np.float64),
        y=_readonly_copy(y, np.float64),
        threshold=float(threshold),
        extraction_method=extraction_method,
        periodic_y=periodic_y,
        y_period=float(y_period),
    )


def _process_worker_frame(frame: NekFramePath) -> LeadingEdgeFrameResult:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Parallel leading-edge worker was not initialized.")
    return process_leading_edge_frame(
        frame,
        context.interpolation_plan,
        context.x,
        context.y,
        context.threshold,
        extraction_method=context.extraction_method,
        periodic_y=context.periodic_y,
        y_period=context.y_period,
    )


def _validated_parallel_frames(
    frame_paths: Sequence[NekFramePath],
) -> tuple[NekFramePath, ...]:
    frames = tuple(frame_paths)
    seen_indices: set[int] = set()
    seen_paths: set[Path] = set()
    for position, frame in enumerate(frames):
        try:
            raw_index = frame.index
            raw_path = frame.path
        except AttributeError as exc:
            raise ValueError(
                f"Frame at position {position} must provide index and path."
            ) from exc
        if not isinstance(raw_index, Integral) or isinstance(
            raw_index, (bool, np.bool_)
        ):
            raise ValueError("Nek frame indices must be non-negative integers.")
        index = int(raw_index)
        path = Path(raw_path)
        if index < 0:
            raise ValueError("Nek frame indices must be non-negative integers.")
        if index in seen_indices:
            raise ValueError("Parallel Nek frames must have unique file indices.")
        if path in seen_paths:
            raise ValueError("Parallel Nek frames must have unique source paths.")
        seen_indices.add(index)
        seen_paths.add(path)
    return frames


def _parallel_error(frame: NekFramePath, message: str) -> str:
    return (
        "Parallel leading-edge processing failed for file index "
        f"{frame.index} ({frame.path}): {message}"
    )


def _parent_result_copy(result: LeadingEdgeFrameResult) -> LeadingEdgeFrameResult:
    """Restore immutable array ownership after process-result deserialization."""
    x_front = _readonly_copy(result.x_front, np.float64)
    success_mask = _readonly_copy(result.success_mask, np.bool_)
    crossing_count = _readonly_copy(result.crossing_count, np.int64)
    if (
        x_front.ndim != 1
        or success_mask.shape != x_front.shape
        or crossing_count.shape != x_front.shape
    ):
        raise ValueError(
            "worker result arrays must be one-dimensional with matching shapes"
        )
    return LeadingEdgeFrameResult(
        file_index=int(result.file_index),
        source_path=Path(result.source_path),
        time=float(result.time),
        extraction_method=str(result.extraction_method),
        x_front=x_front,
        success_mask=success_mask,
        crossing_count=crossing_count,
        finite_leading_edge_fraction=float(
            result.finite_leading_edge_fraction
        ),
        successful_y_count=int(result.successful_y_count),
    )


def process_leading_edge_frames_parallel(
    frame_paths: Sequence[NekFramePath],
    interpolation_plan: SpectralHorizontalSlicePlan,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    threshold: float,
    *,
    extraction_method: str,
    periodic_y: bool,
    y_period: float,
    workers: int,
) -> tuple[LeadingEdgeFrameResult, ...]:
    """Reduce frames in a bounded process pool and return file-index order."""
    worker_count = validate_leading_edge_workers(workers)
    frames = _validated_parallel_frames(frame_paths)
    if not frames:
        return ()

    maximum_in_flight = max(1, 2 * worker_count)
    frame_iterator = iter(frames)
    pending: dict[Future[LeadingEdgeFrameResult], NekFramePath] = {}
    results: list[LeadingEdgeFrameResult] = []
    returned_indices: set[int] = set()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(
            interpolation_plan,
            x,
            y,
            threshold,
            extraction_method,
            periodic_y,
            y_period,
        ),
    ) as executor:

        def cancel_pending() -> None:
            for pending_future in pending:
                pending_future.cancel()

        def submit_next() -> bool:
            try:
                frame = next(frame_iterator)
            except StopIteration:
                return False
            try:
                future = executor.submit(_process_worker_frame, frame)
            except Exception as exc:
                cancel_pending()
                raise ParallelLeadingEdgeFrameError(
                    _parallel_error(frame, f"could not submit task: {exc}")
                ) from exc
            pending[future] = frame
            return True

        for _ in range(min(maximum_in_flight, len(frames))):
            submit_next()

        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                frame = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(frame, str(exc))
                    ) from exc
                if not isinstance(result, LeadingEdgeFrameResult):
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(
                            frame,
                            "worker returned an invalid result object",
                        )
                    )
                if result.file_index in returned_indices:
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(
                            frame,
                            f"worker returned duplicate file index {result.file_index}",
                        )
                    )
                returned_indices.add(result.file_index)
                if (
                    result.file_index != int(frame.index)
                    or Path(result.source_path) != Path(frame.path)
                ):
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(
                            frame,
                            "worker result identity mismatch: returned file index "
                            f"{result.file_index} and source path "
                            f"{result.source_path}",
                        )
                    )
                if result.extraction_method != extraction_method:
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(
                            frame,
                            "worker extraction-method mismatch: returned "
                            f"{result.extraction_method!r}, expected "
                            f"{extraction_method!r}",
                        )
                    )
                try:
                    parent_result = _parent_result_copy(result)
                except Exception as exc:
                    cancel_pending()
                    raise ParallelLeadingEdgeFrameError(
                        _parallel_error(frame, f"invalid worker result: {exc}")
                    ) from exc
                results.append(parent_result)
                submit_next()

    return tuple(sorted(results, key=lambda result: result.file_index))


__all__ = (
    "LeadingEdgeFrameResult",
    "ParallelLeadingEdgeFrameError",
    "process_leading_edge_frame",
    "process_leading_edge_frames_parallel",
    "validate_leading_edge_workers",
)
