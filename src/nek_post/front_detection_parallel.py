"""Process-based preprocessing for later front-detection frames."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nek_post.fixed_grid_interpolation import (
    FixedGridGeometryMismatchError,
    FixedGridInterpolationPlan,
    apply_fixed_grid_interpolation_plan,
)
from nek_post.front_detection_io import NekFramePath
from nek_post.io_nek import get_nek_time, read_nek_file
from nek_post.slicing import extract_y_slice


class ParallelFramePreprocessingError(RuntimeError):
    """Raised when a process-pool frame task fails."""


@dataclass(frozen=True)
class FrontDetectionFrameResult:
    """One fully reduced frame result returned from a worker."""

    file_index: int
    source_path: Path
    time: float
    concentration: NDArray[np.float64]
    finite_fraction: float
    concentration_min: float
    concentration_max: float
    selected_y: float


@dataclass(frozen=True)
class _WorkerContext:
    interpolation_plan: FixedGridInterpolationPlan
    slice_mode: str
    slab_ratio: float
    y_round_decimals: int


_WORKER_CONTEXT: _WorkerContext | None = None


def validate_preprocessing_workers(value: int) -> int:
    """Validate and normalize the requested process count."""
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError("workers must be an integer greater than or equal to 1.")
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be greater than or equal to 1.")
    return workers


def _finite_frame_time(data: object, source_path: Path) -> float:
    raw_time = get_nek_time(data)
    try:
        time = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path} has non-finite Nek time {raw_time!r}."
        ) from exc
    if not np.isfinite(time):
        raise ValueError(
            f"{source_path} has non-finite Nek time {raw_time!r}."
        )
    return time


def preprocess_front_detection_frame(
    frame: NekFramePath,
    interpolation_plan: FixedGridInterpolationPlan,
    *,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
) -> FrontDetectionFrameResult:
    """Read, slice, validate, interpolate, and reduce one later frame."""
    source_path = Path(frame.path)
    try:
        data = read_nek_file(source_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Nek frame {source_path}: {exc}"
        ) from exc
    frame_time = _finite_frame_time(data, source_path)
    try:
        slice_data = extract_y_slice(
            data,
            slab_ratio=slab_ratio,
            mode=slice_mode,
            y_round_decimals=y_round_decimals,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract concentration slice from {source_path}: {exc}"
        ) from exc
    try:
        concentration = apply_fixed_grid_interpolation_plan(
            interpolation_plan,
            slice_data["x"],
            slice_data["z"],
            slice_data["C"],
        )
    except FixedGridGeometryMismatchError as exc:
        raise FixedGridGeometryMismatchError(
            f"Source geometry mismatch for {source_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to interpolate concentration from {source_path}: {exc}"
        ) from exc

    concentration_arr = np.asarray(concentration, dtype=np.float64)
    if concentration_arr.shape != interpolation_plan.target_shape:
        raise ValueError(
            f"Interpolated concentration from {source_path} has shape "
            f"{concentration_arr.shape}; expected "
            f"{interpolation_plan.target_shape}."
        )
    finite = np.isfinite(concentration_arr)
    if not np.any(finite):
        raise ValueError(
            f"Interpolated concentration from {source_path} has no finite values."
        )
    concentration_arr.setflags(write=False)
    selected_y = (
        float(slice_data.get("selected_y", np.nan))
        if slice_mode == "nearest_plane"
        else float("nan")
    )
    return FrontDetectionFrameResult(
        file_index=int(frame.index),
        source_path=source_path,
        time=frame_time,
        concentration=concentration_arr,
        finite_fraction=float(np.count_nonzero(finite) / finite.size),
        concentration_min=float(np.nanmin(concentration_arr)),
        concentration_max=float(np.nanmax(concentration_arr)),
        selected_y=selected_y,
    )


def _initialize_worker(
    interpolation_plan: FixedGridInterpolationPlan,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(
        interpolation_plan=interpolation_plan,
        slice_mode=slice_mode,
        slab_ratio=slab_ratio,
        y_round_decimals=y_round_decimals,
    )


def _preprocess_worker_frame(
    frame: NekFramePath,
) -> FrontDetectionFrameResult:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Parallel preprocessing worker was not initialized.")
    return preprocess_front_detection_frame(
        frame,
        context.interpolation_plan,
        slice_mode=context.slice_mode,
        slab_ratio=context.slab_ratio,
        y_round_decimals=context.y_round_decimals,
    )


def preprocess_front_detection_frames_parallel(
    frame_paths: Sequence[NekFramePath],
    interpolation_plan: FixedGridInterpolationPlan,
    *,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
    workers: int,
) -> tuple[FrontDetectionFrameResult, ...]:
    """Process later frames with bounded submissions and deterministic ordering."""
    worker_count = validate_preprocessing_workers(workers)
    frames = tuple(frame_paths)
    if not frames:
        return ()

    maximum_in_flight = max(1, 2 * worker_count)
    results: list[FrontDetectionFrameResult] = []
    frame_iterator = iter(frames)
    pending: dict[Future[FrontDetectionFrameResult], NekFramePath] = {}

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(
            interpolation_plan,
            slice_mode,
            slab_ratio,
            y_round_decimals,
        ),
    ) as executor:

        def submit_next() -> bool:
            try:
                frame = next(frame_iterator)
            except StopIteration:
                return False
            try:
                future = executor.submit(_preprocess_worker_frame, frame)
            except Exception as exc:
                for pending_future in pending:
                    pending_future.cancel()
                raise ParallelFramePreprocessingError(
                    "Could not submit parallel preprocessing for file index "
                    f"{frame.index} ({frame.path}): {exc}"
                ) from exc
            pending[future] = frame
            return True

        for _ in range(min(maximum_in_flight, len(frames))):
            submit_next()

        while pending:
            completed, _ = wait(
                tuple(pending),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                frame = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise ParallelFramePreprocessingError(
                        "Parallel preprocessing failed for file index "
                        f"{frame.index} ({frame.path}): {exc}"
                    ) from exc
                submit_next()

    return tuple(sorted(results, key=lambda result: result.file_index))
