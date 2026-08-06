"""Multi-frame processing for horizontal-plane leading-edge evolution."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from nek_post.front_detection_io import NekFramePath
from nek_post.io_nek import read_nek_file
from nek_post.leading_edge_parallel import (
    LeadingEdgeFrameResult,
    process_leading_edge_frame,
    process_leading_edge_frames_parallel,
    validate_leading_edge_workers,
)
from nek_post.leading_edge_methods import (
    DEFAULT_LEADING_EDGE_METHOD,
    normalize_extraction_x_min,
    normalize_leading_edge_method,
)
from nek_post.spectral_horizontal_slice import (
    build_spectral_horizontal_slice_plan,
    spectral_horizontal_plan_metadata,
)


@dataclass(frozen=True)
class LeadingEdgeEvolution:
    """Leading-edge curves retained from an ordered Nek snapshot sequence."""

    file_indices: NDArray[np.int64]
    source_files: tuple[Path, ...]
    time: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    finite_leading_edge_fraction: NDArray[np.float64]
    successful_y_count: NDArray[np.int64]
    threshold: float
    z_target: float
    nx: int
    native_ny: int
    dense_ny: int
    y_upsample_factor: int
    horizontal_plan_metadata: Mapping[str, float | int]
    periodic_endpoint_included: bool = False
    extraction_method: str = DEFAULT_LEADING_EDGE_METHOD
    extraction_x_min: float | None = None


@dataclass(frozen=True)
class LeadingEdgeTimeSelection:
    """Immutable subset of evolution frames selected at requested target times."""

    evolution: LeadingEdgeEvolution
    selected_positions: NDArray[np.int64]
    file_indices: NDArray[np.int64]
    source_files: tuple[Path, ...]
    target_time: NDArray[np.float64]
    actual_time: NDArray[np.float64]
    time_error: NDArray[np.float64]
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    finite_leading_edge_fraction: NDArray[np.float64]
    successful_y_count: NDArray[np.int64]
    target_time_spacing: float | None


def _readonly_copy(
    array: object,
    dtype: np.dtype | type,
) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return result


def _validated_frames(
    frame_paths: object,
) -> tuple[NekFramePath, ...]:
    try:
        supplied = tuple(frame_paths)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("At least one Nek frame path is required.") from exc
    if not supplied:
        raise ValueError("At least one Nek frame path is required.")

    normalized: list[NekFramePath] = []
    for position, frame in enumerate(supplied):
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
        if index < 0:
            raise ValueError("Nek frame indices must be non-negative integers.")
        normalized.append(NekFramePath(index=index, path=Path(raw_path)))

    frames = tuple(sorted(normalized, key=lambda frame: frame.index))
    indices = tuple(frame.index for frame in frames)
    if len(set(indices)) != len(indices):
        raise ValueError("Nek frame paths must have unique file indices.")
    paths = tuple(frame.path for frame in frames)
    if len(set(paths)) != len(paths):
        raise ValueError("Nek frame paths must have unique source paths.")
    return frames


def _horizontal_coordinates(
    plan: object,
    expected_nx: int,
) -> tuple[np.ndarray, np.ndarray]:
    Xi = np.asarray(plan.Xi, dtype=np.float64)  # type: ignore[attr-defined]
    Yi = np.asarray(plan.Yi, dtype=np.float64)  # type: ignore[attr-defined]
    target_shape = tuple(plan.target_shape)  # type: ignore[attr-defined]
    dense_ny = int(plan.dense_ny)  # type: ignore[attr-defined]
    if (
        Xi.ndim != 2
        or Yi.ndim != 2
        or Xi.shape != Yi.shape
        or Xi.shape != target_shape
        or Xi.shape != (dense_ny, expected_nx)
    ):
        raise ValueError(
            "Horizontal interpolation plan must provide matching Xi and Yi arrays "
            "with target_shape=(dense_ny, nx)."
        )
    x = Xi[0, :]
    y = Yi[:, 0]
    if not np.array_equal(Xi, np.broadcast_to(x, Xi.shape)):
        raise ValueError("Horizontal plan Xi must vary only along array columns.")
    if not np.array_equal(Yi, np.broadcast_to(y[:, None], Yi.shape)):
        raise ValueError("Horizontal plan Yi must vary only along array rows.")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
        or np.any(np.diff(y) <= 0.0)
    ):
        raise ValueError(
            "Horizontal plan x and y coordinate vectors must be finite and "
            "strictly increasing."
        )
    return x, y


def _plan_metadata(plan: object) -> Mapping[str, float | int]:
    metadata = dict(spectral_horizontal_plan_metadata(plan))  # type: ignore[arg-type]
    diagnostics = plan.inverse_mapping_diagnostics  # type: ignore[attr-defined]
    metadata.update(
        {
            "inverse_mapping_target_count": int(diagnostics.target_point_count),
            "inverse_mapping_success_count": int(diagnostics.inverse_success_count),
            "inverse_mapping_failure_count": int(diagnostics.inverse_failure_count),
            "ambiguous_boundary_point_count": int(
                diagnostics.ambiguous_boundary_point_count
            ),
            "maximum_successful_residual": float(
                diagnostics.maximum_successful_residual
            ),
            "maximum_iteration_count": int(diagnostics.maximum_iteration_count),
        }
    )
    return MappingProxyType(metadata)


def build_leading_edge_evolution(
    frame_paths: object,
    *,
    nx: int,
    z_target: float,
    threshold: float = 0.1,
    y_upsample_factor: int = 2,
    workers: int = 1,
    extraction_method: object = DEFAULT_LEADING_EDGE_METHOD,
    x_min: float | None = None,
    _frame_reader: Any | None = None,
) -> LeadingEdgeEvolution:
    """Read, interpolate, and immediately reduce each frame to one x(y) curve."""
    frames = _validated_frames(frame_paths)
    worker_count = validate_leading_edge_workers(workers)
    canonical_method = normalize_leading_edge_method(extraction_method)
    extraction_x_min = normalize_extraction_x_min(x_min)
    nx_value = _positive_integer(nx, "nx", 2)
    z_value = _finite_float(z_target, "z_target")
    threshold_value = _finite_float(threshold, "threshold")
    upsample_factor = _positive_integer(
        y_upsample_factor, "y_upsample_factor", 1
    )
    if worker_count > 1 and _frame_reader is not None:
        raise ValueError(
            "A custom _frame_reader is supported only with workers=1; child "
            "processes use the standard Nek frame reader."
        )
    reader = read_nek_file if _frame_reader is None else _frame_reader

    first_frame = frames[0]
    try:
        first_data = reader(first_frame.path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Nek frame {first_frame.path}: {exc}"
        ) from exc
    try:
        plan = build_spectral_horizontal_slice_plan(
            first_data,
            nx=nx_value,
            z_target=z_value,
            y_upsample_factor=upsample_factor,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build spectral horizontal interpolation plan from "
            f"{first_frame.path}: {exc}"
        ) from exc
    x, y = _horizontal_coordinates(plan, nx_value)
    plan_metadata = _plan_metadata(plan)
    y_period = float(
        plan_metadata["ymax_periodic_endpoint"] - plan_metadata["ymin"]
    )
    x_domain_options = (
        {} if extraction_x_min is None else {"x_min": extraction_x_min}
    )

    def first_frame_reader(_source_path: Path) -> object:
        return first_data

    try:
        first_result = process_leading_edge_frame(
            first_frame,
            plan,
            x,
            y,
            threshold_value,
            extraction_method=canonical_method,
            periodic_y=True,
            y_period=y_period,
            frame_reader=first_frame_reader,
            **x_domain_options,
        )
    finally:
        del first_data

    later_frames = frames[1:]
    if worker_count == 1:
        later_results = tuple(
            process_leading_edge_frame(
                frame,
                plan,
                x,
                y,
                threshold_value,
                extraction_method=canonical_method,
                periodic_y=True,
                y_period=y_period,
                frame_reader=reader,
                **x_domain_options,
            )
            for frame in later_frames
        )
    elif later_frames:
        later_results = process_leading_edge_frames_parallel(
            later_frames,
            plan,
            x,
            y,
            threshold_value,
            extraction_method=canonical_method,
            periodic_y=True,
            y_period=y_period,
            workers=worker_count,
            **x_domain_options,
        )
    else:
        later_results = ()

    results: tuple[LeadingEdgeFrameResult, ...] = tuple(
        sorted((first_result, *later_results), key=lambda result: result.file_index)
    )
    if len(results) != len(frames):
        raise RuntimeError(
            "Leading-edge frame processing returned an unexpected result count: "
            f"expected {len(frames)}, got {len(results)}."
        )
    for frame, result in zip(frames, results, strict=True):
        if (
            result.file_index != frame.index
            or Path(result.source_path) != frame.path
        ):
            raise RuntimeError(
                "Leading-edge frame result identity mismatch: expected file index "
                f"{frame.index} ({frame.path}), got {result.file_index} "
                f"({result.source_path})."
            )
        if result.extraction_method != canonical_method:
            raise RuntimeError(
                "Leading-edge frame result extraction-method mismatch: expected "
                f"{canonical_method!r}, got {result.extraction_method!r} for "
                f"{frame.path}."
            )

    times = np.asarray([result.time for result in results], dtype=np.float64)
    for frame, frame_time in zip(frames, times, strict=True):
        if not np.isfinite(frame_time):
            raise ValueError(
                f"{frame.path} has non-finite Nek time {frame_time!r}."
            )
    bad_steps = np.flatnonzero(np.diff(times) <= 0.0)
    if bad_steps.size:
        previous_position = int(bad_steps[0])
        current_position = previous_position + 1
        raise ValueError(
            "Nek frame times must be strictly increasing and unique in "
            f"file-index order: {frames[previous_position].path} has time "
            f"{times[previous_position]:.16g}, but "
            f"{frames[current_position].path} has time "
            f"{times[current_position]:.16g}."
        )

    return LeadingEdgeEvolution(
        file_indices=_readonly_copy(
            [frame.index for frame in frames], np.int64
        ),
        source_files=tuple(frame.path for frame in frames),
        time=_readonly_copy(times, np.float64),
        x=_readonly_copy(x, np.float64),
        y=_readonly_copy(y, np.float64),
        x_front=_readonly_copy(
            np.stack([result.x_front for result in results]), np.float64
        ),
        success_mask=_readonly_copy(
            np.stack([result.success_mask for result in results]), np.bool_
        ),
        crossing_count=_readonly_copy(
            np.stack([result.crossing_count for result in results]), np.int64
        ),
        finite_leading_edge_fraction=_readonly_copy(
            [result.finite_leading_edge_fraction for result in results],
            np.float64,
        ),
        successful_y_count=_readonly_copy(
            [result.successful_y_count for result in results], np.int64
        ),
        extraction_method=canonical_method,
        extraction_x_min=extraction_x_min,
        threshold=threshold_value,
        z_target=z_value,
        nx=nx_value,
        native_ny=int(plan.native_ny),  # type: ignore[attr-defined]
        dense_ny=int(plan.dense_ny),  # type: ignore[attr-defined]
        y_upsample_factor=int(plan.y_upsample_factor),  # type: ignore[attr-defined]
        horizontal_plan_metadata=plan_metadata,
        periodic_endpoint_included=False,
    )


def select_leading_edge_times(
    evolution: LeadingEdgeEvolution,
    *,
    spacing: float | None = 0.25,
) -> LeadingEdgeTimeSelection:
    """Select nearest unique frames for regular targets, or all if spacing is None."""
    actual_times = np.asarray(evolution.time, dtype=np.float64)
    if actual_times.ndim != 1 or actual_times.size == 0:
        raise ValueError("Evolution time must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(actual_times)) or np.any(np.diff(actual_times) <= 0.0):
        raise ValueError("Evolution times must be finite and strictly increasing.")

    if spacing is None:
        positions = np.arange(actual_times.size, dtype=np.int64)
        target_times = actual_times.copy()
        spacing_value: float | None = None
    else:
        spacing_value = _finite_float(spacing, "spacing")
        if spacing_value <= 0.0:
            raise ValueError("spacing must be positive when time selection is requested.")
        start = float(actual_times[0])
        end = float(actual_times[-1])
        time_scale = max(abs(start), abs(end), abs(spacing_value), 1.0)
        tolerance = 64.0 * np.finfo(np.float64).eps * time_scale
        maximum_k = int(np.floor((end - start + tolerance) / spacing_value))
        requested = start + spacing_value * np.arange(
            maximum_k + 1, dtype=np.float64
        )
        requested = requested[requested <= end + tolerance]

        selected_positions: list[int] = []
        selected_targets: list[float] = []
        seen_positions: set[int] = set()
        for target in requested:
            distances = np.abs(actual_times - target)
            minimum_distance = float(np.min(distances))
            candidates = np.flatnonzero(distances == minimum_distance)
            position = min(
                (int(candidate) for candidate in candidates),
                key=lambda candidate: (
                    float(actual_times[candidate]),
                    int(evolution.file_indices[candidate]),
                ),
            )
            if position in seen_positions:
                continue
            seen_positions.add(position)
            selected_positions.append(position)
            selected_targets.append(float(target))
        positions = np.asarray(selected_positions, dtype=np.int64)
        target_times = np.asarray(selected_targets, dtype=np.float64)

    if positions.size == 0:
        raise ValueError("Time selection did not retain any evolution frame.")
    if np.any(np.diff(positions) <= 0):
        raise RuntimeError(
            "Selected frames must remain in strictly increasing evolution order."
        )
    selected_actual_times = actual_times[positions]
    return LeadingEdgeTimeSelection(
        evolution=evolution,
        selected_positions=_readonly_copy(positions, np.int64),
        file_indices=_readonly_copy(evolution.file_indices[positions], np.int64),
        source_files=tuple(evolution.source_files[int(position)] for position in positions),
        target_time=_readonly_copy(target_times, np.float64),
        actual_time=_readonly_copy(selected_actual_times, np.float64),
        time_error=_readonly_copy(selected_actual_times - target_times, np.float64),
        x_front=_readonly_copy(evolution.x_front[positions], np.float64),
        success_mask=_readonly_copy(evolution.success_mask[positions], np.bool_),
        crossing_count=_readonly_copy(evolution.crossing_count[positions], np.int64),
        finite_leading_edge_fraction=_readonly_copy(
            evolution.finite_leading_edge_fraction[positions], np.float64
        ),
        successful_y_count=_readonly_copy(
            evolution.successful_y_count[positions], np.int64
        ),
        target_time_spacing=spacing_value,
    )


__all__ = (
    "LeadingEdgeEvolution",
    "LeadingEdgeTimeSelection",
    "build_leading_edge_evolution",
    "select_leading_edge_times",
)
