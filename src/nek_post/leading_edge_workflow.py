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
from nek_post.io_nek import get_nek_time, read_nek_file
from nek_post.leading_edge_extraction import extract_spanwise_leading_edge
from nek_post.spectral_horizontal_slice import (
    apply_spectral_horizontal_slice_plan,
    build_spectral_horizontal_slice_plan,
    spectral_horizontal_plan_metadata,
)
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


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


def _frame_time(data: object, path: Path) -> float:
    raw_time = get_nek_time(data)
    try:
        time = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has non-finite Nek time {raw_time!r}.") from exc
    if not np.isfinite(time):
        raise ValueError(f"{path} has non-finite Nek time {raw_time!r}.")
    return time


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
    _frame_reader: Any | None = None,
) -> LeadingEdgeEvolution:
    """Read, interpolate, and immediately reduce each frame to one x(y) curve."""
    frames = _validated_frames(frame_paths)
    nx_value = _positive_integer(nx, "nx", 2)
    z_value = _finite_float(z_target, "z_target")
    threshold_value = _finite_float(threshold, "threshold")
    upsample_factor = _positive_integer(
        y_upsample_factor, "y_upsample_factor", 1
    )
    reader = read_nek_file if _frame_reader is None else _frame_reader

    plan: object | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    times: list[float] = []
    x_front_rows: list[NDArray[np.float64]] = []
    success_rows: list[NDArray[np.bool_]] = []
    crossing_rows: list[NDArray[np.int64]] = []
    finite_fractions: list[float] = []
    successful_counts: list[int] = []

    for frame in frames:
        source_path = frame.path
        try:
            data = reader(source_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Nek frame {source_path}: {exc}") from exc
        frame_time = _frame_time(data, source_path)
        if times and frame_time <= times[-1]:
            previous = frames[len(times) - 1]
            raise ValueError(
                "Nek frame times must be strictly increasing and unique in "
                f"file-index order: {previous.path} has time {times[-1]:.16g}, "
                f"but {source_path} has time {frame_time:.16g}."
            )

        if plan is None:
            try:
                plan = build_spectral_horizontal_slice_plan(
                    data,
                    nx=nx_value,
                    z_target=z_value,
                    y_upsample_factor=upsample_factor,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to build spectral horizontal interpolation plan from "
                    f"{source_path}: {exc}"
                ) from exc
            x, y = _horizontal_coordinates(plan, nx_value)

        assert plan is not None
        assert x is not None
        assert y is not None
        try:
            concentration = apply_spectral_horizontal_slice_plan(
                plan,  # type: ignore[arg-type]
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
            curve = extract_spanwise_leading_edge(
                x,
                y,
                concentration,
                threshold=threshold_value,
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
        times.append(frame_time)
        x_front_rows.append(curve.x_front)
        success_rows.append(curve.success_mask)
        crossing_rows.append(curve.crossing_count)
        finite_count = int(np.count_nonzero(np.isfinite(curve.x_front)))
        finite_fractions.append(float(finite_count / y.size))
        successful_counts.append(int(np.count_nonzero(curve.success_mask)))

        # Only the extracted one-dimensional arrays survive this iteration.
        del concentration, curve, data

    assert plan is not None
    assert x is not None
    assert y is not None
    return LeadingEdgeEvolution(
        file_indices=_readonly_copy(
            [frame.index for frame in frames], np.int64
        ),
        source_files=tuple(frame.path for frame in frames),
        time=_readonly_copy(times, np.float64),
        x=_readonly_copy(x, np.float64),
        y=_readonly_copy(y, np.float64),
        x_front=_readonly_copy(np.stack(x_front_rows), np.float64),
        success_mask=_readonly_copy(np.stack(success_rows), np.bool_),
        crossing_count=_readonly_copy(np.stack(crossing_rows), np.int64),
        finite_leading_edge_fraction=_readonly_copy(
            finite_fractions, np.float64
        ),
        successful_y_count=_readonly_copy(successful_counts, np.int64),
        threshold=threshold_value,
        z_target=z_value,
        nx=nx_value,
        native_ny=int(plan.native_ny),  # type: ignore[attr-defined]
        dense_ny=int(plan.dense_ny),  # type: ignore[attr-defined]
        y_upsample_factor=int(plan.y_upsample_factor),  # type: ignore[attr-defined]
        horizontal_plan_metadata=_plan_metadata(plan),
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
