"""Pure numerical helpers for concentration-based front detection and tracking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


STATUS_SELECTED_INITIAL = "selected_initial"
STATUS_SELECTED_TRACKED = "selected_tracked"
STATUS_NO_THRESHOLD_COMPONENT = "no_threshold_component"
STATUS_NO_VALID_SPATIAL_CANDIDATE = "no_valid_spatial_candidate"
STATUS_NO_VALID_TEMPORAL_CANDIDATE = "no_valid_temporal_candidate"


@dataclass(frozen=True)
class FrontComponent:
    """Statistics and grid mask for one threshold-connected component."""

    label: int
    pixel_count: int
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    bottom_contact: bool
    mask: NDArray[np.bool_] = field(repr=False, compare=False)


@dataclass(frozen=True)
class FrontTrackingResult:
    """Per-frame front tracking outputs and the configuration that produced them."""

    time: NDArray[np.float64]
    x_front: NDArray[np.float64]
    predicted_x: NDArray[np.float64]
    tracking_error: NDArray[np.float64]
    selected_component_label: NDArray[np.int64]
    selected_component_pixels: NDArray[np.int64]
    selected_component_xmin: NDArray[np.float64]
    selected_component_xmax: NDArray[np.float64]
    selected_bottom_contact: NDArray[np.bool_]
    component_count: NDArray[np.int64]
    spatial_candidate_count: NDArray[np.int64]
    temporal_candidate_count: NDArray[np.int64]
    selected_overlap_pixels: NDArray[np.int64]
    status: tuple[str, ...]
    threshold: float
    min_component_pixels: int
    bottom_rows: int
    max_front_jump: float
    connectivity: int


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to 1.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be greater than or equal to 1.")
    return result


def _validated_connectivity(connectivity: int) -> int:
    if not isinstance(connectivity, Integral) or isinstance(
        connectivity, (bool, np.bool_)
    ):
        raise ValueError("connectivity must be exactly 4 or 8.")
    result = int(connectivity)
    if result not in {4, 8}:
        raise ValueError("connectivity must be exactly 4 or 8.")
    return result


def _validated_threshold(threshold: float) -> float:
    try:
        result = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError("threshold must be finite.")
    return result


def _validated_grid(
    Xi: ArrayLike,
    Zi: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_arr = np.asarray(Xi, dtype=float)
    z_arr = np.asarray(Zi, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("Xi must be a two-dimensional array.")
    if z_arr.ndim != 2:
        raise ValueError("Zi must be a two-dimensional array.")
    if x_arr.shape != z_arr.shape:
        raise ValueError("Xi and Zi must have matching shapes.")
    if not np.any(np.isfinite(z_arr)):
        raise ValueError("Zi must contain at least one finite z level.")
    if not np.any(np.isfinite(x_arr) & np.isfinite(z_arr)):
        raise ValueError("Xi and Zi must contain at least one finite coordinate pair.")
    return x_arr, z_arr


def _bottom_cutoff(Zi: NDArray[np.float64], bottom_rows: int) -> float:
    finite_levels = np.unique(Zi[np.isfinite(Zi)])
    level_count = min(bottom_rows, finite_levels.size)
    return float(finite_levels[level_count - 1])


def detect_front_components(
    Xi: ArrayLike,
    Zi: ArrayLike,
    C_grid: ArrayLike,
    *,
    threshold: float,
    bottom_rows: int,
    connectivity: int = 8,
) -> tuple[FrontComponent, ...]:
    """Return all finite, strictly thresholded connected components in one frame."""
    x_arr, z_arr = _validated_grid(Xi, Zi)
    c_arr = np.asarray(C_grid, dtype=float)
    if c_arr.ndim != 2:
        raise ValueError("C_grid must be a two-dimensional array.")
    if c_arr.shape != x_arr.shape:
        raise ValueError("C_grid must match the Xi and Zi grid shape.")
    threshold_value = _validated_threshold(threshold)
    bottom_row_count = _positive_integer(bottom_rows, "bottom_rows")
    connectivity_value = _validated_connectivity(connectivity)

    active_mask = (
        np.isfinite(x_arr)
        & np.isfinite(z_arr)
        & np.isfinite(c_arr)
        & (c_arr > threshold_value)
    )
    if connectivity_value == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = ndimage.generate_binary_structure(2, 1)
    labeled, component_count = ndimage.label(active_mask, structure=structure)
    cutoff = _bottom_cutoff(z_arr, bottom_row_count)

    components: list[FrontComponent] = []
    for label_value in range(1, component_count + 1):
        component_mask = labeled == label_value
        component_x = x_arr[component_mask]
        component_z = z_arr[component_mask]
        touches_bottom = component_mask & (
            (z_arr <= cutoff)
            | np.isclose(z_arr, cutoff, rtol=1.0e-9, atol=1.0e-12)
        )
        components.append(
            FrontComponent(
                label=label_value,
                pixel_count=int(np.count_nonzero(component_mask)),
                x_min=float(np.min(component_x)),
                x_max=float(np.max(component_x)),
                z_min=float(np.min(component_z)),
                z_max=float(np.max(component_z)),
                bottom_contact=bool(np.any(touches_bottom)),
                mask=component_mask,
            )
        )
    return tuple(components)


def filter_spatial_components(
    components: Iterable[FrontComponent],
    *,
    min_component_pixels: int,
) -> tuple[FrontComponent, ...]:
    """Keep components that are large enough and contact the configured bottom."""
    minimum = _positive_integer(min_component_pixels, "min_component_pixels")
    return tuple(
        component
        for component in components
        if component.pixel_count >= minimum and component.bottom_contact
    )


def select_initial_component(
    components: Iterable[FrontComponent],
) -> FrontComponent | None:
    """Select the largest initial component with deterministic tie breaking."""
    candidates = tuple(components)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda component: (
            -component.pixel_count,
            -component.x_max,
            component.label,
        ),
    )


def predict_front_position(
    successful_times: ArrayLike,
    successful_fronts: ArrayLike,
    current_time: float,
) -> float:
    """Predict the current front from the last one or two successful detections."""
    times = np.asarray(successful_times, dtype=float)
    fronts = np.asarray(successful_fronts, dtype=float)
    if times.ndim != 1 or fronts.ndim != 1:
        raise ValueError("Successful times and fronts must be one-dimensional.")
    if times.size != fronts.size:
        raise ValueError("Successful times and fronts must have matching lengths.")
    if times.size == 0:
        return float("nan")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(fronts)):
        raise ValueError("Successful times and fronts must be finite.")
    current = float(current_time)
    if not np.isfinite(current):
        raise ValueError("current_time must be finite.")
    if times.size == 1:
        return float(fronts[-1])
    time_step = times[-1] - times[-2]
    if time_step <= 0.0:
        raise ValueError("The last two successful times must be strictly increasing.")
    velocity = (fronts[-1] - fronts[-2]) / time_step
    return float(fronts[-1] + velocity * (current - times[-1]))


def _temporal_candidates(
    components: Iterable[FrontComponent],
    predicted_x: float,
    max_front_jump: float,
) -> tuple[FrontComponent, ...]:
    return tuple(
        component
        for component in components
        if abs(component.x_max - predicted_x) <= max_front_jump
    )


def select_temporal_component(
    components: Iterable[FrontComponent],
    *,
    predicted_x: float,
    max_front_jump: float,
    previous_selected_mask: ArrayLike,
) -> tuple[FrontComponent | None, int]:
    """Select a temporally valid component and return its previous-mask overlap."""
    prediction = float(predicted_x)
    if not np.isfinite(prediction):
        raise ValueError("predicted_x must be finite for temporal selection.")
    jump = float(max_front_jump)
    if np.isnan(jump) or jump <= 0.0:
        raise ValueError("max_front_jump must be positive and may be positive infinity.")
    candidates = _temporal_candidates(components, prediction, jump)
    if not candidates:
        return None, 0

    previous_mask = np.asarray(previous_selected_mask, dtype=bool)
    for component in candidates:
        if component.mask.shape != previous_mask.shape:
            raise ValueError("Component masks and previous_selected_mask must match.")

    def ranking_key(component: FrontComponent) -> tuple[float, ...]:
        overlap = int(np.count_nonzero(component.mask & previous_mask))
        distance = abs(component.x_max - prediction)
        return (
            -overlap,
            -component.pixel_count,
            distance,
            -component.x_max,
            component.label,
        )

    selected = min(candidates, key=ranking_key)
    selected_overlap = int(np.count_nonzero(selected.mask & previous_mask))
    return selected, selected_overlap


def _validated_sequence_inputs(
    time: ArrayLike,
    Xi: ArrayLike,
    Zi: ArrayLike,
    C_frames: ArrayLike,
    threshold: float,
    min_component_pixels: int,
    bottom_rows: int,
    max_front_jump: float,
    connectivity: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    int,
    int,
    float,
    int,
]:
    time_arr = np.asarray(time, dtype=float)
    if time_arr.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")
    if time_arr.size == 0:
        raise ValueError("time must contain at least one sample.")
    if not np.all(np.isfinite(time_arr)):
        raise ValueError("time values must be finite.")
    if np.any(np.diff(time_arr) <= 0.0):
        raise ValueError("time values must be strictly increasing and unique.")

    x_arr, z_arr = _validated_grid(Xi, Zi)
    frames_arr = np.asarray(C_frames, dtype=float)
    if frames_arr.ndim != 3:
        raise ValueError("C_frames must be a three-dimensional array.")
    if frames_arr.shape[0] != time_arr.size:
        raise ValueError("C_frames frame count must match time.size.")
    if frames_arr.shape[1:] != x_arr.shape:
        raise ValueError("Every concentration frame must match the Xi and Zi grid shape.")

    threshold_value = _validated_threshold(threshold)
    minimum = _positive_integer(min_component_pixels, "min_component_pixels")
    bottom_row_count = _positive_integer(bottom_rows, "bottom_rows")
    connectivity_value = _validated_connectivity(connectivity)
    jump = float(max_front_jump)
    if np.isnan(jump) or jump <= 0.0:
        raise ValueError("max_front_jump must be positive and may be positive infinity.")

    return (
        time_arr,
        x_arr,
        z_arr,
        frames_arr,
        threshold_value,
        minimum,
        bottom_row_count,
        jump,
        connectivity_value,
    )


def track_concentration_front(
    time: ArrayLike,
    Xi: ArrayLike,
    Zi: ArrayLike,
    C_frames: ArrayLike,
    *,
    threshold: float,
    min_component_pixels: int,
    bottom_rows: int,
    max_front_jump: float,
    connectivity: int = 8,
) -> FrontTrackingResult:
    """Detect and temporally track the selected concentration front."""
    (
        time_arr,
        x_arr,
        z_arr,
        frames_arr,
        threshold_value,
        minimum,
        bottom_row_count,
        jump,
        connectivity_value,
    ) = _validated_sequence_inputs(
        time,
        Xi,
        Zi,
        C_frames,
        threshold,
        min_component_pixels,
        bottom_rows,
        max_front_jump,
        connectivity,
    )

    frame_count = time_arr.size
    x_front = np.full(frame_count, np.nan, dtype=float)
    predicted_x = np.full(frame_count, np.nan, dtype=float)
    tracking_error = np.full(frame_count, np.nan, dtype=float)
    selected_component_label = np.full(frame_count, -1, dtype=np.int64)
    selected_component_pixels = np.zeros(frame_count, dtype=np.int64)
    selected_component_xmin = np.full(frame_count, np.nan, dtype=float)
    selected_component_xmax = np.full(frame_count, np.nan, dtype=float)
    selected_bottom_contact = np.zeros(frame_count, dtype=bool)
    component_count = np.zeros(frame_count, dtype=np.int64)
    spatial_candidate_count = np.zeros(frame_count, dtype=np.int64)
    temporal_candidate_count = np.zeros(frame_count, dtype=np.int64)
    selected_overlap_pixels = np.zeros(frame_count, dtype=np.int64)
    statuses: list[str] = []

    successful_times: list[float] = []
    successful_fronts: list[float] = []
    previous_selected_mask: NDArray[np.bool_] | None = None

    for frame_index, (frame_time, concentration) in enumerate(
        zip(time_arr, frames_arr, strict=True)
    ):
        components = detect_front_components(
            x_arr,
            z_arr,
            concentration,
            threshold=threshold_value,
            bottom_rows=bottom_row_count,
            connectivity=connectivity_value,
        )
        component_count[frame_index] = len(components)

        if successful_times:
            predicted_x[frame_index] = predict_front_position(
                successful_times[-2:],
                successful_fronts[-2:],
                float(frame_time),
            )

        if not components:
            statuses.append(STATUS_NO_THRESHOLD_COMPONENT)
            continue

        spatial_candidates = filter_spatial_components(
            components,
            min_component_pixels=minimum,
        )
        spatial_candidate_count[frame_index] = len(spatial_candidates)
        if not spatial_candidates:
            statuses.append(STATUS_NO_VALID_SPATIAL_CANDIDATE)
            continue

        if not successful_times:
            selected = select_initial_component(spatial_candidates)
            selected_overlap = 0
            status = STATUS_SELECTED_INITIAL
        else:
            assert previous_selected_mask is not None
            temporal_candidates = _temporal_candidates(
                spatial_candidates,
                predicted_x[frame_index],
                jump,
            )
            temporal_candidate_count[frame_index] = len(temporal_candidates)
            if not temporal_candidates:
                statuses.append(STATUS_NO_VALID_TEMPORAL_CANDIDATE)
                continue
            selected, selected_overlap = select_temporal_component(
                temporal_candidates,
                predicted_x=predicted_x[frame_index],
                max_front_jump=jump,
                previous_selected_mask=previous_selected_mask,
            )
            status = STATUS_SELECTED_TRACKED

        assert selected is not None
        x_front[frame_index] = selected.x_max
        if status == STATUS_SELECTED_TRACKED:
            tracking_error[frame_index] = (
                selected.x_max - predicted_x[frame_index]
            )
        selected_component_label[frame_index] = selected.label
        selected_component_pixels[frame_index] = selected.pixel_count
        selected_component_xmin[frame_index] = selected.x_min
        selected_component_xmax[frame_index] = selected.x_max
        selected_bottom_contact[frame_index] = selected.bottom_contact
        selected_overlap_pixels[frame_index] = selected_overlap
        statuses.append(status)

        successful_times.append(float(frame_time))
        successful_fronts.append(selected.x_max)
        previous_selected_mask = selected.mask

    return FrontTrackingResult(
        time=time_arr.copy(),
        x_front=x_front,
        predicted_x=predicted_x,
        tracking_error=tracking_error,
        selected_component_label=selected_component_label,
        selected_component_pixels=selected_component_pixels,
        selected_component_xmin=selected_component_xmin,
        selected_component_xmax=selected_component_xmax,
        selected_bottom_contact=selected_bottom_contact,
        component_count=component_count,
        spatial_candidate_count=spatial_candidate_count,
        temporal_candidate_count=temporal_candidate_count,
        selected_overlap_pixels=selected_overlap_pixels,
        status=tuple(statuses),
        threshold=threshold_value,
        min_component_pixels=minimum,
        bottom_rows=bottom_row_count,
        max_front_jump=jump,
        connectivity=connectivity_value,
    )
