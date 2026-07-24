"""Visualization-only reconstruction of already tracked front components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.front_detection import (
    FrontComponent,
    FrontTrackingResult,
    STATUS_NO_THRESHOLD_COMPONENT,
    STATUS_NO_VALID_SPATIAL_CANDIDATE,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
    detect_front_components,
    filter_spatial_components,
)


SUCCESS_STATUSES = frozenset({STATUS_SELECTED_INITIAL, STATUS_SELECTED_TRACKED})
FAILURE_STATUSES = frozenset(
    {
        STATUS_NO_THRESHOLD_COMPONENT,
        STATUS_NO_VALID_SPATIAL_CANDIDATE,
        STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    }
)


@dataclass(frozen=True)
class FrontFrameDiagnostic:
    """One requested concentration frame and its tracker-selected segmentation."""

    frame_position: int
    file_index: int
    source_file: Path
    time: float
    concentration: NDArray[np.float64]
    components: tuple[FrontComponent, ...]
    spatial_components: tuple[FrontComponent, ...]
    selected_component: FrontComponent | None
    x_front: float
    predicted_x: float
    tracking_error: float
    status: str
    component_count: int
    spatial_candidate_count: int
    temporal_candidate_count: int
    selected_component_label: int
    selected_component_pixels: int
    selected_overlap_pixels: int
    threshold: float
    reference_x: float


def parse_diagnostic_indices(text: str) -> tuple[int, ...]:
    """Parse a nonempty, duplicate-free comma-separated file-index list."""
    if not isinstance(text, str):
        raise ValueError("Diagnostic indices must be provided as comma-separated text.")
    stripped = text.strip()
    if not stripped:
        raise ValueError("At least one diagnostic file index is required.")
    tokens = stripped.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("Diagnostic file indices must not contain empty tokens.")

    indices: list[int] = []
    for token in tokens:
        value_text = token.strip()
        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Diagnostic file index {value_text!r} is not an integer."
            ) from exc
        if value < 0:
            raise ValueError("Diagnostic file indices must be non-negative.")
        if value in indices:
            raise ValueError(f"Duplicate diagnostic file index: {value}.")
        indices.append(value)
    return tuple(indices)


def _requested_indices(values: Iterable[int]) -> tuple[int, ...]:
    requested = tuple(values)
    if not requested:
        raise ValueError("At least one diagnostic file index is required.")
    parsed: list[int] = []
    for value in requested:
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
            raise ValueError("Requested diagnostic file indices must be integers.")
        index = int(value)
        if index < 0:
            raise ValueError("Requested diagnostic file indices must be non-negative.")
        if index in parsed:
            raise ValueError(f"Duplicate requested diagnostic file index: {index}.")
        parsed.append(index)
    return tuple(parsed)


def _reference_arrays(
    reference_front: Mapping[str, Any] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if reference_front is None:
        return None
    time = np.asarray(reference_front["time"], dtype=float)
    x_front = np.abs(np.asarray(reference_front["x_front"], dtype=float))
    if (
        time.ndim != 1
        or x_front.ndim != 1
        or time.size != x_front.size
        or time.size < 2
    ):
        raise ValueError(
            "Reference time and x_front must be matching one-dimensional arrays "
            "with at least two points."
        )
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(x_front)):
        raise ValueError("Reference time and x_front must be finite.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Reference time must be strictly increasing and unique.")
    return time, x_front


def _reference_position(
    time: float,
    reference_arrays: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> float:
    if reference_arrays is None:
        return float("nan")
    reference_time, reference_x = reference_arrays
    if time < reference_time[0] or time > reference_time[-1]:
        return float("nan")
    return float(np.interp(time, reference_time, reference_x))


def _resolve_selected_component(
    components: tuple[FrontComponent, ...],
    spatial_components: tuple[FrontComponent, ...],
    tracking_result: FrontTrackingResult,
    position: int,
) -> FrontComponent | None:
    status = tracking_result.status[position]
    selected_label = int(tracking_result.selected_component_label[position])
    if status in FAILURE_STATUSES:
        if selected_label != -1:
            raise ValueError(
                f"Failed frame at position {position} must have selected label -1; "
                f"found {selected_label}."
            )
        if not np.isnan(tracking_result.x_front[position]):
            raise ValueError(
                f"Failed frame at position {position} must have NaN x_front."
            )
        return None
    if status not in SUCCESS_STATUSES:
        raise ValueError(f"Unknown tracking status {status!r} at position {position}.")
    if selected_label < 1:
        raise ValueError(
            f"Successful frame at position {position} must have a selected label "
            f"greater than or equal to 1; found {selected_label}."
        )

    matches = tuple(
        component for component in components if component.label == selected_label
    )
    if len(matches) != 1:
        raise ValueError(
            f"Reconstructed segmentation at position {position} contains "
            f"{len(matches)} components with tracked label {selected_label}; "
            "expected exactly one."
        )
    selected = matches[0]
    if not any(component.label == selected_label for component in spatial_components):
        raise ValueError(
            f"Tracked label {selected_label} at position {position} is not a "
            "spatially valid reconstructed component."
        )
    expected_pixels = int(tracking_result.selected_component_pixels[position])
    if selected.pixel_count != expected_pixels:
        raise ValueError(
            f"Tracked label {selected_label} pixel count mismatch at position "
            f"{position}: reconstructed {selected.pixel_count}, tracking "
            f"{expected_pixels}."
        )
    expected_xmin = float(tracking_result.selected_component_xmin[position])
    expected_xmax = float(tracking_result.selected_component_xmax[position])
    if not np.isclose(selected.x_min, expected_xmin):
        raise ValueError(
            f"Tracked label {selected_label} x_min mismatch at position {position}: "
            f"reconstructed {selected.x_min:g}, tracking {expected_xmin:g}."
        )
    if not np.isclose(selected.x_max, expected_xmax):
        raise ValueError(
            f"Tracked label {selected_label} x_max mismatch at position {position}: "
            f"reconstructed {selected.x_max:g}, tracking {expected_xmax:g}."
        )
    expected_bottom = bool(tracking_result.selected_bottom_contact[position])
    if selected.bottom_contact != expected_bottom:
        raise ValueError(
            f"Tracked label {selected_label} bottom-contact mismatch at position "
            f"{position}: reconstructed {selected.bottom_contact}, tracking "
            f"{expected_bottom}."
        )
    return selected


def build_front_frame_diagnostics(
    sequence: Any,
    tracking_result: FrontTrackingResult,
    requested_file_indices: Iterable[int],
    *,
    reference_front: Mapping[str, Any] | None = None,
) -> tuple[FrontFrameDiagnostic, ...]:
    """Reconstruct segmentation and resolve selection only by tracked label."""
    requested = _requested_indices(requested_file_indices)
    file_indices = np.asarray(sequence.file_indices)
    if file_indices.ndim != 1:
        raise ValueError("sequence.file_indices must be one-dimensional.")
    if file_indices.size != tracking_result.time.size:
        raise ValueError("Sequence and tracking result lengths must match.")
    positions_by_index = {
        int(file_index): position
        for position, file_index in enumerate(file_indices)
    }
    unavailable = [index for index in requested if index not in positions_by_index]
    if unavailable:
        text = ", ".join(str(index) for index in unavailable)
        raise ValueError(f"Unavailable diagnostic file indices: {text}.")
    reference_arrays = _reference_arrays(reference_front)

    diagnostics: list[FrontFrameDiagnostic] = []
    for file_index in requested:
        position = positions_by_index[file_index]
        concentration = sequence.C_frames[position]
        components = detect_front_components(
            sequence.Xi,
            sequence.Zi,
            concentration,
            threshold=tracking_result.threshold,
            bottom_rows=tracking_result.bottom_rows,
            connectivity=tracking_result.connectivity,
        )
        spatial_components = filter_spatial_components(
            components,
            min_component_pixels=tracking_result.min_component_pixels,
        )
        selected_component = _resolve_selected_component(
            components,
            spatial_components,
            tracking_result,
            position,
        )
        frame_time = float(sequence.time[position])
        diagnostics.append(
            FrontFrameDiagnostic(
                frame_position=position,
                file_index=file_index,
                source_file=Path(sequence.source_files[position]),
                time=frame_time,
                concentration=concentration,
                components=components,
                spatial_components=spatial_components,
                selected_component=selected_component,
                x_front=float(tracking_result.x_front[position]),
                predicted_x=float(tracking_result.predicted_x[position]),
                tracking_error=float(tracking_result.tracking_error[position]),
                status=tracking_result.status[position],
                component_count=int(tracking_result.component_count[position]),
                spatial_candidate_count=int(
                    tracking_result.spatial_candidate_count[position]
                ),
                temporal_candidate_count=int(
                    tracking_result.temporal_candidate_count[position]
                ),
                selected_component_label=int(
                    tracking_result.selected_component_label[position]
                ),
                selected_component_pixels=int(
                    tracking_result.selected_component_pixels[position]
                ),
                selected_overlap_pixels=int(
                    tracking_result.selected_overlap_pixels[position]
                ),
                threshold=float(tracking_result.threshold),
                reference_x=_reference_position(frame_time, reference_arrays),
            )
        )
    return tuple(diagnostics)
