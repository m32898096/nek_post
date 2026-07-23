"""Reference comparison and summary metrics for automatic front detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from nek_post.front_compare import (
    linear_fit_slope,
    max_abs,
    mean_abs,
    relative_difference,
    rms,
)
from nek_post.front_detection import (
    FrontTrackingResult,
    STATUS_NO_THRESHOLD_COMPONENT,
    STATUS_NO_VALID_SPATIAL_CANDIDATE,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
)
from nek_post.front_detection_workflow import ConcentrationSequence


@dataclass(frozen=True)
class FrontDetectionComparison:
    """Automatic/reference front values at overlapping successful times."""

    time: NDArray[np.float64]
    file_indices: NDArray[np.int64]
    x_front_auto: NDArray[np.float64]
    x_front_reference: NDArray[np.float64]
    difference: NDArray[np.float64]
    absolute_difference: NDArray[np.float64]


def compare_detected_front_to_reference(
    tracking_result: FrontTrackingResult,
    file_indices: NDArray[np.int64],
    reference_front: Mapping[str, NDArray[np.float64]],
) -> FrontDetectionComparison:
    """Interpolate reference positions at successful auto times without extrapolation."""
    indices = np.asarray(file_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size != tracking_result.time.size:
        raise ValueError(
            "file_indices must be one-dimensional and match the tracking result length."
        )
    reference_time = np.asarray(reference_front["time"], dtype=float)
    reference_x = np.abs(np.asarray(reference_front["x_front"], dtype=float))
    if (
        reference_time.ndim != 1
        or reference_x.ndim != 1
        or reference_time.size != reference_x.size
        or reference_time.size < 2
    ):
        raise ValueError(
            "Reference front time and x_front must be matching one-dimensional arrays "
            "with at least two points."
        )
    if not np.all(np.isfinite(reference_time)) or not np.all(np.isfinite(reference_x)):
        raise ValueError("Reference front time and x_front must be finite.")
    if np.any(np.diff(reference_time) <= 0.0):
        raise ValueError("Reference front times must be strictly increasing and unique.")

    auto_time = np.asarray(tracking_result.time, dtype=float)
    auto_x = np.abs(np.asarray(tracking_result.x_front, dtype=float))
    successful = np.isfinite(auto_time) & np.isfinite(auto_x)
    overlap = (
        successful
        & (auto_time >= reference_time[0])
        & (auto_time <= reference_time[-1])
    )
    if not np.any(overlap):
        raise ValueError(
            "No successful automatic front detections overlap the reference time range."
        )

    comparison_time = auto_time[overlap]
    comparison_auto = auto_x[overlap]
    comparison_reference = np.interp(
        comparison_time,
        reference_time,
        reference_x,
    )
    difference = comparison_auto - comparison_reference
    return FrontDetectionComparison(
        time=comparison_time,
        file_indices=indices[overlap],
        x_front_auto=comparison_auto,
        x_front_reference=comparison_reference,
        difference=difference,
        absolute_difference=np.abs(difference),
    )


def build_front_detection_summary(
    *,
    case: str,
    sequence: ConcentrationSequence,
    tracking_result: FrontTrackingResult,
    comparison: FrontDetectionComparison,
    reference_path: str | Path,
) -> dict[str, str | int | float]:
    """Build deterministic detection, comparison, and slumping-velocity metrics."""
    status_counts = Counter(tracking_result.status)
    n_input = int(sequence.time.size)
    n_successful = int(np.count_nonzero(np.isfinite(tracking_result.x_front)))
    if hasattr(sequence, "interpolation_method"):
        interpolation_method = sequence.interpolation_method
    else:
        interpolation_method = sequence.grid_metadata.get(
            "interpolation_method", "linear"
        )
    slumping_mask = (comparison.time >= 3.0) & (comparison.time <= 12.0)
    n_slumping = int(np.count_nonzero(slumping_mask))
    auto_velocity = float("nan")
    reference_velocity = float("nan")
    velocity_difference = float("nan")
    velocity_relative_difference = float("nan")
    if n_slumping >= 2:
        slumping_time = comparison.time[slumping_mask]
        auto_velocity = linear_fit_slope(
            slumping_time, comparison.x_front_auto[slumping_mask]
        )
        reference_velocity = linear_fit_slope(
            slumping_time, comparison.x_front_reference[slumping_mask]
        )
        velocity_difference = auto_velocity - reference_velocity
        velocity_relative_difference = relative_difference(
            auto_velocity,
            reference_velocity,
        )

    return {
        "case": case,
        "reference_file": str(reference_path),
        "reference_role": "external_comparison_only",
        "n_input_frames": n_input,
        "n_successful_detections": n_successful,
        "success_fraction": n_successful / n_input,
        "n_selected_initial": status_counts[STATUS_SELECTED_INITIAL],
        "n_selected_tracked": status_counts[STATUS_SELECTED_TRACKED],
        "n_no_threshold_component": status_counts[
            STATUS_NO_THRESHOLD_COMPONENT
        ],
        "n_no_valid_spatial_candidate": status_counts[
            STATUS_NO_VALID_SPATIAL_CANDIDATE
        ],
        "n_no_valid_temporal_candidate": status_counts[
            STATUS_NO_VALID_TEMPORAL_CANDIDATE
        ],
        "time_start": float(sequence.time[0]),
        "time_end": float(sequence.time[-1]),
        "index_start": int(sequence.file_indices[0]),
        "index_end": int(sequence.file_indices[-1]),
        "threshold": tracking_result.threshold,
        "min_component_pixels": tracking_result.min_component_pixels,
        "bottom_rows": tracking_result.bottom_rows,
        "max_front_jump": tracking_result.max_front_jump,
        "connectivity": tracking_result.connectivity,
        "nx": int(sequence.Xi.shape[1]),
        "nz": int(sequence.Xi.shape[0]),
        "interpolation_method": str(interpolation_method),
        "n_comparison_points": int(comparison.time.size),
        "mean_signed_difference": float(np.mean(comparison.difference)),
        "mean_absolute_difference": mean_abs(comparison.difference),
        "rms_difference": rms(comparison.difference),
        "max_absolute_difference": max_abs(comparison.difference),
        "n_slumping_points": n_slumping,
        "auto_slumping_velocity": auto_velocity,
        "reference_slumping_velocity": reference_velocity,
        "slumping_velocity_difference": velocity_difference,
        "slumping_velocity_relative_difference": velocity_relative_difference,
    }
