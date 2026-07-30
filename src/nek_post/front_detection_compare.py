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


COMPARISON_STATUSES = frozenset(
    {"available", "no_time_overlap", "disabled"}
)


def empty_front_detection_comparison() -> FrontDetectionComparison:
    """Return a deterministic typed comparison containing no rows."""
    return FrontDetectionComparison(
        time=np.empty(0, dtype=np.float64),
        file_indices=np.empty(0, dtype=np.int64),
        x_front_auto=np.empty(0, dtype=np.float64),
        x_front_reference=np.empty(0, dtype=np.float64),
        difference=np.empty(0, dtype=np.float64),
        absolute_difference=np.empty(0, dtype=np.float64),
    )


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
        return empty_front_detection_comparison()

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
    comparison_status: str | None = None,
) -> dict[str, str | int | float]:
    """Build deterministic detection, comparison, and slumping-velocity metrics."""
    status = (
        comparison_status
        if comparison_status is not None
        else ("available" if comparison.time.size else "no_time_overlap")
    )
    if status not in COMPARISON_STATUSES:
        raise ValueError(
            "comparison_status must be 'available', 'no_time_overlap', or "
            "'disabled'."
        )
    has_comparison = comparison.time.size > 0
    if status == "available" and not has_comparison:
        raise ValueError(
            "comparison_status='available' requires at least one comparison point."
        )
    if status != "available" and has_comparison:
        raise ValueError(
            f"comparison_status={status!r} requires an empty comparison."
        )
    status_counts = Counter(tracking_result.status)
    n_input = int(sequence.time.size)
    n_successful = int(np.count_nonzero(np.isfinite(tracking_result.x_front)))
    if hasattr(sequence, "interpolation_method"):
        interpolation_method = sequence.interpolation_method
    else:
        interpolation_method = sequence.grid_metadata.get(
            "interpolation_method", "linear"
        )
    sequence_element_shape = getattr(sequence, "spectral_element_shape", None)
    sequence_polynomial_order = getattr(
        sequence, "spectral_polynomial_order", None
    )
    sequence_slice_y = getattr(sequence, "spectral_slice_y", None)
    interpolation_engine = getattr(
        sequence,
        "interpolation_engine",
        f"scattered_{interpolation_method}",
    )
    spectral_element_shape = (
        ""
        if sequence_element_shape is None
        else " x ".join(str(value) for value in sequence_element_shape)
    )
    spectral_polynomial_order = (
        ""
        if sequence_polynomial_order is None
        else " x ".join(str(value) for value in sequence_polynomial_order)
    )
    spectral_slice_y: str | float = (
        "" if sequence_slice_y is None else sequence_slice_y
    )
    difference_mean = float("nan")
    difference_mean_absolute = float("nan")
    difference_rms = float("nan")
    difference_maximum_absolute = float("nan")
    if has_comparison:
        difference_mean = float(np.mean(comparison.difference))
        difference_mean_absolute = mean_abs(comparison.difference)
        difference_rms = rms(comparison.difference)
        difference_maximum_absolute = max_abs(comparison.difference)

    slumping_mask = (
        (comparison.time >= 3.0) & (comparison.time <= 12.0)
        if has_comparison
        else np.empty(0, dtype=bool)
    )
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
        "reference_file": "" if status == "disabled" else str(reference_path),
        "reference_role": (
            "not_requested"
            if status == "disabled"
            else "external_comparison_only"
        ),
        "comparison_status": status,
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
        "interpolation_engine": interpolation_engine,
        "spectral_element_shape": spectral_element_shape,
        "spectral_polynomial_order": spectral_polynomial_order,
        "spectral_slice_y": spectral_slice_y,
        "n_comparison_points": int(comparison.time.size),
        "mean_signed_difference": difference_mean,
        "mean_absolute_difference": difference_mean_absolute,
        "rms_difference": difference_rms,
        "max_absolute_difference": difference_maximum_absolute,
        "n_slumping_points": n_slumping,
        "auto_slumping_velocity": auto_velocity,
        "reference_slumping_velocity": reference_velocity,
        "slumping_velocity_difference": velocity_difference,
        "slumping_velocity_relative_difference": velocity_relative_difference,
    }
