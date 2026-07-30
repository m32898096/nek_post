from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.front_detection import (
    FrontTrackingResult,
    STATUS_NO_THRESHOLD_COMPONENT,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
)
from nek_post.front_detection_compare import (
    FrontDetectionComparison,
    build_front_detection_summary,
    compare_detected_front_to_reference,
)


def _tracking(
    time: list[float],
    x_front: list[float],
    status: tuple[str, ...] | None = None,
) -> FrontTrackingResult:
    n = len(time)
    statuses = status or tuple(STATUS_SELECTED_TRACKED for _ in range(n))
    x = np.asarray(x_front, dtype=float)
    return FrontTrackingResult(
        time=np.asarray(time, dtype=float),
        x_front=x,
        predicted_x=np.full(n, np.nan),
        tracking_error=np.full(n, np.nan),
        selected_component_label=np.where(np.isfinite(x), 1, -1).astype(np.int64),
        selected_component_pixels=np.where(np.isfinite(x), 10, 0).astype(np.int64),
        selected_component_xmin=np.where(np.isfinite(x), x - 1.0, np.nan),
        selected_component_xmax=x.copy(),
        selected_bottom_contact=np.isfinite(x),
        component_count=np.ones(n, dtype=np.int64),
        spatial_candidate_count=np.ones(n, dtype=np.int64),
        temporal_candidate_count=np.ones(n, dtype=np.int64),
        selected_overlap_pixels=np.zeros(n, dtype=np.int64),
        status=statuses,
        threshold=0.01,
        min_component_pixels=50,
        bottom_rows=3,
        max_front_jump=0.5,
        connectivity=8,
    )


def test_comparison_excludes_failures_and_extrapolation_and_aligns_indices() -> None:
    tracking = _tracking(
        [-1.0, 0.0, 1.0, 2.0, 4.0],
        [5.0, np.nan, -2.0, 4.0, 10.0],
    )
    reference = {
        "time": np.array([0.0, 2.0, 3.0]),
        "x_front": np.array([-1.0, -3.0, -4.0]),
    }

    comparison = compare_detected_front_to_reference(
        tracking,
        np.array([10, 11, 12, 13, 14]),
        reference,
    )

    assert_array_equal(comparison.time, [1.0, 2.0])
    assert_array_equal(comparison.file_indices, [12, 13])
    assert_array_equal(comparison.x_front_auto, [2.0, 4.0])
    assert_allclose(comparison.x_front_reference, [2.0, 3.0])
    assert_allclose(comparison.difference, [0.0, 1.0])
    assert_allclose(comparison.absolute_difference, [0.0, 1.0])


def test_comparison_fails_when_no_successful_detection_overlaps() -> None:
    tracking = _tracking([0.0, 1.0], [np.nan, 2.0])
    reference = {
        "time": np.array([2.0, 3.0]),
        "x_front": np.array([1.0, 2.0]),
    }

    with pytest.raises(ValueError, match="No successful automatic.*overlap"):
        compare_detected_front_to_reference(
            tracking,
            np.array([1, 2]),
            reference,
        )


def test_summary_metrics_status_counts_success_fraction_and_slumping_velocity() -> None:
    statuses = (
        STATUS_NO_THRESHOLD_COMPONENT,
        STATUS_SELECTED_INITIAL,
        STATUS_SELECTED_TRACKED,
        STATUS_SELECTED_TRACKED,
        STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    )
    tracking = _tracking(
        [0.0, 3.0, 6.0, 9.0, 13.0],
        [np.nan, 7.0, 13.0, 19.0, np.nan],
        statuses,
    )
    sequence = SimpleNamespace(
        time=np.array([0.0, 3.0, 6.0, 9.0, 13.0]),
        file_indices=np.array([1, 2, 3, 4, 5]),
        Xi=np.zeros((2, 3)),
        interpolation_method="linear",
    )
    comparison = FrontDetectionComparison(
        time=np.array([3.0, 6.0, 9.0]),
        file_indices=np.array([2, 3, 4]),
        x_front_auto=np.array([7.0, 13.0, 19.0]),
        x_front_reference=np.array([4.0, 7.0, 10.0]),
        difference=np.array([3.0, 6.0, 9.0]),
        absolute_difference=np.array([3.0, 6.0, 9.0]),
    )

    summary = build_front_detection_summary(
        case="N7",
        sequence=sequence,
        tracking_result=tracking,
        comparison=comparison,
        reference_path=Path("front_simple.dat"),
    )

    assert summary["n_input_frames"] == 5
    assert summary["n_successful_detections"] == 3
    assert summary["success_fraction"] == pytest.approx(0.6)
    assert summary["n_selected_initial"] == 1
    assert summary["n_selected_tracked"] == 2
    assert summary["n_no_threshold_component"] == 1
    assert summary["n_no_valid_spatial_candidate"] == 0
    assert summary["n_no_valid_temporal_candidate"] == 1
    assert summary["reference_role"] == "external_comparison_only"
    assert summary["interpolation_engine"] == "scattered_linear"
    assert summary["spectral_element_shape"] == ""
    assert summary["spectral_polynomial_order"] == ""
    assert summary["spectral_slice_y"] == ""
    assert summary["mean_signed_difference"] == 6.0
    assert summary["mean_absolute_difference"] == 6.0
    assert summary["rms_difference"] == pytest.approx(np.sqrt(42.0))
    assert summary["max_absolute_difference"] == 9.0
    assert summary["n_slumping_points"] == 3
    assert summary["auto_slumping_velocity"] == pytest.approx(2.0)
    assert summary["reference_slumping_velocity"] == pytest.approx(1.0)
    assert summary["slumping_velocity_difference"] == pytest.approx(1.0)
    assert summary["slumping_velocity_relative_difference"] == pytest.approx(1.0)


def test_summary_returns_nan_velocities_with_fewer_than_two_slumping_points() -> None:
    tracking = _tracking(
        [2.0],
        [1.0],
        (STATUS_SELECTED_INITIAL,),
    )
    sequence = SimpleNamespace(
        time=np.array([2.0]),
        file_indices=np.array([1]),
        Xi=np.zeros((2, 3)),
        interpolation_method="nearest",
    )
    comparison = FrontDetectionComparison(
        time=np.array([2.0]),
        file_indices=np.array([1]),
        x_front_auto=np.array([1.0]),
        x_front_reference=np.array([1.0]),
        difference=np.array([0.0]),
        absolute_difference=np.array([0.0]),
    )

    summary = build_front_detection_summary(
        case="N7",
        sequence=sequence,
        tracking_result=tracking,
        comparison=comparison,
        reference_path="front_simple.dat",
    )

    assert summary["n_slumping_points"] == 0
    assert np.isnan(summary["auto_slumping_velocity"])
    assert np.isnan(summary["reference_slumping_velocity"])
    assert np.isnan(summary["slumping_velocity_difference"])
    assert np.isnan(summary["slumping_velocity_relative_difference"])


def test_summary_records_spectral_preprocessing_metadata() -> None:
    tracking = _tracking([2.0], [1.0], (STATUS_SELECTED_INITIAL,))
    sequence = SimpleNamespace(
        time=np.array([2.0]),
        file_indices=np.array([1]),
        Xi=np.zeros((2, 3)),
        interpolation_method="spectral",
        interpolation_engine="spectral_element",
        spectral_element_shape=(8, 8, 8),
        spectral_polynomial_order=(7, 7, 7),
        spectral_slice_y=0.75,
    )
    comparison = FrontDetectionComparison(
        time=np.array([2.0]),
        file_indices=np.array([1]),
        x_front_auto=np.array([1.0]),
        x_front_reference=np.array([1.0]),
        difference=np.array([0.0]),
        absolute_difference=np.array([0.0]),
    )

    summary = build_front_detection_summary(
        case="N7",
        sequence=sequence,
        tracking_result=tracking,
        comparison=comparison,
        reference_path="front_simple.dat",
    )

    assert summary["interpolation_engine"] == "spectral_element"
    assert summary["spectral_element_shape"] == "8 x 8 x 8"
    assert summary["spectral_polynomial_order"] == "7 x 7 x 7"
    assert summary["spectral_slice_y"] == 0.75
