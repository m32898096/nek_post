from __future__ import annotations

import numpy as np
import pytest

from nek_post.leading_edge_comparison import (
    align_fronts, common_reference_timeline, compare_extraction_methods,
    front_difference, front_statistics,
)


def test_linear_time_alignment_is_exact_on_unequal_time_steps():
    source = np.array([0., .4, 1.3, 2.])
    target = np.array([0., .2, .4, .8, 1.3, 1.7, 2.])
    y = np.arange(5) / 5
    values = 3*source[:, None] + 2*y[None, :] - .7
    result = align_fronts(source, values, np.ones(values.shape, bool), target)
    np.testing.assert_allclose(result.front, 3*target[:, None]+2*y-.7, atol=1e-14)
    assert result.valid.all()
    np.testing.assert_array_equal(result.left_position, [0, 0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(result.right_position, [0, 1, 1, 2, 2, 3, 3])
    assert np.all((result.right_weight >= 0) & (result.right_weight <= 1))


def test_failed_crossings_are_not_bridged_and_exact_samples_do_not_need_neighbors():
    values = np.array([[1., 10.], [np.nan, 11.], [3., 12.]])
    success = np.isfinite(values)
    result = align_fronts([0, 1, 2], values, success, [0, .5, 1, 1.5, 2])
    np.testing.assert_array_equal(result.valid[:, 0], [True, False, False, False, True])
    np.testing.assert_allclose(result.front[:, 1], [10, 10.5, 11, 11.5, 12])
    assert np.isnan(result.front[1:4, 0]).all()
    # A finite but explicitly failed crossing also cannot be silently used.
    values[1, 0] = 2.
    result = align_fronts([0, 1, 2], values, success, [0, .5, 1, 1.5, 2])
    assert np.isnan(result.front[1:4, 0]).all()


@pytest.mark.parametrize("targets", ([-.01, .5], [.5, 2.01]))
def test_time_extrapolation_is_rejected(targets):
    with pytest.raises(ValueError, match="extrapolation"):
        align_fronts([0, 1, 2], np.ones((3, 4)), np.ones((3, 4), bool), targets)


@pytest.mark.parametrize("time", ([0, 0, 2], [0, 2, 1], [0, np.nan, 2]))
def test_invalid_stored_times_rejected(time):
    with pytest.raises(ValueError, match="strictly increasing"):
        align_fronts(time, np.ones((3, 4)), np.ones((3, 4), bool), [0, 1])


def test_common_timeline_uses_reference_times_inside_intersection_only():
    times, interval = common_reference_timeline({"H": [.1, .8, 1.9],
        "VH": [0, 1, 2], "VVH": [0, .5, 1, 1.5, 2]}, "VVH")
    np.testing.assert_array_equal(times, [.5, 1, 1.5])
    assert interval == (.1, 1.9)


def test_bulk_translation_removed_from_shape_metrics():
    reference = np.array([1., 3., 5., 3.])
    result = front_difference(reference+4, reference, np.ones(4, bool))
    assert result["bulk_difference"] == 4
    assert result["total_mae"] == result["total_rms"] == result["total_max_abs"] == 4
    assert result["shape_mae"] == result["shape_rms"] == result["shape_max_abs"] == 0
    assert front_statistics(reference, np.ones(4, bool))["std_front"] == pytest.approx(np.sqrt(2))


def test_shape_metrics_and_bulk_decomposition_use_same_common_support():
    reference = np.array([1., 3., 5., 1e8])
    coarse = reference + [1., 2., 3., -1e8]
    mask = np.array([True, True, True, False])
    result = front_difference(coarse, reference, mask)
    assert result["bulk_difference"] == 2
    assert result["shape_rms"] == pytest.approx(np.sqrt(2/3))
    assert result["total_rms"]**2 == pytest.approx(result["bulk_difference"]**2 + result["shape_rms"]**2)
    assert result["valid_fraction"] == .75
    self_result = front_difference(reference, reference, mask)
    assert self_result["total_rms"] == self_result["shape_rms"] == 0


def test_no_valid_support_returns_nan_metrics_and_zero_coverage():
    result = front_difference([np.nan, 1.], [2., 3.], [False, False])
    assert result["valid_y_count"] == 0
    assert np.isnan(result["total_rms"])
    stats = front_statistics([np.nan, 1.], [False, False])
    assert np.isnan(stats["std_front"])
    assert stats["failed_y_count"] == 2


def test_method_comparison_reports_value_and_mask_differences_without_tolerance():
    a = np.array([1., 2., np.nan, np.nan])
    b = np.array([1.+1e-12, 2., 3., np.nan])
    result = compare_extraction_methods(a, np.isfinite(a), b, np.isfinite(b))
    assert not result["exactly_identical"]
    np.testing.assert_array_equal(result["different_y_positions"], [0, 2])
    assert result["mask_mismatch_count"] == 1
    assert result["common_valid_y_count"] == 2
    assert result["max_abs_difference"] == abs(a[0]-b[0])
    assert compare_extraction_methods(a, np.isfinite(a), a.copy(), np.isfinite(a))["exactly_identical"]
