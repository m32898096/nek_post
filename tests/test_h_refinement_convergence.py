"""Manufactured convergence sequences and common-mask diagnostics."""
import json

import numpy as np
import pytest

from nek_post.h_refinement_convergence import (
    analyze_convergence_triplet,
    solve_generalized_observed_order,
)


def manufactured(h_values, order):
    exact = np.array([[1., 2., 3.], [2., 1., 4.]])
    coefficient = np.array([[1., -2., 3.], [-1., .5, 2.]])
    return [exact + h**order * coefficient for h in h_values]


@pytest.mark.parametrize("h_values,order", [
    ((.8, .4, .2), 2.5),
    ((1., .6, .25), 1.75),
    ((1., .5, .125), 3.),
])
def test_manufactured_equal_and_unequal_ratios_recover_order(h_values, order):
    result = analyze_convergence_triplet(
        *manufactured(h_values, order), h_values=h_values, scalar_h_justified=True)
    assert result["classification"] == "monotonic convergence-like"
    assert result["observed_order"] == pytest.approx(order, abs=1e-11)
    assert result["difference_vector_cosine"] == pytest.approx(1.)
    assert result["order_status"] == "conditional_power_law_estimate"
    assert not result["order_rejection_reasons"]


def test_unequal_ratio_does_not_use_equal_ratio_shortcut():
    h_values = (1., .6, .25)
    result = analyze_convergence_triplet(
        *manufactured(h_values, 1.75), h_values=h_values, scalar_h_justified=True)
    wrong = np.log(1 / result["successive_difference_ratio"]) / np.log(h_values[0]/h_values[1])
    assert abs(wrong - 1.75) > .1
    assert result["observed_order"] == pytest.approx(1.75)


def test_unequal_steps_allow_increasing_successive_differences_for_exact_power_law():
    h_values = (1., .9, .1)
    result = analyze_convergence_triplet(
        *manufactured(h_values, 2.), h_values=h_values, scalar_h_justified=True)
    assert result["successive_difference_ratio"] > 1.05
    assert result["finest_reference_error_ratio"] < .95
    assert result["classification"] == "monotonic convergence-like"
    assert not result["equal_refinement_ratios"]
    assert result["observed_order"] == pytest.approx(2.)


def test_order_requires_independent_scalar_h_justification():
    h_values = (.8, .4, .2)
    result = analyze_convergence_triplet(*manufactured(h_values, 2.), h_values=h_values)
    assert result["classification"] == "monotonic convergence-like"
    assert result["observed_order"] is None
    assert "independently justified" in result["order_rejection_reasons"][0]
    result = analyze_convergence_triplet(*manufactured(h_values, 2.), scalar_h_justified=True)
    assert result["observed_order"] is None
    assert any("No measured" in reason for reason in result["order_rejection_reasons"])


def test_external_time_gate_preserves_scalar_h_justification_and_blocks_fit():
    reason = "snapshot_times_not_identical_temporal_error_uncontrolled"
    h_values = (.8, .4, .2)
    result = analyze_convergence_triplet(
        *manufactured(h_values, 2.), h_values=h_values, scalar_h_justified=True,
        external_order_rejection_reasons=[reason])
    assert result["scalar_h_justified"]
    assert result["classification"] == "monotonic convergence-like"
    assert result["observed_order"] is None
    assert result["order_rejection_reasons"] == [reason]


def test_oscillatory_sequence_rejected_even_when_norms_decrease():
    result = analyze_convergence_triplet(
        np.full((2, 2), 1.), np.full((2, 2), -.5), np.zeros((2, 2)),
        h_values=(1., .5, .25), scalar_h_justified=True)
    assert result["successive_difference_ratio"] < .95
    assert result["finest_reference_error_ratio"] < .95
    assert result["difference_vector_cosine"] == pytest.approx(-1.)
    assert result["classification"] == "non-monotonic"
    assert result["observed_order"] is None


def test_increasing_finest_reference_error_is_non_monotonic():
    result = analyze_convergence_triplet(
        np.full((2, 2), .8), np.full((2, 2), 1.), np.zeros((2, 2)))
    assert result["finest_reference_error_ratio"] == pytest.approx(1.25)
    assert result["classification"] == "non-monotonic"
    assert result["observed_order"] is None


def test_near_equal_successive_differences_on_equal_ratio_family_reject_order():
    result = analyze_convergence_triplet(
        np.full((2, 2), 1.99), np.full((2, 2), .99), np.zeros((2, 2)),
        h_values=(1., .5, .25), scalar_h_justified=True)
    assert result["successive_difference_ratio"] == pytest.approx(.99)
    assert result["classification"] == "monotonic convergence-like"
    assert result["equal_refinement_ratios"]
    assert result["observed_order"] is None
    assert any("Near-equal successive" in reason for reason in result["order_rejection_reasons"])


def test_near_equal_finest_reference_errors_do_not_support_order():
    result = analyze_convergence_triplet(
        np.full((2, 2), 1.), np.full((2, 2), .99), np.zeros((2, 2)),
        h_values=(1., .5, .25), scalar_h_justified=True)
    assert result["finest_reference_error_ratio"] == pytest.approx(.99)
    assert result["classification"] == "approximately unchanged / inconclusive"
    assert result["observed_order"] is None


def test_difference_alignment_required_for_common_leading_error_model():
    result = analyze_convergence_triplet(
        np.array([[2., 1.]]), np.array([[1., 0.]]), np.zeros((1, 2)),
        h_values=(1., .5, .25), scalar_h_justified=True)
    assert result["classification"] == "monotonic convergence-like"
    assert result["difference_vector_cosine"] == pytest.approx(1/np.sqrt(2))
    assert result["observed_order"] is None
    assert any("alignment" in reason for reason in result["order_rejection_reasons"])


def test_every_pair_uses_all_case_common_finite_valid_mask():
    coarse = np.array([[4., 1000., 1000.], [4., 1000., 4.]])
    medium = np.array([[2., np.nan, 2.], [2., 2., 2.]])
    fine = np.array([[1., 1., np.inf], [1., 1., 1.]])
    mask = np.ones((2, 3), dtype=bool)
    mask[1, 1] = False
    result = analyze_convergence_triplet(coarse, medium, fine, mask)
    assert result["valid_point_count"] == 3
    assert result["valid_mask_fraction"] == .5
    for pair, difference in (("coarse_medium", 2.), ("medium_fine", 1.), ("coarse_fine", 3.)):
        metrics = result["pairwise"][pair]
        assert metrics["absolute_l2"] == pytest.approx(difference*np.sqrt(3))
        assert metrics["rms_l2"] == pytest.approx(difference)
        assert metrics["mean_absolute_error"] == difference
        assert metrics["maximum_absolute_error"] == difference
    assert result["finest_reference"]["coarse"]["relative_l2_error"] == 3.
    assert result["finest_reference"]["medium"]["relative_l2_error"] == 1.
    reference = result["finest_reference"]["fine"]
    assert reference["relative_l2_error"] == 0.
    assert reference["is_reference"] and not reference["independent_datapoint"]


def test_zero_differences_and_zero_reference_are_strict_json_serializable():
    result = analyze_convergence_triplet(*(np.zeros((2, 2)) for _ in range(3)))
    assert result["classification"] == "approximately unchanged / inconclusive"
    assert result["observed_order"] is None
    assert result["difference_vector_cosine"] is None
    assert result["successive_difference_ratio"] is None
    assert result["finest_reference"]["coarse"]["relative_l2_error"] is None
    assert result["finest_reference"]["fine"]["relative_l2_error"] == 0.
    json.dumps(result, allow_nan=False)


def test_zero_fine_medium_difference_does_not_imply_infinite_order():
    result = analyze_convergence_triplet(
        np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
        h_values=(1., .5, .25), scalar_h_justified=True)
    assert result["observed_order"] is None
    assert result["classification"] == "approximately unchanged / inconclusive"


@pytest.mark.parametrize("h_values", [
    (1., .5), (1., .5, .5), (1., 2., .5), (1., .5, 0.), (1., .5, np.nan),
])
def test_invalid_mesh_lengths_rejected(h_values):
    with pytest.raises(ValueError, match="strictly decreasing"):
        solve_generalized_observed_order(h_values, 4.)


def test_no_positive_order_root_rejected():
    with pytest.raises(ValueError, match="No positive"):
        solve_generalized_observed_order((1., .5, .25), .8)
    with pytest.raises(ValueError, match="No positive"):
        solve_generalized_observed_order((1., .5, .25), 2.**40)


def test_solver_works_in_log_space_for_small_lengths():
    assert solve_generalized_observed_order((1e-100, 5e-101, 2.5e-101), 8.) == pytest.approx(3.)


@pytest.mark.parametrize("kwargs", [
    {"relative_change_tolerance": 0.}, {"relative_change_tolerance": 1.},
    {"alignment_threshold": 1.1}, {"absolute_difference_tolerance": -1.},
    {"external_order_rejection_reasons": "time mismatch"},
    {"external_order_rejection_reasons": [""]},
])
def test_invalid_criteria_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze_convergence_triplet(*(np.zeros((2, 2)) for _ in range(3)), **kwargs)


def test_empty_common_mask_and_inconsistent_shapes_rejected():
    with pytest.raises(ValueError, match="No common"):
        analyze_convergence_triplet(np.ones((2, 2)), np.ones((2, 2)), np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="matching"):
        analyze_convergence_triplet(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 3)))
