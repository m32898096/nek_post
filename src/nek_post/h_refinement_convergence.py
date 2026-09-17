"""Pure, conservative three-mesh convergence diagnostics on co-located fields.

All norms are unweighted Cartesian sample norms, consistent with Step 3. The
finest field is a numerical reference, never an exact solution. A decreasing
three-grid sequence is evidence of convergence-like behavior, not proof of an
asymptotic regime. Mesh geometry and time alignment must be checked by callers.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from scipy.optimize import brentq

from nek_post.comparison import compare_sampled_grids


def _validated_h(h_values: Sequence[float]) -> tuple[float, float, float]:
    h = np.asarray(h_values, dtype=float)
    if h.shape != (3,) or not np.all(np.isfinite(h)) or not h[0] > h[1] > h[2] > 0:
        raise ValueError("h_values must be three finite positive, strictly decreasing lengths.")
    return tuple(float(value) for value in h)


def _log_expm1(value: float) -> float:
    # Stable for both nearly equal scales and large trial orders.
    return math.log(math.expm1(value)) if value < 50 else value + math.log1p(-math.exp(-value))


def solve_generalized_observed_order(
    h_values: Sequence[float], difference_ratio: float, *,
    order_bounds: tuple[float, float] = (1.0e-6, 32.0),
) -> float:
    """Solve the unequal-ratio three-grid power-law equation for positive p.

    This solves ``D01/D12 = (h0**p-h1**p)/(h1**p-h2**p)`` in logarithmic
    form. It makes no equal-ratio substitution and does not establish that the
    supplied mesh family or solution differences satisfy this power-law model.
    A root outside the declared positive interval is rejected, not clipped.
    """
    h0, h1, h2 = _validated_h(h_values)
    lower, upper = order_bounds
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise ValueError("order_bounds must be finite, positive and increasing.")
    if not np.isfinite(difference_ratio) or difference_ratio <= 0:
        raise ValueError("difference_ratio must be finite and positive.")
    log_r01 = math.log(h0) - math.log(h1)
    log_r12 = math.log(h1) - math.log(h2)
    log_observed = math.log(difference_ratio)

    def residual(order: float) -> float:
        return (order * log_r12 + _log_expm1(order * log_r01)
                - _log_expm1(order * log_r12) - log_observed)

    if residual(lower) > 0 or residual(upper) < 0:
        raise ValueError("No positive observed-order root in the declared order_bounds.")
    return float(brentq(residual, lower, upper, xtol=1.0e-12, rtol=1.0e-12))


def _finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def analyze_convergence_triplet(
    coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray,
    mask: np.ndarray | None = None, *,
    h_values: Sequence[float] | None = None,
    scalar_h_justified: bool = False,
    relative_change_tolerance: float = 0.05,
    alignment_threshold: float = 0.99,
    absolute_difference_tolerance: float = 1.0e-14,
    order_bounds: tuple[float, float] = (1.0e-6, 32.0),
    external_order_rejection_reasons: Sequence[str] = (),
) -> dict:
    """Compare a coarse/medium/fine triplet on one common finite-valid mask.

    Classification uses the finest-reference RMS ratio D12/D02: below
    ``1-tolerance`` is "monotonic convergence-like", above ``1+tolerance``
    is "non-monotonic", and the band between is "approximately unchanged /
    inconclusive". Negative alignment of successive difference vectors also
    makes a sequence non-monotonic. Threshold equality remains inconclusive.
    The successive-difference ratio D12/D01 is diagnostic, not an unconditional
    classification gate: unequal mesh-spacing changes can make D12 exceed D01
    even for an exact, monotone leading-error power law.
    These are explicit diagnostic criteria, not a statistical uncertainty test.

    A conditional observed order additionally requires caller-justified scalar
    h, a monotonic convergence-like classification, non-negligible differences,
    and nearly collinear, same-signed difference vectors. For equal refinement
    ratios, near-equal successive differences also reject an order estimate.
    Unequal refinement ratios use the full nonlinear equation without requiring
    D12 < D01. The estimate does not prove that a
    common leading-error power law or an asymptotic regime has been established.
    Caller-supplied rejection reasons can independently gate time alignment,
    physics equivalence, or other prerequisites without changing mesh metadata.
    Undefined quantities are returned as None for strict JSON serialization.
    """
    if not np.isfinite(relative_change_tolerance) or not 0 < relative_change_tolerance < 1:
        raise ValueError("relative_change_tolerance must lie strictly between zero and one.")
    if not np.isfinite(alignment_threshold) or not 0 < alignment_threshold <= 1:
        raise ValueError("alignment_threshold must lie in (0, 1].")
    if not np.isfinite(absolute_difference_tolerance) or absolute_difference_tolerance < 0:
        raise ValueError("absolute_difference_tolerance must be finite and nonnegative.")
    if (isinstance(external_order_rejection_reasons, (str, bytes))
            or any(not isinstance(reason, str) or not reason.strip()
                   for reason in external_order_rejection_reasons)):
        raise ValueError("external_order_rejection_reasons must be a sequence of nonempty strings.")
    h = None if h_values is None else _validated_h(h_values)
    grids = {"coarse": coarse, "medium": medium, "fine": fine}
    # The shared kernel can subtract Inf-Inf outside its finite-valid mask;
    # those points never enter a metric and are intentionally excluded here.
    with np.errstate(invalid="ignore"):
        comparison = compare_sampled_grids(grids, "fine", valid_mask=mask)
    common = comparison.common_mask
    count = int(common.sum())
    arrays = {case: np.asarray(values, dtype=float) for case, values in grids.items()}
    pairwise = {}
    vectors = {}
    for key, first, second in (("coarse_medium", "coarse", "medium"),
                               ("medium_fine", "medium", "fine"),
                               ("coarse_fine", "coarse", "fine")):
        # Reuse established Step 3 MAE/Linf semantics, retaining the same mask
        # for every pair instead of allowing pair-specific finite intersections.
        with np.errstate(invalid="ignore"):
            pair_result = compare_sampled_grids(
                {first: arrays[first], second: arrays[second]}, second, valid_mask=common)
        metrics = pair_result.error_rows[0]
        difference = arrays[first][common] - arrays[second][common]
        norm = float(np.linalg.norm(difference))
        vectors[key] = difference
        pairwise[key] = {
            "absolute_l2": norm,
            "rms_l2": norm / math.sqrt(count),
            "mean_absolute_error": metrics["mean_absolute_error"],
            "maximum_absolute_error": metrics["maximum_absolute_error"],
        }
    d01, d12, d02 = (pairwise[key]["rms_l2"]
                     for key in ("coarse_medium", "medium_fine", "coarse_fine"))
    if not np.all(np.isfinite([d01, d12, d02])):
        raise ValueError("Difference norms overflowed; rescale the fields before analysis.")
    negligible = min(d01, d12, d02) <= absolute_difference_tolerance
    successive_ratio = d12 / d01 if d01 > absolute_difference_tolerance else None
    reference_ratio = d12 / d02 if d02 > absolute_difference_tolerance else None
    cosine = None
    if min(d01, d12) > absolute_difference_tolerance:
        # Unit-vector products avoid overflow in the product of two norms.
        cosine = float(np.clip(np.dot(
            vectors["coarse_medium"] / pairwise["coarse_medium"]["absolute_l2"],
            vectors["medium_fine"] / pairwise["medium_fine"]["absolute_l2"]), -1., 1.))

    classification = "approximately unchanged / inconclusive"
    reasons = []
    if reference_ratio is not None and reference_ratio > 1 + relative_change_tolerance:
        classification = "non-monotonic"
        reasons.append("The finest-reference RMS error ratio exceeds the upper tolerance bound.")
    if cosine is not None and cosine < 0:
        classification = "non-monotonic"
        reasons.append("Successive difference vectors have negative alignment (oscillatory/cancelling changes).")
    if classification != "non-monotonic":
        if negligible:
            reasons.append("At least one difference is zero or below the absolute tolerance.")
        elif reference_ratio < 1 - relative_change_tolerance:
            classification = "monotonic convergence-like"
            reasons.append("The finest-reference RMS error decreases beyond tolerance with nonnegative difference-vector alignment.")
        else:
            reasons.append("The finest-reference RMS error ratio is within the unchanged/inconclusive band.")

    order_rejections = list(external_order_rejection_reasons)
    if not scalar_h_justified:
        order_rejections.append("A scalar h for a common refinement family has not been independently justified.")
    if h is None:
        order_rejections.append("No measured scalar mesh-resolution triplet was supplied.")
    if classification != "monotonic convergence-like":
        order_rejections.append("The sequence does not meet the monotonic convergence-like criteria.")
    if negligible:
        order_rejections.append("Differences are insufficiently separated from the absolute tolerance.")
    if cosine is None or cosine < alignment_threshold:
        order_rejections.append("Successive difference vectors do not meet the common leading-error alignment threshold.")
    equal_ratio_relative_tolerance = 1.0e-8
    equal_refinement_ratios = None if h is None else bool(np.isclose(
        h[0] / h[1], h[1] / h[2], rtol=equal_ratio_relative_tolerance, atol=0.))
    if (equal_refinement_ratios and successive_ratio is not None
            and abs(successive_ratio - 1.) <= relative_change_tolerance):
        order_rejections.append("Near-equal successive differences on an equal-ratio family do not support a separated positive order estimate.")
    observed_order = None
    if not order_rejections:
        try:
            observed_order = solve_generalized_observed_order(
                h, d01 / d12, order_bounds=order_bounds)
        except ValueError as error:
            order_rejections.append(str(error))

    return {
        "classification": classification,
        "classification_reasons": reasons,
        "pairwise": pairwise,
        "finest_reference": {
            row["case"]: {key: _finite_or_none(value) for key, value in row.items()}
            for row in comparison.error_rows
        },
        "reference_interpretation": "finest available numerical reference, not an exact solution",
        "valid_point_count": count,
        "total_grid_point_count": int(common.size),
        "valid_mask_fraction": count / int(common.size),
        "successive_difference_ratio": successive_ratio,
        "finest_reference_error_ratio": reference_ratio,
        "difference_vector_cosine": cosine,
        "observed_order": observed_order,
        "order_status": "conditional_power_law_estimate" if observed_order is not None else "not_reported",
        "order_rejection_reasons": order_rejections,
        "order_caveat": "Three meshes alone do not establish an asymptotic regime or validate the leading-error model.",
        "h_values": None if h is None else list(h),
        "scalar_h_justified": bool(scalar_h_justified),
        "refinement_ratios": None if h is None else [h[0]/h[1], h[1]/h[2]],
        "equal_refinement_ratios": equal_refinement_ratios,
        "criteria": {
            "relative_change_tolerance": relative_change_tolerance,
            "alignment_threshold_for_order": alignment_threshold,
            "absolute_difference_tolerance": absolute_difference_tolerance,
            "equal_ratio_relative_tolerance": equal_ratio_relative_tolerance,
            "order_bounds": list(order_bounds),
            "norm_definition": "unweighted Cartesian sample norms on one all-case finite-valid mask",
            "ratios": "RMS(medium-fine)/RMS(coarse-medium) and RMS(medium-fine)/RMS(coarse-fine)",
            "classification": "finest-reference RMS ratio > 1+tol or cosine < 0: non-monotonic; finest-reference RMS ratio < 1-tol: monotonic convergence-like; otherwise inconclusive",
            "successive_ratio_role": "diagnostic only; near-equal values reject order only for equal refinement ratios, since unequal h steps need not produce decreasing successive differences",
            "order_equation": "D01/D12 = (h0**p-h1**p)/(h1**p-h2**p)",
        },
    }
