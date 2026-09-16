"""Directional physical mesh evidence for h-refinement studies.

Only complete, affine, axis-aligned tensor partitions are certified.  The
geometry validators and coordinate-storage tolerances are the existing GLL
directional-integration implementation; element counts alone never define h.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from nek_post.gll_directional_integration import (
    FLOAT32_STORAGE_ULP_FACTOR,
    FLOAT64_AFFINE_ROUNDOFF_FACTOR,
    MIN_AFFINE_SPAN_TO_TOLERANCE_RATIO,
    _cluster_intervals,
    _detect_geometry_mapping,
    _validate_complete_topology,
    _validated_geometry,
)
from nek_post.io_nek import read_nek_file, read_nek_header


def _summary(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "min": float(np.min(values)), "max": float(np.max(values)),
        "mean": float(np.mean(values)), "median": float(np.median(values)),
    }
    if weights is not None:
        result["length_weighted_mean"] = float(np.average(values, weights=weights))
    return result


def analyze_mesh(data: object) -> dict[str, Any]:
    """Certify a physical tensor partition and report its directional widths.

    The representative width is the median of unique physical intervals, not
    a GLL-node distance. Geometry is validated in bounded batches to avoid a
    second full float64 coordinate copy for large four-byte Nek snapshots.
    Unsupported/curved/incomplete meshes raise ValueError rather than having
    their bounding boxes silently interpreted as physical element widths.
    """
    try:
        elements = tuple(data.elem)  # type: ignore[attr-defined]
    except (TypeError, AttributeError) as exc:
        raise ValueError("Mesh must provide an iterable elem collection.") from exc
    if not elements:
        raise ValueError("Mesh has no elements.")
    bounds = np.empty((len(elements), 3, 2), dtype=float)
    tolerances = np.empty((len(elements), 3), dtype=float)
    mapping = shape = None
    for start in range(0, len(elements), 512):
        batch = elements[start:start + 512]
        _, geometry, batch_shape, _ = _validated_geometry(SimpleNamespace(elem=batch))
        batch_mapping, _, _, batch_bounds, batch_tolerances = _detect_geometry_mapping(
            geometry, batch_shape)
        if shape is not None and shape != batch_shape:
            raise ValueError("Element-local shapes vary across the mesh.")
        if mapping is not None and mapping != batch_mapping:
            raise ValueError("Physical/reference-axis mapping varies across the mesh.")
        shape, mapping = batch_shape, batch_mapping
        bounds[start:start + len(batch)] = batch_bounds
        tolerances[start:start + len(batch)] = batch_tolerances
    assert shape is not None and mapping is not None
    axis_tolerances = np.max(tolerances, axis=0)
    cells = np.empty((len(elements), 3), dtype=np.int64)
    directional: dict[str, Any] = {}
    for i, axis in enumerate("xyz"):
        intervals, cells[:, i] = _cluster_intervals(bounds[:, i], axis_tolerances[i], axis)
        widths = intervals[:, 1] - intervals[:, 0]
        spacing = _summary(widths, widths)
        directional[axis] = {
            "element_count": len(intervals),
            "domain_bounds": [float(intervals[0, 0]), float(intervals[-1, 1])],
            "element_intervals": intervals.tolist(), "element_widths": widths.tolist(),
            "spacing": spacing,
            "representative_spacing": spacing["median"],
            "representative_definition": "median of unique physical element intervals",
            "coordinate_tolerance": float(axis_tolerances[i]),
            "gll_nodes_per_element_direction": int(shape[mapping[i]]),
            "polynomial_order": int(shape[mapping[i]] - 1),
            "uniform_spacing_within_storage_tolerance": bool(
                np.ptp(widths) <= 4 * axis_tolerances[i]),
        }
    counts = tuple(directional[a]["element_count"] for a in "xyz")
    _validate_complete_topology(cells, counts)
    widths = bounds[:, :, 1] - bounds[:, :, 0]
    aspect = np.max(widths, axis=1) / np.min(widths, axis=1)
    # Comparing widths in different physical directions uses both independent
    # endpoint uncertainty budgets. This describes element shape, separately
    # from whether successive meshes refine isotropically.
    isotropic_elements = np.ptp(widths, axis=1) <= 4 * np.max(tolerances, axis=1)
    return {
        "schema_version": 1,
        "geometry_model": "complete axis-aligned affine tensor partition",
        "number_of_elements": len(elements), "element_local_shape": list(shape),
        "physical_to_reference_axes": dict(zip("xyz", mapping)),
        "directional": directional,
        "complete_tensor_tiling_verified": True,
        "directional_count_product": int(np.prod(counts)),
        "element_aspect_ratio": _summary(aspect),
        "all_elements_physically_isotropic": bool(np.all(isotropic_elements)),
        "tolerances": {
            "source": "existing gll_directional_integration geometry validators",
            "float32_storage_ulp_factor": FLOAT32_STORAGE_ULP_FACTOR,
            "float64_roundoff_factor": FLOAT64_AFFINE_ROUNDOFF_FACTOR,
            "minimum_span_to_tolerance_ratio": MIN_AFFINE_SPAN_TO_TOLERANCE_RATIO,
            "interpretation": "Affine/axis-alignment certified only within coordinate storage uncertainty.",
        },
    }


def read_mesh_report(path: str | Path) -> dict[str, Any]:
    """Read one coordinate-bearing unsplit snapshot through project readers."""
    header = read_nek_header(path)
    if header.nb_files != 1 or header.nb_elems_file != header.nb_elems:
        raise ValueError("Split-file meshes are not supported.")
    if header.nb_dims != 3 or header.nb_vars[0] != 3:
        raise ValueError("Mesh analysis requires stored three-dimensional coordinates.")
    skip = ("ux", "uy", "uz", "pressure", "temperature") + tuple(
        f"s{i:02d}" for i in range(1, header.nb_vars[4] + 1))
    data = read_nek_file(path, skip_vars=skip)
    if len(data.elem) != header.nb_elems:
        raise ValueError("Mesh header and payload element counts disagree.")
    report = analyze_mesh(data)
    report.update(source_file=str(Path(path)), source_time=float(header.time),
                  scope="Every element and coordinate in this snapshot; stationary mesh assumed.")
    return report


def _overlay_ratios(coarse: Mapping[str, Any], fine: Mapping[str, Any]) -> dict[str, Any]:
    a = np.asarray(coarse["element_intervals"], dtype=float)
    b = np.asarray(fine["element_intervals"], dtype=float)
    ct, ft = float(coarse["coordinate_tolerance"]), float(fine["coordinate_tolerance"])
    tolerance = ct + ft
    if not np.allclose(a[[0, -1], [0, 1]], b[[0, -1], [0, 1]], rtol=0, atol=tolerance):
        raise ValueError("Mesh physical domains differ; directional ratios are undefined.")
    segments = []
    i = j = 0
    while i < len(a) and j < len(b):
        lower, upper = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if upper - lower > tolerance:
            cw, fw = a[i, 1] - a[i, 0], b[j, 1] - b[j, 0]
            ratio = cw / fw
            # Each width subtracts two independently rounded coordinates;
            # retain an additional factor two as in the geometry validator.
            uncertainty = ratio * (4 * ct / cw + 4 * ft / fw)
            segments.append({"lower": float(lower), "upper": float(upper),
                             "coarse_width": float(cw), "fine_width": float(fw),
                             "ratio": float(ratio), "ratio_uncertainty": float(uncertainty)})
        if a[i, 1] < b[j, 1] - tolerance:
            i += 1
        elif b[j, 1] < a[i, 1] - tolerance:
            j += 1
        else:
            i += 1
            j += 1
    if not segments:
        raise ValueError("No physical overlap available for local refinement ratios.")
    ratios = np.asarray([s["ratio"] for s in segments])
    lengths = np.asarray([s["upper"] - s["lower"] for s in segments])
    uncertainty = np.asarray([s["ratio_uncertainty"] for s in segments])
    count_ratio = len(b) / len(a)
    constant = bool(np.all(np.abs(ratios - count_ratio) <= uncertainty + 1e-12))
    return {
        "segments": segments, "ratio_summary": _summary(ratios, lengths),
        "directional_count_ratio": float(count_ratio),
        "constant_within_storage_tolerance": constant,
        "strictly_refined_everywhere": bool(np.all(ratios - uncertainty > 1)),
        "no_coarsening_within_storage_tolerance": bool(np.all(ratios + uncertainty >= 1)),
        "maximum_ratio_uncertainty": float(np.max(uncertainty)),
        "matched_physical_length": float(np.sum(lengths)),
        "matched_physical_fraction": float(np.sum(lengths) / (a[-1, 1] - a[0, 0])),
        "endpoint_matching_tolerance": tolerance,
        "matching_method": "overlap of coarse/fine physical element intervals at the same coordinate",
    }


def analyze_mesh_refinement(mesh_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare successive meshes in mapping order without assuming their names.

    A scalar h is eligible only for one common isotropic spatial scaling:
    locally constant ratios in every physical direction, equal directional
    ratios for each transition, and fixed polynomial order/domain topology.
    This geometric eligibility alone does not prove asymptotic convergence.
    """
    cases = list(mesh_reports)
    if len(cases) < 2:
        raise ValueError("At least two mesh reports are required.")
    if any(not report.get("complete_tensor_tiling_verified") for report in mesh_reports.values()):
        raise ValueError("All meshes must have certified complete tensor tiling.")
    transitions = []
    reasons = []
    orders = [tuple(mesh_reports[c]["directional"][axis]["polynomial_order"] for axis in "xyz")
              for c in cases]
    if len(set(orders)) != 1:
        reasons.append("Polynomial orders differ between meshes.")
    for coarse, fine in zip(cases, cases[1:]):
        directional = {axis: _overlay_ratios(mesh_reports[coarse]["directional"][axis],
                                             mesh_reports[fine]["directional"][axis])
                       for axis in "xyz"}
        ratios = np.asarray([directional[a]["directional_count_ratio"] for a in "xyz"])
        uncertainty = max(directional[a]["maximum_ratio_uncertainty"] for a in "xyz")
        constant = all(directional[a]["constant_within_storage_tolerance"] for a in "xyz")
        isotropic = constant and bool(np.ptp(ratios) <= 2 * uncertainty + 1e-12)
        strict = all(directional[a]["strictly_refined_everywhere"] for a in "xyz")
        if not constant:
            reasons.append(f"{coarse} -> {fine}: local ratios vary with physical position.")
        if not isotropic:
            reasons.append(f"{coarse} -> {fine}: no single isotropic directional refinement ratio.")
        if not strict:
            reasons.append(f"{coarse} -> {fine}: strict local refinement in all directions is unproven.")
        transitions.append({"coarse_case": coarse, "fine_case": fine,
                            "directional": directional, "refinement_isotropic": isotropic,
                            "strictly_refined_everywhere": strict,
                            "single_spatially_constant_ratio": constant})
    eligible = not reasons
    return {
        "case_order": cases, "transitions": transitions,
        "same_polynomial_order": len(set(orders)) == 1,
        "same_certified_box_topology": True,
        "scalar_h_justified": eligible,
        "scalar_h_rejection_reasons": reasons,
        "scalar_h_definition": "median physical x element width; other directions scale identically" if eligible else None,
        "scalar_h_by_case": {case: mesh_reports[case]["directional"]["x"]["representative_spacing"]
                              for case in cases} if eligible else None,
        "limitations": [
            "Resolution measures are element widths, not post-processing grid spacing or GLL node spacings.",
            "Geometry eligibility alone does not establish asymptotic solution convergence.",
            "Initial/boundary conditions, physical parameters, time-integration error and mesh stationarity are not verified.",
        ],
    }
