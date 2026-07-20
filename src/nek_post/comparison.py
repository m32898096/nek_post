"""Pure numerical kernels for polynomial-order slice comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nek_post.interpolation import interpolate_to_grid, valid_common_mask
from nek_post.metrics import (
    absolute_linf_error,
    front_position,
    mean_absolute_error,
    relative_l2_error,
    relative_linf_error,
)


Array = np.ndarray
SliceData = Mapping[str, Array]
ComparisonRow = dict[str, object]


@dataclass(frozen=True)
class ConcentrationComparisonResult:
    """Interpolated concentration grids, their common mask, and output rows."""

    grids: dict[str, Array]
    common_mask: Array
    valid_point_count: int
    total_grid_point_count: int
    error_rows: list[ComparisonRow]
    front_rows: list[ComparisonRow]
    threshold: float


@dataclass(frozen=True)
class VelocityComparisonResult:
    """Interpolated component/speed grids, their speed-based mask, and error rows."""

    grids: dict[str, dict[str, Array]]
    common_mask: Array
    valid_point_count: int
    total_grid_point_count: int
    error_rows: list[ComparisonRow]


@dataclass(frozen=True)
class PressureComparisonResult:
    """Interpolated raw/fluctuation pressure grids, their mask, and error rows."""

    grids: dict[str, dict[str, Array]]
    common_mask: Array
    valid_point_count: int
    total_grid_point_count: int
    error_rows: list[ComparisonRow]


def _validate_inputs(
    slice_data_by_case: Mapping[str, SliceData],
    cases: Sequence[str],
    orders: Mapping[str, int],
    case_indices: Mapping[str, int],
    reference_case: str,
) -> None:
    if reference_case not in cases:
        raise ValueError(f"Reference case {reference_case!r} is not present in cases.")

    for case in cases:
        if case not in slice_data_by_case:
            raise ValueError(f"Missing slice data for case {case!r}.")
        if case not in orders:
            raise ValueError(f"Missing polynomial order for case {case!r}.")
        if case not in case_indices:
            raise ValueError(f"Missing file index for case {case!r}.")


def _safe_relative_l2_error(
    values: Array,
    reference: Array,
    mask: Array | None = None,
    denominator_tol: float = 1.0e-14,
) -> float:
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(values_arr) & np.isfinite(ref_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(values_arr) & np.isfinite(ref_arr)

    if not np.any(mask_arr):
        return float("nan")

    denominator = float(np.sum(ref_arr[mask_arr] ** 2))
    if denominator <= denominator_tol:
        return float("nan")

    numerator = float(np.sum((values_arr[mask_arr] - ref_arr[mask_arr]) ** 2))
    return float(np.sqrt(numerator / denominator))


def _safe_relative_linf_error(
    values: Array,
    reference: Array,
    mask: Array | None = None,
    denominator_tol: float = 1.0e-14,
) -> float:
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(values_arr) & np.isfinite(ref_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(values_arr) & np.isfinite(ref_arr)

    if not np.any(mask_arr):
        return float("nan")

    denominator = float(np.max(np.abs(ref_arr[mask_arr])))
    if denominator <= denominator_tol:
        return float("nan")

    return float(np.max(np.abs(values_arr[mask_arr] - ref_arr[mask_arr])) / denominator)


def _pressure_fluctuation(p_grid: Array, mask: Array) -> Array:
    p_arr = np.asarray(p_grid, dtype=float)
    p_prime = p_arr.copy()
    p_prime -= float(np.mean(p_arr[mask]))
    p_prime[~np.isfinite(p_arr)] = np.nan
    return p_prime


def _interpolate_field(
    slice_data_by_case: Mapping[str, SliceData],
    cases: Sequence[str],
    field: str,
    Xi: Array,
    Zi: Array,
    interpolation_method: str,
    duplicate_decimals: int,
) -> dict[str, Array]:
    grids: dict[str, Array] = {}
    for case in cases:
        slice_data = slice_data_by_case[case]
        grids[case] = interpolate_to_grid(
            slice_data["x"],
            slice_data["z"],
            slice_data[field],
            Xi,
            Zi,
            method=interpolation_method,
            duplicate_decimals=duplicate_decimals,
        )
    return grids


def compare_concentration_slices(
    slice_data_by_case: Mapping[str, SliceData],
    cases: Sequence[str],
    orders: Mapping[str, int],
    case_indices: Mapping[str, int],
    reference_case: str,
    comparison_set_name: str | None,
    Xi: Array,
    Zi: Array,
    interpolation_method: str,
    duplicate_decimals: int,
    front_threshold_ratio: float,
) -> ConcentrationComparisonResult:
    """Interpolate concentration and compare every non-reference case.

    Inputs provide per-case scattered slices, case metadata, the common target
    grid, and interpolation settings; no configuration or files are accessed.
    The returned common mask is finite for every case. Error rows omit the
    reference case, while front rows include all cases on the shared threshold.
    """
    _validate_inputs(slice_data_by_case, cases, orders, case_indices, reference_case)
    grids = _interpolate_field(
        slice_data_by_case,
        cases,
        "C",
        Xi,
        Zi,
        interpolation_method,
        duplicate_decimals,
    )
    common_mask = valid_common_mask(*(grids[case] for case in cases))
    valid_point_count = int(np.count_nonzero(common_mask))
    total_grid_point_count = int(common_mask.size)
    if valid_point_count == 0:
        raise ValueError("No finite common grid points are available for comparison.")

    reference_grid = grids[reference_case]
    error_rows: list[ComparisonRow] = []
    for case in cases:
        if case == reference_case:
            continue
        error_rows.append(
            {
                "case": case,
                "order": orders[case],
                "reference_case": reference_case,
                "comparison_set": comparison_set_name or "",
                "index": case_indices[case],
                "reference_index": case_indices[reference_case],
                "relative_L2_C": relative_l2_error(grids[case], reference_grid, common_mask),
                "mean_abs_error_C": mean_absolute_error(grids[case] - reference_grid, common_mask),
                "absolute_Linf_C": absolute_linf_error(grids[case], reference_grid, common_mask),
                "relative_Linf_C": relative_linf_error(grids[case], reference_grid, common_mask),
                "valid_point_count": valid_point_count,
                "total_grid_point_count": total_grid_point_count,
            }
        )

    threshold = float(front_threshold_ratio * np.max(reference_grid[common_mask]))
    front_rows = [
        {
            "case": case,
            "order": orders[case],
            "comparison_set": comparison_set_name or "",
            "index": case_indices[case],
            "threshold": threshold,
            "x_front": front_position(Xi, grids[case], threshold, common_mask),
        }
        for case in cases
    ]
    return ConcentrationComparisonResult(
        grids=grids,
        common_mask=common_mask,
        valid_point_count=valid_point_count,
        total_grid_point_count=total_grid_point_count,
        error_rows=error_rows,
        front_rows=front_rows,
        threshold=threshold,
    )


def compare_velocity_slices(
    slice_data_by_case: Mapping[str, SliceData],
    cases: Sequence[str],
    orders: Mapping[str, int],
    case_indices: Mapping[str, int],
    reference_case: str,
    comparison_set_name: str | None,
    Xi: Array,
    Zi: Array,
    interpolation_method: str,
    duplicate_decimals: int,
) -> VelocityComparisonResult:
    """Interpolate velocity components and compare non-reference speed grids.

    Inputs provide per-case scattered slices, case metadata, the common target
    grid, and interpolation settings; no configuration or files are accessed.
    Component metrics use the same mask that is finite for every speed grid.
    Returned grids retain ``u``, ``v``, ``w``, and derived ``speed`` arrays.
    """
    _validate_inputs(slice_data_by_case, cases, orders, case_indices, reference_case)
    grids: dict[str, dict[str, Array]] = {}
    for case in cases:
        slice_data = slice_data_by_case[case]
        u_grid = interpolate_to_grid(
            slice_data["x"],
            slice_data["z"],
            slice_data["u"],
            Xi,
            Zi,
            method=interpolation_method,
            duplicate_decimals=duplicate_decimals,
        )
        v_grid = interpolate_to_grid(
            slice_data["x"],
            slice_data["z"],
            slice_data["v"],
            Xi,
            Zi,
            method=interpolation_method,
            duplicate_decimals=duplicate_decimals,
        )
        w_grid = interpolate_to_grid(
            slice_data["x"],
            slice_data["z"],
            slice_data["w"],
            Xi,
            Zi,
            method=interpolation_method,
            duplicate_decimals=duplicate_decimals,
        )
        grids[case] = {
            "u": u_grid,
            "v": v_grid,
            "w": w_grid,
            "speed": np.sqrt(u_grid**2 + v_grid**2 + w_grid**2),
        }

    common_mask = valid_common_mask(*(grids[case]["speed"] for case in cases))
    valid_point_count = int(np.count_nonzero(common_mask))
    total_grid_point_count = int(common_mask.size)
    if valid_point_count == 0:
        raise ValueError("No finite common grid points are available for velocity comparison.")

    reference_grids = grids[reference_case]
    reference_speed = reference_grids["speed"]
    error_rows: list[ComparisonRow] = []
    for case in cases:
        if case == reference_case:
            continue
        case_grids = grids[case]
        error_rows.append(
            {
                "case": case,
                "order": orders[case],
                "reference_case": reference_case,
                "comparison_set": comparison_set_name or "",
                "index": case_indices[case],
                "reference_index": case_indices[reference_case],
                "relative_L2_speed": relative_l2_error(case_grids["speed"], reference_speed, common_mask),
                "mean_abs_error_speed": mean_absolute_error(case_grids["speed"] - reference_speed, common_mask),
                "absolute_Linf_speed": absolute_linf_error(case_grids["speed"], reference_speed, common_mask),
                "relative_Linf_speed": relative_linf_error(case_grids["speed"], reference_speed, common_mask),
                "relative_L2_u": _safe_relative_l2_error(case_grids["u"], reference_grids["u"], common_mask),
                "mean_abs_error_u": mean_absolute_error(case_grids["u"] - reference_grids["u"], common_mask),
                "relative_L2_v": _safe_relative_l2_error(case_grids["v"], reference_grids["v"], common_mask),
                "mean_abs_error_v": mean_absolute_error(case_grids["v"] - reference_grids["v"], common_mask),
                "relative_L2_w": _safe_relative_l2_error(case_grids["w"], reference_grids["w"], common_mask),
                "mean_abs_error_w": mean_absolute_error(case_grids["w"] - reference_grids["w"], common_mask),
                "valid_point_count": valid_point_count,
                "total_grid_point_count": total_grid_point_count,
            }
        )

    return VelocityComparisonResult(
        grids=grids,
        common_mask=common_mask,
        valid_point_count=valid_point_count,
        total_grid_point_count=total_grid_point_count,
        error_rows=error_rows,
    )


def compare_pressure_slices(
    slice_data_by_case: Mapping[str, SliceData],
    cases: Sequence[str],
    orders: Mapping[str, int],
    case_indices: Mapping[str, int],
    reference_case: str,
    comparison_set_name: str | None,
    Xi: Array,
    Zi: Array,
    interpolation_method: str,
    duplicate_decimals: int,
) -> PressureComparisonResult:
    """Interpolate pressure and compare non-reference pressure fluctuations.

    Inputs provide per-case scattered slices, case metadata, the common target
    grid, and interpolation settings; no configuration or files are accessed.
    Raw pressure grids define the shared finite mask. Each returned ``p_prime``
    grid subtracts its own mean over that mask and preserves raw-pressure NaNs.
    """
    _validate_inputs(slice_data_by_case, cases, orders, case_indices, reference_case)
    raw_grids = _interpolate_field(
        slice_data_by_case,
        cases,
        "p",
        Xi,
        Zi,
        interpolation_method,
        duplicate_decimals,
    )
    common_mask = valid_common_mask(*(raw_grids[case] for case in cases))
    valid_point_count = int(np.count_nonzero(common_mask))
    total_grid_point_count = int(common_mask.size)
    if valid_point_count == 0:
        raise ValueError("No finite common grid points are available for pressure comparison.")

    grids = {
        case: {"p": raw_grids[case], "p_prime": _pressure_fluctuation(raw_grids[case], common_mask)}
        for case in cases
    }
    reference_p_prime = grids[reference_case]["p_prime"]
    error_rows: list[ComparisonRow] = []
    for case in cases:
        if case == reference_case:
            continue
        p_prime = grids[case]["p_prime"]
        error_rows.append(
            {
                "case": case,
                "order": orders[case],
                "reference_case": reference_case,
                "comparison_set": comparison_set_name or "",
                "index": case_indices[case],
                "reference_index": case_indices[reference_case],
                "relative_L2_p_prime": _safe_relative_l2_error(p_prime, reference_p_prime, common_mask),
                "mean_abs_error_p_prime": mean_absolute_error(p_prime - reference_p_prime, common_mask),
                "absolute_Linf_p_prime": absolute_linf_error(p_prime, reference_p_prime, common_mask),
                "relative_Linf_p_prime": _safe_relative_linf_error(p_prime, reference_p_prime, common_mask),
                "valid_point_count": valid_point_count,
                "total_grid_point_count": total_grid_point_count,
            }
        )

    return PressureComparisonResult(
        grids=grids,
        common_mask=common_mask,
        valid_point_count=valid_point_count,
        total_grid_point_count=total_grid_point_count,
        error_rows=error_rows,
    )
