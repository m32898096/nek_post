import numpy as np
import pytest

from nek_post.comparison import (
    compare_concentration_slices,
    compare_pressure_slices,
    compare_velocity_slices,
)


CASES = ["N5", "N11"]
ORDERS = {"N5": 5, "N11": 11}
CASE_INDICES = {"N5": 79, "N11": 40}
CONCENTRATION_KEYS = {
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_C",
    "mean_abs_error_C",
    "absolute_Linf_C",
    "relative_Linf_C",
    "valid_point_count",
    "total_grid_point_count",
}
FRONT_KEYS = {"case", "order", "comparison_set", "index", "threshold", "x_front"}
VELOCITY_KEYS = {
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_speed",
    "mean_abs_error_speed",
    "absolute_Linf_speed",
    "relative_Linf_speed",
    "relative_L2_u",
    "mean_abs_error_u",
    "relative_L2_v",
    "mean_abs_error_v",
    "relative_L2_w",
    "mean_abs_error_w",
    "valid_point_count",
    "total_grid_point_count",
}
PRESSURE_KEYS = {
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_p_prime",
    "mean_abs_error_p_prime",
    "absolute_Linf_p_prime",
    "relative_Linf_p_prime",
    "valid_point_count",
    "total_grid_point_count",
}


def _grid() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(0.0, 1.0, 3)
    return np.meshgrid(coordinates, coordinates)


def _points() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])


def _slice(x: np.ndarray, z: np.ndarray, **fields: np.ndarray) -> dict[str, np.ndarray]:
    return {"x": x, "z": z, **fields}


def _common_arguments(Xi: np.ndarray, Zi: np.ndarray) -> dict[str, object]:
    return {
        "cases": CASES,
        "orders": ORDERS,
        "case_indices": CASE_INDICES,
        "reference_case": "N11",
        "comparison_set_name": "synthetic",
        "Xi": Xi,
        "Zi": Zi,
        "interpolation_method": "linear",
        "duplicate_decimals": 10,
    }


def test_concentration_interpolation_rows_threshold_and_reference_behavior() -> None:
    x, z = _points()
    Xi, Zi = _grid()
    slices = {
        "N5": _slice(x, z, C=1.0 + 1.5 * x + z),
        "N11": _slice(x, z, C=2.0 + x + 2.0 * z),
    }

    result = compare_concentration_slices(
        slice_data_by_case=slices,
        front_threshold_ratio=0.25,
        **_common_arguments(Xi, Zi),
    )

    assert np.allclose(result.grids["N11"], 2.0 + Xi + 2.0 * Zi)
    assert len(result.error_rows) == 1
    assert result.error_rows[0]["case"] == "N5"
    assert set(result.error_rows[0]) == CONCENTRATION_KEYS
    assert {row["case"] for row in result.front_rows} == set(CASES)
    assert all(set(row) == FRONT_KEYS for row in result.front_rows)
    expected_threshold = 0.25 * np.max(result.grids["N11"][result.common_mask])
    assert result.threshold == expected_threshold
    assert all(row["threshold"] == expected_threshold for row in result.front_rows)


def test_velocity_speed_schema_mask_and_safe_component_metric() -> None:
    x, z = _points()
    Xi, Zi = _grid()
    triangle = np.array([0, 1, 2])
    slices = {
        "N5": _slice(
            x[triangle],
            z[triangle],
            u=(x + z)[triangle],
            v=(2.0 + x)[triangle],
            w=(3.0 + z)[triangle],
        ),
        "N11": _slice(
            x,
            z,
            u=np.zeros_like(x),
            v=2.0 + 0.5 * x,
            w=3.0 + 0.5 * z,
        ),
    }

    result = compare_velocity_slices(slice_data_by_case=slices, **_common_arguments(Xi, Zi))
    row = result.error_rows[0]

    for grids in result.grids.values():
        assert np.allclose(
            grids["speed"],
            np.sqrt(grids["u"] ** 2 + grids["v"] ** 2 + grids["w"] ** 2),
            equal_nan=True,
        )
    expected_mask = np.isfinite(result.grids["N5"]["speed"]) & np.isfinite(result.grids["N11"]["speed"])
    assert np.array_equal(result.common_mask, expected_mask)
    expected_u_mae = np.mean(np.abs(result.grids["N5"]["u"][expected_mask]))
    assert row["mean_abs_error_u"] == pytest.approx(expected_u_mae)
    assert np.isnan(row["relative_L2_u"])
    assert row["case"] == "N5"
    assert set(row) == VELOCITY_KEYS


def test_pressure_fluctuation_schema_and_safe_relative_metrics() -> None:
    x, z = _points()
    Xi, Zi = _grid()
    slices = {
        "N5": _slice(x, z, p=8.0 + x + 2.0 * z),
        "N11": _slice(x, z, p=np.full_like(x, 7.0)),
    }

    result = compare_pressure_slices(slice_data_by_case=slices, **_common_arguments(Xi, Zi))
    row = result.error_rows[0]

    for grids in result.grids.values():
        assert np.mean(grids["p_prime"][result.common_mask]) == pytest.approx(0.0, abs=1.0e-15)
        assert np.array_equal(np.isnan(grids["p_prime"]), np.isnan(grids["p"]))
    assert set(row) == PRESSURE_KEYS
    assert row["case"] == "N5"
    assert np.isnan(row["relative_L2_p_prime"])
    assert np.isnan(row["relative_Linf_p_prime"])


@pytest.mark.parametrize(
    ("comparison", "extra", "message"),
    [
        (compare_concentration_slices, {"front_threshold_ratio": 0.01}, "No finite common grid points"),
        (compare_velocity_slices, {}, "No finite common grid points.*velocity"),
        (compare_pressure_slices, {}, "No finite common grid points.*pressure"),
    ],
)
def test_comparisons_fail_without_finite_common_points(comparison, extra, message: str) -> None:
    x, z = _points()
    outside = np.linspace(2.0, 3.0, 2)
    Xi, Zi = np.meshgrid(outside, outside)
    fields = {
        "C": 1.0 + x + z,
        "u": 1.0 + x,
        "v": 2.0 + z,
        "w": 3.0 + x,
        "p": 4.0 + x + z,
    }
    slices = {case: _slice(x, z, **fields) for case in CASES}

    with pytest.raises(ValueError, match=message):
        comparison(slice_data_by_case=slices, **_common_arguments(Xi, Zi), **extra)


def test_comparison_rejects_unknown_reference_case() -> None:
    x, z = _points()
    Xi, Zi = _grid()
    slices = {case: _slice(x, z, C=1.0 + x + z) for case in CASES}
    arguments = _common_arguments(Xi, Zi)
    arguments["reference_case"] = "N9"

    with pytest.raises(ValueError, match="Reference case 'N9' is not present"):
        compare_concentration_slices(
            slice_data_by_case=slices,
            front_threshold_ratio=0.01,
            **arguments,
        )
