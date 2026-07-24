from __future__ import annotations

from dataclasses import fields

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import nek_post.fixed_grid_interpolation as interpolation_module
from nek_post.fixed_grid_interpolation import (
    FixedGridGeometryMismatchError,
    apply_fixed_grid_interpolation_plan,
    build_fixed_grid_interpolation_plan,
    validate_fixed_grid_source_geometry,
)
from nek_post.interpolation import interpolate_to_grid


def _square_geometry() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    )


def _target_grid() -> tuple[np.ndarray, np.ndarray]:
    xi = np.asarray([-0.1, 0.15, 0.55, 0.9, 1.1], dtype=np.float64)
    zi = np.asarray([-0.1, 0.2, 0.65, 1.1], dtype=np.float64)
    return np.meshgrid(xi, zi)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_reusable_plan_matches_griddata_for_stationary_geometry(
    method: str,
) -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    values = 2.0 * x - 0.5 * z + 3.0

    plan = build_fixed_grid_interpolation_plan(
        x, z, Xi, Zi, method=method
    )
    actual = apply_fixed_grid_interpolation_plan(plan, x, z, values)
    expected = interpolate_to_grid(x, z, values, Xi, Zi, method=method)

    assert_array_equal(np.isnan(actual), np.isnan(expected))
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.dtype == np.dtype("float64")


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_duplicate_values_are_averaged_like_existing_interpolation(
    method: str,
) -> None:
    x = np.asarray([0.0, 4.0e-11, 1.0, 0.0, 1.0])
    z = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    values = np.asarray([1.0, 3.0, 4.0, 6.0, 8.0])
    Xi, Zi = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 3))

    plan = build_fixed_grid_interpolation_plan(
        x, z, Xi, Zi, method=method
    )
    actual = apply_fixed_grid_interpolation_plan(plan, x, z, values)
    expected = interpolate_to_grid(x, z, values, Xi, Zi, method=method)

    assert_array_equal(plan.duplicate_group_counts, [2, 1, 1, 1])
    assert plan.deduplicated_x[0] == 2.0e-11
    assert_array_equal(np.isnan(actual), np.isnan(expected))
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_nonfinite_duplicate_value_does_not_poison_finite_group_value(
    method: str,
) -> None:
    x = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, np.nan])
    z = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    values = np.asarray([np.nan, 2.0, 4.0, 6.0, 8.0, 999.0])
    Xi, Zi = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 3))

    plan = build_fixed_grid_interpolation_plan(
        x, z, Xi, Zi, method=method
    )
    actual = apply_fixed_grid_interpolation_plan(plan, x, z, values)
    expected = interpolate_to_grid(x, z, values, Xi, Zi, method=method)

    assert not plan.finite_coordinate_mask[-1]
    assert_array_equal(np.isnan(actual), np.isnan(expected))
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_group_without_finite_values_produces_nan_not_zero(
    method: str,
) -> None:
    x, z = _square_geometry()
    Xi = np.asarray([[0.0, 1.0]], dtype=np.float64)
    Zi = np.asarray([[0.0, 0.0]], dtype=np.float64)
    values = np.asarray([np.nan, 4.0, 6.0, 8.0])

    plan = build_fixed_grid_interpolation_plan(
        x, z, Xi, Zi, method=method
    )
    actual = apply_fixed_grid_interpolation_plan(plan, x, z, values)

    assert np.isnan(actual[0, 0])
    if method == "nearest":
        assert actual[0, 1] == 4.0


def test_linear_targets_outside_convex_hull_remain_nan() -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(
        x, z, Xi, Zi, method="linear"
    )

    actual = apply_fixed_grid_interpolation_plan(
        plan, x, z, np.asarray([1.0, 2.0, 3.0, 4.0])
    )

    assert_array_equal(np.isnan(actual), ~plan.valid_target_mask)
    assert np.all(np.isnan(actual[[0, -1], :]))


def test_geometry_count_mismatch_reports_expected_and_actual_counts() -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(x, z, Xi, Zi)

    with pytest.raises(
        FixedGridGeometryMismatchError, match=r"expected 4, got 3"
    ):
        validate_fixed_grid_source_geometry(plan, x[:-1], z[:-1])


def test_geometry_coordinate_mismatch_reports_first_source_position() -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(x, z, Xi, Zi)
    changed_x = x.copy()
    changed_x[2] = 0.25

    with pytest.raises(
        FixedGridGeometryMismatchError,
        match=r"source position 2.*expected rounded key.*got",
    ):
        validate_fixed_grid_source_geometry(plan, changed_x, z)


def test_geometry_finite_pattern_and_raw_order_mismatches_fail() -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(x, z, Xi, Zi)
    changed_z = z.copy()
    changed_z[1] = np.nan
    with pytest.raises(
        FixedGridGeometryMismatchError,
        match=r"source position 1.*non-finite",
    ):
        validate_fixed_grid_source_geometry(plan, x, changed_z)

    order = np.asarray([1, 0, 2, 3])
    with pytest.raises(
        FixedGridGeometryMismatchError, match=r"source position 0"
    ):
        validate_fixed_grid_source_geometry(plan, x[order], z[order])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"z": np.asarray([0.0, 1.0])}, "matching"),
        ({"Xi": np.zeros(3), "Zi": np.zeros(3)}, "two-dimensional"),
        ({"Xi": np.zeros((2, 2)), "Zi": np.zeros((2, 3))}, "two-dimensional"),
        ({"x": np.asarray([np.nan]), "z": np.asarray([0.0])}, "No finite"),
        ({"method": "cubic"}, "linear.*nearest"),
        ({"duplicate_decimals": -1}, "duplicate_decimals"),
    ],
)
def test_plan_input_validation(
    updates: dict[str, object],
    message: str,
) -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    arguments: dict[str, object] = {
        "x": x,
        "z": z,
        "Xi": Xi,
        "Zi": Zi,
        "method": "linear",
        "duplicate_decimals": 10,
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=message):
        build_fixed_grid_interpolation_plan(**arguments)


def test_linear_rejects_too_few_or_collinear_unique_points() -> None:
    Xi, Zi = np.meshgrid(np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2))
    with pytest.raises(ValueError, match="three unique"):
        build_fixed_grid_interpolation_plan(
            [0.0, 1.0],
            [0.0, 0.0],
            Xi,
            Zi,
            method="linear",
        )
    with pytest.raises(ValueError, match="non-collinear"):
        build_fixed_grid_interpolation_plan(
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            Xi,
            Zi,
            method="linear",
        )


def test_values_count_mismatch_fails_before_application() -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(x, z, Xi, Zi)

    with pytest.raises(ValueError, match=r"values.*expected 4, got 3"):
        apply_fixed_grid_interpolation_plan(plan, x, z, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_plan_arrays_are_deterministic_and_read_only(method: str) -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    first = build_fixed_grid_interpolation_plan(x, z, Xi, Zi, method=method)
    second = build_fixed_grid_interpolation_plan(x, z, Xi, Zi, method=method)

    assert {field.name for field in fields(first)} >= {
        "expected_rounded_source_keys",
        "duplicate_group_inverse_indices",
        "duplicate_group_counts",
        "valid_target_mask",
    }
    for name in (
        "finite_coordinate_mask",
        "expected_rounded_source_keys",
        "duplicate_group_inverse_indices",
        "duplicate_group_counts",
        "deduplicated_x",
        "deduplicated_z",
        "valid_target_mask",
        "target_simplex_indices",
        "source_vertex_indices",
        "barycentric_weights",
        "nearest_source_indices",
    ):
        first_array = getattr(first, name)
        second_array = getattr(second, name)
        if first_array is None:
            assert second_array is None
        else:
            assert_array_equal(first_array, second_array)
            assert not first_array.flags.writeable


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_application_does_not_rebuild_spatial_geometry(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, z = _square_geometry()
    Xi, Zi = _target_grid()
    plan = build_fixed_grid_interpolation_plan(x, z, Xi, Zi, method=method)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("spatial geometry was rebuilt")

    monkeypatch.setattr(interpolation_module, "Delaunay", fail)
    monkeypatch.setattr(interpolation_module, "cKDTree", fail)

    result = apply_fixed_grid_interpolation_plan(
        plan, x, z, np.asarray([1.0, 2.0, 3.0, 4.0])
    )
    assert result.shape == Xi.shape
