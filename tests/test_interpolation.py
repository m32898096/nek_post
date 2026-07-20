import numpy as np
import pytest

import nek_post.interpolation as interpolation
from nek_post.interpolation import (
    average_duplicate_xz_points,
    create_common_xz_grid,
    interpolate_to_grid,
    valid_common_mask,
)


def test_common_grid_uses_exact_domain_intersection_and_meshgrid_orientation() -> None:
    slices = {
        "N5": {"x": np.array([-2.0, 4.0]), "z": np.array([-3.0, 5.0])},
        "N7": {"x": np.array([0.0, 6.0]), "z": np.array([-1.0, 3.0])},
        "N9": {"x": np.array([-1.0, 2.0]), "z": np.array([-2.0, 4.0])},
    }

    Xi, Zi, xi, zi, metadata = create_common_xz_grid(slices, nx=3, nz=5)

    np.testing.assert_array_equal(xi, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(zi, [-1.0, 0.0, 1.0, 2.0, 3.0])
    assert xi.shape == (3,)
    assert zi.shape == (5,)
    assert Xi.shape == (5, 3)
    assert Zi.shape == (5, 3)
    np.testing.assert_array_equal(Xi, np.array([[0.0, 1.0, 2.0]] * 5))
    np.testing.assert_array_equal(
        Zi,
        np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
    )
    assert metadata == {
        "xmin": 0.0,
        "xmax": 2.0,
        "zmin": -1.0,
        "zmax": 3.0,
        "nx": 3,
        "nz": 5,
    }


@pytest.mark.parametrize(
    "empty_case",
    [
        {"x": np.array([]), "z": np.array([0.0])},
        {"x": np.array([0.0]), "z": np.array([])},
    ],
)
def test_common_grid_rejects_case_without_xz_points(empty_case: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="Slice data for N7 contains no x-z points"):
        create_common_xz_grid({"N5": {"x": [0.0], "z": [0.0]}, "N7": empty_case}, nx=2, nz=2)


def test_common_grid_rejects_nonoverlapping_x_domain() -> None:
    slices = {
        "N5": {"x": [0.0, 1.0], "z": [0.0, 2.0]},
        "N7": {"x": [2.0, 3.0], "z": [1.0, 3.0]},
    }

    with pytest.raises(ValueError, match="do not share an overlapping x-z domain"):
        create_common_xz_grid(slices, nx=2, nz=2)


def test_common_grid_rejects_nonoverlapping_z_domain() -> None:
    slices = {
        "N5": {"x": [0.0, 2.0], "z": [0.0, 1.0]},
        "N7": {"x": [1.0, 3.0], "z": [2.0, 3.0]},
    }

    with pytest.raises(ValueError, match="do not share an overlapping x-z domain"):
        create_common_xz_grid(slices, nx=2, nz=2)


def test_duplicate_averaging_excludes_nonfinite_data_and_reports_exact_stats() -> None:
    x = np.array([0.0, 0.0, 1.0, 1.0004, np.nan, 2.0, 3.0])
    z = np.array([0.0, 0.0, 1.0, 1.0004, 1.0, np.inf, 3.0])
    values = np.array([2.0, 4.0, 10.0, 14.0, 5.0, 6.0, np.nan])

    x_avg, z_avg, values_avg, stats = average_duplicate_xz_points(x, z, values, decimals=3)

    np.testing.assert_allclose(x_avg, [0.0, 1.0002], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(z_avg, [0.0, 1.0002], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(values_avg, [3.0, 12.0], rtol=0.0, atol=1e-15)
    assert np.all(np.isfinite(x_avg))
    assert np.all(np.isfinite(z_avg))
    assert np.all(np.isfinite(values_avg))
    assert stats == {
        "original_point_count": 7,
        "finite_point_count": 4,
        "unique_point_count": 2,
        "duplicate_point_count": 2,
        "max_multiplicity": 2,
        "decimals": 3,
    }


def test_duplicate_rounding_precision_changes_grouping() -> None:
    x = [0.0, 0.0, 1.0, 1.0004]
    z = [0.0, 0.0, 1.0, 1.0004]
    values = [2.0, 4.0, 10.0, 14.0]

    coarse = average_duplicate_xz_points(x, z, values, decimals=3)
    fine = average_duplicate_xz_points(x, z, values, decimals=4)

    assert coarse[3]["unique_point_count"] == 2
    assert coarse[3]["duplicate_point_count"] == 2
    assert fine[3]["unique_point_count"] == 3
    assert fine[3]["duplicate_point_count"] == 1
    np.testing.assert_allclose(fine[0], [0.0, 1.0, 1.0004], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(fine[2], [3.0, 10.0, 14.0], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("x", "z", "values"),
    [
        ([0.0, 1.0], [0.0], [1.0, 2.0]),
        ([0.0], [0.0, 1.0], [1.0, 2.0]),
        ([0.0, 1.0], [0.0, 1.0], [1.0]),
    ],
)
def test_duplicate_averaging_rejects_mismatched_lengths(x, z, values) -> None:
    with pytest.raises(ValueError, match="matching one-dimensional sizes"):
        average_duplicate_xz_points(x, z, values)


def test_duplicate_averaging_rejects_all_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="No finite x-z-value points"):
        average_duplicate_xz_points([np.nan, 0.0], [0.0, np.inf], [1.0, 2.0])


def test_linear_interpolation_reproduces_plane_on_structured_grid() -> None:
    x = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 1.0, 1.0])
    values = 2.0 * x + 3.0 * z + 1.0
    xi = np.linspace(0.0, 1.0, 5)
    zi = np.linspace(0.0, 1.0, 4)
    Xi, Zi = np.meshgrid(xi, zi)

    actual = interpolate_to_grid(x, z, values, Xi, Zi, method="linear")

    assert actual.shape == Xi.shape == Zi.shape
    np.testing.assert_allclose(actual, 2.0 * Xi + 3.0 * Zi + 1.0, rtol=1e-14, atol=1e-14)


def test_interpolation_averages_duplicate_inputs_before_linear_interpolation() -> None:
    x = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    values = np.array([1.0, 3.0, 3.0, 4.0, 6.0])
    Xi, Zi = np.meshgrid([0.0, 1.0], [0.0, 1.0])

    actual = interpolate_to_grid(x, z, values, Xi, Zi, method="linear", deduplicate=True)

    np.testing.assert_allclose(actual, [[2.0, 3.0], [4.0, 6.0]], rtol=0.0, atol=1e-15)


def test_duplicate_decimals_is_forwarded_to_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_griddata(points, values, grid, method):
        calls.append((points[0].copy(), values.copy()))
        return np.zeros_like(grid[0], dtype=float)

    monkeypatch.setattr(interpolation, "griddata", fake_griddata)
    x = [0.0, 0.004, 1.0, 0.0]
    z = [0.0, 0.004, 0.0, 1.0]
    values = [1.0, 3.0, 3.0, 4.0]
    Xi, Zi = np.meshgrid([0.0, 1.0], [0.0, 1.0])

    interpolate_to_grid(x, z, values, Xi, Zi, duplicate_decimals=2)
    interpolate_to_grid(x, z, values, Xi, Zi, duplicate_decimals=3)

    assert calls[0][0].size == 3
    np.testing.assert_allclose(calls[0][1], [2.0, 4.0, 3.0], rtol=0.0, atol=1e-15)
    assert calls[1][0].size == 4


def test_interpolation_without_deduplication_flattens_inputs_and_respects_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_griddata(points, values, grid, method):
        captured.update(points=points, values=values, grid=grid, method=method)
        return np.full_like(grid[0], 7.0, dtype=float)

    monkeypatch.setattr(interpolation, "griddata", fake_griddata)
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    z = np.array([[0.0, 0.0], [1.0, 1.0]])
    values = 2.0 * x + 3.0 * z + 1.0
    Xi, Zi = np.meshgrid([0.25, 0.75], [0.25, 0.75])

    actual = interpolate_to_grid(x, z, values, Xi, Zi, method="cubic", deduplicate=False)

    np.testing.assert_array_equal(captured["points"][0], np.ravel(x))
    np.testing.assert_array_equal(captured["points"][1], np.ravel(z))
    np.testing.assert_array_equal(captured["values"], np.ravel(values))
    assert captured["method"] == "cubic"
    assert captured["grid"] == (Xi, Zi)
    np.testing.assert_array_equal(actual, np.full(Xi.shape, 7.0))


def test_valid_common_mask_for_one_array_has_boolean_dtype_and_matching_shape() -> None:
    values = np.array([[1.0, np.nan], [np.inf, -2.0]])

    mask = valid_common_mask(values)

    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
    assert mask.dtype == np.bool_
    assert mask.shape == values.shape


def test_valid_common_mask_intersects_all_finite_positions() -> None:
    first = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    second = np.array([[1.0, 2.0, np.inf], [4.0, -np.inf, 6.0]])
    third = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]])

    mask = valid_common_mask(first, second, third)

    np.testing.assert_array_equal(mask, [[True, False, False], [False, False, True]])


def test_valid_common_mask_requires_at_least_one_array() -> None:
    with pytest.raises(ValueError, match="At least one array is required"):
        valid_common_mask()
