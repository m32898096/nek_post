from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from nek_post.slicing import extract_y_slice, get_y_range, save_slice_npz


FIELD_NAMES = ("x", "y", "z", "C", "u", "v", "w", "p")


def _element(
    x: np.ndarray,
    y: np.ndarray,
    *,
    concentration_source: str = "temp",
) -> SimpleNamespace:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = x + 100.0
    concentration = x + 1_000.0
    fields = {
        "pos": (x, y, z),
        "vel": (x + 2_000.0, x + 3_000.0, x + 4_000.0),
        "pres": (x + 5_000.0,),
    }
    if concentration_source == "temp":
        fields.update(temp=(concentration,), scal=None)
    else:
        fields.update(temp=None, scal=(concentration,))
    return SimpleNamespace(**fields)


def _dataset(*elements: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(elem=list(elements))


def test_get_y_range_across_multidimensional_elements() -> None:
    data = _dataset(
        _element(np.arange(6).reshape(2, 3), np.array([[2.0, -3.0, 1.0], [4.0, 0.0, 5.0]])),
        _element(np.arange(4).reshape(2, 2), np.array([[7.0, 6.0], [-2.0, 3.0]])),
    )

    assert get_y_range(data) == (-3.0, 7.0)


def test_get_y_range_rejects_dataset_without_coordinate_points() -> None:
    with pytest.raises(
        ValueError,
        match="Could not determine y range because no coordinate points were found",
    ):
        get_y_range(_dataset())


def test_default_y0_is_complete_range_midpoint_and_selects_nearest_plane() -> None:
    data = _dataset(
        _element(np.array([[10.0, 11.0], [12.0, 13.0]]), np.array([[-2.0, 1.0], [3.0, 4.0]])),
        _element(np.array([[20.0, 21.0]]), np.array([[0.0, 2.0]])),
    )

    result = extract_y_slice(data)

    assert result["y0"] == 1.0
    assert result["selected_y"] == 1.0
    np.testing.assert_array_equal(result["x"], [11.0])
    np.testing.assert_array_equal(result["y"], [1.0])


def test_nearest_plane_collects_aligned_raveled_fields_across_elements() -> None:
    first = _element(
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        np.array([[0.0, 1.00000000004, 2.0], [1.00000000003, 3.0, 4.0]]),
        concentration_source="temp",
    )
    second = _element(
        np.array([[20.0, 21.0], [22.0, 23.0]]),
        np.array([[1.00000000001, 5.0], [0.0, 1.00000000002]]),
        concentration_source="scal",
    )

    result = extract_y_slice(
        _dataset(first, second),
        y0=1.1,
        slab_ratio=0.025,
        mode="nearest_plane",
        y_round_decimals=10,
    )

    expected_x = np.array([11.0, 13.0, 20.0, 23.0])
    np.testing.assert_array_equal(result["x"], expected_x)
    np.testing.assert_allclose(
        result["y"],
        [1.00000000004, 1.00000000003, 1.00000000001, 1.00000000002],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(result["z"], expected_x + 100.0)
    np.testing.assert_array_equal(result["C"], expected_x + 1_000.0)
    np.testing.assert_array_equal(result["u"], expected_x + 2_000.0)
    np.testing.assert_array_equal(result["v"], expected_x + 3_000.0)
    np.testing.assert_array_equal(result["w"], expected_x + 4_000.0)
    np.testing.assert_array_equal(result["p"], expected_x + 5_000.0)

    assert result["mode"] == "nearest_plane"
    assert result["y0"] == 1.1
    assert result["slab_ratio"] == 0.025
    assert result["y_round_decimals"] == 10
    assert result["selected_y"] == 1.0
    assert result["rounded_unique_y_count_before_selection"] == 6
    assert result["point_count"] == 4
    assert "dy_tol" not in result


def test_y_round_decimals_controls_nearest_plane_grouping() -> None:
    data = _dataset(
        _element(
            np.array([10.0, 11.0, 12.0]),
            np.array([1.0004, 1.00049, 2.0]),
        )
    )

    coarse = extract_y_slice(data, y0=1.0, y_round_decimals=3)
    fine = extract_y_slice(data, y0=1.0, y_round_decimals=5)

    np.testing.assert_array_equal(coarse["x"], [10.0, 11.0])
    np.testing.assert_array_equal(fine["x"], [10.0])
    assert coarse["rounded_unique_y_count_before_selection"] == 2
    assert fine["rounded_unique_y_count_before_selection"] == 3


def test_slab_uses_exact_tolerance_and_inclusive_boundaries() -> None:
    element = _element(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        np.array([0.0, 3.999, 4.0, 5.5, 6.0, 6.001, 10.0]),
    )

    result = extract_y_slice(_dataset(element), y0=5.0, slab_ratio=0.1, mode="slab")

    assert result["dy_tol"] == 0.1 * (10.0 - 0.0) == 1.0
    np.testing.assert_array_equal(result["x"], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result["y"], [4.0, 5.5, 6.0])
    np.testing.assert_array_equal(result["C"], [1_002.0, 1_003.0, 1_004.0])
    assert result["mode"] == "slab"
    assert result["y0"] == 5.0
    assert result["slab_ratio"] == 0.1
    assert result["y_round_decimals"] == 10
    assert result["point_count"] == 3
    assert "selected_y" not in result
    assert "rounded_unique_y_count_before_selection" not in result


def test_slab_rejects_request_without_points_and_recommends_larger_ratio() -> None:
    data = _dataset(_element(np.array([0.0, 1.0]), np.array([0.0, 10.0])))

    with pytest.raises(ValueError, match=r"No points found in y slice.*Increase slab_ratio\."):
        extract_y_slice(data, y0=100.0, slab_ratio=0.01, mode="slab")


def test_invalid_mode_names_both_valid_modes() -> None:
    data = _dataset(_element(np.array([0.0, 1.0]), np.array([0.0, 1.0])))

    with pytest.raises(ValueError) as exc_info:
        extract_y_slice(data, mode="invalid")

    message = str(exc_info.value)
    assert "nearest_plane" in message
    assert "slab" in message


@pytest.mark.parametrize("mode", ["nearest_plane", "slab"])
def test_save_slice_npz_round_trip_schema_and_extra_metadata(tmp_path: Path, mode: str) -> None:
    element = _element(
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[0.0, 1.0], [1.0, 2.0]]),
    )
    if mode == "nearest_plane":
        slice_data = extract_y_slice(_dataset(element), y0=1.0, mode=mode)
        mode_keys = {"selected_y", "rounded_unique_y_count_before_selection"}
    else:
        slice_data = extract_y_slice(_dataset(element), y0=1.0, slab_ratio=0.5, mode=mode)
        mode_keys = {"dy_tol"}

    path = tmp_path / "missing" / "parents" / f"{mode}.npz"
    save_slice_npz(slice_data, path, metadata={"case": "N5", "index": 17})

    assert path.is_file()
    with zipfile.ZipFile(path) as archive:
        assert archive.infolist()
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())

    common_metadata = {"mode", "y0", "slab_ratio", "y_round_decimals", "point_count"}
    with np.load(path) as loaded:
        assert set(loaded.files) == set(FIELD_NAMES) | common_metadata | mode_keys | {"case", "index"}
        for name in FIELD_NAMES:
            np.testing.assert_array_equal(loaded[name], slice_data[name])
        assert loaded["mode"].item() == mode
        assert loaded["case"].item() == "N5"
        assert loaded["index"].item() == 17
