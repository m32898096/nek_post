from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.gll_directional_workflow import (
    compute_gll_directional_integral,
    gll_directional_field_getter,
    gll_directional_integral_path,
    load_gll_directional_integral_npz,
    normalize_gll_directional_field,
    save_gll_directional_integral_npz,
)


def _data(
    *,
    bad_concentration_shape: bool = False,
    velocity_components: tuple[float, float, float] | None = None,
) -> SimpleNamespace:
    nodes = gll_nodes(3)
    z, y, x = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    concentration = np.ones_like(x)
    if bad_concentration_shape:
        concentration = concentration[0]
    velocity = (
        (x + 1.0, y + 2.0, z + 3.0)
        if velocity_components is None
        else tuple(np.full_like(x, component) for component in velocity_components)
    )
    return SimpleNamespace(
        elem=(
            SimpleNamespace(
                pos=(x, y, z),
                temp=(concentration,),
                vel=velocity,
                pres=(x - y + z,),
            ),
        )
    )


@pytest.mark.parametrize("field", ("concentration", "u", "v", "w", "speed", "pressure"))
def test_supported_fields_use_existing_accessors(field: str) -> None:
    integral = compute_gll_directional_integral(
        _data(), field=field, direction="y"
    )

    assert integral.field == field
    assert integral.plan.direction == "y"
    assert integral.result.values.shape == (3, 3)


def test_speed_uses_all_velocity_components_numerically() -> None:
    data = _data(velocity_components=(3.0, 4.0, 12.0))
    speed = gll_directional_field_getter("speed")(data.elem[0])
    np.testing.assert_allclose(speed, 13.0)

    integral = compute_gll_directional_integral(
        data, field="speed", direction="y"
    )
    np.testing.assert_allclose(integral.result.values, 26.0)


def test_workflow_preserves_directional_output_orientation() -> None:
    nodes = gll_nodes(3)
    z, y, x = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    data = SimpleNamespace(
        elem=(
            SimpleNamespace(
                pos=(x, y, z),
                temp=(1.0 + x + 2.0 * y + 3.0 * z,),
                vel=(x, y, z),
                pres=(x + y + z,),
            ),
        )
    )

    expected = {
        "x": ("y", "z", lambda horizontal, vertical: 2.0 + 4.0 * horizontal + 6.0 * vertical),
        "y": ("x", "z", lambda horizontal, vertical: 2.0 + 2.0 * horizontal + 6.0 * vertical),
        "z": ("x", "y", lambda horizontal, vertical: 2.0 + 2.0 * horizontal + 4.0 * vertical),
    }
    for direction, (horizontal_name, vertical_name, expected_values) in expected.items():
        integral = compute_gll_directional_integral(
            data, field="concentration", direction=direction
        )
        horizontal, vertical = np.meshgrid(
            integral.result.horizontal_coordinates,
            integral.result.vertical_coordinates,
        )

        assert integral.result.horizontal_coordinate_name == horizontal_name
        assert integral.result.vertical_coordinate_name == vertical_name
        np.testing.assert_allclose(
            integral.result.values,
            expected_values(horizontal, vertical),
        )


def test_field_normalization_and_shape_validation_are_clear() -> None:
    assert normalize_gll_directional_field(" Speed ") == "speed"
    with pytest.raises(ValueError, match="Unsupported field 'temperature'"):
        normalize_gll_directional_field("temperature")
    with pytest.raises(ValueError, match="Scalar field shape mismatch"):
        compute_gll_directional_integral(
            _data(bad_concentration_shape=True),
            field="concentration",
            direction="z",
        )


def test_npz_artifact_round_trip_and_deterministic_path(tmp_path: Path) -> None:
    integral = compute_gll_directional_integral(
        _data(), field="concentration", direction="y"
    )
    path = gll_directional_integral_path(
        tmp_path,
        case="N7",
        index=79,
        field="concentration",
        direction="y",
    )
    assert path == (
        tmp_path
        / "N7"
        / "concentration"
        / "y"
        / "N7_f00079_concentration_integrate_y.npz"
    )

    save_gll_directional_integral_npz(
        path,
        integral,
        case="N7",
        index=79,
        time=19.5,
        source_file="/data/Nek5000_data/case_N7/GC0.f00079",
    )
    loaded = load_gll_directional_integral_npz(path)

    assert loaded["field"] == "concentration"
    assert loaded["direction"] == "y"
    assert loaded["horizontal_coordinate_name"] == "x"
    assert loaded["vertical_coordinate_name"] == "z"
    assert loaded["case"] == "N7"
    assert loaded["index"] == 79
    assert loaded["time"] == 19.5
    np.testing.assert_allclose(loaded["values"], 2.0)
    np.testing.assert_array_equal(loaded["element_shape"], [3, 3, 3])
    np.testing.assert_array_equal(loaded["element_interval_counts"], [1, 1, 1])
    np.testing.assert_array_equal(loaded["physical_to_reference_axes"], [2, 1, 0])


def test_loader_rejects_inconsistent_shape_and_coordinate_names(tmp_path: Path) -> None:
    shape_path = tmp_path / "bad-shape.npz"
    np.savez_compressed(
        shape_path,
        values=np.ones((2, 2)),
        horizontal_coordinates=np.array([0.0, 1.0, 2.0]),
        vertical_coordinates=np.array([0.0, 1.0]),
        field="u",
        direction="x",
        horizontal_coordinate_name="y",
        vertical_coordinate_name="z",
        case="N7",
        index=1,
        time=0.5,
        source_file="source",
        element_count=1,
        element_shape=np.array([3, 3, 3]),
        element_interval_counts=np.array([1, 1, 1]),
        physical_to_reference_axes=np.array([2, 1, 0]),
    )
    with pytest.raises(ValueError, match="Artifact values shape"):
        load_gll_directional_integral_npz(shape_path)

    names_path = tmp_path / "bad-names.npz"
    np.savez_compressed(
        names_path,
        values=np.ones((2, 2)),
        horizontal_coordinates=np.array([0.0, 1.0]),
        vertical_coordinates=np.array([0.0, 1.0]),
        field="u",
        direction="x",
        horizontal_coordinate_name="x",
        vertical_coordinate_name="z",
        case="N7",
        index=1,
        time=0.5,
        source_file="source",
        element_count=1,
        element_shape=np.array([3, 3, 3]),
        element_interval_counts=np.array([1, 1, 1]),
        physical_to_reference_axes=np.array([2, 1, 0]),
    )
    with pytest.raises(ValueError, match="coordinate names are inconsistent"):
        load_gll_directional_integral_npz(names_path)
