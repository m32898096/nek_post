from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.gll_directional_integration import (
    GLLDirectionalIntegrationPlan,
    apply_gll_directional_integration_plan,
    build_gll_directional_integration_plan,
)


DEFAULT_MAPPING = (2, 1, 0)  # physical x/y/z -> reference axes
DEFAULT_X_EDGES = (-1.0, 0.25, 2.0)
DEFAULT_Y_EDGES = (0.0, 0.4, 1.5)
DEFAULT_Z_EDGES = (-0.5, 0.1, 1.0)


def _linear_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    return 1.0 + x + 2.0 * y + 3.0 * z


def _make_element(
    cell: tuple[int, int, int],
    *,
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
    z_edges: tuple[float, ...],
    shape: tuple[int, int, int],
    mapping: tuple[int, int, int],
    field_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
    reverse: Callable[[tuple[int, int, int], int], bool] | None = None,
    serialize_float32: bool = False,
) -> SimpleNamespace:
    reference = np.meshgrid(
        *(gll_nodes(size) for size in shape),
        indexing="ij",
    )
    physical_edges = (x_edges, y_edges, z_edges)
    coordinates = []
    for physical_axis in range(3):
        lower = physical_edges[physical_axis][cell[physical_axis]]
        upper = physical_edges[physical_axis][cell[physical_axis] + 1]
        q = reference[mapping[physical_axis]]
        if reverse is not None and reverse(cell, physical_axis):
            q = -q
        coordinate = 0.5 * (lower + upper) + 0.5 * (upper - lower) * q
        if serialize_float32:
            coordinate = coordinate.astype(np.float32).astype(np.float64)
        coordinates.append(coordinate)
    scalar = field_function(*coordinates)
    return SimpleNamespace(
        pos=tuple(coordinates),
        scalar=np.asarray(scalar, dtype=np.float64),
        cell=cell,
    )


def _make_mesh(
    *,
    x_edges: tuple[float, ...] = DEFAULT_X_EDGES,
    y_edges: tuple[float, ...] = DEFAULT_Y_EDGES,
    z_edges: tuple[float, ...] = DEFAULT_Z_EDGES,
    shape: tuple[int, int, int] = (4, 4, 4),
    mapping: tuple[int, int, int] = DEFAULT_MAPPING,
    field_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = _linear_field,
    reverse: Callable[[tuple[int, int, int], int], bool] | None = None,
    shuffle: bool = True,
    serialize_float32: bool = False,
) -> SimpleNamespace:
    elements = [
        _make_element(
            (ix, iy, iz),
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            shape=shape,
            mapping=mapping,
            field_function=field_function,
            reverse=reverse,
            serialize_float32=serialize_float32,
        )
        for iz in range(len(z_edges) - 1)
        for iy in range(len(y_edges) - 1)
        for ix in range(len(x_edges) - 1)
    ]
    if shuffle:
        np.random.default_rng(20260810).shuffle(elements)
    return SimpleNamespace(elem=elements)


def _global_gll_coordinates(
    edges: tuple[float, ...],
    node_count: int,
) -> np.ndarray:
    nodes = gll_nodes(node_count)
    pieces = []
    for index, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        profile = 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes
        pieces.append(profile if index == 0 else profile[1:])
    return np.concatenate(pieces)


def _apply(
    data: SimpleNamespace,
    direction: str,
) -> tuple[GLLDirectionalIntegrationPlan, object]:
    plan = build_gll_directional_integration_plan(
        data, direction=direction
    )
    result = apply_gll_directional_integration_plan(
        plan,
        data,
        field_getter=lambda element: element.scalar,
    )
    return plan, result


@pytest.mark.parametrize(
    (
        "direction",
        "horizontal_name",
        "vertical_name",
        "expected_shape",
        "domain_length",
    ),
    (
        ("x", "y", "z", (7, 7), 3.0),
        ("y", "x", "z", (7, 7), 1.5),
        ("z", "x", "y", (7, 7), 1.5),
    ),
)
def test_constant_field_all_directions_and_output_orientation(
    direction: str,
    horizontal_name: str,
    vertical_name: str,
    expected_shape: tuple[int, int],
    domain_length: float,
) -> None:
    data = _make_mesh(
        field_function=lambda x, _y, _z: np.ones_like(x)
    )

    plan, result = _apply(data, direction)

    assert plan.physical_to_reference_axes == DEFAULT_MAPPING
    assert plan.element_interval_counts == (2, 2, 2)
    assert result.direction == direction
    assert result.horizontal_coordinate_name == horizontal_name
    assert result.vertical_coordinate_name == vertical_name
    assert result.values.shape == expected_shape
    np.testing.assert_allclose(result.values, domain_length, atol=2.0e-14)
    assert not result.horizontal_coordinates.flags.writeable
    assert not result.vertical_coordinates.flags.writeable
    assert not result.values.flags.writeable


def test_shuffled_mesh_detection_recovers_real_n7_axis_permutation() -> None:
    data = _make_mesh(shuffle=True)

    plan = build_gll_directional_integration_plan(data, direction="x")

    assert plan.physical_to_reference_axes == (2, 1, 0)
    assert plan.integration_reference_axis == 2
    assert plan.horizontal_reference_axis == 1
    assert plan.vertical_reference_axis == 0
    assert plan.element_count == 8


@pytest.mark.parametrize("direction", ("x", "y", "z"))
def test_linear_physical_field_matches_analytic_directional_integral(
    direction: str,
) -> None:
    data = _make_mesh()

    _plan, result = _apply(data, direction)
    horizontal, vertical = np.meshgrid(
        result.horizontal_coordinates,
        result.vertical_coordinates,
    )
    if direction == "x":
        lower, upper = DEFAULT_X_EDGES[0], DEFAULT_X_EDGES[-1]
        expected = (
            (upper - lower) * (1.0 + 2.0 * horizontal + 3.0 * vertical)
            + 0.5 * (upper**2 - lower**2)
        )
    elif direction == "y":
        lower, upper = DEFAULT_Y_EDGES[0], DEFAULT_Y_EDGES[-1]
        expected = (
            (upper - lower) * (1.0 + horizontal + 3.0 * vertical)
            + upper**2
            - lower**2
        )
    else:
        lower, upper = DEFAULT_Z_EDGES[0], DEFAULT_Z_EDGES[-1]
        expected = (
            (upper - lower) * (1.0 + horizontal + 2.0 * vertical)
            + 1.5 * (upper**2 - lower**2)
        )

    np.testing.assert_allclose(result.values, expected, rtol=2.0e-14, atol=3.0e-14)


def test_nonuniform_element_widths_sum_to_complete_domain_length() -> None:
    x_edges = (-3.0, -2.8, 0.4, 0.45, 5.0)
    data = _make_mesh(
        x_edges=x_edges,
        field_function=lambda x, _y, _z: np.ones_like(x),
    )

    plan, result = _apply(data, "x")

    assert plan.element_interval_counts == (4, 2, 2)
    np.testing.assert_allclose(result.values, 8.0, atol=3.0e-14)


def test_alternate_axis_permutation_and_noncubic_shape_are_supported() -> None:
    mapping = (0, 2, 1)
    shape = (3, 4, 5)
    data = _make_mesh(mapping=mapping, shape=shape)

    plan, result = _apply(data, "y")

    assert plan.physical_to_reference_axes == mapping
    assert plan.integration_reference_axis == 2
    expected_x = _global_gll_coordinates(DEFAULT_X_EDGES, shape[mapping[0]])
    expected_z = _global_gll_coordinates(DEFAULT_Z_EDGES, shape[mapping[2]])
    np.testing.assert_allclose(result.horizontal_coordinates, expected_x)
    np.testing.assert_allclose(result.vertical_coordinates, expected_z)
    assert result.values.shape == (expected_z.size, expected_x.size)
    horizontal, vertical = np.meshgrid(
        result.horizontal_coordinates,
        result.vertical_coordinates,
    )
    lower, upper = DEFAULT_Y_EDGES[0], DEFAULT_Y_EDGES[-1]
    expected_values = (
        (upper - lower) * (1.0 + horizontal + 3.0 * vertical)
        + upper**2
        - lower**2
    )
    np.testing.assert_allclose(
        result.values,
        expected_values,
        rtol=2.0e-14,
        atol=3.0e-14,
    )


def test_reversed_local_reference_orientations_are_normalized_physically() -> None:
    data = _make_mesh(
        reverse=lambda cell, physical: (sum(cell) + physical) % 2 == 1
    )

    _plan, result = _apply(data, "y")
    horizontal, vertical = np.meshgrid(
        result.horizontal_coordinates,
        result.vertical_coordinates,
    )
    lower, upper = DEFAULT_Y_EDGES[0], DEFAULT_Y_EDGES[-1]
    expected = (
        (upper - lower) * (1.0 + horizontal + 3.0 * vertical)
        + upper**2
        - lower**2
    )

    np.testing.assert_allclose(result.values, expected, rtol=2.0e-14, atol=3.0e-14)


def test_transverse_shared_nodes_are_deduplicated_not_summed() -> None:
    data = _make_mesh(
        field_function=lambda x, _y, _z: np.ones_like(x)
    )

    _plan, result = _apply(data, "x")

    expected_y = _global_gll_coordinates(DEFAULT_Y_EDGES, 4)
    expected_z = _global_gll_coordinates(DEFAULT_Z_EDGES, 4)
    np.testing.assert_allclose(result.horizontal_coordinates, expected_y)
    np.testing.assert_allclose(result.vertical_coordinates, expected_z)
    assert np.count_nonzero(
        np.isclose(
            result.horizontal_coordinates,
            DEFAULT_Y_EDGES[1],
            rtol=0.0,
            atol=2.0e-16,
        )
    ) == 1
    assert np.count_nonzero(
        np.isclose(
            result.vertical_coordinates,
            DEFAULT_Z_EDGES[1],
            rtol=0.0,
            atol=2.0e-16,
        )
    ) == 1
    np.testing.assert_allclose(result.values, 3.0, atol=2.0e-14)


def test_float32_serialized_affine_coordinates_are_accepted() -> None:
    data = _make_mesh(
        x_edges=(-17.0, -16.875),
        y_edges=(1.0, 1.125),
        z_edges=(0.6816, 0.8386),
        shape=(8, 8, 8),
        serialize_float32=True,
    )

    plan = build_gll_directional_integration_plan(data, direction="x")

    assert plan.physical_to_reference_axes == DEFAULT_MAPPING


def test_narrow_curved_element_is_rejected_using_local_affine_tolerance() -> None:
    data = _make_mesh(
        x_edges=(-17.0, -16.99, 17.0),
        y_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        shape=(8, 8, 8),
        shuffle=False,
        serialize_float32=True,
    )
    narrow_element = next(
        element for element in data.elem if element.cell[0] == 0
    )
    qx = np.meshgrid(
        *(gll_nodes(size) for size in (8, 8, 8)),
        indexing="ij",
    )[DEFAULT_MAPPING[0]]
    perturbation = 2.0e-5 * (1.0 - qx**2)
    old_global_tolerance = 8.0 * np.finfo(np.float32).eps * 34.0
    assert np.max(np.abs(perturbation)) < old_global_tolerance
    narrow_element.pos = (
        (
            (narrow_element.pos[0] + perturbation)
            .astype(np.float32)
            .astype(np.float64)
        ),
        narrow_element.pos[1],
        narrow_element.pos[2],
    )

    with pytest.raises(ValueError, match="materially non-affine"):
        build_gll_directional_integration_plan(data, direction="x")


def test_element_below_float32_affine_resolution_is_rejected() -> None:
    data = _make_mesh(
        x_edges=(17.0, 17.0005),
        shape=(8, 8, 8),
        shuffle=False,
        serialize_float32=True,
    )

    with pytest.raises(ValueError, match="too small relative to coordinate storage"):
        build_gll_directional_integration_plan(data, direction="x")


@pytest.mark.parametrize(
    "direction",
    ("", "xy", "streamwise", "X", " x ", 1, None, True),
)
def test_invalid_direction_is_rejected(direction: object) -> None:
    with pytest.raises(ValueError, match="direction"):
        build_gll_directional_integration_plan(
            _make_mesh(), direction=direction
        )


def test_empty_snapshot_is_rejected() -> None:
    with pytest.raises(ValueError, match="no elements"):
        build_gll_directional_integration_plan(
            SimpleNamespace(elem=[]), direction="x"
        )


def test_scalar_field_shape_mismatch_is_rejected() -> None:
    data = _make_mesh()
    plan = build_gll_directional_integration_plan(data, direction="x")
    data.elem[0].scalar = np.zeros((3, 4, 4))

    with pytest.raises(ValueError, match="shape mismatch"):
        apply_gll_directional_integration_plan(
            plan,
            data,
            field_getter=lambda element: element.scalar,
        )


def test_nonfinite_scalar_field_is_rejected() -> None:
    data = _make_mesh()
    plan = build_gll_directional_integration_plan(data, direction="z")
    data.elem[0].scalar[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        apply_gll_directional_integration_plan(
            plan,
            data,
            field_getter=lambda element: element.scalar,
        )


def test_nonfinite_coordinate_is_rejected() -> None:
    data = _make_mesh()
    data.elem[0].pos[0][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite x"):
        build_gll_directional_integration_plan(data, direction="x")


def test_inconsistent_element_coordinate_shapes_are_rejected() -> None:
    data = _make_mesh(shuffle=False)
    element = data.elem[1]
    element.pos = tuple(coordinate[:-1] for coordinate in element.pos)

    with pytest.raises(ValueError, match="one consistent shape"):
        build_gll_directional_integration_plan(data, direction="x")


def test_ambiguous_physical_reference_mapping_is_rejected() -> None:
    data = _make_mesh(
        x_edges=(0.0, 1.0),
        y_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        shuffle=False,
    )
    q0, _q1, q2 = np.meshgrid(
        *(gll_nodes(size) for size in (4, 4, 4)),
        indexing="ij",
    )
    data.elem[0].pos = (
        q0 + q2,
        data.elem[0].pos[1],
        data.elem[0].pos[2],
    )

    with pytest.raises(ValueError, match="Ambiguous physical/reference-axis"):
        build_gll_directional_integration_plan(data, direction="x")


def test_inconsistent_reference_axis_mapping_between_elements_is_rejected() -> None:
    default = _make_mesh(
        x_edges=(0.0, 1.0),
        y_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        mapping=(2, 1, 0),
        shuffle=False,
    )
    alternate = _make_mesh(
        x_edges=(0.0, 1.0),
        y_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        mapping=(0, 2, 1),
        shuffle=False,
    )
    data = SimpleNamespace(elem=[default.elem[0], alternate.elem[0]])

    with pytest.raises(ValueError, match="inconsistent between elements"):
        build_gll_directional_integration_plan(data, direction="x")


def test_missing_structured_element_is_rejected() -> None:
    data = _make_mesh(z_edges=(0.0, 1.0), shuffle=False)
    data.elem.pop()

    with pytest.raises(ValueError, match="topology is incomplete"):
        build_gll_directional_integration_plan(data, direction="y")


def test_duplicate_structured_element_is_rejected() -> None:
    data = _make_mesh(z_edges=(0.0, 1.0), shuffle=False)
    data.elem.append(deepcopy(data.elem[0]))

    with pytest.raises(ValueError, match="duplicate elements"):
        build_gll_directional_integration_plan(data, direction="z")


def test_materially_curved_coordinate_is_rejected() -> None:
    data = _make_mesh(
        x_edges=(0.0, 1.0),
        y_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        shuffle=False,
    )
    q2 = np.meshgrid(
        *(gll_nodes(size) for size in (4, 4, 4)),
        indexing="ij",
    )[2]
    x_coordinate = data.elem[0].pos[0]
    x_coordinate[...] = x_coordinate + 0.05 * q2**2

    with pytest.raises(ValueError, match="materially non-affine"):
        build_gll_directional_integration_plan(data, direction="x")


def test_changed_geometry_is_rejected_when_reusing_plan() -> None:
    data = _make_mesh()
    plan = build_gll_directional_integration_plan(data, direction="x")
    changed = deepcopy(data)
    changed.elem[0].pos[0][0, 0, 0] += 1.0e-12

    with pytest.raises(ValueError, match="coordinate signature changed"):
        apply_gll_directional_integration_plan(
            plan,
            changed,
            field_getter=lambda element: element.scalar,
            source_file="changed.f00002",
        )


def test_disagreeing_transverse_field_replicas_are_rejected() -> None:
    data = _make_mesh(
        x_edges=(0.0, 1.0),
        y_edges=(0.0, 0.5, 1.0),
        z_edges=(0.0, 1.0),
        field_function=lambda x, _y, _z: np.ones_like(x),
        shuffle=False,
    )
    plan = build_gll_directional_integration_plan(data, direction="x")
    for element in data.elem:
        if element.cell[1] == 1:
            element.scalar *= 2.0

    with pytest.raises(ValueError, match="transverse element replicas"):
        apply_gll_directional_integration_plan(
            plan,
            data,
            field_getter=lambda element: element.scalar,
        )
