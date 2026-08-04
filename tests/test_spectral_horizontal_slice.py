from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Callable

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.gll import gll_nodes
from nek_post.spectral_horizontal_slice import (
    SPECTRAL_HORIZONTAL_ALGORITHM_VERSION,
    apply_spectral_horizontal_slice_plan,
    build_spectral_horizontal_slice_plan,
    spectral_horizontal_plan_metadata,
    validate_spectral_horizontal_geometry,
)
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


FieldFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _default_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    return 1.25 + 0.7 * x - 0.4 * y + 1.1 * z + 0.2 * x * y - 0.3 * y * z


def _element(
    *,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
    z_bounds: tuple[float, float] = (0.0, 1.0),
    shape: tuple[int, int, int] = (4, 4, 4),
    field: FieldFunction = _default_field,
) -> SimpleNamespace:
    q0, q1, q2 = np.meshgrid(
        *(gll_nodes(size) for size in shape),
        indexing="ij",
    )

    def physical(q: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        lower, upper = bounds
        return 0.5 * (lower + upper) + 0.5 * (upper - lower) * q

    x = physical(q0, x_bounds)
    y = physical(q1, y_bounds)
    z = physical(q2, z_bounds)
    concentration = np.asarray(field(x, y, z), dtype=np.float64)
    return SimpleNamespace(
        pos=np.stack((x, y, z)),
        temp=np.asarray([concentration], dtype=np.float64),
        scal=None,
    )


def _mesh(
    *,
    x_intervals: tuple[tuple[float, float], ...] = ((0.0, 1.0),),
    y_intervals: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.0)),
    z_bounds: tuple[float, float] = (0.0, 1.0),
    shape: tuple[int, int, int] = (4, 4, 4),
    field: FieldFunction = _default_field,
) -> SimpleNamespace:
    elements = [
        _element(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            shape=shape,
            field=field,
        )
        for y_bounds in y_intervals
        for x_bounds in x_intervals
    ]
    return SimpleNamespace(elem=elements)


def test_polynomial_exactness_on_fixed_physical_z_plane() -> None:
    def polynomial(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return (
            2.0
            + 0.5 * x
            - 0.8 * y
            + 1.7 * z
            + 0.3 * x * y
            - 0.2 * x * z
            + 0.4 * y * z
            + 0.1 * z**3
        )

    data = _mesh(
        x_intervals=((0.0, 1.0), (1.0, 2.0)),
        field=polynomial,
    )
    plan = build_spectral_horizontal_slice_plan(
        data,
        nx=9,
        z_target=0.37,
    )

    result = apply_spectral_horizontal_slice_plan(plan, data)
    expected = polynomial(plan.Xi, plan.Yi, np.full(plan.target_shape, 0.37))

    assert np.all(plan.target_valid_mask)
    assert_allclose(result, expected, rtol=3.0e-12, atol=3.0e-12)


def test_grid_shape_orientation_and_fixed_z_evaluation() -> None:
    data = _mesh()
    plan = build_spectral_horizontal_slice_plan(
        data,
        nx=5,
        z_target=0.23,
    )
    result = apply_spectral_horizontal_slice_plan(plan, data)

    assert plan.target_shape == (12, 5)
    assert plan.Xi.shape == plan.Yi.shape == result.shape == (12, 5)
    assert_allclose(plan.Xi[0], np.linspace(0.0, 1.0, 5))
    assert_allclose(plan.Xi, np.broadcast_to(plan.Xi[0], plan.target_shape))
    assert_allclose(plan.Yi[:, 0], np.linspace(0.0, 1.0, 12, endpoint=False))
    assert_allclose(
        plan.Yi,
        np.broadcast_to(plan.Yi[:, :1], plan.target_shape),
    )
    assert_allclose(
        result,
        _default_field(plan.Xi, plan.Yi, np.full(plan.target_shape, 0.23)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_native_resolution_deduplicates_x_replicas_and_shared_y_interfaces() -> None:
    data = _mesh(
        x_intervals=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)),
        y_intervals=((0.0, 0.4), (0.4, 1.0)),
    )

    plan = build_spectral_horizontal_slice_plan(data, nx=7, z_target=0.4)

    # Two degree-three y elements have 2*3+1 physical levels. The two physical
    # periodic endpoints describe one seam, so there are six independent
    # periodic intervals rather than seven levels.
    assert plan.native_ny == 6
    assert plan.y_upsample_factor == 2
    assert plan.dense_ny == 12 == 2 * plan.native_ny
    assert plan.target_shape == (12, 7)


def test_scale_aware_deduplication_accepts_roundoff_at_shared_interface() -> None:
    data = _mesh(y_intervals=((0.0, 0.5), (0.5 + 2.0e-13, 1.0)))

    plan = build_spectral_horizontal_slice_plan(data, nx=4, z_target=0.4)

    assert plan.native_ny == 6
    assert plan.dense_ny == 12


def test_periodic_target_excludes_ymax_and_metadata_retains_period() -> None:
    plan = build_spectral_horizontal_slice_plan(
        _mesh(),
        nx=3,
        z_target=0.4,
    )
    metadata = spectral_horizontal_plan_metadata(plan)

    expected_y = np.linspace(0.0, 1.0, 12, endpoint=False)
    assert_allclose(plan.Yi[:, 0], expected_y, rtol=0.0, atol=0.0)
    assert not np.any(plan.Yi == 1.0)
    assert plan.Yi[-1, 0] == pytest.approx(11.0 / 12.0)
    assert metadata["ymin"] == 0.0
    assert metadata["ymax_periodic_endpoint"] == 1.0
    assert metadata["native_ny"] == 6
    assert metadata["dense_ny"] == 12
    assert metadata["spectral_horizontal_algorithm_version"] == (
        SPECTRAL_HORIZONTAL_ALGORITHM_VERSION
    )
    with pytest.raises(TypeError):
        metadata["dense_ny"] = 13  # type: ignore[index]


def test_multiple_x_and_y_elements_have_deterministic_interface_ownership() -> None:
    data = _mesh(x_intervals=((0.0, 1.0), (1.0, 2.0)))

    first = build_spectral_horizontal_slice_plan(data, nx=5, z_target=0.5)
    second = build_spectral_horizontal_slice_plan(data, nx=5, z_target=0.5)

    assert_array_equal(first.owner_element_index, second.owner_element_index)
    # Element order is lower-y/left-x, lower-y/right-x, upper-y/left-x,
    # upper-y/right-x. At x=1 and y=0.25, elements 0 and 1 both contain the
    # target and the residual/index tie break deterministically selects 0.
    y_index = int(np.flatnonzero(np.isclose(first.Yi[:, 0], 0.25))[0])
    x_index = int(np.flatnonzero(np.isclose(first.Xi[0], 1.0))[0])
    assert first.owner_element_index[y_index, x_index] == 0
    assert first.inverse_mapping_diagnostics.ambiguous_boundary_point_count > 0


def test_plan_reuse_accepts_changed_concentration_with_unchanged_geometry() -> None:
    data = _mesh(x_intervals=((0.0, 1.0), (1.0, 2.0)))
    plan = build_spectral_horizontal_slice_plan(data, nx=7, z_target=0.31)
    changed = deepcopy(data)

    def replacement(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return 9.0 - 0.5 * x + 1.2 * y + 0.7 * z

    for element in changed.elem:
        x, y, z = element.pos
        element.temp[0] = replacement(x, y, z)

    validate_spectral_horizontal_geometry(plan, changed, source_file="same.f00002")
    result = apply_spectral_horizontal_slice_plan(
        plan,
        changed,
        source_file="same.f00002",
    )

    assert_allclose(
        result,
        replacement(plan.Xi, plan.Yi, np.full(plan.target_shape, 0.31)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_geometry_reuse_rejects_changed_element_count() -> None:
    data = _mesh(x_intervals=((0.0, 1.0), (1.0, 2.0)))
    plan = build_spectral_horizontal_slice_plan(data, nx=5, z_target=0.4)

    with pytest.raises(
        SpectralGeometryMismatchError,
        match=r"later\.f00002.*expected 4 elements, got 3",
    ):
        validate_spectral_horizontal_geometry(
            plan,
            SimpleNamespace(elem=data.elem[:-1]),
            source_file="later.f00002",
        )


def test_geometry_reuse_rejects_changed_element_shape() -> None:
    data = _mesh()
    plan = build_spectral_horizontal_slice_plan(data, nx=5, z_target=0.4)
    changed = deepcopy(data)
    changed.elem[0] = _element(
        y_bounds=(0.0, 0.5),
        shape=(5, 5, 5),
    )

    with pytest.raises(
        SpectralGeometryMismatchError,
        match=r"reshaped\.f00002, element 0.*expected shape",
    ):
        validate_spectral_horizontal_geometry(
            plan,
            changed,
            source_file="reshaped.f00002",
        )


def test_geometry_reuse_rejects_changed_coordinates() -> None:
    data = _mesh()
    plan = build_spectral_horizontal_slice_plan(data, nx=5, z_target=0.4)
    changed = deepcopy(data)
    changed.elem[1].pos[0, 1, 1, 1] += 1.0e-12

    with pytest.raises(
        SpectralGeometryMismatchError,
        match=r"changed\.f00002, element 1.*coordinate signature",
    ):
        validate_spectral_horizontal_geometry(
            plan,
            changed,
            source_file="changed.f00002",
        )


@pytest.mark.parametrize("nx", (1, 0, -2, 2.5, True))
def test_invalid_nx_is_rejected(nx: object) -> None:
    with pytest.raises(ValueError, match="nx"):
        build_spectral_horizontal_slice_plan(
            _mesh(),
            nx=nx,  # type: ignore[arg-type]
            z_target=0.4,
        )


@pytest.mark.parametrize("factor", (0, -1, 1.5, True))
def test_invalid_y_upsample_factor_is_rejected(factor: object) -> None:
    with pytest.raises(ValueError, match="y_upsample_factor"):
        build_spectral_horizontal_slice_plan(
            _mesh(),
            nx=3,
            z_target=0.4,
            y_upsample_factor=factor,  # type: ignore[arg-type]
        )


def test_z_target_outside_mesh_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"z_target=1\.5.*mesh z range"):
        build_spectral_horizontal_slice_plan(
            _mesh(),
            nx=3,
            z_target=1.5,
        )


def test_plan_arrays_are_immutable() -> None:
    plan = build_spectral_horizontal_slice_plan(_mesh(), nx=5, z_target=0.4)

    for name in (
        "Xi",
        "Yi",
        "target_valid_mask",
        "owner_element_index",
        "valid_target_flat_indices",
        "q0",
        "q1",
        "q2",
        "basis_q0",
        "basis_q1",
        "basis_q2",
    ):
        assert not getattr(plan, name).flags.writeable, name
    with pytest.raises(ValueError):
        plan.Xi[0, 0] = -1.0


def test_unmapped_targets_remain_nan_between_disconnected_x_elements() -> None:
    data = _mesh(
        x_intervals=((0.0, 1.0), (2.0, 3.0)),
        y_intervals=((0.0, 1.0),),
    )
    plan = build_spectral_horizontal_slice_plan(data, nx=7, z_target=0.4)

    result = apply_spectral_horizontal_slice_plan(plan, data)

    gap_column = int(np.flatnonzero(np.isclose(plan.Xi[0], 1.5))[0])
    assert np.all(~plan.target_valid_mask[:, gap_column])
    assert np.all(plan.owner_element_index[:, gap_column] == -1)
    assert np.all(np.isnan(result[:, gap_column]))
    assert np.all(np.isfinite(result[:, [0, 1, 2, 4, 5, 6]]))


def test_nonstructured_spanwise_geometry_is_rejected() -> None:
    data = _mesh(y_intervals=((0.0, 1.0),))
    data.elem[0].pos[1] += 0.02 * data.elem[0].pos[0]

    with pytest.raises(ValueError, match="Structured spanwise geometry"):
        build_spectral_horizontal_slice_plan(data, nx=3, z_target=0.4)
