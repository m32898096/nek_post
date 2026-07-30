from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_nodes,
)
from nek_post.spectral_interpolation import (
    SpectralGeometryMismatchError,
    apply_spectral_slice_interpolation_plan,
    build_spectral_slice_interpolation_plan,
    evaluate_tensor_bary3,
    invert_map_newton_3d,
    validate_spectral_geometry,
)


def _reference_grid(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(
        *(gll_nodes(size) for size in shape),
        indexing="ij",
    )


def _element(
    *,
    shape: tuple[int, int, int] = (4, 4, 4),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    matrix: np.ndarray | None = None,
    curved: bool = False,
    concentration: np.ndarray | None = None,
) -> SimpleNamespace:
    q0, q1, q2 = _reference_grid(shape)
    transform = np.eye(3) if matrix is None else np.asarray(matrix, dtype=float)
    reference = np.stack((q0, q1, q2))
    physical = np.einsum("ij,jabc->iabc", transform, reference)
    physical += np.asarray(origin, dtype=float)[:, None, None, None]
    if curved:
        physical[0] += 0.08 * q0 * q1
        physical[1] += 0.05 * q1 * q2
        physical[2] += 0.06 * q0 * q2
    if concentration is None:
        concentration = 0.4 + q0 - 0.5 * q1 + 0.25 * q2
    return SimpleNamespace(
        pos=physical,
        temp=np.asarray([concentration], dtype=float),
        scal=None,
    )


def _data(*elements: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(elem=list(elements))


def _basis(nodes: np.ndarray, q: float) -> np.ndarray:
    return barycentric_basis_and_derivative(
        nodes, barycentric_weights(nodes), q
    )[0]


@pytest.mark.parametrize("shape", ((4, 5, 6), (8, 8, 8)))
def test_tensor_interpolation_is_exact_for_supported_separable_polynomial(
    shape: tuple[int, int, int],
) -> None:
    q0, q1, q2 = _reference_grid(shape)
    values = (
        (1.0 + q0 + q0 ** (shape[0] - 1))
        * (0.5 - q1 + 0.2 * q1 ** (shape[1] - 1))
        * (2.0 + q2 ** (shape[2] - 1))
    )
    target = (0.17, -0.23, 0.61)
    rows = tuple(
        _basis(gll_nodes(size), q)
        for size, q in zip(shape, target, strict=True)
    )
    expected = (
        (1.0 + target[0] + target[0] ** (shape[0] - 1))
        * (0.5 - target[1] + 0.2 * target[1] ** (shape[1] - 1))
        * (2.0 + target[2] ** (shape[2] - 1))
    )

    assert evaluate_tensor_bary3(values, *rows) == pytest.approx(
        expected, rel=2.0e-12, abs=2.0e-12
    )


def test_tensor_interpolation_is_exact_at_all_nodes_and_for_constant_linear_fields() -> None:
    shape = (3, 4, 5)
    q0, q1, q2 = _reference_grid(shape)
    values = 2.5 + 3.0 * q0 - 0.2 * q1 + 1.7 * q2

    for index in np.ndindex(shape):
        rows = tuple(
            _basis(gll_nodes(shape[axis]), gll_nodes(shape[axis])[index[axis]])
            for axis in range(3)
        )
        assert evaluate_tensor_bary3(values, *rows) == values[index]

    rows = tuple(
        _basis(gll_nodes(shape[axis]), q)
        for axis, q in enumerate((0.2, -0.7, 0.4))
    )
    assert evaluate_tensor_bary3(np.full(shape, 7.25), *rows) == pytest.approx(7.25)
    assert evaluate_tensor_bary3(values, *rows) == pytest.approx(
        2.5 + 3.0 * 0.2 - 0.2 * -0.7 + 1.7 * 0.4
    )


@pytest.mark.parametrize(
    ("origin", "matrix", "q"),
    (
        ((0.0, 0.0, 0.0), np.eye(3), (0.2, -0.4, 0.7)),
        (
            (4.0, -2.0, 1.0),
            np.diag((3.0, 0.5, 2.0)),
            (-0.61, 0.13, 0.45),
        ),
        (
            (1.0, 2.0, -1.0),
            np.asarray(
                (
                    (0.8, -0.6, 0.1),
                    (0.6, 0.8, 0.2),
                    (0.0, 0.1, 1.3),
                )
            ),
            (0.71, -0.26, -0.42),
        ),
    ),
)
def test_inverse_mapping_recovers_affine_cube_stretch_and_rotation(
    origin: tuple[float, float, float],
    matrix: np.ndarray,
    q: tuple[float, float, float],
) -> None:
    element = _element(origin=origin, matrix=matrix)
    target = np.asarray(origin) + matrix @ np.asarray(q)

    result = invert_map_newton_3d(*element.pos, target)

    assert result.converged
    np.testing.assert_allclose(result.q, q, rtol=0.0, atol=3.0e-12)
    assert result.residual < 1.0e-10


def test_inverse_mapping_recovers_smooth_curved_element() -> None:
    element = _element(shape=(6, 6, 6), origin=(1.0, -0.2, 2.0), curved=True)
    q = np.asarray((0.31, -0.44, 0.58))
    q0, q1, q2 = q
    target = np.asarray(
        (
            1.0 + q0 + 0.08 * q0 * q1,
            -0.2 + q1 + 0.05 * q1 * q2,
            2.0 + q2 + 0.06 * q0 * q2,
        )
    )

    result = invert_map_newton_3d(*element.pos, target)

    assert result.converged
    np.testing.assert_allclose(result.q, q, rtol=0.0, atol=2.0e-11)


def test_inverse_mapping_rejects_outside_point_and_singular_jacobian() -> None:
    regular = _element()
    outside = invert_map_newton_3d(*regular.pos, (2.0, 0.0, 0.0))
    assert not outside.converged
    assert outside.reason == "outside_reference_element"

    q0, q1, q2 = _reference_grid((4, 4, 4))
    del q2
    singular = invert_map_newton_3d(
        q0,
        q1,
        np.zeros_like(q0),
        (0.2, -0.3, 0.1),
    )
    assert not singular.converged
    assert singular.reason == "singular_jacobian"


def test_plan_detects_order_assigns_adjacent_elements_without_interface_gaps() -> None:
    left = _element(
        shape=(8, 8, 8),
        origin=(0.0, 0.0, 0.0),
        matrix=np.diag((1.0, 1.0, 1.0)),
    )
    right = _element(
        shape=(8, 8, 8),
        origin=(2.0, 0.0, 0.0),
        matrix=np.diag((1.0, 1.0, 1.0)),
    )
    plan = build_spectral_slice_interpolation_plan(
        _data(left, right), nx=5, nz=3, y_target=0.0
    )

    assert plan.element_shape == (8, 8, 8)
    assert plan.polynomial_order == (7, 7, 7)
    assert np.all(plan.target_valid_mask)
    assert np.all(plan.owner_element_index[:, :3] == 0)
    assert np.all(plan.owner_element_index[:, 3:] == 1)
    assert plan.inverse_mapping_diagnostics.ambiguous_boundary_point_count == 3


def test_non_n7_order_is_detected_and_factorized_basis_avoids_tensor_weights() -> None:
    plan = build_spectral_slice_interpolation_plan(
        _data(_element(shape=(6, 6, 6))), nx=4, nz=3, y_target=0.0
    )

    assert plan.element_shape == (6, 6, 6)
    assert plan.polynomial_order == (5, 5, 5)
    valid_count = int(np.count_nonzero(plan.target_valid_mask))
    assert plan.basis_q0.shape == (valid_count, 6)
    assert plan.basis_q1.shape == (valid_count, 6)
    assert plan.basis_q2.shape == (valid_count, 6)
    assert not hasattr(plan, "tensor_weights")


def test_points_outside_disconnected_mesh_remain_nan() -> None:
    left = _element(origin=(-2.0, 0.0, 0.0))
    right = _element(origin=(2.0, 0.0, 0.0))
    plan = build_spectral_slice_interpolation_plan(
        _data(left, right), nx=7, nz=3, y_target=0.0
    )
    result = apply_spectral_slice_interpolation_plan(
        plan, _data(left, right)
    )

    assert np.all(np.isnan(result[:, 3]))
    np.testing.assert_array_equal(np.isfinite(result), plan.target_valid_mask)


def test_plan_application_preserves_overshoots_without_clipping() -> None:
    shape = (4, 4, 4)
    q0, _q1, _q2 = _reference_grid(shape)
    concentration = 2.0 * q0
    element = _element(shape=shape, concentration=concentration)
    plan = build_spectral_slice_interpolation_plan(
        _data(element), nx=3, nz=3, y_target=0.0
    )

    result = apply_spectral_slice_interpolation_plan(plan, _data(element))

    assert np.nanmin(result) == pytest.approx(-2.0)
    assert np.nanmax(result) == pytest.approx(2.0)


def test_inconsistent_element_and_concentration_shapes_fail() -> None:
    with pytest.raises(ValueError, match="same shape"):
        build_spectral_slice_interpolation_plan(
            _data(_element(shape=(4, 4, 4)), _element(shape=(5, 5, 5))),
            nx=3,
            nz=3,
        )

    element = _element(shape=(4, 4, 4))
    element.temp = np.zeros((1, 4, 4, 3))
    with pytest.raises(ValueError, match="concentration shape mismatch"):
        build_spectral_slice_interpolation_plan(
            _data(element), nx=3, nz=3
        )


def test_geometry_reuse_passes_unchanged_and_rejects_coordinate_count_and_shape() -> None:
    data = _data(_element(), _element(origin=(2.0, 0.0, 0.0)))
    plan = build_spectral_slice_interpolation_plan(
        data, nx=5, nz=3, y_target=0.0
    )
    validate_spectral_geometry(plan, deepcopy(data), source_file="same.f00002")

    changed_coordinate = deepcopy(data)
    changed_coordinate.elem[1].pos[0, 1, 1, 1] += 1.0e-12
    with pytest.raises(
        SpectralGeometryMismatchError, match=r"changed\.f00002, element 1.*signature"
    ):
        validate_spectral_geometry(
            plan, changed_coordinate, source_file="changed.f00002"
        )

    with pytest.raises(SpectralGeometryMismatchError, match="expected 2 elements"):
        validate_spectral_geometry(plan, _data(data.elem[0]), source_file="short")

    changed_shape = deepcopy(data)
    changed_shape.elem[0] = _element(shape=(5, 5, 5))
    with pytest.raises(
        SpectralGeometryMismatchError, match=r"element 0.*expected shape"
    ):
        validate_spectral_geometry(plan, changed_shape, source_file="reshaped")
