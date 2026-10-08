"""Small analytic elements; no simulation files or finite differences."""

from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import barycentric_basis_and_derivative, barycentric_weights, gll_nodes
from nek_post.spectral_derivatives import (
    element_jacobian,
    element_velocity_gradient,
    physical_gradient,
    reference_derivatives,
    transform_reference_gradient,
    velocity_gradient_tensor,
)
from nek_post.spectral_interpolation import _mapping_and_jacobian


def reference_grid(shape=(5, 6, 7)):
    t, s, r = np.meshgrid(*(gll_nodes(n) for n in shape), indexing="ij")
    return r, s, t


def affine_coordinates(matrix, shape=(5, 6, 7), origin=(0.3, -0.7, 1.2)):
    reference = np.stack(reference_grid(shape))
    return np.einsum("ij,j...->i...", matrix, reference) + np.asarray(origin)[:, None, None, None]


def curved_coordinates(shape=(5, 6, 7)):
    r, s, t = reference_grid(shape)
    return np.stack((r + .15*s**2, s + .1*t**2, t + .1*r**2))


@pytest.mark.parametrize("shape", ((2, 2, 2), (3, 4, 5), (8, 8, 8)))
def test_constant_and_linear_reference_fields(shape):
    r, s, t = reference_grid(shape)
    np.testing.assert_allclose(reference_derivatives(np.full(shape, 7.25)), 0, atol=3e-13)
    result = reference_derivatives(2*r - 3*s + 5*t + .25)
    assert result.shape == shape + (3,)
    np.testing.assert_allclose(result, np.broadcast_to([2, -3, 5], result.shape), atol=3e-13)


def test_tensor_polynomial_up_to_each_non_cubic_axis_degree():
    r, s, t = reference_grid((4, 5, 6))
    f = (1+r**5)*(1+2*s**4)*(1+3*t**3)
    expected = np.stack((
        5*r**4*(1+2*s**4)*(1+3*t**3),
        (1+r**5)*8*s**3*(1+3*t**3),
        (1+r**5)*(1+2*s**4)*9*t**2,
    ), axis=-1)
    np.testing.assert_allclose(reference_derivatives(f), expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("matrix", (
    np.diag([2., 3., .5]),
    np.array([[1.2, .3, -.2], [.1, 1.4, .4], [.2, -.1, .9]]),
    np.array([[0., 1., 0.], [0., 0., 2.], [3., 0., 0.]]),
    np.diag([-2., 3., .5]),
))
def test_affine_jacobian_signed_orientation_and_physical_gradient(matrix):
    xyz = affine_coordinates(matrix)
    jacobian, determinant = element_jacobian(xyz)
    np.testing.assert_allclose(jacobian, np.broadcast_to(matrix, jacobian.shape), atol=3e-13)
    np.testing.assert_allclose(determinant, np.linalg.det(matrix), atol=3e-13, rtol=3e-13)
    coefficients = np.array([2.5, -3.2, .7])
    f = np.einsum("i,i...->...", coefficients, xyz) + 4
    expected = np.broadcast_to(coefficients, f.shape + (3,))
    np.testing.assert_allclose(physical_gradient(f, xyz), expected, atol=4e-13, rtol=4e-13)
    np.testing.assert_allclose(
        transform_reference_gradient(reference_derivatives(f), jacobian), expected,
        atol=4e-13, rtol=4e-13,
    )


def test_curved_mapping_jacobian_and_analytic_physical_polynomial():
    r, s, t = reference_grid()
    xyz = curved_coordinates()
    x, y, z = xyz
    jacobian, determinant = element_jacobian(xyz)
    expected_j = np.broadcast_to(np.eye(3), jacobian.shape).copy()
    expected_j[..., 0, 1] = .3*s
    expected_j[..., 1, 2] = .2*t
    expected_j[..., 2, 0] = .2*r
    np.testing.assert_allclose(jacobian, expected_j, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(determinant, 1 + .012*r*s*t, atol=3e-13, rtol=3e-13)
    f = x*x + 2*y*y + 3*z*z + 4*x*y + 5*y*z + 6*z*x
    expected = np.stack((2*x+4*y+6*z, 4*y+4*x+5*z, 6*z+5*y+6*x), axis=-1)
    np.testing.assert_allclose(physical_gradient(f, xyz), expected, atol=3e-12, rtol=3e-12)


def test_jacobian_column_reversal_matches_existing_interpolation_convention():
    xyz = curved_coordinates()
    jacobian, _ = element_jacobian(xyz)
    for index in ((0, 2, 3), (2, 3, 4), (4, 5, 6)):
        basis_data = tuple(
            barycentric_basis_and_derivative(gll_nodes(n), barycentric_weights(gll_nodes(n)), gll_nodes(n)[i])
            for n, i in zip(xyz.shape[1:], index)
        )
        _, array_axis_jacobian = _mapping_and_jacobian(tuple(xyz), basis_data)
        np.testing.assert_allclose(jacobian[index], array_axis_jacobian[:, ::-1], atol=3e-13)


def test_velocity_tensor_component_order_and_reader_element_adapter():
    xyz = affine_coordinates(np.array([[1.2, .3, -.2], [.1, 1.4, .4], [.2, -.1, .9]]))
    matrix = np.array([[1., 2., 3.], [-4., 5., -6.], [7., -8., 9.]])
    velocity = np.einsum("ij,j...->i...", matrix, xyz) + np.array([.1, .2, .3])[:, None, None, None]
    tensor = velocity_gradient_tensor(xyz, velocity)
    assert tensor.shape == xyz.shape[1:] + (3, 3)
    np.testing.assert_allclose(tensor, np.broadcast_to(matrix, tensor.shape), atol=3e-12, rtol=3e-12)
    element = SimpleNamespace(pos=xyz, vel=velocity)
    np.testing.assert_array_equal(element_velocity_gradient(element), tensor)


def test_curved_velocity_gradients_against_analytic_component_derivatives():
    xyz = curved_coordinates()
    x, y, z = xyz
    velocity = (x*x + 2*y*z, 3*y*y + x*z, z*z + x*y)
    expected = np.stack((
        np.stack((2*x, 2*z, 2*y), axis=-1),
        np.stack((z, 6*y, x), axis=-1),
        np.stack((y, x, 2*z), axis=-1),
    ), axis=-2)
    np.testing.assert_allclose(velocity_gradient_tensor(xyz, velocity), expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("kind", ("collapsed", "dependent", "near_singular", "folded", "overflow"))
def test_singular_or_invalid_jacobians_are_rejected(kind):
    r, s, t = reference_grid((4, 4, 4))
    if kind == "collapsed": xyz = (r, s, np.zeros_like(t))
    elif kind == "dependent": xyz = (r+s, 2*(r+s), t)
    elif kind == "near_singular": xyz = (r, s, 1e-14*t)
    elif kind == "folded": xyz = (r*r, s, t)
    else: xyz = (1e110*r, 1e110*s, 1e110*t)
    with pytest.raises(ValueError, match="[Jj]acobian"):
        element_jacobian(xyz)


def test_jacobian_tolerance_is_scale_relative_and_configurable():
    r, s, t = reference_grid((3, 4, 5))
    xyz = (1e-20*r, 1e-20*s, 1e-20*t)
    _, determinant = element_jacobian(xyz)
    np.testing.assert_allclose(determinant, 1e-60, rtol=3e-13, atol=0)
    expected = np.broadcast_to([2., 3., -4.], r.shape + (3,))
    np.testing.assert_allclose(physical_gradient(2*xyz[0]+3*xyz[1]-4*xyz[2], xyz), expected, atol=3e-13)
    element_jacobian((r, s, 1e-14*t), singular_rtol=1e-16)


@pytest.mark.parametrize("tolerance", (-1, 1, np.nan, np.inf, True, None, "bad"))
def test_invalid_singular_tolerances(tolerance):
    with pytest.raises(ValueError, match="singular_rtol"):
        element_jacobian(reference_grid(), singular_rtol=tolerance)


@pytest.mark.parametrize("values", (
    np.zeros((4, 4)), np.zeros((2, 1, 3)), np.full((3, 3, 3), np.nan),
    np.full((3, 3, 3), np.inf), np.ones((3, 3, 3), dtype=complex), "bad",
))
def test_invalid_scalar_arrays(values):
    with pytest.raises(ValueError):
        reference_derivatives(values)


def test_coordinate_velocity_and_transform_shape_validation():
    xyz = reference_grid()
    jacobian, _ = element_jacobian(xyz)
    with pytest.raises(ValueError, match="exactly three"):
        element_jacobian(xyz[:2])
    with pytest.raises(ValueError, match="shapes must match"):
        element_jacobian((xyz[0], xyz[1], np.zeros((3, 4, 5))))
    with pytest.raises(ValueError, match="Scalar field shape"):
        physical_gradient(np.ones((3, 4, 5)), xyz)
    with pytest.raises(ValueError, match="Velocity field shape"):
        velocity_gradient_tensor(xyz, np.zeros((3, 3, 4, 5)))
    with pytest.raises(ValueError, match="exactly three"):
        velocity_gradient_tensor(xyz, xyz[:2])
    with pytest.raises(ValueError, match="derivative shape"):
        transform_reference_gradient(np.zeros((3, 4, 5, 3)), jacobian)
    with pytest.raises(ValueError, match="jacobian must have shape"):
        transform_reference_gradient(np.zeros((3, 4, 5, 3)), np.eye(3))
    invalid = jacobian.copy()
    invalid[1, 2, 3] = 0
    with pytest.raises(ValueError, match=r"index \(1, 2, 3\)"):
        transform_reference_gradient(reference_derivatives(xyz[0]), invalid)


def test_finite_and_real_validation_applies_to_coordinates_and_velocities():
    xyz = np.stack(reference_grid())
    bad = xyz.copy()
    bad[1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        element_jacobian(bad)
    with pytest.raises(ValueError, match="real"):
        velocity_gradient_tensor(xyz, xyz.astype(complex))
    with pytest.raises(ValueError, match="finite"):
        velocity_gradient_tensor(xyz, bad)


def test_float32_and_read_only_inputs_are_preserved():
    xyz = np.stack(reference_grid()).astype(np.float32)
    velocity = np.zeros_like(xyz)
    xyz.setflags(write=False)
    velocity.setflags(write=False)
    before = xyz.copy()
    tensor = velocity_gradient_tensor(xyz, velocity)
    assert tensor.dtype == np.float64
    np.testing.assert_array_equal(tensor, 0)
    np.testing.assert_array_equal(xyz, before)
    assert not xyz.flags.writeable and not velocity.flags.writeable
