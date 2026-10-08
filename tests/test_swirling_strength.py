"""Deterministic analytic spectra and small projected-gradient integration."""

from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.gll import gll_nodes
from nek_post.l2_projection import build_l2_projection_operator, project_velocity_gradient
from nek_post.spectral_derivatives import velocity_gradient_tensor
from nek_post.swirling_strength import compute_swirling_strength


def rotation(omega, axial=0., transverse=0.):
    return np.array([[transverse, -omega, 0.], [omega, transverse, 0.], [0., 0., axial]])


@pytest.mark.parametrize("matrix", (
    np.zeros((3, 3)),
    np.array([[0., 3., 0.], [0., 0., 0.], [0., 0., 0.]]),
    np.array([[0., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
    np.diag([1., -2., 3.]),
    np.diag([-4., 0., -4.]),
    np.array([[1., 5., -7.], [0., 2., 3.], [0., 0., -3.]]),
))
def test_zero_shear_and_real_strain_spectra(matrix):
    result = compute_swirling_strength(matrix)
    assert result.lambda_ci.shape == ()
    assert result.lambda_ci.dtype == np.float64
    assert result.lambda_ci == 0
    assert result.lambda_ci_squared == 0


@pytest.mark.parametrize("omega", (-7., -1., 0., .25, 3., 1e-12))
@pytest.mark.parametrize("axial,transverse", ((0., 0.), (4., 0.), (-3., 0.), (5., -2.)))
def test_rotation_with_extension_compression_and_real_pair_offset(omega, axial, transverse):
    result = compute_swirling_strength(rotation(omega, axial, transverse))
    assert result.lambda_ci == pytest.approx(abs(omega), rel=1e-14, abs=1e-25)
    assert result.lambda_ci_squared == pytest.approx(omega**2, rel=2e-14, abs=1e-30)
    assert result.lambda_ci >= 0


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_rotation_about_each_physical_axis(axis):
    matrix = np.roll(np.roll(rotation(2.5, axial=-1.), axis, axis=0), axis, axis=1)
    assert compute_swirling_strength(matrix).lambda_ci == pytest.approx(2.5)


def test_tensor_components_and_stage1_cartesian_convention():
    t, s, r = np.meshgrid(*(gll_nodes(n) for n in (3, 4, 5)), indexing="ij")
    coordinates = (2*r, 3*s, .5*t)
    x, y, z = coordinates
    # Eigenvalues are +/-6i and -2; unequal cross-components detect reshaping errors.
    gradient = velocity_gradient_tensor(coordinates, (-4*y, 9*x, -2*z))
    expected = np.array([[0., -4., 0.], [9., 0., 0.], [0., 0., -2.]])
    np.testing.assert_allclose(gradient, np.broadcast_to(expected, gradient.shape), atol=2e-13)
    result = compute_swirling_strength(gradient)
    assert result.lambda_ci.shape == (3, 4, 5)
    np.testing.assert_allclose(result.lambda_ci, 6., atol=5e-14)
    np.testing.assert_allclose(result.lambda_ci_squared, 36., atol=6e-13)


@pytest.mark.parametrize("shape", ((7,), (5, 2), (7, 3, 4, 2)))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_multiple_nodes_chunked_equals_one_chunk_and_preserves_input(shape, dtype):
    omega = np.arange(np.prod(shape)).reshape(shape)/4 - 2
    gradients = np.zeros((*shape, 3, 3), dtype=dtype)
    gradients[..., 0, 1] = -omega
    gradients[..., 1, 0] = omega
    gradients[..., 2, 2] = -3
    gradients.setflags(write=False)
    saved = gradients.copy()
    chunked = compute_swirling_strength(gradients, chunk_size=2)
    one_chunk = compute_swirling_strength(gradients, chunk_size=100)
    assert chunked.lambda_ci.shape == shape
    np.testing.assert_allclose(chunked.lambda_ci, np.abs(omega), atol=5e-14)
    np.testing.assert_array_equal(chunked.lambda_ci, one_chunk.lambda_ci)
    np.testing.assert_array_equal(chunked.lambda_ci_squared, chunked.lambda_ci**2)
    np.testing.assert_array_equal(gradients, saved)
    assert chunked.lambda_ci.dtype == chunked.lambda_ci_squared.dtype == np.float64
    assert (chunked.lambda_ci >= 0).all()


def test_noncontiguous_input_and_integer_matrices():
    values = np.broadcast_to(rotation(-3.), (9, 4, 2, 3, 3)).copy()[::2, ::-1]
    assert not values.flags.c_contiguous
    result = compute_swirling_strength(values, chunk_size=2)
    np.testing.assert_allclose(result.lambda_ci, 3.)
    np.testing.assert_allclose(compute_swirling_strength(rotation(4.).astype(np.int64)).lambda_ci, 4.)


def test_chunk_bound_and_no_whole_float64_input_copy(monkeypatch):
    original = np.linalg.eigvals
    values = np.broadcast_to(rotation(2.), (7, 2, 3, 4, 3, 3)).copy()
    seen = []
    def observe(chunk):
        seen.append((chunk.shape, chunk.dtype, np.shares_memory(chunk, values)))
        return original(chunk)
    monkeypatch.setattr(np.linalg, "eigvals", observe)
    compute_swirling_strength(values, chunk_size=np.int64(2))
    assert [v[0][0] for v in seen] == [2, 2, 2, 1]
    assert all(dtype == np.float64 and shares for _, dtype, shares in seen)
    seen.clear()
    compute_swirling_strength(values.astype(np.float32), chunk_size=2)
    assert max(shape[0] for shape, _, _ in seen) == 2
    assert all(dtype == np.float64 for _, dtype, _ in seen)


def test_project_gradient_before_eigenvalues_not_scalar_afterward():
    t, s, r = np.meshgrid(*(gll_nodes(3) for _ in range(3)), indexing="ij")
    data = SimpleNamespace(elem=[SimpleNamespace(pos=np.stack((cx+(r+1)/2, (s+1)/2, (t+1)/2))) for cx in (0, 1)])
    raw = np.empty((2, 3, 3, 3, 3, 3))
    raw[0], raw[1] = rotation(2.), rotation(-2.)
    operator = build_l2_projection_operator(data, periodic_axes=())
    projected = project_velocity_gradient(operator, raw)
    result = compute_swirling_strength(projected.values, chunk_size=1)
    np.testing.assert_array_equal(result.lambda_ci[0, :, :, -1], 0)
    np.testing.assert_array_equal(result.lambda_ci[1, :, :, 0], 0)
    np.testing.assert_allclose(result.lambda_ci[0, :, :, 0], 2.)
    np.testing.assert_allclose(result.lambda_ci[1, :, :, -1], 2.)
    np.testing.assert_allclose(compute_swirling_strength(raw).lambda_ci, 2.)


@pytest.mark.parametrize("bad", (None, 1., [], np.zeros(3), np.zeros((2, 2)), np.zeros((3, 4)), np.zeros((2, 3, 2)), np.zeros((0, 3, 3)), np.zeros((2, 0, 3, 3)), [[1], [1, 2]]))
def test_invalid_shapes(bad):
    with pytest.raises(ValueError): compute_swirling_strength(bad)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_nonfinite_input_in_later_chunk(bad):
    values = np.broadcast_to(rotation(1.), (5, 3, 3)).copy()
    values[-1, 0, 0] = bad
    with pytest.raises(ValueError, match=r"\[4:5\]"):
        compute_swirling_strength(values, chunk_size=2)


@pytest.mark.parametrize("dtype", (complex, object, str, bool))
def test_nonreal_or_nonnumeric_inputs_rejected(dtype):
    with pytest.raises(ValueError, match="real numeric"):
        compute_swirling_strength(rotation(1.).astype(dtype))


@pytest.mark.parametrize("chunk_size", (0, -1, True, np.bool_(False), 1.5, None, "2"))
def test_invalid_chunk_sizes(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        compute_swirling_strength(rotation(1.), chunk_size=chunk_size)


def test_eigensolver_failure_has_chunk_context(monkeypatch):
    def fail(_): raise np.linalg.LinAlgError("test failure")
    monkeypatch.setattr(np.linalg, "eigvals", fail)
    with pytest.raises(np.linalg.LinAlgError, match=r"\[0:1\]"):
        compute_swirling_strength(rotation(1.))


@pytest.mark.parametrize("bad", (np.nan, np.inf, complex(0., np.inf)))
def test_nonfinite_eigenvalues_rejected(monkeypatch, bad):
    monkeypatch.setattr(np.linalg, "eigvals", lambda a: np.full(a.shape[:-1], bad))
    with pytest.raises(FloatingPointError, match="eigenvalues"):
        compute_swirling_strength(rotation(1.))


def test_squared_overflow_rejected_and_small_strength_not_thresholded():
    with pytest.raises(FloatingPointError, match="squared"):
        compute_swirling_strength(rotation(1e200))
    result = compute_swirling_strength(rotation(1e-100))
    assert result.lambda_ci > 0
    assert result.lambda_ci == pytest.approx(1e-100, rel=1e-14, abs=0)
    assert result.lambda_ci_squared == pytest.approx(1e-200, rel=2e-14, abs=0)
