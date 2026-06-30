import numpy as np

from nek_post.metrics import linf_error, mean_absolute_error, relative_l2_error


def test_relative_l2_error() -> None:
    values = np.array([2.0, 4.0, 6.0])
    reference = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.norm(values - reference) / np.linalg.norm(reference)
    assert relative_l2_error(values, reference) == expected


def test_linf_error() -> None:
    values = np.array([1.0, -1.0, 4.0])
    reference = np.array([0.0, 2.0, 1.0])
    assert linf_error(values, reference) == 3.0


def test_mean_absolute_error_uses_valid_mask_and_finite_values() -> None:
    difference = np.array([1.0, -2.0, np.nan, 8.0])
    mask = np.array([True, True, True, False])
    assert mean_absolute_error(difference, mask) == 1.5


def test_mean_absolute_error_returns_nan_without_valid_points() -> None:
    difference = np.array([np.nan, 1.0])
    mask = np.array([True, False])
    assert np.isnan(mean_absolute_error(difference, mask))
