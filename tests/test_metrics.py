import numpy as np

from nek_post.metrics import linf_error, relative_l2_error


def test_relative_l2_error() -> None:
    values = np.array([2.0, 4.0, 6.0])
    reference = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.norm(values - reference) / np.linalg.norm(reference)
    assert relative_l2_error(values, reference) == expected


def test_linf_error() -> None:
    values = np.array([1.0, -1.0, 4.0])
    reference = np.array([0.0, 2.0, 1.0])
    assert linf_error(values, reference) == 3.0
