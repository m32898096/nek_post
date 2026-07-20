import numpy as np
import pytest
from scipy.signal import savgol_filter

from nek_post.front_kinematics import (
    compute_kinematics,
    front_velocity,
    integrate_velocity,
    moving_average,
    odd_smoothing_window,
    reconstruction_error,
    smooth_velocity,
)


def test_odd_smoothing_window_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="Cannot smooth an empty array"):
        odd_smoothing_window(3, 0)


@pytest.mark.parametrize("requested", [-5, 0])
def test_odd_smoothing_window_clips_nonpositive_request_to_one(requested: int) -> None:
    assert odd_smoothing_window(requested, 5) == 1


def test_odd_smoothing_window_increases_even_request() -> None:
    assert odd_smoothing_window(4, 9) == 5


def test_odd_smoothing_window_clips_oversized_request() -> None:
    assert odd_smoothing_window(99, 5) == 5
    assert odd_smoothing_window(99, 6) == 5


def test_odd_smoothing_window_for_one_point_is_one() -> None:
    assert odd_smoothing_window(99, 1) == 1


def test_moving_average_matches_edge_normalized_script_behavior() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    expected = np.array([2.0, 3.0, 5.0, 6.0])

    np.testing.assert_allclose(moving_average(values, window=3), expected)


def test_moving_average_window_one_returns_copy_without_modifying_input() -> None:
    values = np.array([1.0, 2.0, 3.0])
    original = values.copy()

    actual = moving_average(values, window=1)

    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(values, original)
    assert actual is not values
    assert actual.shape == values.shape


def test_moving_average_oversized_window_uses_largest_available_odd_window() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])

    actual = moving_average(values, window=99)

    np.testing.assert_allclose(actual, [1.5, 2.0, 3.0, 3.5])
    assert actual.shape == values.shape


def test_smooth_velocity_moving_average_method_delegates_to_existing_average() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])

    actual = smooth_velocity(values, "moving_average", window=3, polyorder=2)

    np.testing.assert_allclose(actual, moving_average(values, 3))


def test_savgol_preserves_quadratic_polynomial() -> None:
    coordinate = np.arange(7, dtype=float)
    values = 2.0 * coordinate**2 - 3.0 * coordinate + 4.0

    actual = smooth_velocity(values, "savgol", window=5, polyorder=2)

    np.testing.assert_allclose(actual, values, rtol=1e-13, atol=1e-13)


def test_savgol_converts_even_window_and_clips_oversized_window() -> None:
    values = np.array([0.0, 2.0, 1.0, 5.0, 3.0, 4.0])

    even = smooth_velocity(values, "savgol", window=4, polyorder=2)
    oversized = smooth_velocity(values, "savgol", window=99, polyorder=2)
    expected = savgol_filter(values, window_length=5, polyorder=2, mode="interp")

    np.testing.assert_allclose(even, expected)
    np.testing.assert_allclose(oversized, expected)


def test_savgol_clips_polynomial_order_below_window_length() -> None:
    values = np.array([2.0, -1.0, 4.0, 0.5, 3.0])

    actual = smooth_velocity(values, "savgol", window=5, polyorder=99)

    np.testing.assert_allclose(actual, savgol_filter(values, 5, 4, mode="interp"), rtol=1e-13, atol=1e-13)


def test_savgol_effective_polyorder_below_one_falls_back_to_moving_average(
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    actual = smooth_velocity(values, "savgol", window=3, polyorder=0)

    np.testing.assert_allclose(actual, moving_average(values, 3))
    assert "falling back to moving_average" in capsys.readouterr().out


def test_unsupported_smoothing_method_currently_uses_savgol_path() -> None:
    values = np.array([0.0, 2.0, 1.0, 5.0, 3.0])

    actual = smooth_velocity(values, "unsupported", window=5, polyorder=2)

    np.testing.assert_allclose(actual, savgol_filter(values, 5, 2, mode="interp"))


def test_front_velocity_for_linear_position_is_constant() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])

    np.testing.assert_allclose(front_velocity(time, 2.5 * time + 4.0), 2.5, rtol=0.0, atol=1e-14)


def test_front_velocity_differentiates_quadratic_position() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])

    np.testing.assert_allclose(front_velocity(time, time**2), 2.0 * time, rtol=0.0, atol=1e-14)


def test_front_velocity_handles_nonuniform_time_coordinates() -> None:
    time = np.array([0.0, 0.5, 2.0, 4.0])

    np.testing.assert_allclose(front_velocity(time, time**2), 2.0 * time, rtol=0.0, atol=1e-14)


def test_front_velocity_two_points_uses_first_order_edges() -> None:
    np.testing.assert_allclose(front_velocity(np.array([0.0, 2.0]), np.array([1.0, 7.0])), [3.0, 3.0])


def test_integrate_velocity_uses_nonuniform_trapezoidal_rule() -> None:
    time = np.array([0.0, 1.0, 3.0])
    velocity = np.array([2.0, 4.0, 8.0])

    np.testing.assert_allclose(integrate_velocity(time, velocity, x0=10.0), [10.0, 13.0, 25.0])


def test_integrate_constant_velocity_reconstructs_line_and_preserves_initial_position() -> None:
    time = np.array([0.0, 0.5, 2.0, 4.0])
    velocity = np.full(time.shape, 3.0)

    actual = integrate_velocity(time, velocity, x0=7.0)

    np.testing.assert_allclose(actual, 7.0 + 3.0 * time)
    assert actual[0] == 7.0
    assert actual.shape == velocity.shape


def test_reconstruction_error_is_reconstructed_minus_original() -> None:
    reconstructed = np.array([1.0, 3.0, 4.0])
    original = np.array([2.0, 2.0, 5.0])

    np.testing.assert_array_equal(reconstruction_error(reconstructed, original), [-1.0, 1.0, -1.0])
    np.testing.assert_array_equal(reconstruction_error(original, original), np.zeros(3))


def test_compute_kinematics_preserves_linear_front_under_unit_window() -> None:
    front = {"time": np.array([0.0, 1.0, 2.0]), "x_front": np.array([5.0, 7.0, 9.0])}
    kinematics = compute_kinematics(front, method="moving_average", window=1, polyorder=3)

    assert set(kinematics) == {
        "time",
        "x_front",
        "v_raw",
        "v_smooth",
        "x_reconstructed",
        "x_reconstruction_error",
    }
    np.testing.assert_array_equal(kinematics["time"], front["time"])
    np.testing.assert_array_equal(kinematics["x_front"], front["x_front"])
    np.testing.assert_allclose(kinematics["v_raw"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(kinematics["v_smooth"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(kinematics["x_reconstructed"], front["x_front"])
    np.testing.assert_allclose(kinematics["x_reconstruction_error"], np.zeros(3))
    assert kinematics["x_reconstructed"][0] == front["x_front"][0]


def test_compute_kinematics_smoothing_changes_noisy_reconstruction_without_mutating_inputs() -> None:
    front = {
        "time": np.arange(6, dtype=float),
        "x_front": np.array([0.0, 1.0, 4.0, 4.5, 8.0, 9.0]),
    }
    original_time = front["time"].copy()
    original_x = front["x_front"].copy()

    unsmoothed = compute_kinematics(front, method="moving_average", window=1, polyorder=3)
    smoothed = compute_kinematics(front, method="moving_average", window=3, polyorder=3)

    assert not np.allclose(smoothed["x_reconstructed"], unsmoothed["x_reconstructed"])
    np.testing.assert_array_equal(front["time"], original_time)
    np.testing.assert_array_equal(front["x_front"], original_x)
