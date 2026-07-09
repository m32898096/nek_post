import numpy as np

from nek_post.front_kinematics import compute_kinematics, integrate_velocity, moving_average


def test_moving_average_matches_edge_normalized_script_behavior() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    expected = np.array([2.0, 3.0, 5.0, 6.0])
    np.testing.assert_allclose(moving_average(values, window=3), expected)


def test_integrate_velocity_uses_trapezoidal_rule() -> None:
    time = np.array([0.0, 1.0, 3.0])
    velocity = np.array([2.0, 4.0, 8.0])
    np.testing.assert_allclose(integrate_velocity(time, velocity, x0=10.0), np.array([10.0, 13.0, 25.0]))


def test_compute_kinematics_preserves_linear_front_under_unit_window() -> None:
    front = {"time": np.array([0.0, 1.0, 2.0]), "x_front": np.array([5.0, 7.0, 9.0])}
    kinematics = compute_kinematics(front, method="moving_average", window=1, polyorder=3)

    np.testing.assert_allclose(kinematics["v_raw"], np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(kinematics["v_smooth"], np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(kinematics["x_reconstructed"], front["x_front"])
    np.testing.assert_allclose(kinematics["x_reconstruction_error"], np.zeros(3))
