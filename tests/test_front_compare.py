import numpy as np

from nek_post.front_compare import compare_front_to_paper, mean_abs, rms, slumping_velocity_metrics


def test_compare_front_to_paper_interpolates_onto_overlapping_paper_times() -> None:
    front = {"time": np.array([0.0, 2.0, 4.0]), "x": np.array([0.0, 4.0, 8.0])}
    paper = {"time": np.array([1.0, 3.0, 5.0]), "x": np.array([1.0, 7.0, 10.0])}

    comparison = compare_front_to_paper(
        front,
        paper,
        front_x_key="x",
        interpolated_key="simulation_x_interp",
        error_key="x_error",
    )

    np.testing.assert_allclose(comparison["time"], np.array([1.0, 3.0]))
    np.testing.assert_allclose(comparison["simulation_x_interp"], np.array([2.0, 6.0]))
    np.testing.assert_allclose(comparison["x_error"], np.array([1.0, -1.0]))
    np.testing.assert_allclose(comparison["relative_error"], np.array([1.0, -1.0 / 7.0]))


def test_slumping_velocity_metrics_uses_requested_window_and_relative_difference() -> None:
    comparison = {
        "time": np.array([0.0, 3.0, 6.0, 12.0, 15.0]),
        "paper_x": np.array([0.0, 6.0, 12.0, 24.0, 30.0]),
        "simulation_x_interp": np.array([0.0, 9.0, 18.0, 36.0, 45.0]),
        "relative_error": np.array([np.nan, 0.5, 0.5, 0.5, 0.5]),
    }

    metrics = slumping_velocity_metrics(
        comparison,
        compared_x_key="simulation_x_interp",
        compared_label="simulation",
        tmin=3.0,
        tmax=12.0,
        include_relative_error_stats=True,
    )

    assert metrics["n_slumping_points"] == 3
    assert np.isclose(metrics["paper_slumping_velocity"], 2.0)
    assert np.isclose(metrics["simulation_slumping_velocity"], 3.0)
    assert np.isclose(metrics["slumping_velocity_difference"], 1.0)
    assert np.isclose(metrics["slumping_velocity_relative_difference"], 0.5)
    assert np.isclose(metrics["slumping_mean_abs_relative_error"], 0.5)


def test_front_compare_metrics_can_ignore_nonfinite_values() -> None:
    values = np.array([3.0, np.nan, -4.0])
    assert mean_abs(values, finite_only=True) == 3.5
    assert np.isclose(rms(values, finite_only=True), 5.0 / np.sqrt(2.0))
