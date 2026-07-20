"""Front-position kinematics helpers."""

from __future__ import annotations

import numpy as np


def odd_smoothing_window(window: int, n_points: int) -> int:
    """Return an odd smoothing window clipped to the available samples."""
    if n_points < 1:
        raise ValueError("Cannot smooth an empty array.")
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return max(1, window)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth values using the existing edge-normalized moving average."""
    values_arr = np.asarray(values, dtype=float)
    window = odd_smoothing_window(window, values_arr.size)
    if window == 1:
        return values_arr.copy()
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values_arr, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values_arr, dtype=float), kernel, mode="same")
    return numerator / denominator


def smooth_velocity(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    """Smooth velocity by moving average or Savitzky-Golay with scipy fallback."""
    values_arr = np.asarray(values, dtype=float)
    window = odd_smoothing_window(window, values_arr.size)
    if method == "moving_average" or window == 1:
        return moving_average(values_arr, window)

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("WARNING: scipy is unavailable; falling back to moving_average smoothing.")
        return moving_average(values_arr, window)

    polyorder = min(max(0, int(polyorder)), window - 1)
    if polyorder < 1:
        print("WARNING: Savitzky-Golay window is too small for requested polyorder; falling back to moving_average.")
        return moving_average(values_arr, window)
    return savgol_filter(values_arr, window_length=window, polyorder=polyorder, mode="interp")


def front_velocity(time: np.ndarray, x_front: np.ndarray) -> np.ndarray:
    """Compute ``dx_front/dt`` using ``numpy.gradient``."""
    time_arr = np.asarray(time, dtype=float)
    x_arr = np.asarray(x_front, dtype=float)
    edge_order = 2 if time_arr.size >= 3 else 1
    return np.gradient(x_arr, time_arr, edge_order=edge_order)


def integrate_velocity(time: np.ndarray, velocity: np.ndarray, x0: float) -> np.ndarray:
    """Integrate velocity using the trapezoidal rule from the initial position."""
    time_arr = np.asarray(time, dtype=float)
    velocity_arr = np.asarray(velocity, dtype=float)
    reconstructed = np.empty_like(velocity_arr, dtype=float)
    reconstructed[0] = x0
    increments = 0.5 * (velocity_arr[1:] + velocity_arr[:-1]) * np.diff(time_arr)
    reconstructed[1:] = x0 + np.cumsum(increments)
    return reconstructed


def reconstruction_error(x_reconstructed: np.ndarray, x_front: np.ndarray) -> np.ndarray:
    """Return reconstructed minus original front position."""
    return np.asarray(x_reconstructed, dtype=float) - np.asarray(x_front, dtype=float)


def compute_kinematics(front: dict[str, np.ndarray], method: str, window: int, polyorder: int) -> dict[str, np.ndarray]:
    """Compute raw/smoothed front velocity and reconstructed front position."""
    time = np.asarray(front["time"], dtype=float)
    x_front = np.asarray(front["x_front"], dtype=float)
    v_raw = front_velocity(time, x_front)
    v_smooth = smooth_velocity(v_raw, method, window, polyorder)
    x_reconstructed = integrate_velocity(time, v_smooth, float(x_front[0]))
    error = reconstruction_error(x_reconstructed, x_front)
    return {
        "time": time,
        "x_front": x_front,
        "v_raw": v_raw,
        "v_smooth": v_smooth,
        "x_reconstructed": x_reconstructed,
        "x_reconstruction_error": error,
    }
