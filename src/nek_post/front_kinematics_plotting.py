"""Matplotlib figures for front-kinematics analysis."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

from nek_post.front_kinematics_io import (
    ensure_writable_output,
    front_position_figure_path,
    raw_velocity_figure_path,
    reconstruction_error_figure_path,
    reconstruction_figure_path,
    smoothed_velocity_figure_path,
)


def save_figure(fig, path: Path, overwrite: bool) -> None:
    """Save and close a figure using the established layout and DPI."""
    try:
        ensure_writable_output(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def plot_front_overlay(
    path: Path,
    kinematics_by_case: Mapping[str, Mapping[str, np.ndarray]],
    y_key: str,
    title: str,
    ylabel: str,
    overwrite: bool,
) -> None:
    """Write one multi-case front-kinematics overlay."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, kinematics in kinematics_by_case.items():
        ax.plot(kinematics["time"], kinematics[y_key], label=case)
    if y_key == "x_reconstruction_error":
        ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def plot_front_reconstruction(
    path: Path,
    case: str,
    kinematics: Mapping[str, np.ndarray],
    overwrite: bool,
) -> None:
    """Write one original-versus-reconstructed case figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kinematics["time"], kinematics["x_front"], label="original x_front")
    ax.plot(kinematics["time"], kinematics["x_reconstructed"], label="reconstructed x_front")
    ax.set_title(f"Front reconstruction: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("front position")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def write_front_kinematics_plots(
    output_dir: Path,
    kinematics_by_case: Mapping[str, Mapping[str, np.ndarray]],
    overwrite: bool,
) -> list[Path]:
    """Write all established front-kinematics figures in output order."""
    plot_specs = [
        (front_position_figure_path(output_dir), "x_front", "Front position vs time", "x_front"),
        (raw_velocity_figure_path(output_dir), "v_raw", "Raw front velocity vs time", "v_raw"),
        (
            smoothed_velocity_figure_path(output_dir),
            "v_smooth",
            "Smoothed front velocity vs time",
            "v_smooth",
        ),
        (
            reconstruction_error_figure_path(output_dir),
            "x_reconstruction_error",
            "Front reconstruction error vs time",
            "x_reconstructed - x_front",
        ),
    ]
    figure_paths: list[Path] = []
    for path, y_key, title, ylabel in plot_specs:
        plot_front_overlay(path, kinematics_by_case, y_key, title, ylabel, overwrite)
        figure_paths.append(path)

    for case, kinematics in kinematics_by_case.items():
        path = reconstruction_figure_path(output_dir, case)
        plot_front_reconstruction(path, case, kinematics, overwrite)
        figure_paths.append(path)
    return figure_paths
