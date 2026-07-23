"""Matplotlib figures for energy-budget closure diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt

from nek_post.energy_budget import EnergyBudgetDiagnostics
from nek_post.energy_budget_io import (
    closure_drift_figure_path,
    closure_residual_figure_path,
    component_figure_path,
    differential_closure_residual_figure_path,
    energy_closure_figure_path,
    ensure_writable_output,
    epsilon_figure_path,
)


def save_figure(fig, path: Path, overwrite: bool) -> None:
    """Save and close a figure with the established layout and resolution."""
    try:
        ensure_writable_output(path, overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def plot_energy_components(
    path: Path,
    case: str,
    diagnostics: EnergyBudgetDiagnostics,
    target: float,
    overwrite: bool,
) -> None:
    """Write one case's energy-budget components and closure figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = slice(None, None, 2)
    component_specs = (
        (diagnostics.E_k, "E_k", "black"),
        (diagnostics.E_p, "E_p", "red"),
        (diagnostics.E_total, "E_total", "blue"),
        (diagnostics.epsilon, "epsilon", "purple"),
    )
    for values, label, color in component_specs:
        ax.plot(
            diagnostics.time[sample],
            values[sample],
            linestyle="none",
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )
    ax.plot(
        diagnostics.time,
        diagnostics.energy_closure,
        color="green",
        label="energy closure",
    )
    ax.axhline(
        target,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"target {target:g}",
    )
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 15.0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks(range(16))
    ax.set_title(f"Energy budget and closure: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def plot_energy_overlay(
    path: Path,
    diagnostics_by_case: Mapping[str, EnergyBudgetDiagnostics],
    y_key: str,
    title: str,
    ylabel: str,
    overwrite: bool,
    *,
    target: float | None = None,
) -> None:
    """Write one multi-case energy diagnostic overlay figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, diagnostics in diagnostics_by_case.items():
        ax.plot(diagnostics.time, getattr(diagnostics, y_key), label=case)
    if target is not None:
        ax.axhline(target, linestyle="--", linewidth=1.0, label=f"target {target:g}")
    if y_key in {
        "closure_residual_from_target",
        "closure_drift_from_initial",
        "differential_closure_residual",
    }:
        ax.axhline(0.0, linewidth=1.0)
    if y_key == "energy_closure":
        ax.set_xlim(0.0, 20.0)
        ax.set_ylim(10.0, 13.0)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, path, overwrite)


def write_energy_budget_plots(
    output_dir: Path,
    diagnostics_by_case: Mapping[str, EnergyBudgetDiagnostics],
    target: float,
    overwrite: bool,
) -> list[Path]:
    """Write all established component and overlay figures in output order."""
    figure_paths: list[Path] = []
    for case, diagnostics in diagnostics_by_case.items():
        path = component_figure_path(output_dir, case)
        plot_energy_components(path, case, diagnostics, target, overwrite)
        figure_paths.append(path)

    plot_specs = [
        (epsilon_figure_path(output_dir), "epsilon", "Dissipation rate by case", "epsilon", None),
        (
            energy_closure_figure_path(output_dir),
            "energy_closure",
            "Energy closure by case",
            "E_total + integral epsilon dt",
            target,
        ),
        (
            closure_residual_figure_path(output_dir),
            "closure_residual_from_target",
            "Closure residual from target by case",
            "energy_closure - target",
            None,
        ),
        (
            closure_drift_figure_path(output_dir),
            "closure_drift_from_initial",
            "Closure drift from initial by case",
            "energy_closure - energy_closure[0]",
            None,
        ),
        (
            differential_closure_residual_figure_path(output_dir),
            "differential_closure_residual",
            "Differential closure residual by case",
            "dE_total/dt + epsilon",
            None,
        ),
    ]
    for path, y_key, title, ylabel, target_line in plot_specs:
        plot_energy_overlay(
            path,
            diagnostics_by_case,
            y_key,
            title,
            ylabel,
            overwrite,
            target=target_line,
        )
        figure_paths.append(path)
    return figure_paths
