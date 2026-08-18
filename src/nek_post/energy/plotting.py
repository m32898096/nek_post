"""Plots for derivative-based energy-budget closure analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt

from nek_post.energy.analysis import EnergyBudgetAnalysis


def _save(fig, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def _axes(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def write_energy_plots(output_dir: Path, analysis: EnergyBudgetAnalysis) -> tuple[Path, ...]:
    """Write the four required unsmoothed figures on the original time coordinates."""
    paths = (
        output_dir / "energy_evolution.png",
        output_dir / "energy_derivatives.png",
        output_dir / "energy_closure.png",
        output_dir / "energy_closure_residual.png",
    )

    fig, ax = _axes("Energy evolution", "energy")
    ax.plot(analysis.time, analysis.E_k, label="E_k")
    ax.plot(analysis.time, analysis.E_p, label="E_p")
    ax.plot(analysis.time, analysis.E_total, label="E_total")
    ax.legend()
    _save(fig, paths[0])

    fig, ax = _axes("Energy derivatives", "energy rate")
    ax.plot(analysis.time, analysis.dE_k_dt, label="dE_k/dt")
    ax.plot(analysis.time, analysis.dE_p_dt, label="dE_p/dt")
    ax.legend()
    _save(fig, paths[1])

    fig, ax = _axes("Differential energy closure", "energy rate")
    ax.plot(analysis.time, analysis.dE_total_dt, label="dE_total/dt")
    ax.plot(analysis.time, analysis.minus_epsilon, label="-epsilon")
    ax.legend()
    _save(fig, paths[2])

    fig, ax = _axes("Energy closure residual", "dE_total/dt + epsilon")
    ax.plot(analysis.time, analysis.closure_residual, label="closure residual")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="zero")
    ax.legend()
    _save(fig, paths[3])
    return paths
