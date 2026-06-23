"""Plotting helpers for Nek5000 comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _colorbar_extend(values, vmin, vmax) -> str:
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0 or vmin is None or vmax is None:
        return "neither"

    below = np.min(finite_values) < vmin
    above = np.max(finite_values) > vmax
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _save_contour(X, Z, values, output_path, title, label, vmin=None, vmax=None, ticks=None) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    levels = np.linspace(vmin, vmax, 80) if vmin is not None and vmax is not None else 80
    extend = _colorbar_extend(values, vmin, vmax)
    contour = ax.contourf(X, Z, values, levels=levels, vmin=vmin, vmax=vmax, extend=extend)
    colorbar = fig.colorbar(contour, ax=ax, ticks=ticks)
    colorbar.set_label(label)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_contour(X, Z, values, output_path, title, label="C", vmin=None, vmax=None, ticks=None):
    """Plot a contour field and save it to disk."""
    _save_contour(X, Z, values, output_path, title, label, vmin=vmin, vmax=vmax, ticks=ticks)


def plot_difference(X, Z, diff, output_path, title, label="|difference|", vmin=None, vmax=None, ticks=None):
    """Plot a difference field and save it to disk."""
    _save_contour(X, Z, diff, output_path, title, label, vmin=vmin, vmax=vmax, ticks=ticks)


def plot_error_vs_order(orders, errors, output_path, title):
    """Plot error trends versus polynomial order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(orders, errors, marker="o")
    ax.set_xlabel("Polynomial order N")
    ax.set_ylabel("Relative L2 error of C")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_front_position(orders, x_front, output_path, title):
    """Plot front position versus polynomial order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(orders, x_front, marker="o")
    ax.set_xlabel("Polynomial order N")
    ax.set_ylabel("Front position x_f")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
