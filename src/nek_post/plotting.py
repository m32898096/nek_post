"""Plotting helpers for Nek5000 comparison figures."""

from __future__ import annotations


def plot_contour(X, Z, values, output_path, title):
    """Plot a contour field and save it to disk."""
    raise NotImplementedError


def plot_difference(X, Z, diff, output_path, title):
    """Plot a difference field and save it to disk."""
    raise NotImplementedError


def plot_error_vs_order(orders, errors, output_path):
    """Plot error trends versus polynomial order."""
    raise NotImplementedError
