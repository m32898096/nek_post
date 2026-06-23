"""Field access helpers for Nek5000 data."""

from __future__ import annotations

from typing import Any

import numpy as np


def _component(field: Any, index: int, field_name: str) -> Any:
    """Return one field component with a clear error if it is unavailable."""
    if field is None:
        raise AttributeError(f"Element does not contain {field_name}.")
    try:
        return field[index]
    except (IndexError, TypeError) as exc:
        raise AttributeError(f"Element does not contain {field_name}[{index}].") from exc


def _field_exists(element: Any, name: str) -> bool:
    return getattr(element, name, None) is not None


def _component_shapes(element: Any, name: str) -> list[tuple[int, ...]]:
    field = getattr(element, name, None)
    if field is None:
        return []

    try:
        return [tuple(np.shape(component)) for component in field]
    except TypeError:
        return []


def get_coordinates(element: Any):
    """Return coordinate arrays for one element."""
    try:
        return (
            _component(element.pos, 0, "pos"),
            _component(element.pos, 1, "pos"),
            _component(element.pos, 2, "pos"),
        )
    except AttributeError as exc:
        raise AttributeError("Element does not contain complete coordinate arrays in pos[0:3].") from exc


def get_concentration(element: Any):
    """Return the concentration field for one element."""
    try:
        return _component(getattr(element, "temp", None), 0, "temp")
    except AttributeError:
        pass

    try:
        return _component(getattr(element, "scal", None), 0, "scal")
    except AttributeError as exc:
        raise AttributeError("Element does not contain concentration data in temp[0] or scal[0].") from exc


def get_velocity(element: Any):
    """Return the velocity field for one element."""
    try:
        return (
            _component(element.vel, 0, "vel"),
            _component(element.vel, 1, "vel"),
            _component(element.vel, 2, "vel"),
        )
    except AttributeError as exc:
        raise AttributeError("Element does not contain complete velocity arrays in vel[0:3].") from exc


def get_pressure(element: Any):
    """Return the pressure field for one element."""
    try:
        return _component(element.pres, 0, "pres")
    except AttributeError as exc:
        raise AttributeError("Element does not contain pressure data in pres[0].") from exc


def get_speed(u, v, w):
    """Compute speed magnitude from velocity components."""
    return np.sqrt(u**2 + v**2 + w**2)


def subtract_mean_pressure(p):
    """Return pressure with its mean removed."""
    return p - np.mean(p)


def _concentration_source(element: Any) -> str:
    try:
        _component(getattr(element, "temp", None), 0, "temp")
        return "temp[0]"
    except AttributeError:
        pass

    try:
        _component(getattr(element, "scal", None), 0, "scal")
        return "scal[0]"
    except AttributeError:
        return "not found"


def summarize_element_fields(element: Any) -> dict[str, Any]:
    """Summarize available field arrays on one Nek5000 element."""
    return {
        "has_pos": _field_exists(element, "pos"),
        "has_vel": _field_exists(element, "vel"),
        "has_pres": _field_exists(element, "pres"),
        "has_temp": _field_exists(element, "temp"),
        "has_scal": _field_exists(element, "scal"),
        "pos_shapes": _component_shapes(element, "pos"),
        "vel_shapes": _component_shapes(element, "vel"),
        "pres_shapes": _component_shapes(element, "pres"),
        "temp_shapes": _component_shapes(element, "temp"),
        "scal_shapes": _component_shapes(element, "scal"),
        "concentration_source": _concentration_source(element),
    }
