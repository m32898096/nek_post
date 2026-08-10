"""Cantero-equivalent-height preprocessing with composite GLL quadrature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_concentration
from nek_post.gll_directional_integration import (
    GLLDirectionalIntegrationPlan,
    apply_gll_directional_integration_plan,
    build_gll_directional_integration_plan,
    composite_physical_axis_quadrature_weights,
)


_REQUIRED_ARTIFACT_KEYS = (
    "x",
    "y",
    "local_equivalent_height",
    "span_averaged_height",
    "spanwise_length",
    "spanwise_quadrature_weights",
    "case",
    "index",
    "time",
    "source_file",
    "element_count",
    "element_shape",
    "element_interval_counts",
    "physical_to_reference_axes",
)


def _readonly_float(values: object) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _finite_scalar(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    return parsed


def _integer_vector(
    values: object,
    name: str,
    *,
    minimum: int,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != 3 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a length-three integer vector.")
    try:
        converted = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a length-three integer vector.") from exc
    if not np.all(np.equal(raw, converted)) or np.any(converted < minimum):
        raise ValueError(f"{name} must be a length-three integer vector.")
    return converted


def _validate_coordinates(values: object, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        coordinates = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if (
        coordinates.ndim != 1
        or coordinates.size == 0
        or not np.all(np.isfinite(coordinates))
        or np.any(np.diff(coordinates) <= 0.0)
    ):
        raise ValueError(f"{name} must be a finite increasing vector.")
    return coordinates


@dataclass(frozen=True)
class CanteroEquivalentHeightPlan:
    """Reusable z-integration and spanwise-GLL quadrature geometry plan."""

    z_integration_plan: GLLDirectionalIntegrationPlan
    x_coordinates: NDArray[np.float64]
    y_coordinates: NDArray[np.float64]
    spanwise_quadrature_weights: NDArray[np.float64]
    spanwise_length: float

    def __post_init__(self) -> None:
        if not isinstance(self.z_integration_plan, GLLDirectionalIntegrationPlan):
            raise ValueError("z_integration_plan must be a GLLDirectionalIntegrationPlan.")
        if self.z_integration_plan.direction != "z":
            raise ValueError("z_integration_plan direction must be 'z'.")
        x = _validate_coordinates(self.x_coordinates, "x_coordinates")
        y = _validate_coordinates(self.y_coordinates, "y_coordinates")
        weights = _readonly_float(self.spanwise_quadrature_weights)
        spanwise_length = _finite_scalar(self.spanwise_length, "spanwise_length")
        if spanwise_length <= 0.0:
            raise ValueError("spanwise_length must be positive.")
        if (
            weights.shape != y.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "spanwise_quadrature_weights must be finite positive and match y_coordinates."
            )
        coordinate_length = float(y[-1] - y[0])
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if not np.isclose(spanwise_length, coordinate_length, rtol=1.0e-12, atol=1.0e-13):
            raise ValueError("spanwise_length must equal y_coordinates[-1] - y_coordinates[0].")
        if not np.isclose(weight_sum, spanwise_length, rtol=1.0e-12, atol=1.0e-13):
            raise ValueError("spanwise_quadrature_weights must sum to spanwise_length.")
        if not np.array_equal(x, self.z_integration_plan.horizontal_coordinates):
            raise ValueError("x_coordinates must match the z-integration plan output.")
        if not np.array_equal(y, self.z_integration_plan.vertical_coordinates):
            raise ValueError("y_coordinates must match the z-integration plan output.")
        object.__setattr__(self, "x_coordinates", _readonly_float(x))
        object.__setattr__(self, "y_coordinates", _readonly_float(y))
        object.__setattr__(self, "spanwise_quadrature_weights", weights)
        object.__setattr__(self, "spanwise_length", spanwise_length)


@dataclass(frozen=True)
class CanteroEquivalentHeightResult:
    """Local and span-averaged equivalent heights for one snapshot."""

    x_coordinates: NDArray[np.float64]
    y_coordinates: NDArray[np.float64]
    local_equivalent_height: NDArray[np.float64]
    span_averaged_height: NDArray[np.float64]
    spanwise_length: float

    def __post_init__(self) -> None:
        x = _validate_coordinates(self.x_coordinates, "x_coordinates")
        y = _validate_coordinates(self.y_coordinates, "y_coordinates")
        local = _readonly_float(self.local_equivalent_height)
        span_averaged = _readonly_float(self.span_averaged_height)
        spanwise_length = _finite_scalar(self.spanwise_length, "spanwise_length")
        if spanwise_length <= 0.0:
            raise ValueError("spanwise_length must be positive.")
        if local.shape != (y.size, x.size):
            raise ValueError(
                "local_equivalent_height must have shape "
                "(y_coordinates.size, x_coordinates.size)."
            )
        if span_averaged.shape != (x.size,):
            raise ValueError(
                "span_averaged_height must have shape (x_coordinates.size,)."
            )
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(span_averaged)):
            raise ValueError("Equivalent-height values must be finite.")
        if not np.isclose(
            spanwise_length,
            float(y[-1] - y[0]),
            rtol=1.0e-12,
            atol=1.0e-13,
        ):
            raise ValueError("spanwise_length must match the physical y extent.")
        object.__setattr__(self, "x_coordinates", _readonly_float(x))
        object.__setattr__(self, "y_coordinates", _readonly_float(y))
        object.__setattr__(self, "local_equivalent_height", local)
        object.__setattr__(self, "span_averaged_height", span_averaged)
        object.__setattr__(self, "spanwise_length", spanwise_length)


def build_cantero_equivalent_height_plan(data: object) -> CanteroEquivalentHeightPlan:
    """Build the reusable z-GLL and composite-y-GLL geometry plan."""
    z_plan = build_gll_directional_integration_plan(data, direction="z")
    y_coordinates, y_weights = composite_physical_axis_quadrature_weights(
        z_plan, "y"
    )
    spanwise_length = float(y_coordinates[-1] - y_coordinates[0])
    return CanteroEquivalentHeightPlan(
        z_integration_plan=z_plan,
        x_coordinates=z_plan.horizontal_coordinates,
        y_coordinates=y_coordinates,
        spanwise_quadrature_weights=y_weights,
        spanwise_length=spanwise_length,
    )


def apply_cantero_equivalent_height_plan(
    plan: CanteroEquivalentHeightPlan,
    data: object,
    *,
    source_file: object | None = None,
) -> CanteroEquivalentHeightResult:
    """Apply a reusable plan to concentration on compatible stationary geometry."""
    if not isinstance(plan, CanteroEquivalentHeightPlan):
        raise ValueError("plan must be a CanteroEquivalentHeightPlan.")
    z_result = apply_gll_directional_integration_plan(
        plan.z_integration_plan,
        data,
        field_getter=get_concentration,
        source_file=source_file,
    )
    if (
        not np.array_equal(z_result.horizontal_coordinates, plan.x_coordinates)
        or not np.array_equal(z_result.vertical_coordinates, plan.y_coordinates)
    ):
        raise ValueError("Applied z-integration result does not match plan coordinates.")
    local_equivalent_height = z_result.values
    span_averaged_height = (
        np.tensordot(
            plan.spanwise_quadrature_weights,
            local_equivalent_height,
            axes=(0, 0),
        )
        / plan.spanwise_length
    )
    return CanteroEquivalentHeightResult(
        x_coordinates=plan.x_coordinates,
        y_coordinates=plan.y_coordinates,
        local_equivalent_height=local_equivalent_height,
        span_averaged_height=span_averaged_height,
        spanwise_length=plan.spanwise_length,
    )


def compute_cantero_equivalent_height(
    data: object,
    *,
    source_file: object | None = None,
) -> tuple[CanteroEquivalentHeightPlan, CanteroEquivalentHeightResult]:
    """Build and apply a Cantero-equivalent-height plan for one snapshot."""
    plan = build_cantero_equivalent_height_plan(data)
    return plan, apply_cantero_equivalent_height_plan(
        plan, data, source_file=source_file
    )


def cantero_equivalent_height_path(
    root: str | Path,
    *,
    case: object,
    index: object,
) -> Path:
    """Return the deterministic NPZ path for one snapshot artifact."""
    normalized_case = _nonempty_string(case, "case")
    normalized_index = _nonnegative_integer(index, "index")
    return (
        Path(root)
        / normalized_case
        / f"{normalized_case}_f{normalized_index:05d}_cantero_equivalent_height.npz"
    )


def save_cantero_equivalent_height_npz(
    output_path: str | Path,
    plan: CanteroEquivalentHeightPlan,
    result: CanteroEquivalentHeightResult,
    *,
    case: object,
    index: object,
    time: object,
    source_file: object,
) -> Path:
    """Save a portable, allow-pickle-free equivalent-height artifact."""
    if not isinstance(plan, CanteroEquivalentHeightPlan):
        raise ValueError("plan must be a CanteroEquivalentHeightPlan.")
    if not isinstance(result, CanteroEquivalentHeightResult):
        raise ValueError("result must be a CanteroEquivalentHeightResult.")
    if (
        not np.array_equal(result.x_coordinates, plan.x_coordinates)
        or not np.array_equal(result.y_coordinates, plan.y_coordinates)
        or result.spanwise_length != plan.spanwise_length
    ):
        raise ValueError("result coordinates and spanwise length must match plan.")
    z_plan = plan.z_integration_plan
    payload = {
        "x": result.x_coordinates,
        "y": result.y_coordinates,
        "local_equivalent_height": result.local_equivalent_height,
        "span_averaged_height": result.span_averaged_height,
        "spanwise_length": result.spanwise_length,
        "spanwise_quadrature_weights": plan.spanwise_quadrature_weights,
        "case": _nonempty_string(case, "case"),
        "index": _nonnegative_integer(index, "index"),
        "time": _finite_scalar(time, "time"),
        "source_file": _nonempty_string(source_file, "source_file"),
        "element_count": z_plan.element_count,
        "element_shape": np.asarray(z_plan.element_shape, dtype=np.int64),
        "element_interval_counts": np.asarray(
            z_plan.element_interval_counts, dtype=np.int64
        ),
        "physical_to_reference_axes": np.asarray(
            z_plan.physical_to_reference_axes, dtype=np.int64
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _load_array(
    archive: Mapping[str, Any], name: str
) -> NDArray[np.float64]:
    values = archive[name]
    if np.iscomplexobj(values):
        raise ValueError(f"Artifact {name} must be real-valued.")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Artifact {name} must be numeric.") from exc


def _load_string(archive: Mapping[str, Any], name: str) -> str:
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"Artifact {name} must be a scalar string.")
    return _nonempty_string(str(value.reshape(()).item()), name)


def load_cantero_equivalent_height_npz(path: str | Path) -> dict[str, object]:
    """Load and validate a single-snapshot equivalent-height artifact."""
    artifact_path = Path(path)
    try:
        archive_context = np.load(artifact_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load Cantero-equivalent-height artifact {artifact_path}: {exc}"
        ) from exc
    with archive_context as archive:
        missing = [name for name in _REQUIRED_ARTIFACT_KEYS if name not in archive]
        if missing:
            raise ValueError(
                "Cantero-equivalent-height artifact is missing required metadata: "
                + ", ".join(missing)
                + "."
            )
        x = _validate_coordinates(_load_array(archive, "x"), "Artifact x")
        y = _validate_coordinates(_load_array(archive, "y"), "Artifact y")
        local = _load_array(archive, "local_equivalent_height")
        span_averaged = _load_array(archive, "span_averaged_height")
        weights = _load_array(archive, "spanwise_quadrature_weights")
        if local.shape != (y.size, x.size):
            raise ValueError(
                "Artifact local_equivalent_height must have shape (len(y), len(x))."
            )
        if span_averaged.shape != (x.size,):
            raise ValueError(
                "Artifact span_averaged_height must have shape (len(x),)."
            )
        if (
            weights.shape != y.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "Artifact spanwise_quadrature_weights must be finite positive and match y."
            )
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(span_averaged)):
            raise ValueError("Artifact equivalent-height values must be finite.")
        spanwise_length_raw = np.asarray(archive["spanwise_length"])
        if spanwise_length_raw.size != 1:
            raise ValueError("Artifact spanwise_length must be a scalar.")
        spanwise_length = _finite_scalar(
            spanwise_length_raw.reshape(()).item(), "spanwise_length"
        )
        if spanwise_length <= 0.0 or not np.isclose(
            spanwise_length,
            float(y[-1] - y[0]),
            rtol=1.0e-12,
            atol=1.0e-13,
        ):
            raise ValueError("Artifact spanwise_length is inconsistent with y.")
        if not np.isclose(
            float(np.sum(weights, dtype=np.float64)),
            spanwise_length,
            rtol=1.0e-12,
            atol=1.0e-13,
        ):
            raise ValueError(
                "Artifact spanwise_quadrature_weights do not sum to spanwise_length."
            )
        case = _load_string(archive, "case")
        source_file = _load_string(archive, "source_file")
        index_raw = np.asarray(archive["index"])
        time_raw = np.asarray(archive["time"])
        element_count_raw = np.asarray(archive["element_count"])
        if index_raw.size != 1 or time_raw.size != 1 or element_count_raw.size != 1:
            raise ValueError("Artifact index, time, and element_count must be scalars.")
        index = _nonnegative_integer(index_raw.reshape(()).item(), "index")
        time = _finite_scalar(time_raw.reshape(()).item(), "time")
        element_count = _nonnegative_integer(
            element_count_raw.reshape(()).item(), "element_count"
        )
        if element_count == 0:
            raise ValueError("Artifact element_count must be positive.")
        element_shape = _integer_vector(
            archive["element_shape"], "element_shape", minimum=2
        )
        element_interval_counts = _integer_vector(
            archive["element_interval_counts"],
            "element_interval_counts",
            minimum=1,
        )
        physical_to_reference_axes = _integer_vector(
            archive["physical_to_reference_axes"],
            "physical_to_reference_axes",
            minimum=0,
        )
        if tuple(sorted(physical_to_reference_axes.tolist())) != (0, 1, 2):
            raise ValueError(
                "Artifact physical_to_reference_axes must be a permutation of 0, 1, 2."
            )

    return {
        "x": _readonly_float(x),
        "y": _readonly_float(y),
        "local_equivalent_height": _readonly_float(local),
        "span_averaged_height": _readonly_float(span_averaged),
        "spanwise_length": spanwise_length,
        "spanwise_quadrature_weights": _readonly_float(weights),
        "case": case,
        "index": index,
        "time": time,
        "source_file": source_file,
        "element_count": element_count,
        "element_shape": element_shape,
        "element_interval_counts": element_interval_counts,
        "physical_to_reference_axes": physical_to_reference_axes,
    }


__all__ = (
    "CanteroEquivalentHeightPlan",
    "CanteroEquivalentHeightResult",
    "apply_cantero_equivalent_height_plan",
    "build_cantero_equivalent_height_plan",
    "cantero_equivalent_height_path",
    "compute_cantero_equivalent_height",
    "load_cantero_equivalent_height_npz",
    "save_cantero_equivalent_height_npz",
)
