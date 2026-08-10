"""Single-snapshot workflow and NPZ artifacts for directional GLL integrals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import (
    get_concentration,
    get_pressure,
    get_speed,
    get_velocity,
)
from nek_post.gll_directional_integration import (
    GLLDirectionalIntegrationPlan,
    GLLDirectionalIntegrationResult,
    apply_gll_directional_integration_plan,
    build_gll_directional_integration_plan,
    normalize_integration_direction,
)


SUPPORTED_GLL_DIRECTIONAL_FIELDS = (
    "concentration",
    "u",
    "v",
    "w",
    "speed",
    "pressure",
)

_OUTPUT_COORDINATE_NAMES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}

_REQUIRED_ARTIFACT_KEYS = (
    "values",
    "horizontal_coordinates",
    "vertical_coordinates",
    "field",
    "direction",
    "horizontal_coordinate_name",
    "vertical_coordinate_name",
    "case",
    "index",
    "time",
    "source_file",
    "element_count",
    "element_shape",
    "element_interval_counts",
    "physical_to_reference_axes",
)


def normalize_gll_directional_field(field: object) -> str:
    """Normalize a user-facing scalar-field name for this workflow."""
    if not isinstance(field, str):
        raise ValueError(
            "field must be one of: "
            + ", ".join(SUPPORTED_GLL_DIRECTIONAL_FIELDS)
            + "."
        )
    normalized = field.strip().lower()
    if normalized not in SUPPORTED_GLL_DIRECTIONAL_FIELDS:
        raise ValueError(
            "Unsupported field "
            f"{field!r}. Choose one of: "
            + ", ".join(SUPPORTED_GLL_DIRECTIONAL_FIELDS)
            + "."
        )
    return normalized


def gll_directional_field_getter(field: object) -> Callable[[Any], object]:
    """Return the existing Nek field accessor for one supported scalar name."""
    normalized = normalize_gll_directional_field(field)
    if normalized == "concentration":
        return get_concentration
    if normalized == "pressure":
        return get_pressure
    if normalized in {"u", "v", "w"}:
        component = {"u": 0, "v": 1, "w": 2}[normalized]

        def velocity_component(element: Any) -> object:
            return get_velocity(element)[component]

        return velocity_component

    def speed(element: Any) -> object:
        u, v, w = get_velocity(element)
        return get_speed(u, v, w)

    return speed


@dataclass(frozen=True)
class GLLDirectionalIntegral:
    """One field integral together with the validated geometry plan used."""

    field: str
    plan: GLLDirectionalIntegrationPlan
    result: GLLDirectionalIntegrationResult

    def __post_init__(self) -> None:
        field = normalize_gll_directional_field(self.field)
        if not isinstance(self.plan, GLLDirectionalIntegrationPlan):
            raise ValueError("plan must be a GLLDirectionalIntegrationPlan.")
        if not isinstance(self.result, GLLDirectionalIntegrationResult):
            raise ValueError("result must be a GLLDirectionalIntegrationResult.")
        if self.plan.direction != self.result.direction:
            raise ValueError("plan and result directions must match.")
        if (
            self.plan.horizontal_coordinate_name
            != self.result.horizontal_coordinate_name
            or self.plan.vertical_coordinate_name
            != self.result.vertical_coordinate_name
        ):
            raise ValueError("plan and result output coordinate names must match.")
        object.__setattr__(self, "field", field)


def compute_gll_directional_integral(
    data: object,
    *,
    field: object,
    direction: object,
    source_file: object | None = None,
) -> GLLDirectionalIntegral:
    """Integrate one supported scalar field in one physical direction.

    Geometry and scalar-array compatibility are validated by the underlying
    directional-integration plan and apply APIs; no field reshaping or
    interpolation is performed here.
    """
    normalized_field = normalize_gll_directional_field(field)
    normalized_direction = normalize_integration_direction(direction)
    plan = build_gll_directional_integration_plan(
        data, direction=normalized_direction
    )
    result = apply_gll_directional_integration_plan(
        plan,
        data,
        field_getter=gll_directional_field_getter(normalized_field),
        source_file=source_file,
    )
    return GLLDirectionalIntegral(
        field=normalized_field,
        plan=plan,
        result=result,
    )


def gll_directional_integral_path(
    root: str | Path,
    *,
    case: str,
    index: int,
    field: object,
    direction: object,
) -> Path:
    """Return the deterministic path for one directional-GLL NPZ artifact."""
    normalized_field = normalize_gll_directional_field(field)
    normalized_direction = normalize_integration_direction(direction)
    normalized_case = _nonempty_string(case, "case")
    normalized_index = _nonnegative_integer(index, "index")
    return (
        Path(root)
        / normalized_case
        / normalized_field
        / normalized_direction
        / (
            f"{normalized_case}_f{normalized_index:05d}_{normalized_field}"
            f"_integrate_{normalized_direction}.npz"
        )
    )


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
    array = np.asarray(values)
    if array.ndim != 1 or array.size != 3 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a length-three integer vector.")
    try:
        converted = np.asarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a length-three integer vector.") from exc
    if not np.all(np.equal(array, converted)) or np.any(converted < minimum):
        raise ValueError(f"{name} must be a length-three integer vector.")
    return converted


def _artifact_payload(
    integral: GLLDirectionalIntegral,
    *,
    case: object,
    index: object,
    time: object,
    source_file: object,
) -> dict[str, object]:
    if not isinstance(integral, GLLDirectionalIntegral):
        raise ValueError("integral must be a GLLDirectionalIntegral.")
    return {
        "values": integral.result.values,
        "horizontal_coordinates": integral.result.horizontal_coordinates,
        "vertical_coordinates": integral.result.vertical_coordinates,
        "field": integral.field,
        "direction": integral.result.direction,
        "horizontal_coordinate_name": integral.result.horizontal_coordinate_name,
        "vertical_coordinate_name": integral.result.vertical_coordinate_name,
        "case": _nonempty_string(case, "case"),
        "index": _nonnegative_integer(index, "index"),
        "time": _finite_scalar(time, "time"),
        "source_file": _nonempty_string(source_file, "source_file"),
        "element_count": integral.plan.element_count,
        "element_shape": np.asarray(integral.plan.element_shape, dtype=np.int64),
        "element_interval_counts": np.asarray(
            integral.plan.element_interval_counts, dtype=np.int64
        ),
        "physical_to_reference_axes": np.asarray(
            integral.plan.physical_to_reference_axes, dtype=np.int64
        ),
    }


def save_gll_directional_integral_npz(
    output_path: str | Path,
    integral: GLLDirectionalIntegral,
    *,
    case: object,
    index: object,
    time: object,
    source_file: object,
) -> Path:
    """Save one validated directional integral as a portable compressed NPZ."""
    path = Path(output_path)
    payload = _artifact_payload(
        integral,
        case=case,
        index=index,
        time=time,
        source_file=source_file,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _load_array(archive: Mapping[str, Any], name: str) -> NDArray[np.float64]:
    raw = archive[name]
    if np.iscomplexobj(raw):
        raise ValueError(f"Artifact {name} must be real-valued.")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Artifact {name} must be numeric.") from exc


def _load_scalar_string(archive: Mapping[str, Any], name: str) -> str:
    raw = np.asarray(archive[name])
    if raw.size != 1:
        raise ValueError(f"Artifact {name} must be a scalar string.")
    return _nonempty_string(str(raw.reshape(()).item()), name)


def load_gll_directional_integral_npz(path: str | Path) -> dict[str, object]:
    """Load and validate a directional-GLL NPZ artifact without pickle data."""
    artifact_path = Path(path)
    try:
        archive_context = np.load(artifact_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load directional-GLL artifact {artifact_path}: {exc}"
        ) from exc
    with archive_context as archive:
        missing = [name for name in _REQUIRED_ARTIFACT_KEYS if name not in archive]
        if missing:
            raise ValueError(
                "Directional-GLL artifact is missing required metadata: "
                + ", ".join(missing)
                + "."
            )
        values = _load_array(archive, "values")
        horizontal = _load_array(archive, "horizontal_coordinates")
        vertical = _load_array(archive, "vertical_coordinates")
        if values.ndim != 2:
            raise ValueError("Artifact values must be two-dimensional.")
        for name, coordinates in (
            ("horizontal_coordinates", horizontal),
            ("vertical_coordinates", vertical),
        ):
            if (
                coordinates.ndim != 1
                or coordinates.size == 0
                or not np.all(np.isfinite(coordinates))
                or np.any(np.diff(coordinates) <= 0.0)
            ):
                raise ValueError(
                    f"Artifact {name} must be a finite increasing vector."
                )
        if values.shape != (vertical.size, horizontal.size):
            raise ValueError(
                "Artifact values shape must equal "
                "(len(vertical_coordinates), len(horizontal_coordinates))."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Artifact values must be finite.")

        field = normalize_gll_directional_field(_load_scalar_string(archive, "field"))
        direction = normalize_integration_direction(
            _load_scalar_string(archive, "direction")
        )
        horizontal_name = _load_scalar_string(archive, "horizontal_coordinate_name")
        vertical_name = _load_scalar_string(archive, "vertical_coordinate_name")
        expected_names = _OUTPUT_COORDINATE_NAMES[direction]
        if (horizontal_name, vertical_name) != expected_names:
            raise ValueError(
                "Artifact coordinate names are inconsistent with direction "
                f"{direction!r}: expected {expected_names}, got "
                f"{(horizontal_name, vertical_name)}."
            )
        case = _load_scalar_string(archive, "case")
        source_file = _load_scalar_string(archive, "source_file")
        index_raw = np.asarray(archive["index"])
        time_raw = np.asarray(archive["time"])
        if index_raw.size != 1:
            raise ValueError("Artifact index must be a scalar.")
        if time_raw.size != 1:
            raise ValueError("Artifact time must be a scalar.")
        index = _nonnegative_integer(index_raw.reshape(()).item(), "index")
        time = _finite_scalar(time_raw.reshape(()).item(), "time")
        element_count_raw = np.asarray(archive["element_count"])
        if element_count_raw.size != 1:
            raise ValueError("Artifact element_count must be a scalar.")
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
        "values": values,
        "horizontal_coordinates": horizontal,
        "vertical_coordinates": vertical,
        "field": field,
        "direction": direction,
        "horizontal_coordinate_name": horizontal_name,
        "vertical_coordinate_name": vertical_name,
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
    "GLLDirectionalIntegral",
    "SUPPORTED_GLL_DIRECTIONAL_FIELDS",
    "compute_gll_directional_integral",
    "gll_directional_field_getter",
    "gll_directional_integral_path",
    "load_gll_directional_integral_npz",
    "normalize_gll_directional_field",
    "save_gll_directional_integral_npz",
)
