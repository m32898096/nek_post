"""Persistent fixed-grid concentration-sequence cache."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np

from nek_post.front_detection_io import NekFramePath
from nek_post.front_detection_workflow import ConcentrationSequence


CACHE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
ARRAY_FILENAMES = MappingProxyType(
    {
        "time": "time.npy",
        "file_indices": "file_indices.npy",
        "Xi": "Xi.npy",
        "Zi": "Zi.npy",
        "C_frames": "C_frames.npy",
        "finite_fraction": "finite_fraction.npy",
        "concentration_min": "concentration_min.npy",
        "concentration_max": "concentration_max.npy",
        "selected_y": "selected_y.npy",
    }
)
MEMMAP_ARRAYS = frozenset({"Xi", "Zi", "C_frames"})
FLOAT_ARRAYS = frozenset(
    {
        "time",
        "Xi",
        "Zi",
        "C_frames",
        "finite_fraction",
        "concentration_min",
        "concentration_max",
        "selected_y",
    }
)
REQUIRED_GRID_METADATA = frozenset({"xmin", "xmax", "zmin", "zmax", "nx", "nz"})
MISMATCH_GUIDANCE = (
    "Use --rebuild-cache to replace this cache or --no-cache to bypass it."
)


class FrontDetectionCacheError(RuntimeError):
    """Base error for concentration-sequence cache operations."""


class FrontDetectionCacheMismatchError(FrontDetectionCacheError):
    """Raised when a valid cache was built from different preprocessing inputs."""


class FrontDetectionCacheCorruptionError(FrontDetectionCacheError):
    """Raised when cache files or manifest content are incomplete or invalid."""


@dataclass(frozen=True)
class SourceFileSignature:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class FrontDetectionCacheSpec:
    schema_version: int
    case: str
    file_prefix: str
    file_indices: tuple[int, ...]
    source_files: tuple[SourceFileSignature, ...]
    nx: int
    nz: int
    slice_mode: str
    slab_ratio: float
    y_round_decimals: int
    interpolation_method: str


@dataclass(frozen=True)
class ConcentrationSequenceAcquisition:
    sequence: ConcentrationSequence
    mode: str
    cache_dir: Path | None


def _positive_grid_size(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    parsed = int(value)
    if parsed < 2:
        raise ValueError(f"{name} must be greater than or equal to 2.")
    return parsed


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _validated_preprocessing(
    *,
    nx: int,
    nz: int,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
    interpolation_method: str,
) -> tuple[int, int, str, float, int, str]:
    nx_value = _positive_grid_size(nx, "nx")
    nz_value = _positive_grid_size(nz, "nz")
    if slice_mode not in {"nearest_plane", "slab"}:
        raise ValueError("slice_mode must be exactly 'nearest_plane' or 'slab'.")
    try:
        slab_value = float(slab_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("slab_ratio must be finite and non-negative.") from exc
    if not np.isfinite(slab_value) or slab_value < 0.0:
        raise ValueError("slab_ratio must be finite and non-negative.")
    decimals = _nonnegative_integer(y_round_decimals, "y_round_decimals")
    if interpolation_method not in {"linear", "nearest"}:
        raise ValueError(
            "interpolation_method must be exactly 'linear' or 'nearest'."
        )
    return (
        nx_value,
        nz_value,
        slice_mode,
        slab_value,
        decimals,
        interpolation_method,
    )


def build_front_detection_cache_spec(
    *,
    case: str,
    file_prefix: str,
    frame_paths: Sequence[NekFramePath],
    nx: int,
    nz: int,
    slice_mode: str,
    slab_ratio: float,
    y_round_decimals: int,
    interpolation_method: str,
) -> FrontDetectionCacheSpec:
    """Build a preprocessing-only specification with ordered source signatures."""
    case_value = _nonempty_text(case, "case")
    prefix_value = _nonempty_text(file_prefix, "file_prefix")
    frames = tuple(frame_paths)
    if not frames:
        raise ValueError("At least one Nek frame path is required for caching.")
    (
        nx_value,
        nz_value,
        slice_mode_value,
        slab_value,
        decimals,
        interpolation_value,
    ) = _validated_preprocessing(
        nx=nx,
        nz=nz,
        slice_mode=slice_mode,
        slab_ratio=slab_ratio,
        y_round_decimals=y_round_decimals,
        interpolation_method=interpolation_method,
    )

    indices: list[int] = []
    signatures: list[SourceFileSignature] = []
    normalized_paths: set[str] = set()
    for frame in frames:
        index = _nonnegative_integer(frame.index, "Nek frame index")
        if index in indices:
            raise ValueError(f"Duplicate Nek frame index in cache specification: {index}.")
        source_path = Path(frame.path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(
                f"Cache source file is missing or not a regular file: {source_path}"
            )
        normalized = str(source_path.resolve())
        if normalized in normalized_paths:
            raise ValueError(
                f"Duplicate normalized source path in cache specification: {normalized}"
            )
        stat = source_path.stat()
        indices.append(index)
        normalized_paths.add(normalized)
        signatures.append(
            SourceFileSignature(
                path=normalized,
                size=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
            )
        )

    return FrontDetectionCacheSpec(
        schema_version=CACHE_SCHEMA_VERSION,
        case=case_value,
        file_prefix=prefix_value,
        file_indices=tuple(indices),
        source_files=tuple(signatures),
        nx=nx_value,
        nz=nz_value,
        slice_mode=slice_mode_value,
        slab_ratio=slab_value,
        y_round_decimals=decimals,
        interpolation_method=interpolation_value,
    )


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "value"


def default_front_detection_cache_path(
    cache_root: str | Path,
    *,
    case: str,
    file_prefix: str,
    frame_paths: Sequence[NekFramePath],
    nx: int,
    nz: int,
    slice_mode: str,
    interpolation_method: str,
) -> Path:
    """Return a readable cache directory derived from selected preprocessing."""
    frames = tuple(frame_paths)
    if not frames:
        raise ValueError("At least one Nek frame path is required for a cache path.")
    nx_value = _positive_grid_size(nx, "nx")
    nz_value = _positive_grid_size(nz, "nz")
    indices = tuple(
        _nonnegative_integer(frame.index, "Nek frame index") for frame in frames
    )
    case_name = _sanitize_path_component(_nonempty_text(case, "case"))
    prefix_name = _sanitize_path_component(
        _nonempty_text(file_prefix, "file_prefix")
    )
    slice_name = _sanitize_path_component(
        _nonempty_text(slice_mode, "slice_mode")
    )
    interpolation_name = _sanitize_path_component(
        _nonempty_text(interpolation_method, "interpolation_method")
    )
    directory_name = (
        f"{prefix_name}_f{indices[0]:05d}-f{indices[-1]:05d}_"
        f"n{len(indices)}_{nx_value}x{nz_value}_{slice_name}_"
        f"{interpolation_name}"
    )
    return Path(cache_root) / case_name / directory_name


def cache_spec_to_dict(spec: FrontDetectionCacheSpec) -> dict[str, Any]:
    """Serialize a cache specification into JSON-compatible primitives."""
    payload = asdict(spec)
    payload["file_indices"] = list(spec.file_indices)
    payload["source_files"] = [asdict(source) for source in spec.source_files]
    return payload


def _required_key(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest is missing required key {context}.{key}."
        )
    return mapping[key]


def _manifest_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest field {context} must be an integer."
        )
    return value


def _manifest_float(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest field {context} must be numeric."
        )
    parsed = float(value)
    if not np.isfinite(parsed):
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest field {context} must be finite."
        )
    return parsed


def _manifest_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest field {context} must be a nonempty string."
        )
    return value


def cache_spec_from_dict(payload: Mapping[str, Any]) -> FrontDetectionCacheSpec:
    """Parse and type-check a cache specification from manifest content."""
    if not isinstance(payload, Mapping):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest spec must be a JSON object."
        )
    schema_version = _manifest_int(
        _required_key(payload, "schema_version", "spec"),
        "spec.schema_version",
    )
    file_indices_raw = _required_key(payload, "file_indices", "spec")
    source_files_raw = _required_key(payload, "source_files", "spec")
    if not isinstance(file_indices_raw, list):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest field spec.file_indices must be a list."
        )
    if not isinstance(source_files_raw, list):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest field spec.source_files must be a list."
        )
    file_indices = tuple(
        _manifest_int(value, f"spec.file_indices[{index}]")
        for index, value in enumerate(file_indices_raw)
    )
    source_files: list[SourceFileSignature] = []
    for index, source in enumerate(source_files_raw):
        if not isinstance(source, Mapping):
            raise FrontDetectionCacheCorruptionError(
                f"Cache manifest field spec.source_files[{index}] must be an object."
            )
        source_files.append(
            SourceFileSignature(
                path=_manifest_text(
                    _required_key(source, "path", f"spec.source_files[{index}]"),
                    f"spec.source_files[{index}].path",
                ),
                size=_manifest_int(
                    _required_key(source, "size", f"spec.source_files[{index}]"),
                    f"spec.source_files[{index}].size",
                ),
                mtime_ns=_manifest_int(
                    _required_key(
                        source,
                        "mtime_ns",
                        f"spec.source_files[{index}]",
                    ),
                    f"spec.source_files[{index}].mtime_ns",
                ),
            )
        )
    spec = FrontDetectionCacheSpec(
        schema_version=schema_version,
        case=_manifest_text(_required_key(payload, "case", "spec"), "spec.case"),
        file_prefix=_manifest_text(
            _required_key(payload, "file_prefix", "spec"),
            "spec.file_prefix",
        ),
        file_indices=file_indices,
        source_files=tuple(source_files),
        nx=_manifest_int(_required_key(payload, "nx", "spec"), "spec.nx"),
        nz=_manifest_int(_required_key(payload, "nz", "spec"), "spec.nz"),
        slice_mode=_manifest_text(
            _required_key(payload, "slice_mode", "spec"),
            "spec.slice_mode",
        ),
        slab_ratio=_manifest_float(
            _required_key(payload, "slab_ratio", "spec"),
            "spec.slab_ratio",
        ),
        y_round_decimals=_manifest_int(
            _required_key(payload, "y_round_decimals", "spec"),
            "spec.y_round_decimals",
        ),
        interpolation_method=_manifest_text(
            _required_key(payload, "interpolation_method", "spec"),
            "spec.interpolation_method",
        ),
    )
    if not spec.file_indices:
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest spec.file_indices must not be empty."
        )
    if len(spec.file_indices) != len(spec.source_files):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest spec file_indices and source_files lengths disagree."
        )
    if len(set(spec.file_indices)) != len(spec.file_indices):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest spec.file_indices contains duplicates."
        )
    if len({source.path for source in spec.source_files}) != len(spec.source_files):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest spec.source_files contains duplicate paths."
        )
    try:
        _nonempty_text(spec.case, "case")
        _nonempty_text(spec.file_prefix, "file_prefix")
        _validated_preprocessing(
            nx=spec.nx,
            nz=spec.nz,
            slice_mode=spec.slice_mode,
            slab_ratio=spec.slab_ratio,
            y_round_decimals=spec.y_round_decimals,
            interpolation_method=spec.interpolation_method,
        )
        for index in spec.file_indices:
            _nonnegative_integer(index, "Nek frame index")
        for source in spec.source_files:
            if source.size < 0 or source.mtime_ns < 0:
                raise ValueError(
                    "Source file size and mtime_ns must be non-negative."
                )
    except ValueError as exc:
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest specification is invalid: {exc}"
        ) from exc
    return spec


def _json_grid_metadata(metadata: Mapping[str, Any]) -> dict[str, float | int]:
    converted: dict[str, float | int] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("Grid metadata keys must be strings.")
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            converted[key] = int(value)
        elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
            parsed = float(value)
            if not np.isfinite(parsed):
                raise ValueError(f"Grid metadata value {key!r} must be finite.")
            converted[key] = parsed
        else:
            raise ValueError(f"Grid metadata value {key!r} must be numeric.")
    return converted


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "filename": ARRAY_FILENAMES[name],
            "shape": list(np.shape(array)),
            "dtype": str(np.asarray(array).dtype),
        }
        for name, array in arrays.items()
    }


def _validate_sequence(
    sequence: ConcentrationSequence,
    spec: FrontDetectionCacheSpec,
) -> dict[str, np.ndarray]:
    n_frames = len(spec.file_indices)
    arrays = {
        name: np.asanyarray(getattr(sequence, name))
        for name in ARRAY_FILENAMES
    }
    for name in FLOAT_ARRAYS:
        if arrays[name].dtype != np.dtype("float64"):
            raise ValueError(f"Sequence {name} must have dtype float64.")
    if arrays["file_indices"].dtype != np.dtype("int64"):
        raise ValueError("Sequence file_indices must have dtype int64.")
    if arrays["time"].shape != (n_frames,):
        raise ValueError("Sequence time length must match the cache specification.")
    if arrays["file_indices"].shape != (n_frames,):
        raise ValueError(
            "Sequence file_indices length must match the cache specification."
        )
    if tuple(int(value) for value in arrays["file_indices"]) != spec.file_indices:
        raise ValueError(
            "Sequence file indices must exactly match the cache specification."
        )
    normalized_sources = tuple(str(Path(path).resolve()) for path in sequence.source_files)
    expected_sources = tuple(source.path for source in spec.source_files)
    if normalized_sources != expected_sources:
        raise ValueError(
            "Sequence source files and order must match the cache specification."
        )
    grid_shape = (spec.nz, spec.nx)
    if arrays["Xi"].shape != grid_shape or arrays["Zi"].shape != grid_shape:
        raise ValueError(f"Sequence Xi and Zi must have shape {grid_shape}.")
    if arrays["C_frames"].shape != (n_frames, spec.nz, spec.nx):
        raise ValueError(
            "Sequence C_frames must have shape "
            f"{(n_frames, spec.nz, spec.nx)}."
        )
    for name in (
        "finite_fraction",
        "concentration_min",
        "concentration_max",
        "selected_y",
    ):
        if arrays[name].shape != (n_frames,):
            raise ValueError(f"Sequence {name} must have length {n_frames}.")
    if not np.all(np.isfinite(arrays["time"])):
        raise ValueError("Sequence time values must be finite.")
    if np.any(np.diff(arrays["time"]) <= 0.0):
        raise ValueError("Sequence time values must be strictly increasing and unique.")
    if sequence.interpolation_method != spec.interpolation_method:
        raise ValueError(
            "Sequence interpolation method must match the cache specification."
        )
    missing_grid_metadata = sorted(
        REQUIRED_GRID_METADATA - set(sequence.grid_metadata)
    )
    if missing_grid_metadata:
        raise ValueError(
            "Sequence grid metadata is missing: "
            + ", ".join(missing_grid_metadata)
            + "."
        )
    _json_grid_metadata(sequence.grid_metadata)
    metadata_nx = sequence.grid_metadata["nx"]
    metadata_nz = sequence.grid_metadata["nz"]
    if (
        not isinstance(metadata_nx, Integral)
        or isinstance(metadata_nx, (bool, np.bool_))
        or not isinstance(metadata_nz, Integral)
        or isinstance(metadata_nz, (bool, np.bool_))
    ):
        raise ValueError("Sequence grid metadata nx and nz must be integers.")
    if int(metadata_nx) != spec.nx or int(metadata_nz) != spec.nz:
        raise ValueError("Sequence grid metadata nx and nz must match the cache spec.")
    return arrays


def _manifest_payload(
    sequence: ConcentrationSequence,
    spec: FrontDetectionCacheSpec,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "spec": cache_spec_to_dict(spec),
        "source_file_order": [source.path for source in spec.source_files],
        "grid_metadata": _json_grid_metadata(sequence.grid_metadata),
        "interpolation_method": sequence.interpolation_method,
        "arrays": _array_manifest(arrays),
    }


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def save_concentration_sequence_cache(
    cache_dir: str | Path,
    sequence: ConcentrationSequence,
    spec: FrontDetectionCacheSpec,
    *,
    overwrite: bool,
) -> Path:
    """Atomically save a validated concentration sequence directory cache."""
    target = Path(cache_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Concentration cache exists: {target}. Use --rebuild-cache to replace it."
        )
    arrays = _validate_sequence(sequence, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    backup: Path | None = None
    try:
        for name, filename in ARRAY_FILENAMES.items():
            np.save(temporary / filename, arrays[name], allow_pickle=False)
        payload = _manifest_payload(sequence, spec, arrays)
        with (temporary / MANIFEST_FILENAME).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        if target.exists():
            backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            _remove_path(backup)
            backup = None
    finally:
        if temporary.exists():
            _remove_path(temporary)
        if backup is not None and backup.exists():
            if not target.exists():
                os.replace(backup, target)
            else:
                _remove_path(backup)
    return target


def load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    """Load and structurally validate a cache manifest."""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest is missing: {manifest_path}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest contains invalid JSON: {manifest_path}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise FrontDetectionCacheCorruptionError(
            f"Could not read cache manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest root must be a JSON object."
        )
    schema_version = _manifest_int(
        _required_key(payload, "schema_version", "manifest"),
        "manifest.schema_version",
    )
    if schema_version != CACHE_SCHEMA_VERSION:
        raise FrontDetectionCacheCorruptionError(
            f"Unsupported cache schema version {schema_version}; "
            f"expected {CACHE_SCHEMA_VERSION}."
        )
    spec_payload = _required_key(payload, "spec", "manifest")
    spec = cache_spec_from_dict(spec_payload)
    if spec.schema_version != schema_version:
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest schema version disagrees with spec.schema_version."
        )
    arrays = _required_key(payload, "arrays", "manifest")
    if not isinstance(arrays, Mapping):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest arrays field must be an object."
        )
    for name, expected_filename in ARRAY_FILENAMES.items():
        metadata = _required_key(arrays, name, "arrays")
        if not isinstance(metadata, Mapping):
            raise FrontDetectionCacheCorruptionError(
                f"Cache array metadata for {name} must be an object."
            )
        filename = _manifest_text(
            _required_key(metadata, "filename", f"arrays.{name}"),
            f"arrays.{name}.filename",
        )
        if filename != expected_filename:
            raise FrontDetectionCacheCorruptionError(
                f"Cache array {name} must use filename {expected_filename!r}."
            )
        shape = _required_key(metadata, "shape", f"arrays.{name}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            raise FrontDetectionCacheCorruptionError(
                f"Cache array metadata arrays.{name}.shape must be non-negative integers."
            )
        dtype = _manifest_text(
            _required_key(metadata, "dtype", f"arrays.{name}"),
            f"arrays.{name}.dtype",
        )
        try:
            np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise FrontDetectionCacheCorruptionError(
                f"Cache array metadata arrays.{name}.dtype is invalid."
            ) from exc
    grid_metadata = _required_key(payload, "grid_metadata", "manifest")
    if not isinstance(grid_metadata, Mapping):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest grid_metadata must be an object."
        )
    missing_grid = sorted(REQUIRED_GRID_METADATA - set(grid_metadata))
    if missing_grid:
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest grid_metadata is missing: "
            + ", ".join(missing_grid)
            + "."
        )
    try:
        parsed_grid_metadata = _json_grid_metadata(grid_metadata)
    except ValueError as exc:
        raise FrontDetectionCacheCorruptionError(
            f"Cache manifest grid_metadata is invalid: {exc}"
        ) from exc
    for name in ("nx", "nz"):
        if not isinstance(parsed_grid_metadata[name], int):
            raise FrontDetectionCacheCorruptionError(
                f"Cache manifest grid_metadata.{name} must be an integer."
            )
    _manifest_text(
        _required_key(payload, "interpolation_method", "manifest"),
        "manifest.interpolation_method",
    )
    source_order = _required_key(payload, "source_file_order", "manifest")
    if not isinstance(source_order, list) or any(
        not isinstance(value, str) for value in source_order
    ):
        raise FrontDetectionCacheCorruptionError(
            "Cache manifest source_file_order must be a list of strings."
        )
    return payload


def _spec_mismatches(
    stored: FrontDetectionCacheSpec,
    expected: FrontDetectionCacheSpec,
) -> list[str]:
    mismatches: list[str] = []
    for name in (
        "schema_version",
        "case",
        "file_prefix",
        "file_indices",
        "nx",
        "nz",
        "slice_mode",
        "slab_ratio",
        "y_round_decimals",
        "interpolation_method",
    ):
        if getattr(stored, name) != getattr(expected, name):
            mismatches.append(name)
    if len(stored.source_files) != len(expected.source_files):
        mismatches.append("source_files.length")
    else:
        for index, (stored_source, expected_source) in enumerate(
            zip(stored.source_files, expected.source_files, strict=True)
        ):
            for name in ("path", "size", "mtime_ns"):
                if getattr(stored_source, name) != getattr(expected_source, name):
                    mismatches.append(f"source_files[{index}].{name}")
    return mismatches


def _load_arrays(
    cache_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, filename in ARRAY_FILENAMES.items():
        path = cache_dir / filename
        if not path.is_file():
            raise FrontDetectionCacheCorruptionError(
                f"Required cache array is missing: {path}"
            )
        mmap_mode = "r" if name in MEMMAP_ARRAYS else None
        try:
            array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        except Exception as exc:
            raise FrontDetectionCacheCorruptionError(
                f"Could not load cache array {name} from {path}: {exc}"
            ) from exc
        metadata = manifest["arrays"][name]
        expected_shape = tuple(metadata["shape"])
        expected_dtype = np.dtype(metadata["dtype"])
        if array.shape != expected_shape:
            raise FrontDetectionCacheCorruptionError(
                f"Cache array {name} shape {array.shape} does not match manifest "
                f"shape {expected_shape}."
            )
        if array.dtype != expected_dtype:
            raise FrontDetectionCacheCorruptionError(
                f"Cache array {name} dtype {array.dtype} does not match manifest "
                f"dtype {expected_dtype}."
            )
        arrays[name] = array
    return arrays


def load_concentration_sequence_cache(
    cache_dir: str | Path,
    expected_spec: FrontDetectionCacheSpec,
) -> ConcentrationSequence:
    """Load, validate, and reconstruct a memory-mapped concentration sequence."""
    cache_path = Path(cache_dir)
    manifest = load_cache_manifest(cache_path)
    stored_spec = cache_spec_from_dict(manifest["spec"])
    mismatches = _spec_mismatches(stored_spec, expected_spec)
    if mismatches:
        raise FrontDetectionCacheMismatchError(
            "Cache specification mismatch: "
            + ", ".join(mismatches)
            + ". "
            + MISMATCH_GUIDANCE
        )
    if manifest["interpolation_method"] != expected_spec.interpolation_method:
        raise FrontDetectionCacheCorruptionError(
            "Cache interpolation_method disagrees with its specification."
        )
    expected_order = [source.path for source in expected_spec.source_files]
    if manifest["source_file_order"] != expected_order:
        raise FrontDetectionCacheCorruptionError(
            "Cache source_file_order disagrees with the cache specification."
        )

    arrays = _load_arrays(cache_path, manifest)
    grid_metadata_raw = manifest["grid_metadata"]
    try:
        grid_metadata = _json_grid_metadata(grid_metadata_raw)
    except ValueError as exc:
        raise FrontDetectionCacheCorruptionError(str(exc)) from exc
    sequence = ConcentrationSequence(
        time=arrays["time"],
        file_indices=arrays["file_indices"],
        source_files=tuple(
            Path(source.path) for source in expected_spec.source_files
        ),
        Xi=arrays["Xi"],
        Zi=arrays["Zi"],
        C_frames=arrays["C_frames"],
        finite_fraction=arrays["finite_fraction"],
        concentration_min=arrays["concentration_min"],
        concentration_max=arrays["concentration_max"],
        grid_metadata=MappingProxyType(grid_metadata),
        selected_y=arrays["selected_y"],
        interpolation_method=manifest["interpolation_method"],
    )
    try:
        _validate_sequence(sequence, expected_spec)
    except ValueError as exc:
        raise FrontDetectionCacheCorruptionError(
            f"Cached sequence is inconsistent: {exc}"
        ) from exc
    return sequence


def acquire_concentration_sequence(
    *,
    cache_dir: str | Path,
    spec: FrontDetectionCacheSpec,
    builder: Callable[[], ConcentrationSequence],
    use_cache: bool,
    rebuild_cache: bool,
) -> ConcentrationSequenceAcquisition:
    """Acquire a sequence; ``cache_dir`` is ``None`` only when caching is disabled."""
    path = Path(cache_dir)
    if not use_cache:
        return ConcentrationSequenceAcquisition(
            sequence=builder(),
            mode="cache_disabled",
            cache_dir=None,
        )
    if rebuild_cache:
        sequence = builder()
        save_concentration_sequence_cache(path, sequence, spec, overwrite=True)
        return ConcentrationSequenceAcquisition(
            sequence=sequence,
            mode="cache_rebuilt",
            cache_dir=path,
        )
    if path.exists():
        return ConcentrationSequenceAcquisition(
            sequence=load_concentration_sequence_cache(path, spec),
            mode="cache_hit",
            cache_dir=path,
        )
    sequence = builder()
    save_concentration_sequence_cache(path, sequence, spec, overwrite=False)
    return ConcentrationSequenceAcquisition(
        sequence=sequence,
        mode="cache_created",
        cache_dir=path,
    )
