from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import nek_post.front_detection_cache as cache_module
from nek_post.front_detection_cache import (
    ARRAY_FILENAMES,
    CACHE_SCHEMA_VERSION,
    FrontDetectionCacheCorruptionError,
    FrontDetectionCacheMismatchError,
    FrontDetectionCacheSpec,
    SourceFileSignature,
    acquire_concentration_sequence,
    build_front_detection_cache_spec,
    default_front_detection_cache_path,
    load_concentration_sequence_cache,
    save_concentration_sequence_cache,
)
from nek_post.front_detection_io import NekFramePath
from nek_post.front_detection_workflow import ConcentrationSequence


def _frames(tmp_path: Path, indices: tuple[int, ...] = (1, 9)) -> tuple[NekFramePath, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in indices:
        path = tmp_path / f"GC0.f{index:05d}"
        path.write_bytes(bytes([index % 256]) * (index + 3))
        frames.append(NekFramePath(index=index, path=path))
    return tuple(frames)


def _spec(
    tmp_path: Path,
    *,
    indices: tuple[int, ...] = (1, 9),
    **overrides: object,
) -> FrontDetectionCacheSpec:
    values = {
        "case": "N7",
        "file_prefix": "GC0",
        "frame_paths": _frames(tmp_path, indices),
        "nx": 3,
        "nz": 2,
        "slice_mode": "nearest_plane",
        "slab_ratio": 0.05,
        "y_round_decimals": 10,
        "interpolation_method": "linear",
        "interpolation_engine": "scattered_linear",
    }
    values.update(overrides)
    return build_front_detection_cache_spec(**values)


def _sequence(spec: FrontDetectionCacheSpec, *, offset: float = 0.0) -> ConcentrationSequence:
    x = np.linspace(0.0, 2.0, spec.nx, dtype=np.float64)
    z = np.linspace(0.0, 1.0, spec.nz, dtype=np.float64)
    Xi, Zi = np.meshgrid(x, z)
    n_frames = len(spec.file_indices)
    concentration = np.arange(
        n_frames * spec.nz * spec.nx, dtype=np.float64
    ).reshape(n_frames, spec.nz, spec.nx)
    concentration = concentration + offset
    concentration[0, 0, 0] = np.nan
    return ConcentrationSequence(
        time=np.arange(n_frames, dtype=np.float64) + 0.25,
        file_indices=np.asarray(spec.file_indices, dtype=np.int64),
        source_files=tuple(Path(source.path) for source in spec.source_files),
        Xi=Xi.astype(np.float64),
        Zi=Zi.astype(np.float64),
        C_frames=concentration,
        finite_fraction=np.full(n_frames, 0.9, dtype=np.float64),
        concentration_min=np.arange(n_frames, dtype=np.float64),
        concentration_max=np.arange(n_frames, dtype=np.float64) + 10.0,
        grid_metadata=MappingProxyType(
            {
                "xmin": 0.0,
                "xmax": 2.0,
                "zmin": 0.0,
                "zmax": 1.0,
                "nx": spec.nx,
                "nz": spec.nz,
            }
        ),
        selected_y=np.arange(n_frames, dtype=np.float64) + 0.5,
        interpolation_method=spec.interpolation_method,
        interpolation_engine=spec.interpolation_engine,
        spectral_element_shape=(
            (4, 4, 4)
            if spec.interpolation_engine == "spectral_element"
            else None
        ),
        spectral_polynomial_order=(
            (3, 3, 3)
            if spec.interpolation_engine == "spectral_element"
            else None
        ),
        spectral_slice_y=(
            (
                spec.slice_y_value
                if spec.slice_y_value is not None
                else 0.75
            )
            if spec.interpolation_engine == "spectral_element"
            else None
        ),
    )


def _save(tmp_path: Path) -> tuple[Path, FrontDetectionCacheSpec, ConcentrationSequence]:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    spec = _spec(source_dir)
    sequence = _sequence(spec)
    cache_dir = tmp_path / "cache"
    save_concentration_sequence_cache(cache_dir, sequence, spec, overwrite=False)
    return cache_dir, spec, sequence


def _manifest(cache_dir: Path) -> dict[str, object]:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(cache_dir: Path, payload: dict[str, object]) -> None:
    (cache_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_spec_records_order_indices_size_and_nanosecond_mtime(tmp_path: Path) -> None:
    frames = _frames(tmp_path, (9, 1))
    spec = _spec(tmp_path, indices=(9, 1), frame_paths=frames)

    assert spec.file_indices == (9, 1)
    assert tuple(source.path for source in spec.source_files) == tuple(
        str(frame.path.resolve()) for frame in frames
    )
    assert tuple(source.size for source in spec.source_files) == tuple(
        frame.path.stat().st_size for frame in frames
    )
    assert tuple(source.mtime_ns for source in spec.source_files) == tuple(
        frame.path.stat().st_mtime_ns for frame in frames
    )


def test_spec_rejects_missing_duplicate_sources_and_indices(tmp_path: Path) -> None:
    missing = NekFramePath(1, tmp_path / "missing.f00001")
    with pytest.raises(FileNotFoundError, match="missing or not a regular file"):
        _spec(tmp_path, frame_paths=(missing,))

    frames = _frames(tmp_path, (1, 2))
    with pytest.raises(ValueError, match="Duplicate Nek frame index"):
        _spec(
            tmp_path,
            frame_paths=(
                frames[0],
                NekFramePath(index=1, path=frames[1].path),
            ),
        )
    with pytest.raises(ValueError, match="Duplicate normalized source path"):
        _spec(
            tmp_path,
            frame_paths=(
                frames[0],
                NekFramePath(index=2, path=frames[0].path),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nx", 1, "nx"),
        ("nz", 1, "nz"),
        ("slice_mode", "middle", "slice_mode"),
        ("slab_ratio", -0.1, "slab_ratio"),
        ("slab_ratio", np.nan, "slab_ratio"),
        ("y_round_decimals", -1, "y_round_decimals"),
        ("interpolation_method", "cubic", "interpolation_method"),
        ("interpolation_engine", "unknown", "interpolation_engine"),
    ],
)
def test_spec_rejects_invalid_preprocessing(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(tmp_path, **{field: value})


def test_spec_contains_preprocessing_but_no_tracking_settings(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    names = {field.name for field in fields(spec)}

    assert {
        "case",
        "file_prefix",
        "file_indices",
        "source_files",
        "nx",
        "nz",
        "slice_mode",
        "slab_ratio",
        "y_round_decimals",
        "interpolation_method",
        "interpolation_engine",
        "spectral_algorithm_version",
        "slice_y_mode",
        "slice_y_value",
        "per_frame_griddata",
    } <= names
    assert names.isdisjoint(
        {
            "threshold",
            "min_component_pixels",
            "bottom_rows",
            "max_front_jump",
            "connectivity",
            "reference_file",
            "diagnostic_indices",
            "no_plots",
            "output_dir",
        }
    )


def test_default_cache_path_is_readable_deterministic_and_sanitized(
    tmp_path: Path,
) -> None:
    subset = _frames(tmp_path / "subset", tuple(range(1, 10)))
    full = _frames(tmp_path / "full", tuple(range(1, 82)))
    subset_path = default_front_detection_cache_path(
        tmp_path / "cache-root",
        case="N7",
        file_prefix="GC0",
        frame_paths=subset,
        nx=500,
        nz=200,
        slice_mode="nearest_plane",
        interpolation_method="linear",
        interpolation_engine="scattered_linear",
    )
    full_path = default_front_detection_cache_path(
        tmp_path / "cache-root",
        case="N7",
        file_prefix="GC0",
        frame_paths=full,
        nx=500,
        nz=200,
        slice_mode="nearest_plane",
        interpolation_method="linear",
        interpolation_engine="scattered_linear",
    )
    unsafe_path = default_front_detection_cache_path(
        tmp_path / "cache-root",
        case="N 7/unsafe",
        file_prefix="GC 0?",
        frame_paths=subset,
        nx=500,
        nz=200,
        slice_mode="nearest/plane",
        interpolation_method="linear",
        interpolation_engine="scattered_linear",
    )
    nearest_path = default_front_detection_cache_path(
        tmp_path / "cache-root",
        case="N7",
        file_prefix="GC0",
        frame_paths=subset,
        nx=500,
        nz=200,
        slice_mode="nearest_plane",
        interpolation_method="nearest",
        interpolation_engine="scattered_nearest",
    )
    spectral_path = default_front_detection_cache_path(
        tmp_path / "cache-root",
        case="N7",
        file_prefix="GC0",
        frame_paths=subset,
        nx=500,
        nz=200,
        slice_mode="nearest_plane",
        interpolation_method="spectral",
        interpolation_engine="spectral_element",
    )

    assert subset_path.parent.name == "N7"
    assert subset_path.name == (
        "GC0_f00001-f00009_n9_500x200_nearest_plane_linear_scattered_linear"
    )
    assert full_path.name == (
        "GC0_f00001-f00081_n81_500x200_nearest_plane_linear_scattered_linear"
    )
    assert unsafe_path.parent.name == "N_7_unsafe"
    assert unsafe_path.name == (
        "GC_0_f00001-f00009_n9_500x200_nearest_plane_linear_scattered_linear"
    )
    assert nearest_path.name.endswith("_nearest_scattered_nearest")
    assert spectral_path.name.endswith(
        "_spectral_spectral_element_ydomain_midpoint"
    )
    assert nearest_path != subset_path
    assert spectral_path != subset_path
    assert subset_path != full_path


def test_round_trip_preserves_all_fields_and_memory_maps_large_arrays(
    tmp_path: Path,
) -> None:
    cache_dir, spec, original = _save(tmp_path)
    loaded = load_concentration_sequence_cache(cache_dir, spec)

    for name in ARRAY_FILENAMES:
        np.testing.assert_equal(getattr(loaded, name), getattr(original, name))
        assert getattr(loaded, name).shape == getattr(original, name).shape
        assert getattr(loaded, name).dtype == getattr(original, name).dtype
    assert np.isnan(loaded.C_frames[0, 0, 0])
    assert loaded.source_files == original.source_files
    assert dict(loaded.grid_metadata) == dict(original.grid_metadata)
    assert loaded.interpolation_method == original.interpolation_method
    assert loaded.interpolation_engine == original.interpolation_engine
    assert isinstance(loaded.Xi, np.memmap)
    assert isinstance(loaded.Zi, np.memmap)
    assert isinstance(loaded.C_frames, np.memmap)
    assert isinstance(loaded.grid_metadata, MappingProxyType)
    manifest = _manifest(cache_dir)
    assert manifest["interpolation_engine"] == "scattered_linear"
    assert manifest["spec"]["interpolation_engine"] == "scattered_linear"


def test_spectral_metadata_and_polynomial_order_round_trip(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path / "sources",
        interpolation_method="spectral",
        interpolation_engine="spectral_element",
        slice_y=0.75,
    )
    sequence = _sequence(spec)
    cache_dir = tmp_path / "spectral-cache"

    save_concentration_sequence_cache(
        cache_dir, sequence, spec, overwrite=False
    )
    loaded = load_concentration_sequence_cache(cache_dir, spec)
    manifest = _manifest(cache_dir)

    assert spec.slice_y_mode == "explicit"
    assert spec.spectral_algorithm_version is not None
    assert loaded.spectral_element_shape == (4, 4, 4)
    assert loaded.spectral_polynomial_order == (3, 3, 3)
    assert loaded.spectral_slice_y == 0.75
    assert manifest["spectral_element_shape"] == [4, 4, 4]
    assert manifest["spectral_polynomial_order"] == [3, 3, 3]
    assert manifest["resolved_slice_y"] == 0.75


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("file_indices", "file_indices"),
        ("source_path", "source_files\\[0\\]\\.path"),
        ("source_size", "source_files\\[0\\]\\.size"),
        ("source_mtime", "source_files\\[0\\]\\.mtime_ns"),
        ("nx", "nx"),
        ("nz", "nz"),
        ("slice_mode", "slice_mode"),
        ("slab_ratio", "slab_ratio"),
        ("y_round_decimals", "y_round_decimals"),
        ("interpolation_method", "interpolation_method"),
        ("interpolation_engine", "interpolation_engine"),
    ],
)
def test_load_lists_spec_mismatches_with_recovery_guidance(
    tmp_path: Path, field: str, expected_message: str
) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    if field == "file_indices":
        expected = replace(spec, file_indices=(2, 9))
    elif field == "source_path":
        expected = replace(
            spec,
            source_files=(
                replace(spec.source_files[0], path="/different/source"),
                spec.source_files[1],
            ),
        )
    elif field == "source_size":
        expected = replace(
            spec,
            source_files=(
                replace(spec.source_files[0], size=999),
                spec.source_files[1],
            ),
        )
    elif field == "source_mtime":
        expected = replace(
            spec,
            source_files=(
                replace(spec.source_files[0], mtime_ns=999),
                spec.source_files[1],
            ),
        )
    else:
        values = {
            "nx": spec.nx + 1,
            "nz": spec.nz + 1,
            "slice_mode": "slab",
            "slab_ratio": spec.slab_ratio + 0.1,
            "y_round_decimals": spec.y_round_decimals + 1,
            "interpolation_method": "nearest",
            "interpolation_engine": "scattered_nearest",
        }
        expected = replace(spec, **{field: values[field]})

    with pytest.raises(
        FrontDetectionCacheMismatchError, match=expected_message
    ) as error:
        load_concentration_sequence_cache(cache_dir, expected)

    assert "--rebuild-cache" in str(error.value)
    assert "--no-cache" in str(error.value)


def test_load_lists_all_mismatching_fields(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    expected = replace(spec, nx=4, nz=3, slice_mode="slab")

    with pytest.raises(FrontDetectionCacheMismatchError) as error:
        load_concentration_sequence_cache(cache_dir, expected)

    assert all(name in str(error.value) for name in ("nx", "nz", "slice_mode"))


def test_missing_invalid_and_unsupported_manifest_fail(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with pytest.raises(FrontDetectionCacheCorruptionError, match="manifest is missing"):
        load_concentration_sequence_cache(
            cache_dir,
            FrontDetectionCacheSpec(
                CACHE_SCHEMA_VERSION,
                "N7",
                "GC0",
                (1,),
                (SourceFileSignature("/source", 1, 1),),
                2,
                2,
                "nearest_plane",
                0.05,
                10,
                "linear",
                "scattered_linear",
                None,
                "not_applicable",
                None,
                False,
            ),
        )

    (cache_dir / "manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(FrontDetectionCacheCorruptionError, match="invalid JSON"):
        cache_module.load_cache_manifest(cache_dir)

    (cache_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 999}), encoding="utf-8"
    )
    with pytest.raises(FrontDetectionCacheCorruptionError, match="Unsupported"):
        cache_module.load_cache_manifest(cache_dir)


def test_old_schema_cache_requires_explicit_rebuild(tmp_path: Path) -> None:
    assert CACHE_SCHEMA_VERSION == 3
    cache_dir, _, _ = _save(tmp_path)
    payload = _manifest(cache_dir)
    payload["schema_version"] = 2
    payload["spec"]["schema_version"] = 2
    _write_manifest(cache_dir, payload)

    with pytest.raises(
        FrontDetectionCacheCorruptionError,
        match=r"schema version 2.*--rebuild-cache",
    ):
        cache_module.load_cache_manifest(cache_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_key", "missing required key"),
        ("invalid_type", "must be an integer"),
        ("missing_array_metadata", "arrays.C_frames"),
    ],
)
def test_manifest_structure_corruption_fails(
    tmp_path: Path, mutation: str, message: str
) -> None:
    cache_dir, _, _ = _save(tmp_path)
    payload = _manifest(cache_dir)
    if mutation == "missing_key":
        del payload["grid_metadata"]
    elif mutation == "invalid_type":
        payload["spec"]["nx"] = "3"
    else:
        del payload["arrays"]["C_frames"]
    _write_manifest(cache_dir, payload)

    with pytest.raises(FrontDetectionCacheCorruptionError, match=message):
        cache_module.load_cache_manifest(cache_dir)


def test_missing_array_file_fails(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    (cache_dir / "C_frames.npy").unlink()

    with pytest.raises(FrontDetectionCacheCorruptionError, match="array is missing"):
        load_concentration_sequence_cache(cache_dir, spec)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("C_frames", np.zeros((2, 2, 2), dtype=np.float64), "shape"),
        ("C_frames", np.zeros((2, 2, 3), dtype=np.float32), "dtype"),
    ],
)
def test_wrong_array_shape_or_dtype_fails(
    tmp_path: Path, name: str, replacement: np.ndarray, message: str
) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    np.save(cache_dir / ARRAY_FILENAMES[name], replacement, allow_pickle=False)

    with pytest.raises(FrontDetectionCacheCorruptionError, match=message):
        load_concentration_sequence_cache(cache_dir, spec)


def test_inconsistent_sequence_length_fails(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    replacement = np.asarray([0.25], dtype=np.float64)
    np.save(cache_dir / "time.npy", replacement, allow_pickle=False)
    payload = _manifest(cache_dir)
    payload["arrays"]["time"]["shape"] = [1]
    _write_manifest(cache_dir, payload)

    with pytest.raises(
        FrontDetectionCacheCorruptionError, match="time length"
    ):
        load_concentration_sequence_cache(cache_dir, spec)


def test_overwrite_false_preserves_existing_cache(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    manifest_before = (cache_dir / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="rebuild-cache"):
        save_concentration_sequence_cache(
            cache_dir, _sequence(spec, offset=100.0), spec, overwrite=False
        )

    assert (cache_dir / "manifest.json").read_bytes() == manifest_before
    assert np.load(cache_dir / "C_frames.npy")[0, 0, 1] == 1.0


def test_overwrite_true_replaces_existing_cache(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)

    save_concentration_sequence_cache(
        cache_dir, _sequence(spec, offset=100.0), spec, overwrite=True
    )

    assert np.load(cache_dir / "C_frames.npy")[0, 0, 1] == 101.0


def test_failed_write_cleans_temp_and_never_leaves_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    spec = _spec(source_dir)
    sequence = _sequence(spec)
    cache_dir = tmp_path / "cache"
    original_save = cache_module.np.save
    calls = 0

    def fail_on_second_save(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        original_save(*args, **kwargs)

    monkeypatch.setattr(cache_module.np, "save", fail_on_second_save)
    with pytest.raises(OSError, match="synthetic write failure"):
        save_concentration_sequence_cache(
            cache_dir, sequence, spec, overwrite=False
        )

    assert not cache_dir.exists()
    assert list(tmp_path.glob(".cache.tmp-*")) == []


def test_failed_rebuild_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    original_bytes = (cache_dir / "C_frames.npy").read_bytes()

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic rebuild failure")

    monkeypatch.setattr(cache_module.np, "save", fail_save)
    with pytest.raises(OSError, match="synthetic rebuild failure"):
        save_concentration_sequence_cache(
            cache_dir, _sequence(spec, offset=100.0), spec, overwrite=True
        )

    assert (cache_dir / "C_frames.npy").read_bytes() == original_bytes
    assert list(tmp_path.glob(".cache.tmp-*")) == []


def test_acquisition_create_hit_rebuild_and_disabled_modes(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    spec = _spec(source_dir)
    cache_dir = tmp_path / "cache"
    calls = 0

    def builder() -> ConcentrationSequence:
        nonlocal calls
        calls += 1
        return _sequence(spec, offset=float(calls))

    created = acquire_concentration_sequence(
        cache_dir=cache_dir,
        spec=spec,
        builder=builder,
        use_cache=True,
        rebuild_cache=False,
    )
    assert created.mode == "cache_created"
    assert created.cache_dir == cache_dir
    assert calls == 1
    assert cache_dir.is_dir()

    hit = acquire_concentration_sequence(
        cache_dir=cache_dir,
        spec=spec,
        builder=builder,
        use_cache=True,
        rebuild_cache=False,
    )
    assert hit.mode == "cache_hit"
    assert calls == 1

    rebuilt = acquire_concentration_sequence(
        cache_dir=cache_dir,
        spec=spec,
        builder=builder,
        use_cache=True,
        rebuild_cache=True,
    )
    assert rebuilt.mode == "cache_rebuilt"
    assert calls == 2

    unused_cache = tmp_path / "unused"
    disabled = acquire_concentration_sequence(
        cache_dir=unused_cache,
        spec=spec,
        builder=builder,
        use_cache=False,
        rebuild_cache=False,
    )
    assert disabled.mode == "cache_disabled"
    assert disabled.cache_dir is None
    assert calls == 3
    assert not unused_cache.exists()


def test_mismatched_cache_never_calls_builder_or_rebuilds(tmp_path: Path) -> None:
    cache_dir, spec, _ = _save(tmp_path)
    manifest_before = (cache_dir / "manifest.json").read_bytes()
    calls = 0

    def builder() -> ConcentrationSequence:
        nonlocal calls
        calls += 1
        return _sequence(spec)

    with pytest.raises(FrontDetectionCacheMismatchError):
        acquire_concentration_sequence(
            cache_dir=cache_dir,
            spec=replace(spec, slab_ratio=0.2),
            builder=builder,
            use_cache=True,
            rebuild_cache=False,
        )

    assert calls == 0
    assert (cache_dir / "manifest.json").read_bytes() == manifest_before
