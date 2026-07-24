from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.fixed_grid_interpolation import FixedGridGeometryMismatchError
from nek_post.front_detection_io import NekFramePath
from nek_post import front_detection_workflow
from nek_post.front_detection_workflow import build_concentration_sequence


def _frame(index: int) -> NekFramePath:
    return NekFramePath(index=index, path=Path(f"GC0.f{index:05d}"))


def _install_pipeline_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    times: dict[int, float] | None = None,
    grids: dict[int, np.ndarray] | None = None,
):
    times = times or {1: 0.25, 2: 0.5}
    grids = grids or {
        1: np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
        2: np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
    }
    fixed_Xi, fixed_Zi = np.meshgrid(np.arange(3.0), np.arange(2.0))
    calls = {
        "grid": [],
        "plan": [],
        "application": [],
        "interpolation": [],
    }
    plan = object()

    def read(path):
        index = int(Path(path).name[-5:])
        return {"index": index, "time": times[index]}

    def get_time(data):
        return data["time"]

    def slice_data(data, **kwargs):
        index = data["index"]
        return {
            "x": np.array([0.0, 1.0, 2.0]),
            "z": np.array([0.0, 1.0, 0.0]),
            "C": np.full(3, float(index)),
            "selected_y": 0.1 * index,
        }

    def create_grid(data_by_case, nx, nz):
        calls["grid"].append((data_by_case, nx, nz))
        return (
            fixed_Xi,
            fixed_Zi,
            np.arange(3.0),
            np.arange(2.0),
            {"xmin": 0.0, "xmax": 2.0, "zmin": 0.0, "zmax": 1.0, "nx": nx, "nz": nz},
        )

    def interpolate(x, z, values, Xi, Zi, **kwargs):
        index = int(values[0])
        calls["interpolation"].append((Xi, Zi, kwargs))
        return grids[index].copy()

    def build_plan(x, z, Xi, Zi, **kwargs):
        calls["plan"].append((x, z, Xi, Zi, kwargs))
        return plan

    def apply_plan(received_plan, x, z, values):
        index = int(values[0])
        calls["application"].append((received_plan, x, z))
        return grids[index].copy()

    monkeypatch.setattr(front_detection_workflow, "read_nek_file", read)
    monkeypatch.setattr(front_detection_workflow, "get_nek_time", get_time)
    monkeypatch.setattr(front_detection_workflow, "extract_y_slice", slice_data)
    monkeypatch.setattr(
        front_detection_workflow, "create_common_xz_grid", create_grid
    )
    monkeypatch.setattr(front_detection_workflow, "interpolate_to_grid", interpolate)
    monkeypatch.setattr(
        front_detection_workflow,
        "build_fixed_grid_interpolation_plan",
        build_plan,
    )
    monkeypatch.setattr(
        front_detection_workflow,
        "apply_fixed_grid_interpolation_plan",
        apply_plan,
    )
    return calls, fixed_Xi, fixed_Zi, plan


def test_first_slice_defines_fixed_grid_and_collects_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, fixed_Xi, fixed_Zi, plan = _install_pipeline_fakes(monkeypatch)

    sequence = build_concentration_sequence(
        [_frame(2), _frame(1)],
        nx=3,
        nz=2,
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
    )

    assert len(calls["grid"]) == 1
    assert list(calls["grid"][0][0]) == ["current_case"]
    assert len(calls["plan"]) == 1
    assert calls["plan"][0][2] is sequence.Xi
    assert calls["plan"][0][3] is sequence.Zi
    assert len(calls["application"]) == 2
    assert all(call[0] is plan for call in calls["application"])
    assert calls["interpolation"] == []
    assert_array_equal(sequence.Xi, fixed_Xi)
    assert_array_equal(sequence.Zi, fixed_Zi)
    assert_array_equal(sequence.file_indices, [1, 2])
    assert sequence.source_files == (Path("GC0.f00001"), Path("GC0.f00002"))
    assert_allclose(sequence.time, [0.25, 0.5])
    assert sequence.C_frames.shape == (2, 2, 3)
    assert np.isnan(sequence.C_frames[0, 0, 1])
    assert_allclose(sequence.finite_fraction, [5.0 / 6.0, 1.0])
    assert_allclose(sequence.concentration_min, [1.0, 2.0])
    assert_allclose(sequence.concentration_max, [6.0, 7.0])
    assert_allclose(sequence.selected_y, [0.1, 0.2])
    assert sequence.interpolation_method == "linear"
    assert sequence.interpolation_engine == "precomputed_geometry"
    assert dict(sequence.grid_metadata) == {
        "xmin": 0.0,
        "xmax": 2.0,
        "zmin": 0.0,
        "zmax": 1.0,
        "nx": 3,
        "nz": 2,
    }


def test_slab_mode_records_nan_selected_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(monkeypatch)

    sequence = build_concentration_sequence(
        [_frame(1)],
        nx=3,
        nz=2,
        slice_mode="slab",
        slab_ratio=0.01,
        y_round_decimals=10,
    )

    assert np.isnan(sequence.selected_y[0])


def test_legacy_mode_uses_griddata_for_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _, _ = _install_pipeline_fakes(monkeypatch)

    sequence = build_concentration_sequence(
        [_frame(1), _frame(2)],
        nx=3,
        nz=2,
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
        reuse_interpolation_geometry=False,
    )

    assert calls["plan"] == []
    assert calls["application"] == []
    assert len(calls["interpolation"]) == 2
    assert sequence.interpolation_engine == "per_frame_griddata"


def test_geometry_mismatch_identifies_later_source_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(monkeypatch)
    original_apply = front_detection_workflow.apply_fixed_grid_interpolation_plan

    def apply_with_mismatch(plan, x, z, values):
        if int(values[0]) == 2:
            raise FixedGridGeometryMismatchError(
                "source position 1 changed"
            )
        return original_apply(plan, x, z, values)

    monkeypatch.setattr(
        front_detection_workflow,
        "apply_fixed_grid_interpolation_plan",
        apply_with_mismatch,
    )

    with pytest.raises(
        FixedGridGeometryMismatchError,
        match=r"GC0\.f00002.*source position 1",
    ):
        build_concentration_sequence(
            [_frame(1), _frame(2)],
            nx=3,
            nz=2,
            slice_mode="nearest_plane",
            slab_ratio=0.01,
            y_round_decimals=10,
        )


def test_frame_with_no_finite_interpolated_concentration_fails_with_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(
        monkeypatch,
        times={1: 0.25},
        grids={1: np.full((2, 3), np.nan)},
    )

    with pytest.raises(ValueError, match=r"GC0\.f00001.*no finite values"):
        build_concentration_sequence(
            [_frame(1)],
            nx=3,
            nz=2,
            slice_mode="nearest_plane",
            slab_ratio=0.01,
            y_round_decimals=10,
        )


def test_nonfinite_nek_time_fails_with_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(monkeypatch, times={1: np.nan})

    with pytest.raises(ValueError, match=r"GC0\.f00001.*non-finite Nek time"):
        build_concentration_sequence(
            [_frame(1)],
            nx=3,
            nz=2,
            slice_mode="nearest_plane",
            slab_ratio=0.01,
            y_round_decimals=10,
        )


def test_nonincreasing_times_fail_in_file_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(monkeypatch, times={1: 1.0, 2: 0.5})

    with pytest.raises(ValueError, match="strictly increasing and unique"):
        build_concentration_sequence(
            [_frame(2), _frame(1)],
            nx=3,
            nz=2,
            slice_mode="nearest_plane",
            slab_ratio=0.01,
            y_round_decimals=10,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"frame_paths": []}, "At least one"),
        ({"nx": 1}, "nx"),
        ({"nz": 1}, "nz"),
        ({"interpolation_method": "cubic"}, "linear.*nearest"),
    ],
)
def test_workflow_input_validation(
    updates: dict[str, object],
    message: str,
) -> None:
    arguments = {
        "frame_paths": [_frame(1)],
        "nx": 3,
        "nz": 2,
        "slice_mode": "nearest_plane",
        "slab_ratio": 0.01,
        "y_round_decimals": 10,
        "interpolation_method": "linear",
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=message):
        build_concentration_sequence(**arguments)


def test_workflow_writes_no_intermediate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_fakes(monkeypatch, times={1: 0.25})
    frame = NekFramePath(index=1, path=tmp_path / "GC0.f00001")

    build_concentration_sequence(
        [frame],
        nx=3,
        nz=2,
        slice_mode="nearest_plane",
        slab_ratio=0.01,
        y_round_decimals=10,
    )

    assert list(tmp_path.iterdir()) == []
