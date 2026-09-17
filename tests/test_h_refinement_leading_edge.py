from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pymech.core import HexaData
from pymech.neksuite import writenek

import nek_post.h_refinement_leading_edge as production
from nek_post.front_detection_io import NekFramePath
from nek_post.gll import gll_nodes


@pytest.fixture
def frames(tmp_path):
    qz, qy, qx = np.meshgrid(*(gll_nodes(8),)*3, indexing="ij")
    frames = []
    for index, time in enumerate((0., .7, 2.), 1):
        data = HexaData(3, 4, (8, 8, 8), (3, 0, 0, 1, 0))
        data.wdsz, data.endian, data.time, data.istep = 8, "little", time, index
        for cell, element in enumerate(data.elem):
            element.pos[0] = -.5 + cell % 2 + .5*qx
            element.pos[1] = .25 + .5*(cell//2) + .25*qy
            element.pos[2] = .5*(qz+1.)
            element.temp[0] = .1 + (.35 + .1*time) - element.pos[0]
        path = tmp_path / f"GC0.f{index:05d}"
        writenek(str(path), data)
        frames.append(NekFramePath(index, path))
    return tuple(frames)


def test_streaming_production_samples_once_for_both_methods(frames, tmp_path, monkeypatch):
    applications, extractions, builds = [], [], []
    original_apply = production.apply_spectral_horizontal_slice_plan
    original_extract = production.extract_leading_edge
    original_build = production.build_spectral_horizontal_slice_plan

    def build(*a, **kw):
        plan = original_build(*a, **kw)
        builds.append(plan)
        return plan

    def apply(plan, data, **kw):
        applications.append(id(plan))
        return original_apply(plan, data, **kw)

    def extract(x, y, plane, **kw):
        extractions.append((id(plane), kw["method"]))
        return original_extract(x, y, plane, **kw)

    monkeypatch.setattr(production, "build_spectral_horizontal_slice_plan", build)
    monkeypatch.setattr(production, "apply_spectral_horizontal_slice_plan", apply)
    monkeypatch.setattr(production, "extract_leading_edge", extract)
    output = tmp_path / "raw"
    metadata = production.extract_case("test", frames, output,
        expected_x=np.linspace(-1, 1, 11), expected_y=np.linspace(0, 1, 13, endpoint=False),
        y_endpoint=1., progress=lambda *a, **kw: None)
    assert len(builds) == 1
    assert applications == [id(builds[0])]*3
    assert len(extractions) == 6
    assert all(extractions[i][0] == extractions[i+1][0] for i in (0, 2, 4))
    assert metadata["status"] == "complete"
    assert metadata["coordinate_signature_validated_frames"] == 3
    assert metadata["temporal_interpolation"] is False
    assert len(list((output / "frames").glob("*.npz"))) == 3
    with np.load(output / "raw_curves.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["time"], [0., .7, 2.])
        expected = np.broadcast_to(np.array([.35, .42, .55])[:, None], (3, 13))
        for method in production.METHODS:
            prefix = production.method_key(method)
            np.testing.assert_allclose(raw[prefix+"_front"], expected, atol=1e-14)
            assert raw[prefix+"_success"].all()
    assert json.loads((output / "metadata.json").read_text())["frame_count"] == 3
    with pytest.raises(FileExistsError):
        production.extract_case("test", frames, output, expected_x=[0, 1],
                                expected_y=[0, .5], y_endpoint=1.)


def test_production_rejects_different_physical_grid(frames, tmp_path):
    with pytest.raises(ValueError, match="physical x/y/z grid differs"):
        production.extract_case("test", frames, tmp_path / "bad",
            expected_x=np.linspace(-.9, 1, 11), expected_y=np.linspace(0, 1, 13, endpoint=False),
            y_endpoint=1., progress=lambda *a, **kw: None)
