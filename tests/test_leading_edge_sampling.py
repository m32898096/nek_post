from __future__ import annotations

from dataclasses import fields
from functools import partial
import gc
import importlib.util
import json
import multiprocessing
from pathlib import Path
import pickle
import weakref

import numpy as np
import pytest
from pymech.core import HexaData
from pymech.neksuite import writenek

import nek_post.leading_edge_parallel as parallel
import nek_post.leading_edge_workflow as workflow
import nek_post.refined_gll_leading_edge as adapter
from nek_post.config import load_yaml
from nek_post.front_detection_io import NekFramePath
from nek_post.gll import gll_nodes
from nek_post.leading_edge_io import write_leading_edge_csvs
from nek_post.paths import ProjectPaths
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


@pytest.fixture
def frames(tmp_path: Path) -> tuple[NekFramePath, ...]:
    qz, qy, qx = np.meshgrid(*(gll_nodes(8),)*3, indexing="ij")
    result = []
    for index in (1, 2, 3):
        data = HexaData(3, 4, (8, 8, 8), (3, 0, 0, 1, 0))
        data.wdsz, data.endian, data.time, data.istep = 8, "little", index/4, index
        for cell, element in enumerate(data.elem):
            ix, iy = cell % 2, cell // 2
            element.pos[0] = -0.5 + ix + 0.5*qx
            element.pos[1] = 0.25 + 0.5*iy + 0.25*qy
            element.pos[2] = 0.5*(qz+1.)
            element.temp[0] = 0.1 + (0.25+0.1*index) - element.pos[0]
        path = tmp_path / f"GC0.f{index:05d}"
        writenek(str(path), data)
        result.append(NekFramePath(index, path))
    return tuple(reversed(result))


@pytest.mark.parametrize("method", ("rightmost-crossing", "moore-boundary"))
@pytest.mark.parametrize("count", (8, 10))
def test_refined_frames_reuse_one_plan_and_release_planes(frames, monkeypatch, method, count):
    builds, applied, references = [], [], []
    original_build = workflow.build_refined_gll_horizontal_slice_plan
    original_apply = adapter.apply_refined_gll_horizontal_slice_plan

    def build(*args, **kwargs):
        plan = original_build(*args, **kwargs)
        builds.append(plan)
        return plan

    def apply(data, plan):
        gc.collect()
        assert all(ref() is None for ref in references)
        applied.append(id(plan))
        plane = original_apply(data, plan)
        references.append(weakref.ref(plane))
        return plane

    monkeypatch.setattr(workflow, "build_refined_gll_horizontal_slice_plan", build)
    monkeypatch.setattr(adapter, "apply_refined_gll_horizontal_slice_plan", apply)
    evolution = workflow.build_leading_edge_evolution(
        frames, sampling_mode="refined-gll", target_node_count=count,
        z_target=.04, threshold=.25, x_min=0., extraction_method=method,
    )
    assert len(builds) == 1
    assert applied == [id(builds[0])]*3
    assert all(ref() is None for ref in references)
    assert evolution.x_front.shape == (3, 2*(count-1))
    assert evolution.nx == 2*(count-1)+1
    assert evolution.native_ny == 14
    assert evolution.y_upsample_factor is None
    assert evolution.source_node_count == 8
    assert evolution.target_node_count == count
    np.testing.assert_array_equal(evolution.file_indices, [1, 2, 3])
    np.testing.assert_allclose(evolution.x_front, np.broadcast_to(
        np.array([.2, .3, .4])[:, None], evolution.x_front.shape), rtol=0., atol=1e-15)
    assert evolution.plan_build_seconds > 0.
    assert evolution.processing_runtime_seconds >= evolution.plan_build_seconds
    # A spawn worker receives this same pickleable plan; it never rebuilds it.
    restored = pickle.loads(pickle.dumps(builds[0]))
    np.testing.assert_array_equal(restored.x, builds[0].x)


@pytest.mark.parametrize("method", ("rightmost-crossing", "moore-boundary"))
def test_real_spawn_pool_matches_serial_exactly(frames, monkeypatch, method):
    options = dict(sampling_mode="refined-gll", target_node_count=10,
                   z_target=.04, threshold=.1, x_min=0., extraction_method=method)
    serial = workflow.build_leading_edge_evolution(frames, workers=1, **options)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", partial(
        parallel.ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")))
    concurrent = workflow.build_leading_edge_evolution(frames, workers=2, **options)
    for name in ("file_indices", "time", "x", "y", "x_front", "success_mask", "crossing_count"):
        np.testing.assert_array_equal(getattr(serial, name), getattr(concurrent, name))


def test_default_and_explicit_uniform_artifacts_are_byte_identical(frames, tmp_path):
    default = workflow.build_leading_edge_evolution(frames, nx=11, z_target=.04)
    explicit = workflow.build_leading_edge_evolution(
        frames, nx=11, z_target=.04, sampling_mode="uniform-spectral")
    for field in fields(default):
        a, b = getattr(default, field.name), getattr(explicit, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    paths = []
    for label, evolution in (("default", default), ("explicit", explicit)):
        selection = workflow.select_leading_edge_times(evolution, spacing=None)
        paths.append(write_leading_edge_csvs(tmp_path/label, "N7", evolution, selection, False))
    for a, b in zip(*paths, strict=True):
        assert a.read_bytes() == b.read_bytes()
        assert "sampling_mode" not in a.read_text()
    assert default.y_upsample_factor == 2


@pytest.mark.parametrize("options", [
    {"sampling_mode": "bad"}, {"sampling_mode": None},
    {"sampling_mode": "uniform-spectral", "nx": 11, "target_node_count": 10},
    {"sampling_mode": "refined-gll"},
    {"sampling_mode": "refined-gll", "target_node_count": True},
    {"sampling_mode": "refined-gll", "target_node_count": 10, "nx": 1000},
    {"sampling_mode": "refined-gll", "target_node_count": 10, "y_upsample_factor": 2},
])
def test_invalid_sampling_parameters_fail_before_reading(frames, options):
    def forbidden(_path):
        pytest.fail("invalid options read a frame")
    with pytest.raises(ValueError, match="sampling|node_count|nx"):
        workflow.build_leading_edge_evolution(frames, z_target=.04, _frame_reader=forbidden, **options)


def test_refined_geometry_change_has_frame_context(frames):
    original_reader = workflow.read_nek_file
    def reader(path):
        data = original_reader(path)
        if path.name.endswith("00002"):
            data.elem[0].pos[0, 0, 0, 0] += 1e-8
        return data
    with pytest.raises(SpectralGeometryMismatchError, match="GC0.f00002"):
        workflow.build_leading_edge_evolution(frames, sampling_mode="refined-gll",
            target_node_count=10, z_target=.04, _frame_reader=reader)


@pytest.mark.parametrize("method", ("rightmost-crossing", "moore-boundary"))
def test_refined_x_min_and_json_metadata(frames, tmp_path, method):
    evolution = workflow.build_leading_edge_evolution(frames, sampling_mode="refined-gll",
        target_node_count=10, z_target=.04, x_min=.9, extraction_method=method)
    assert not evolution.success_mask.any()
    selection = workflow.select_leading_edge_times(evolution, spacing=None)
    output = write_leading_edge_csvs(tmp_path / method, "N7", evolution, selection, False)
    assert [p.suffix for p in output] == [".csv", ".json"]
    metadata = json.loads(output[1].read_text())
    assert metadata["sampling_mode"] == "refined-gll"
    assert metadata["source_polynomial_order"] == 7
    assert metadata["source_node_count"] == 8
    assert metadata["target_node_count"] == 10
    assert metadata["native_ny"] == 14
    assert metadata["output_ny"] == 18
    assert metadata["y_upsample_factor"] is None
    assert metadata["extraction_method"] == method
    assert metadata["extraction_x_min"] == .9
    assert metadata["extraction_x_condition"] == "strict-greater-than"
    before = [p.read_bytes() for p in output]
    with pytest.raises(FileExistsError):
        write_leading_edge_csvs(tmp_path / method, "N7", evolution, selection, False)
    assert before == [p.read_bytes() for p in output]


@pytest.fixture
def cli_context(tmp_path):
    spec = importlib.util.spec_from_file_location("sampling_cli", Path(__file__).parents[1]/"scripts/19_compute_leading_edge_evolution.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    paths = ProjectPaths(data_root=tmp_path, case_dirs={"N7": tmp_path},
        postproc_root=tmp_path/"post", results_root=tmp_path/"results",
        cantero_fig5a_re3450_csv=tmp_path/"a.csv", cantero_fig5a_re8950_csv=tmp_path/"b.csv")
    config = load_yaml(Path(__file__).parents[1]/"config/cases.yaml")
    return script, paths, config


@pytest.mark.parametrize("arguments", [
    ["--target-node-count", "10"],
    ["--sampling-mode", "refined-gll"],
    ["--sampling-mode", "refined-gll", "--target-node-count", "10", "--nx", "1000"],
    ["--sampling-mode", "refined-gll", "--target-node-count", "10", "--y-upsample-factor", "2"],
    ["--sampling-mode", "unknown"],
])
def test_cli_rejects_mixed_or_missing_route_options(cli_context, arguments):
    script, paths, config = cli_context
    with pytest.raises(SystemExit) as caught:
        script._parse_args(paths, config, arguments)
    assert caught.value.code == 2


def test_cli_default_and_explicit_uniform_arguments_match(cli_context):
    script, paths, config = cli_context
    assert vars(script._parse_args(paths, config, [])) == vars(
        script._parse_args(paths, config, ["--sampling-mode", "uniform-spectral"]))
    args = script._parse_args(paths, config, ["--sampling-mode", "refined-gll", "--target-node-count", "10"])
    assert args.nx is None and args.y_upsample_factor is None
    assert "leading_edge_gll_refinement" in str(args.output_dir)


def test_cli_runs_refined_real_synthetic_files(frames, cli_context, monkeypatch):
    script, paths, config = cli_context
    # Fixture frame files reside in the same tmp_path as the configured case.
    monkeypatch.setattr(script, "load_project_paths", lambda _: paths)
    monkeypatch.setattr(script, "load_yaml", lambda _: config)
    script.main(["--sampling-mode", "refined-gll", "--target-node-count", "10",
                 "--workers", "1", "--all-frames"])
    args = script._parse_args(paths, config, ["--sampling-mode", "refined-gll", "--target-node-count", "10"])
    metadata = json.loads((args.output_dir/"N7_leading_edge_sampling_metadata.json").read_text())
    assert metadata["n_input_frames"] == 3
    assert metadata["n_selected_frames"] == 3
