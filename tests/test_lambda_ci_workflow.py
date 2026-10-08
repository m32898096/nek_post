"""Small actual Nek files and VTK-reader round trips; no N7_H snapshot."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("vtkmodules")
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkFiltersCore import vtkContourFilter

from nek_post.gll import gll_nodes
from nek_post.io_nek import read_nek_file
from nek_post.lambda_ci_export import (
    RectilinearLambdaFields, assemble_rectilinear_lambda_fields,
    write_rectilinear_lambda_vtr, write_single_frame_pvd,
)
from nek_post.lambda_ci_workflow import inspect_single_frame, run_single_frame_lambda_ci
from nek_post.l2_projection import build_structured_node_map


def synthetic_file(tmp_path, *, shape=(3, 3, 3), opposite=False):
    from pymech.core import HexaData
    from pymech.neksuite import writenek
    case = tmp_path / "synthetic"
    case.mkdir()
    data = HexaData(3, 2, list(shape[::-1]), [3, 3, 1, 1, 0])
    data.wdsz, data.endian, data.time, data.istep = 4, "little", .123456789, 17
    t, s, r = np.meshgrid(*(gll_nodes(n) for n in shape), indexing="ij")
    for i, e in enumerate(data.elem):
        e.pos[:] = np.stack((i+(r+1)/2, (s+1)/2, (t+1)/2))
        x, y, z = e.pos
        omega = -2. if opposite and i else 2.
        e.vel[:] = np.stack((-omega*y, omega*x, .5*z))
    source = case / "GC0.f00001"
    writenek(str(source), data)
    return case, source


def read_vtr(path):
    reader = vtkXMLRectilinearGridReader()
    reader.SetFileName(str(path)); reader.Update()
    assert reader.GetErrorCode() == 0
    return reader.GetOutput()


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/33_compute_lambda_ci.py"
    spec = importlib.util.spec_from_file_location("lambda_ci_single_frame_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_arguments_and_exactly_one_index():
    cli = cli_module()
    args = cli.parse_args(["--index", "79", "--chunk-size", "32", "--diagnostics", "--dry-run"])
    assert args.index == 79 and args.chunk_size == 32 and args.diagnostics and args.dry_run
    assert args.output_dir is None
    assert args.periodic_axes == ["x", "y"]
    assert args.expected_element_counts == [272, 12, 8]
    for argv in (["--index", "0"], ["--index", "1,2"], ["--index", "1", "--all-frames"], ["--index", "1", "--chunk-size", "0"]):
        with pytest.raises(SystemExit): cli.parse_args(argv)


def test_header_only_dry_run_creates_nothing_and_does_not_load(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as module
    case, source = synthetic_file(tmp_path)
    output = tmp_path / "absent"
    monkeypatch.setattr(module, "read_nek_file", lambda *a, **k: pytest.fail("dry run loaded data"))
    report = run_single_frame_lambda_ci(case, 1, output, dry_run=True, expected_element_counts=(2, 1, 1))
    assert report["time"] == .123456789 and report["istep"] == 17
    assert report["expected_projection_nodes"] == 24
    assert report["estimated_visual_nodes"] == 45
    assert not output.exists() and source.exists()


def test_complete_workflow_native_gradient_projection_eigenvalues_roundtrip(tmp_path):
    case, source = synthetic_file(tmp_path)
    original = source.read_bytes()
    report = run_single_frame_lambda_ci(case, 1, tmp_path/"export", diagnostics=True, chunk_size=1, expected_element_counts=(2, 1, 1))
    assert source.read_bytes() == original
    assert report["source_sha256"] == sha256(original).hexdigest()
    assert report["source_unchanged"]
    expected_dir = tmp_path/"export"/"synthetic"/"f00001"
    assert Path(report["output_paths"]["vtr"]).parent == expected_dir
    assert Path(report["output_paths"]["pvd"]).parent == expected_dir
    assert Path(report["output_paths"]["metadata"]).parent == expected_dir
    assert json.loads(Path(report["output_paths"]["metadata"]).read_text())["output_paths"] == report["output_paths"]
    assert report["projection"]["global_node_count"] == 24
    np.testing.assert_array_equal(report["projection"]["discontinuity_after"], 0)
    assert report["lambda_ci"]["min"] == pytest.approx(2.)
    assert report["lambda_ci"]["max"] == pytest.approx(2.)
    assert report["peak_rss_bytes"] > 0
    grid = read_vtr(report["output_paths"]["vtr"])
    assert grid.GetDimensions() == (5, 3, 3)
    assert grid.GetNumberOfPoints() == 45 and grid.GetNumberOfCells() == 16
    assert grid.GetBounds() == (0., 2., 0., 1., 0., 1.)
    np.testing.assert_allclose(vtk_to_numpy(grid.GetPointData().GetArray("lambda_ci")), 2.)
    np.testing.assert_allclose(vtk_to_numpy(grid.GetPointData().GetArray("lambda_ci_squared")), 4.)
    assert grid.GetFieldData().GetArray("TIME_VALUE").GetValue(0) == .123456789
    entry = ET.parse(report["output_paths"]["pvd"]).find(".//DataSet")
    assert float(entry.attrib["timestep"]) == .123456789
    assert entry.attrib["file"] == Path(report["output_paths"]["vtr"]).name
    assert not report["visualization"]["paraview_visually_verified"]


def test_workflow_uses_projected_tensor_before_swirling(tmp_path):
    case, _ = synthetic_file(tmp_path, opposite=True)
    report = run_single_frame_lambda_ci(case, 1, tmp_path/"export", periodic_axes=(), expected_element_counts=(2, 1, 1))
    grid = read_vtr(report["output_paths"]["vtr"])
    values = vtk_to_numpy(grid.GetPointData().GetArray("lambda_ci")).reshape(3, 3, 5)
    np.testing.assert_array_equal(values[:, :, 2], 0)
    np.testing.assert_allclose(values[:, :, 0], 2.)
    np.testing.assert_allclose(values[:, :, -1], 2.)


def test_export_native_coordinate_ordering_implicit_connectivity_and_contour(tmp_path):
    case, source = synthetic_file(tmp_path, shape=(3, 4, 5))
    data = read_nek_file(source, dtype="float32")
    mapping = build_structured_node_map(data, periodic_axes=("x", "y"))
    scalar = np.stack([e.pos[0] for e in data.elem]).astype(np.float64)
    visual = assemble_rectilinear_lambda_fields(data, mapping, scalar, scalar**2, chunk_size=1)
    path = write_rectilinear_lambda_vtr(tmp_path/"ordered.vtr", visual, time=1.2)
    grid = read_vtr(path)
    assert grid.GetDimensions() == (9, 4, 3)
    assert grid.GetNumberOfCells() == 8*3*2
    nx, ny, nz = grid.GetDimensions()
    expected_ids = [0, 1, nx, nx+1, nx*ny, nx*ny+1, nx*ny+nx, nx*ny+nx+1]
    cell = grid.GetCell(0)
    assert cell.GetCellType() == 11  # VTK_VOXEL: axis-aligned linear hexahedron
    assert [cell.GetPointId(i) for i in range(8)] == expected_ids
    values = vtk_to_numpy(grid.GetPointData().GetArray("lambda_ci"))
    for point in range(grid.GetNumberOfPoints()):
        assert values[point] == grid.GetPoint(point)[0]
    assert vtk_to_numpy(grid.GetXCoordinates())[0] == 0
    assert vtk_to_numpy(grid.GetXCoordinates())[-1] == 2
    assert not mapping.global_node_count == grid.GetNumberOfPoints()
    contour = vtkContourFilter(); contour.SetInputData(grid); contour.SetValue(0, .5); contour.Update()
    assert contour.GetOutput().GetNumberOfCells() > 0
    assert contour.GetOutput().GetBounds()[0:2] == (.5, .5)


def test_output_is_deterministic_and_collision_does_not_reload(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as module
    case, _ = synthetic_file(tmp_path)
    a = run_single_frame_lambda_ci(case, 1, tmp_path/"a", expected_element_counts=(2, 1, 1))
    b = run_single_frame_lambda_ci(case, 1, tmp_path/"b", chunk_size=1, expected_element_counts=(2, 1, 1))
    for kind in ("vtr", "pvd"):
        assert Path(a["output_paths"][kind]).read_bytes() == Path(b["output_paths"][kind]).read_bytes()
    monkeypatch.setattr(module, "read_nek_file", lambda *a, **k: pytest.fail("collision loaded data"))
    with pytest.raises(FileExistsError):
        run_single_frame_lambda_ci(case, 1, tmp_path/"a", expected_element_counts=(2, 1, 1))


def test_cli_synthetic_single_frame_execution(tmp_path, capsys):
    case, _ = synthetic_file(tmp_path)
    cli_module().main(["--case-dir", str(case), "--index", "1", "--output-dir", str(tmp_path/"cli"), "--expected-element-counts", "2", "1", "1"])
    captured = capsys.readouterr()
    assert '"frame_index": 1' in captured.out
    assert "Reading one snapshot" in captured.err
    assert len(list((tmp_path/"cli"/"synthetic"/"f00001").glob("*.vtr"))) == 1


def test_default_root_uses_case_and_frame_and_never_repository(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as workflow
    case, _ = synthetic_file(tmp_path)
    default_root = tmp_path/"results"/"lambda_ci"
    monkeypatch.setattr(workflow, "DEFAULT_OUTPUT_ROOT", default_root)
    plan = inspect_single_frame(case, 1, expected_element_counts=(2, 1, 1))
    expected = default_root/"synthetic"/"f00001"
    assert {Path(path).parent for path in plan["output_paths"].values()} == {expected}
    assert not Path(plan["output_paths"]["vtr"]).is_relative_to(Path(__file__).resolve().parents[1])
    assert not default_root.exists()


def test_bad_input_output_paths_and_expected_counts(tmp_path):
    case, _ = synthetic_file(tmp_path)
    with pytest.raises(FileNotFoundError): inspect_single_frame(tmp_path/"missing", 1, tmp_path/"out")
    with pytest.raises(FileNotFoundError): inspect_single_frame(case, 2, tmp_path/"out")
    with pytest.raises(ValueError, match="outside"): inspect_single_frame(case, 1, case/"out")
    occupied = tmp_path/"not_a_directory"; occupied.write_text("preserve")
    with pytest.raises(ValueError, match="directory"): inspect_single_frame(case, 1, occupied)
    with pytest.raises(ValueError, match="element count"): inspect_single_frame(case, 1, tmp_path/"out")
    assert occupied.read_text() == "preserve"


def test_memory_gate_stops_before_snapshot_load(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as module
    case, _ = synthetic_file(tmp_path)
    monkeypatch.setattr(module, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(module, "read_nek_file", lambda *a, **k: pytest.fail("memory gate loaded data"))
    with pytest.raises(MemoryError):
        run_single_frame_lambda_ci(case, 1, tmp_path/"out", expected_element_counts=(2, 1, 1))
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("kind", ("nan", "inf", "negative", "square", "shared", "shape"))
def test_invalid_export_scalar_results_rejected(tmp_path, kind):
    case, source = synthetic_file(tmp_path)
    data = read_nek_file(source, dtype="float32")
    mapping = build_structured_node_map(data, periodic_axes=("x", "y"))
    scalar = np.ones((2, 3, 3, 3))
    squared = scalar**2
    if kind == "nan": scalar[0, 0, 0, 0] = np.nan
    elif kind == "inf": scalar[0, 0, 0, 0] = np.inf
    elif kind == "negative": scalar[0, 0, 0, 0] = -1
    elif kind == "square": squared += 1
    elif kind == "shared": scalar[1, :, :, 0] = 2; squared = scalar**2
    elif kind == "shape": scalar = scalar[0]
    with pytest.raises(ValueError): assemble_rectilinear_lambda_fields(data, mapping, scalar, squared)


def test_strict_export_rejects_geometry_average_and_never_overwrites(tmp_path):
    case, source = synthetic_file(tmp_path)
    data = read_nek_file(source, dtype="float32")
    data.elem[0].pos[1, 1, 0, 0] += np.float32(1e-7)
    mapping = build_structured_node_map(data, periodic_axes=())
    scalar = np.ones((2, 3, 3, 3))
    with pytest.raises(ValueError, match="exactly rectilinear"):
        assemble_rectilinear_lambda_fields(data, mapping, scalar, scalar)
    fields = RectilinearLambdaFields((np.arange(2.),)*3, np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    path = tmp_path/"preserve.vtr"; path.write_bytes(b"original")
    with pytest.raises(FileExistsError): write_rectilinear_lambda_vtr(path, fields, time=0)
    assert path.read_bytes() == b"original"


def test_disk_gate_and_dangling_output_collision(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as module
    case, _ = synthetic_file(tmp_path)
    plan = inspect_single_frame(case, 1, tmp_path/"out", expected_element_counts=(2, 1, 1))
    Path(plan["output_paths"]["vtr"]).parent.mkdir(parents=True)
    Path(plan["output_paths"]["vtr"]).symlink_to(tmp_path/"missing_target")
    with pytest.raises(FileExistsError):
        inspect_single_frame(case, 1, tmp_path/"out", expected_element_counts=(2, 1, 1))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    monkeypatch.setattr(module, "read_nek_file", lambda *a, **k: pytest.fail("disk gate loaded data"))
    with pytest.raises(MemoryError):
        run_single_frame_lambda_ci(case, 1, tmp_path/"other", expected_element_counts=(2, 1, 1))


@pytest.mark.parametrize("attribute,value", (("nb_dims", 2), ("nb_vars", (3, 0, 1, 1, 0)), ("nb_files", 2), ("nb_elems_file", 1), ("wdsz", 16), ("time", np.nan)))
def test_invalid_headers_rejected_without_loading(tmp_path, monkeypatch, attribute, value):
    import nek_post.lambda_ci_workflow as module
    case, _ = synthetic_file(tmp_path)
    h = SimpleNamespace(nb_dims=3, nb_vars=(3, 3, 1, 1, 0), nb_files=1,
        nb_elems_file=2, nb_elems=2, wdsz=4, time=.123456789, istep=17, orders=(3, 3, 3))
    setattr(h, attribute, value)
    monkeypatch.setattr(module, "read_nek_header", lambda p: h)
    with pytest.raises(ValueError):
        inspect_single_frame(case, 1, tmp_path/"out", expected_element_counts=(2, 1, 1))


def test_writer_invalid_fields_paths_and_single_frame_pvd(tmp_path):
    good = RectilinearLambdaFields((np.arange(2.),)*3, np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    with pytest.raises(ValueError): write_rectilinear_lambda_vtr(tmp_path/"bad.vtu", good, time=0)
    with pytest.raises(ValueError): write_rectilinear_lambda_vtr(tmp_path/"bad.vtr", good, time=np.inf)
    bad = RectilinearLambdaFields(good.coordinates, np.full((2, 2, 2), np.nan), good.lambda_ci_squared)
    with pytest.raises(ValueError): write_rectilinear_lambda_vtr(tmp_path/"bad.vtr", bad, time=0)
    assert not (tmp_path/"bad.vtr").exists()
    vtr = write_rectilinear_lambda_vtr(tmp_path/"one.vtr", good, time=0)
    with pytest.raises(ValueError): write_single_frame_pvd(tmp_path/"elsewhere/one.pvd", vtr, time=0)
    pvd = write_single_frame_pvd(tmp_path/"one.pvd", vtr, time=0)
    original = pvd.read_bytes()
    with pytest.raises(FileExistsError): write_single_frame_pvd(pvd, vtr, time=0)
    assert pvd.read_bytes() == original


def test_nonfinite_swirling_result_stops_without_output(tmp_path, monkeypatch):
    import nek_post.lambda_ci_workflow as module
    case, _ = synthetic_file(tmp_path)
    shape = (2, 3, 3, 3)
    monkeypatch.setattr(module, "compute_swirling_strength", lambda *a, **k: SimpleNamespace(lambda_ci=np.full(shape, np.nan), lambda_ci_squared=np.full(shape, np.nan)))
    with pytest.raises(ValueError):
        run_single_frame_lambda_ci(case, 1, tmp_path/"out", expected_element_counts=(2, 1, 1))
    assert not (tmp_path/"out").exists()
