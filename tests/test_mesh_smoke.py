"""Small 3-D end-to-end checks using a fresh native Gmsh process per case.

Run from the repository root with ``python -m pytest tests/test_mesh_smoke.py``.
The synthetic binary has variation in every spatial direction and two water rows.
No Spyro, Firedrake, or SeismicMesh installation is needed or imported.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEST_ROOT if (TEST_ROOT / "RobustMesher/__init__.py").is_file() else TEST_ROOT.parent
RESULT_FILE = TEST_ROOT / "validation/mesh_smoke_results.json"
CASES = [
    ("tetra_none", "tetra", None, True, False),
    ("tetra_rectangular", "tetra", "rectangular", True, False),
    ("tetra_hyperelliptical", "tetra", "hyperelliptical", True, False),
    ("hex_none_winslow", "hexahedron", None, True, False),
    ("hex_rectangular_winslow", "hexahedron", "rectangular", True, False),
    ("direct_automatic_tetra", "tetra", None, False, True),
]


@pytest.fixture(scope="module", autouse=True)
def _initialize_result_file():
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text("[]\n")


def _run_worker(case_name, work, report):
    """Execute mesh construction and validation in an isolated native process."""
    import importlib.abc
    import numpy as np

    forbidden_attempts = []

    class RejectExternalMesher(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0].lower() == "seismicmesh":
                forbidden_attempts.append(fullname)
                raise AssertionError(f"Unexpected SeismicMesh import attempt: {fullname}")
            return None

    assert not any(name.lower().startswith("seismicmesh") for name in sys.modules)
    sys.meta_path.insert(0, RejectExternalMesher())
    sys.path.insert(0, str(ROOT))
    from RobustMesher import generate_mesh
    from RobustMesher.sizing_utils import create_sizing_function3D
    import gmsh
    import meshio

    _, expected_type, padding, water, direct = next(item for item in CASES if item[0] == case_name)
    structured = expected_type == "hexahedron"
    nz, nx, ny = 9, 11, 7
    length_z, length_x, length_y = 400.0, 600.0, 300.0
    padding_x, padding_y, padding_z = 120.0, 80.0, 100.0
    zz, xx, yy = np.indices((nz, nx, ny))
    velocity = (
        2100.0 + 180.0 * np.sin(0.85 * zz)
        + 120.0 * np.sin(0.9 * xx)
        + 90.0 * np.cos(0.8 * yy)
        + 12.0 * zz + 8.0 * xx + 6.0 * yy
    )
    velocity[:2] = 1500.0
    work.mkdir(parents=True, exist_ok=True)
    binary = work / "synthetic_zxy.bin"
    np.asarray(velocity, dtype=">f4").ravel(order="F").tofile(binary)
    output = work / f"{case_name}.msh"

    sizing_inputs = dict(
        fname=binary, hmin=60.0,
        bbox=(-length_z, 0.0, 0.0, length_x, 0.0, length_y),
        wl=2.0, freq=10.0, vp_water=1500.0,
        nz=nz, nx=nx, ny=ny, byte_order="big", axes_order=(0, 1, 2),
        axes_order_sort="F", dtype="float32", pad_type=padding,
        pad_size_x=padding_x, pad_size_y=padding_y, pad_size_z=padding_z,
    )
    sizing = create_sizing_function3D(**sizing_inputs, grade=0.6)
    ungraded = create_sizing_function3D(**sizing_inputs, grade=None)
    queries = np.column_stack((
        -zz.ravel() * length_z / (nz - 1),
        xx.ravel() * length_x / (nx - 1),
        yy.ravel() * length_y / (ny - 1),
    ))
    smoothing_change = float(np.max(np.abs(sizing[0](queries) - ungraded[0](queries))))
    assert smoothing_change > 1e-3, "The nonzero grade did not affect this nonlinear velocity field"
    assert sizing[3:] == (nz, nx, ny)
    assert 0.0 < sizing[1] <= sizing[2]
    parameters = dict(
        length_z=length_z, length_x=length_x, length_y=length_y,
        padding_type=padding, padding_x=padding_x, padding_y=padding_y,
        padding_z=padding_z, hyper_n=4.0,
        water_interface=water, water_search_value=1500.0, vp_water=1500.0,
        structured_mesh=structured, min_element_size=100.0,
        apply_winslow=structured, winslow_implementation="numba",
        winslow_iterations=3, winslow_omega=0.1,
        extend_segy=True, h_padding=150.0,
        output_filename=str(output),
        segy_nz=nz, segy_nx=nx, segy_ny=ny,
        segy_dz=length_z / (nz - 1), segy_dx=length_x / (nx - 1),
        segy_dy=length_y / (ny - 1), segy_byte_order="big",
        segy_axes_order=(0, 1, 2), segy_axes_order_sort="F",
        segy_dtype="float32", gmsh_parallel=False, gmsh_num_threads=1,
    )
    if direct:
        from RobustMesher.generation_utils.automatic_mesh import AutomaticMesh
        from RobustMesher.generation_utils.parameters import MeshingParameters
        from RobustMesher.generation_utils.bindings import _prepared_sizing
        assert _prepared_sizing.get() is None
        dictionary = dict(
            parameters, mesh_type="gmsh_mesh", dimension=3,
            segy_velocity_model=str(binary), abc_pad_length=0.0,
            edge_length=None, source_frequency=10.0,
            cells_per_wavelength=2.0, grade=0.6, hmin_segy=60.0,
        )
        configuration = MeshingParameters(input_mesh_dictionary=dictionary, dimension=3)
        mesh = AutomaticMesh(mesh_parameters=configuration).create_gmsh_3D_mesh()
    else:
        mesh = generate_mesh(sizing_function=sizing, dimension=3, velocity_file=binary, **parameters)
    assert output.is_file()
    saved = meshio.read(output)
    assert len(mesh.points) == len(saved.points) > 0
    assert np.all(np.isfinite(saved.points))
    volume_types = {"tetra", "hexahedron", "wedge", "pyramid"}
    actual_volume_types = {block.type for block in saved.cells if block.type in volume_types}
    assert actual_volume_types == {expected_type}, actual_volume_types
    volume_count = sum(len(block.data) for block in saved.cells if block.type == expected_type)
    assert volume_count > 0
    assert "gmsh:physical" in saved.cell_data

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    minimum_jacobian = float("inf")
    physical_volume_counts = {}
    try:
        gmsh.open(str(output))
        for dimension, tag in gmsh.model.getPhysicalGroups():
            if dimension != 3:
                continue
            name = gmsh.model.getPhysicalName(dimension, tag)
            counts = 0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dimension, tag):
                _, element_tags, _ = gmsh.model.mesh.getElements(3, int(entity))
                counts += sum(len(tags) for tags in element_tags)
            physical_volume_counts[name] = int(counts)
        required = {"Water", "Subsurface"} if water else {"SubsurfaceAndWater"}
        if padding is not None:
            required.add("Padding")
        assert required <= physical_volume_counts.keys(), physical_volume_counts
        assert all(physical_volume_counts[name] > 0 for name in required), physical_volume_counts
        for element_type in gmsh.model.mesh.getElementTypes(dim=3):
            local_coordinates, _ = gmsh.model.mesh.getIntegrationPoints(int(element_type), "Gauss2")
            _, determinants, _ = gmsh.model.mesh.getJacobians(int(element_type), local_coordinates)
            determinants = np.asarray(determinants)
            assert determinants.size and np.all(np.isfinite(determinants))
            element_minimum = float(np.min(determinants))
            assert element_minimum > 0.0, f"Nonpositive Gauss2 Jacobian: {element_minimum}"
            minimum_jacobian = min(minimum_jacobian, element_minimum)
    finally:
        gmsh.finalize()

    minimum = saved.points.min(axis=0)
    maximum = saved.points.max(axis=0)
    if padding is None:
        expected_minimum = np.array([0.0, 0.0, 0.0])
        expected_maximum = np.array([length_z, length_x, length_y])
    else:
        expected_minimum = np.array([-length_z - padding_z, 0.0, 0.0])
        expected_maximum = np.array([0.0, length_x + 2.0 * padding_x, length_y + 2.0 * padding_y])
    if padding != "hyperelliptical":
        assert np.allclose(minimum, expected_minimum, atol=1e-7), (minimum, expected_minimum)
        assert np.allclose(maximum, expected_maximum, atol=1e-7), (maximum, expected_maximum)
    else:
        # Faceted spline padding samples the analytic hyperellipsoid envelope.
        tolerance = 0.03 * np.maximum(expected_maximum - expected_minimum, 1.0)
        assert np.all(minimum >= expected_minimum - tolerance), minimum
        assert np.all(maximum <= expected_maximum + tolerance), maximum
        assert np.all(np.abs(minimum - expected_minimum) <= tolerance), minimum
        assert np.all(np.abs(maximum - expected_maximum) <= tolerance), maximum

    assert not forbidden_attempts, forbidden_attempts
    assert "spyro" not in sys.modules
    assert "firedrake" not in sys.modules
    result = dict(
        case=case_name, status="passed", mesh_type=expected_type,
        padding_type=padding, water_interface=water,
        direct_automatic_mesh=direct, grade=0.6,
        smoothing_maximum_change_metres=smoothing_change,
        sizing_minimum_metres=float(sizing[1]), sizing_maximum_metres=float(sizing[2]),
        original_sample_counts=list(sizing[3:]),
        winslow_iterations=3 if structured else 0,
        nodes=len(saved.points), volume_elements=volume_count,
        physical_volume_element_counts=physical_volume_counts,
        minimum_gauss2_jacobian=minimum_jacobian,
        mesh_coordinate_minimum_zxy=minimum.tolist(),
        mesh_coordinate_maximum_zxy=maximum.tolist(),
        seismicmesh_import_attempts=forbidden_attempts,
        spyro_imported=False, firedrake_imported=False,
    )
    report.write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("case", [item[0] for item in CASES])
def test_native_3d_mesh_smoke(tmp_path, case):
    started = time.monotonic()
    report = tmp_path / "result.json"
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("NUMBA_NUM_THREADS", "2")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--smoke-worker", case, str(tmp_path), str(report)],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=120,
    )
    if completed.returncode == 0 and report.is_file():
        result = json.loads(report.read_text())
    else:
        result = dict(case=case, status="failed", returncode=completed.returncode,
                      output=(completed.stdout + completed.stderr)[-10000:])
    result["seconds"] = round(time.monotonic() - started, 3)
    records = json.loads(RESULT_FILE.read_text())
    records.append(result)
    RESULT_FILE.write_text(json.dumps(records, indent=2) + "\n")
    assert completed.returncode == 0, result.get("output", "Missing child report")
    assert report.is_file()


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--smoke-worker":
    _run_worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
