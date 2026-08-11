from __future__ import annotations

import builtins
import sys
import types
import weakref

import numpy as np
import pytest

from grid_generator import (
    ChannelGridSpec,
    GlobalGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
    generate_grid_to_netcdf,
)
from grid_generator import _netcdf
from grid_generator import _streaming
from grid_generator.cutting import cut_grid
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec


def unit_rows(points):
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]


def test_netcdf_field_profiles_have_audited_contract_sizes():
    profiles = _netcdf.ICON_NETCDF_FIELD_PROFILES

    assert set(profiles) == {"full", "reduced", "icon", "icon4py"}
    assert len(_netcdf.ICON_NETCDF_FIELD_NAMES) == 88
    assert len(profiles["full"]) == 88
    assert len(profiles["reduced"]) == 46
    assert len(profiles["icon"]) == 38
    assert len(profiles["icon4py"]) == 26
    assert profiles["reduced"] == profiles["icon"] | profiles["icon4py"]


@pytest.mark.parametrize("profile", ["full", "reduced", "icon", "icon4py"])
def test_in_memory_netcdf_field_profile_writes_exact_contract(profile, tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    output = generate_grid("R01B00").to_netcdf(
        tmp_path / f"{profile}.nc",
        fields=profile,
    )
    expected = [
        name
        for name in _netcdf.ICON_NETCDF_FIELD_NAMES
        if name in _netcdf.ICON_NETCDF_FIELD_PROFILES[profile]
    ]

    with netcdf4.Dataset(output) as dataset:
        assert list(dataset.variables) == expected
        for variable in dataset.variables.values():
            if "bounds" in variable.ncattrs():
                assert variable.getncattr("bounds") in dataset.variables
            if "coordinates" in variable.ncattrs():
                assert set(variable.getncattr("coordinates").split()) <= set(
                    dataset.variables
                )


def test_netcdf_explicit_field_collection_is_validated_and_schema_ordered(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    output = generate_grid("R01B00").to_netcdf(
        tmp_path / "custom.nc",
        fields=["vertex_of_cell", "vlon", "clon"],
    )

    with netcdf4.Dataset(output) as dataset:
        assert list(dataset.variables) == ["clon", "vlon", "vertex_of_cell"]
        assert "bounds" not in dataset.variables["clon"].ncattrs()

    with pytest.raises(ValueError, match="unknown NetCDF field profile"):
        generate_grid("R01B00").to_netcdf(
            tmp_path / "bad-profile.nc",
            fields="portable",
        )
    with pytest.raises(ValueError, match="unknown NetCDF field name"):
        generate_grid("R01B00").to_netcdf(
            tmp_path / "bad-field.nc",
            fields=["clon", "not_a_grid_field"],
        )
    with pytest.raises(TypeError, match="must be a string"):
        generate_grid("R01B00").to_netcdf(
            tmp_path / "bad-type.nc",
            fields=["clon", 42],
        )
    with pytest.raises(ValueError, match="unknown NetCDF field profile"):
        generate_grid_to_netcdf(
            "R02B12",
            tmp_path / "never-generated.nc",
            {"max_cells": None},
            fields="portable",
        )
    assert not (tmp_path / ".never-generated.nc.work").exists()


def test_default_netcdf_includes_established_compatibility_fields(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    output = generate_grid("R02B01").to_netcdf(tmp_path / "grid.nc")

    with netcdf4.Dataset(output) as dataset:
        assert dataset.variables["quadrilateral_area"].dimensions == ("edge",)
        assert np.all(dataset.variables["quadrilateral_area"][:] == 0.0)
        assert dataset.variables["vlon_vertices"].dimensions == ("vertex", "ne")
        assert dataset.variables["vlat_vertices"].dimensions == ("vertex", "ne")
        assert dataset.variables["vlon"].getncattr("bounds") == "vlon_vertices"
        assert dataset.variables["vlat"].getncattr("bounds") == "vlat_vertices"


def test_streamed_global_netcdf_matches_in_memory_export(monkeypatch, tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    pytest.importorskip("numba")
    options = {"max_cells": None, "accelerator": "numba"}
    reference_path = generate_grid("R01B02", options).to_netcdf(
        tmp_path / "reference.nc"
    )
    monkeypatch.setattr(_streaming, "DEFAULT_IN_MEMORY_BASE_CELLS", 80)
    streamed_path = generate_grid_to_netcdf(
        "R01B02",
        tmp_path / "streamed.nc",
        options,
        chunk_size=17,
    )

    with (
        netcdf4.Dataset(reference_path) as reference,
        netcdf4.Dataset(streamed_path) as streamed,
    ):
        assert list(streamed.dimensions) == list(reference.dimensions)
        assert list(streamed.variables) == list(reference.variables)
        for name, expected_variable in reference.variables.items():
            actual_variable = streamed.variables[name]
            assert actual_variable.dtype == expected_variable.dtype
            assert actual_variable.dimensions == expected_variable.dimensions
            assert set(actual_variable.ncattrs()) == set(expected_variable.ncattrs())
            expected = expected_variable[:]
            actual = actual_variable[:]
            if np.issubdtype(expected.dtype, np.floating):
                assert np.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-12), name
            else:
                assert np.array_equal(actual, expected), name
        for name in (
            "uuidOfHGrid",
            "grid_root",
            "grid_level",
            "sphere_radius",
            "global_optimization",
        ):
            assert streamed.getncattr(name) == reference.getncattr(name)

    checkpoint = tmp_path / ".streamed.nc.work" / "R01B02"
    assert (checkpoint / "manifest.json").is_file()
    assert not (tmp_path / "streamed.nc.partial").exists()


@pytest.mark.parametrize(
    "fields",
    [
        "reduced",
        "icon",
        "icon4py",
        ["clon", "vertex_of_cell", "edge_length"],
    ],
    ids=["reduced", "icon", "icon4py", "custom"],
)
def test_streamed_netcdf_field_selection_matches_in_memory_export(
    fields,
    monkeypatch,
    tmp_path,
):
    netcdf4 = pytest.importorskip("netCDF4")
    pytest.importorskip("numba")
    options = {"max_cells": None, "accelerator": "numba"}
    reference_path = generate_grid("R01B01", options).to_netcdf(
        tmp_path / "reference.nc",
        fields=fields,
    )
    monkeypatch.setattr(_streaming, "DEFAULT_IN_MEMORY_BASE_CELLS", 20)
    streamed_path = generate_grid_to_netcdf(
        "R01B01",
        tmp_path / "streamed.nc",
        options,
        chunk_size=17,
        fields=fields,
    )

    with (
        netcdf4.Dataset(reference_path) as reference,
        netcdf4.Dataset(streamed_path) as streamed,
    ):
        assert list(streamed.variables) == list(reference.variables)
        for name, expected_variable in reference.variables.items():
            actual_variable = streamed.variables[name]
            assert actual_variable.dimensions == expected_variable.dimensions
            assert set(actual_variable.ncattrs()) == set(expected_variable.ncattrs())
            expected = expected_variable[:]
            actual = actual_variable[:]
            if np.issubdtype(expected.dtype, np.floating):
                assert np.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-12), name
            else:
                assert np.array_equal(actual, expected), name


def test_custom_streamed_mesh_fields_skip_cell_and_edge_center_construction(
    monkeypatch,
    tmp_path,
):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid("R01B00")
    compact = _streaming._compact_from_icon_grid(grid)

    def fail_centers(*_args, **_kwargs):
        raise AssertionError("unselected centers must not be constructed")

    monkeypatch.setattr(
        "grid_generator.grid_generator._cell_centers",
        fail_centers,
    )
    monkeypatch.setattr(
        "grid_generator.grid_generator._edge_centers",
        fail_centers,
    )
    output = _streaming._write_compact_global_grid(
        compact,
        tmp_path / "mesh.nc",
        chunk_size=17,
        selected_fields=frozenset({"vlon", "vlat", "vertex_of_cell"}),
    )

    with netcdf4.Dataset(output) as dataset:
        assert list(dataset.variables) == ["vlon", "vlat", "vertex_of_cell"]
        assert {
            "mean_cell_area",
            "mean_dual_cell_area",
            "mean_edge_length",
            "mean_dual_edge_length",
        }.isdisjoint(dataset.ncattrs())


def test_streamed_generation_rejects_non_global_grid(tmp_path):
    with pytest.raises(TypeError, match="global grids only"):
        generate_grid_to_netcdf(
            TorusGridSpec(nx=3, ny=4, edge_length=1.0),
            tmp_path / "torus.nc",
        )


@pytest.mark.parametrize(
    ("spec", "expected_base_bisection"),
    [
        (GlobalGridSpec(root=2, bisections=12), 7),
        (GlobalGridSpec(root=32, bisections=8), 3),
        (GlobalGridSpec(root=32, bisections=6), 3),
    ],
)
def test_streamed_base_stage_is_selected_by_cell_count(
    spec,
    expected_base_bisection,
):
    assert (
        _streaming._in_memory_base_bisection(spec) == expected_base_bisection
    )


def test_streamed_generation_rejects_unbounded_root_stage(tmp_path):
    with pytest.raises(ValueError, match="different grid identity"):
        generate_grid_to_netcdf(
            GlobalGridSpec(root=257, bisections=0),
            tmp_path / "grid.nc",
            {"max_cells": None},
        )


def test_streamed_generation_rebuilds_non_exportable_intermediate_checkpoint(
    monkeypatch, tmp_path
):
    pytest.importorskip("netCDF4")
    pytest.importorskip("numba")
    monkeypatch.setattr(_streaming, "DEFAULT_IN_MEMORY_BASE_CELLS", 80)
    options = {"max_cells": None, "accelerator": "numba"}
    work_dir = tmp_path / "checkpoints"

    generate_grid_to_netcdf("R01B03", tmp_path / "b3.nc", options, work_dir=work_dir)
    intermediate_manifest = work_dir / "R01B02" / "manifest.json"
    assert '"has_provenance": false' in intermediate_manifest.read_text()

    output = generate_grid_to_netcdf(
        "R01B02", tmp_path / "b2.nc", options, work_dir=work_dir
    )
    assert output.is_file()
    assert '"has_provenance": true' in intermediate_manifest.read_text()


def test_compact_refinement_releases_parent_and_center_arrays_before_incidence(
    monkeypatch,
):
    pytest.importorskip("numba")
    options = {"accelerator": "numba", "spring_iterations": 0}
    parent = _streaming._compact_from_icon_grid(generate_grid("R01B00", options))
    parent_array_refs = [
        weakref.ref(value)
        for value in (
            parent.vertices,
            parent.cells,
            parent.edges,
            parent.cell_edges,
            parent.edge_cells,
            parent.edges_of_vertex,
            parent.edge_primal_normal_cartesian,
        )
    ]
    center_refs = []
    from grid_generator import grid_generator as gg

    cell_centers = gg._cell_centers
    edge_centers = gg._edge_centers
    compact_edges = _streaming._compact_edges_of_vertex

    def tracked_cell_centers(*args, **kwargs):
        values = cell_centers(*args, **kwargs)
        center_refs.append(weakref.ref(values))
        return values

    def tracked_edge_centers(*args, **kwargs):
        values = edge_centers(*args, **kwargs)
        center_refs.append(weakref.ref(values))
        return values

    def checked_compact_edges(*args, **kwargs):
        assert all(reference() is None for reference in parent_array_refs)
        assert all(reference() is None for reference in center_refs)
        return compact_edges(*args, **kwargs)

    monkeypatch.setattr(gg, "_cell_centers", tracked_cell_centers)
    monkeypatch.setattr(gg, "_edge_centers", tracked_edge_centers)
    monkeypatch.setattr(_streaming, "_compact_edges_of_vertex", checked_compact_edges)

    child = _streaming._refine_compact_global_grid(
        parent,
        GlobalGridSpec(root=1, bisections=1),
        terminal=True,
    )

    assert child.dims == {"cell": 80, "edge": 120, "vertex": 42}
    assert parent.dims == {"cell": 0, "edge": 0, "vertex": 0}


def test_checkpoint_interruption_preserves_previous_snapshot(monkeypatch, tmp_path):
    options = {"optimize_global": False}
    source = _streaming._compact_from_icon_grid(generate_grid("R01B00", options))
    store_a = _streaming._CheckpointStore(tmp_path, "configuration-a")
    store_a.save(source)
    original_manifest = (tmp_path / "R01B00" / "manifest.json").read_text()
    original_vertices = source.vertices.copy()

    replacement = _streaming._compact_from_icon_grid(generate_grid("R01B00", options))
    replacement.vertices = replacement.vertices.copy()
    replacement.vertices[0, 0] = 123.0
    store_b = _streaming._CheckpointStore(tmp_path, "configuration-b")
    replace = _streaming.os.replace
    replacement_count = 0

    def interrupt_after_first_array(source_path, destination_path):
        nonlocal replacement_count
        replacement_count += 1
        replace(source_path, destination_path)
        if replacement_count == 1:
            raise RuntimeError("simulated checkpoint interruption")

    monkeypatch.setattr(_streaming.os, "replace", interrupt_after_first_array)
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        store_b.save(replacement)

    assert (tmp_path / "R01B00" / "manifest.json").read_text() == original_manifest
    resumed = store_a.load(source.spec, source.options)
    assert resumed is not None
    assert np.array_equal(resumed.vertices, original_vertices)
    assert store_b.load(replacement.spec, replacement.options) is None


def test_streamed_generation_does_not_publish_incomplete_output(monkeypatch, tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("numba")
    monkeypatch.setattr(_streaming, "DEFAULT_IN_MEMORY_BASE_CELLS", 80)

    def fail_metrics(*_args, **_kwargs):
        raise RuntimeError("injected metric failure")

    monkeypatch.setattr(_streaming, "_write_metric_fields", fail_metrics)
    output = tmp_path / "failed.nc"
    with pytest.raises(RuntimeError, match="injected metric failure"):
        generate_grid_to_netcdf(
            "R01B02",
            output,
            {"max_cells": None, "accelerator": "numba"},
            chunk_size=17,
        )

    assert not output.exists()
    assert (tmp_path / "failed.nc.partial").is_file()


def test_high_resolution_streaming_fails_early_without_accelerator(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(_streaming, "DEFAULT_IN_MEMORY_BASE_CELLS", 80)
    monkeypatch.setattr(_streaming._accelerated, "should_use_numba", lambda *_: False)

    with pytest.raises(ModuleNotFoundError, match="streamed high-resolution"):
        generate_grid_to_netcdf(
            "R01B02",
            tmp_path / "grid.nc",
            {"max_cells": None},
        )


def test_torus_netcdf_export_contains_complete_periodic_grid(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(TorusGridSpec(nx=4, ny=4, edge_length=2.0))
    path = grid.to_netcdf(tmp_path / "torus.nc")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.dimensions["cell"].size == 32
        assert dataset.dimensions["vertex"].size == 16
        assert dataset.dimensions["edge"].size == 48
        assert dataset.getncattr("grid_geometry") == 2
        assert dataset.getncattr("periodic_layout") == "rectangular"
        assert dataset.getncattr("torus_nx") == 4
        assert dataset.getncattr("torus_ny") == 4
        assert np.array_equal(
            dataset.variables["adjacent_cell_of_edge"][:], grid.edge_cells.T + 1
        )
        assert np.allclose(
            dataset.variables["cell_area"][:], grid.geometry["cell_area"]
        )
        assert np.allclose(
            dataset.variables["edge_length"][:], grid.geometry["edge_length"]
        )
        assert dataset.variables["edgequad_area"].getncattr("units") == "m2"

        vertices = np.column_stack(
            (
                dataset.variables["cartesian_x_vertices"][:],
                dataset.variables["cartesian_y_vertices"][:],
            )
        )
        edges = dataset.variables["edge_vertices"][:].T - 1
        vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        vectors[:, 0] -= dataset.getncattr("domain_length") * np.rint(
            vectors[:, 0] / dataset.getncattr("domain_length")
        )
        vectors[:, 1] -= dataset.getncattr("domain_height") * np.rint(
            vectors[:, 1] / dataset.getncattr("domain_height")
        )
        assert np.allclose(np.linalg.norm(vectors, axis=1), 2.0)


def test_planar_cut_netcdf_retains_physical_edgequad_area(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    parent = generate_grid(TorusGridSpec(nx=6, ny=6, edge_length=2.0))
    grid = cut_grid(
        parent,
        Region.lonlat_box(
            lon_min=-120.0,
            lon_max=0.0,
            lat_min=-90.0,
            lat_max=90.0,
        ),
    )

    path = grid.to_netcdf(tmp_path / "planar-cut.nc")

    with netcdf4.Dataset(path) as dataset:
        assert np.allclose(
            dataset.variables["edgequad_area"][:],
            grid.geometry["edgequad_area"],
        )
        assert dataset.variables["edgequad_area"].getncattr("units") == "m2"
        assert np.any(dataset.variables["neighbor_cell_index"][:] == -1)
        assert np.any(dataset.variables["adjacent_cell_of_edge"][:] == -1)
        assert np.any(dataset.variables["cells_of_vertex"][:] == -1)
        assert not np.any(dataset.variables["neighbor_cell_index"][:] == 0)


def test_refine_last_limited_area_netcdf_has_icon_boundary_contract(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(
        LimitedAreaGridSpec(
            parent="R02B02",
            region=Region.circle(lon=0.0, lat=0.0, radius_degrees=45.0),
            local_optimization_iterations=0,
        ),
        optimize_global=False,
    )
    path = grid.to_netcdf(tmp_path / "limited.nc", fields="full")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.getncattr("grid_geometry") == 1
        assert dataset.getncattr("open_boundary") == 1
        assert dataset.getncattr("grid_root") == 2
        assert dataset.getncattr("grid_level") == 2
        assert dataset.getncattr("boundary_depth_index") == 14
        assert dataset.getncattr("construction_parent_grid_name") == "R02B01"
        assert dataset.getncattr("uuidOfParHGrid") == grid.metadata["uuidOfParHGrid"]
        assert np.any(dataset.variables["neighbor_cell_index"][:] == -1)
        assert np.any(dataset.variables["adjacent_cell_of_edge"][:] == -1)
        assert np.array_equal(
            dataset.variables["refin_c_ctrl"][:],
            grid.refinement["refin_c_ctrl"],
        )
        assert set(np.unique(dataset.variables["parent_cell_type"][:])) == {
            200,
            201,
            202,
            203,
        }
        assert np.any(dataset.variables["parent_vertex_index"][:] < 0)


def test_limited_area_icon_profile_selects_spherical_reconstruction(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(
        LimitedAreaGridSpec(
            parent="R02B01",
            region=Region.circle(lon=0.0, lat=0.0, radius_degrees=60.0),
            local_optimization_iterations=0,
        ),
        optimize_global=False,
    )
    path = grid.to_netcdf(tmp_path / "limited-icon.nc", fields="icon")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.getncattr("grid_geometry") == 1
        assert dataset.getncattr("open_boundary") == 1
        assert "cartesian_x_vertices" not in dataset.variables
        assert np.any(dataset.variables["neighbor_cell_index"][:] == -1)
        assert np.any(dataset.variables["adjacent_cell_of_edge"][:] == -1)


@pytest.mark.parametrize(
    ("spec", "expected_geometry"),
    [
        (StretchedTorusGridSpec(nx=4, ny=4, edge_length=2.0), 2),
        (ChannelGridSpec(nx=3, ny=2, edge_length=2.0), 3),
        (ParallelogramGridSpec(nx=3, ny=2, edge_length=2.0), 4),
        (RaggedOrthogonalGridSpec(nx=3, ny=2, dx=2.0, dy=1.5), 4),
    ],
)
def test_planar_netcdf_uses_icon_geometry_enum(
    spec, expected_geometry, tmp_path
):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(spec)
    path = grid.to_netcdf(tmp_path / f"planar-{expected_geometry}.nc")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.getncattr("grid_geometry") == expected_geometry
        assert "open_boundary" not in dataset.ncattrs()
        assert np.allclose(
            dataset.variables["cartesian_x_vertices"][:],
            grid.vertices[:, 0],
        )


def test_to_netcdf_writes_expected_icon_grid_content(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid("R01B00")
    path = grid.to_netcdf(tmp_path / "nested" / "r01b00.nc")

    assert path == tmp_path / "nested" / "r01b00.nc"
    assert path.exists()

    second_path = grid.to_netcdf(tmp_path / "r01b00-via-method.nc")
    assert second_path.exists()

    with netcdf4.Dataset(path) as dataset:
        assert dataset.dimensions["cell"].size == 20
        assert dataset.dimensions["vertex"].size == 12
        assert dataset.dimensions["edge"].size == 30
        assert dataset.dimensions["nc"].size == 2
        assert dataset.dimensions["nv"].size == 3
        assert dataset.dimensions["ne"].size == 6
        assert dataset.dimensions["no"].size == 4
        assert dataset.dimensions["max_chdom"].size == 1
        assert dataset.dimensions["cell_grf"].size == 14
        assert dataset.dimensions["edge_grf"].size == 24
        assert dataset.dimensions["vert_grf"].size == 13
        assert dataset.getncattr("title") == "Pure Python ICON grid R01B00"
        assert (
            dataset.getncattr("source") == "grid_generator Python ICON grid generator"
        )
        assert dataset.getncattr("uuidOfHGrid") == grid.metadata["uuidOfHGrid"]
        assert dataset.getncattr("grid_root") == 1
        assert dataset.getncattr("grid_level") == 0
        assert dataset.getncattr("sphere_radius") == grid.options.sphere_radius
        assert dataset.getncattr("grid_ID") == 1
        assert dataset.getncattr("parent_grid_ID") == 0
        assert dataset.getncattr("no_of_subgrids") == 1
        assert dataset.getncattr("start_subgrid_id") == 0
        assert dataset.getncattr("max_childdom") == 1
        assert dataset.getncattr("boundary_depth_index") == 0
        assert np.array_equal(dataset.getncattr("rotation_vector"), np.zeros(3))
        assert np.array_equal(dataset.getncattr("domain_cartesian_center"), np.zeros(3))
        assert dataset.getncattr("domain_length") == pytest.approx(
            2.0 * np.pi * grid.options.sphere_radius
        )
        assert dataset.getncattr("domain_height") == pytest.approx(
            2.0 * np.pi * grid.options.sphere_radius
        )
        for attr in ("revision", "history", "date", "user_name", "os_name"):
            assert attr in dataset.ncattrs()
        assert dataset.variables["clon"].dimensions == ("cell",)
        assert dataset.variables["edge_of_cell"].dimensions == ("nv", "cell")
        assert dataset.variables["adjacent_cell_of_edge"].dimensions == ("nc", "edge")
        assert dataset.variables["cells_of_vertex"].dimensions == ("ne", "vertex")
        assert dataset.variables["child_edge_index"].dimensions == ("no", "edge")
        assert dataset.variables["elon_vertices"].dimensions == ("edge", "no")
        assert dataset.variables["elat_vertices"].dimensions == ("edge", "no")
        assert {
            "clon",
            "clat",
            "vlon",
            "vlat",
            "elon",
            "elat",
            "elon_vertices",
            "elat_vertices",
            "edge_of_cell",
            "vertex_of_cell",
            "neighbor_cell_index",
            "adjacent_cell_of_edge",
            "edge_vertices",
            "cells_of_vertex",
            "edges_of_vertex",
            "vertices_of_vertex",
            "cell_area",
            "dual_area",
            "edge_length",
            "dual_edge_length",
            "edge_cell_distance",
            "edge_vert_distance",
            "edgequad_area",
            "orientation_of_normal",
            "edge_system_orientation",
            "edge_orientation",
            "cell_circumcenter_cartesian_x",
            "cell_circumcenter_cartesian_y",
            "cell_circumcenter_cartesian_z",
            "edge_middle_cartesian_x",
            "edge_middle_cartesian_y",
            "edge_middle_cartesian_z",
            "edge_primal_normal_cartesian_x",
            "edge_primal_normal_cartesian_y",
            "edge_primal_normal_cartesian_z",
            "edge_dual_normal_cartesian_x",
            "edge_dual_normal_cartesian_y",
            "edge_dual_normal_cartesian_z",
            "zonal_normal_primal_edge",
            "meridional_normal_primal_edge",
            "zonal_normal_dual_edge",
            "meridional_normal_dual_edge",
        } <= set(dataset.variables)
        assert np.allclose(dataset.variables["clon"][:], np.radians(grid.lon))
        assert np.allclose(dataset.variables["vlon"][:], np.radians(grid.vertex_lon))
        assert np.allclose(dataset.variables["elon"][:], np.radians(grid.edge_lon))
        assert dataset.variables["elon_vertices"].shape == (30, 4)
        assert dataset.variables["elat_vertices"].shape == (30, 4)
        assert np.all(np.isfinite(dataset.variables["elon_vertices"][:]))
        assert np.all(np.isfinite(dataset.variables["elat_vertices"][:]))
        for variable_name, expected_attrs in _netcdf.ICON_VARIABLE_ATTRS.items():
            variable = dataset.variables[variable_name]
            assert set(expected_attrs) <= set(variable.ncattrs())
            for attr_name, attr_value in expected_attrs.items():
                assert variable.getncattr(attr_name) == attr_value
        assert np.array_equal(
            dataset.variables["edge_of_cell"][:], grid.cell_edges.T + 1
        )
        assert np.array_equal(dataset.variables["vertex_of_cell"][:], grid.cells.T + 1)
        assert np.array_equal(
            dataset.variables["adjacent_cell_of_edge"][:], grid.edge_cells.T + 1
        )
        assert np.array_equal(dataset.variables["edge_vertices"][:], grid.edges.T + 1)
        assert np.allclose(
            dataset.variables["cell_area"][:], grid.geometry["cell_area"]
        )
        assert np.allclose(
            dataset.variables["dual_area"][:], grid.geometry["dual_area"]
        )
        assert np.allclose(
            dataset.variables["edge_length"][:], grid.geometry["edge_length"]
        )
        assert np.allclose(
            dataset.variables["dual_edge_length"][:], grid.geometry["dual_edge_length"]
        )
        assert np.allclose(
            dataset.variables["edge_cell_distance"][:],
            grid.geometry["edge_cell_distance"].T,
        )
        assert np.allclose(
            dataset.variables["edge_vert_distance"][:],
            grid.geometry["edge_vert_distance"].T,
        )
        assert np.allclose(
            dataset.variables["edgequad_area"][:],
            grid.geometry["edgequad_area"] / grid.options.sphere_radius**2,
        )
        assert dataset.variables["edgequad_area"].getncattr("units") == "1"
        assert "sphere_radius squared" in dataset.variables["edgequad_area"].getncattr(
            "normalization"
        )
        assert np.array_equal(
            dataset.variables["orientation_of_normal"][:],
            grid.geometry["orientation_of_normal"].T,
        )
        assert np.array_equal(
            dataset.variables["edge_system_orientation"][:],
            grid.geometry["edge_system_orientation"],
        )
        assert np.array_equal(
            dataset.variables["edge_orientation"][:],
            grid.geometry["edge_orientation"].T,
        )
        unit_centers = unit_rows(grid.cell_center_xyz)
        unit_edge_centers = unit_rows(grid.edge_center_xyz)
        assert np.allclose(
            dataset.variables["cell_circumcenter_cartesian_x"][:], unit_centers[:, 0]
        )
        assert np.allclose(
            dataset.variables["cell_circumcenter_cartesian_y"][:], unit_centers[:, 1]
        )
        assert np.allclose(
            dataset.variables["cell_circumcenter_cartesian_z"][:], unit_centers[:, 2]
        )
        assert np.allclose(
            dataset.variables["edge_middle_cartesian_x"][:], unit_edge_centers[:, 0]
        )
        assert np.allclose(
            dataset.variables["edge_middle_cartesian_y"][:], unit_edge_centers[:, 1]
        )
        assert np.allclose(
            dataset.variables["edge_middle_cartesian_z"][:], unit_edge_centers[:, 2]
        )
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_x"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 0],
        )
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_y"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 1],
        )
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_z"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 2],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_x"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 0],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_y"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 1],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_z"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 2],
        )
        assert np.allclose(
            dataset.variables["zonal_normal_primal_edge"][:],
            grid.geometry["zonal_normal_primal_edge"],
        )
        assert np.allclose(
            dataset.variables["meridional_normal_primal_edge"][:],
            grid.geometry["meridional_normal_primal_edge"],
        )
        assert np.allclose(
            dataset.variables["zonal_normal_dual_edge"][:],
            grid.geometry["zonal_normal_dual_edge"],
        )
        assert np.allclose(
            dataset.variables["meridional_normal_dual_edge"][:],
            grid.geometry["meridional_normal_dual_edge"],
        )
        assert np.array_equal(dataset.variables["refin_c_ctrl"][:], np.full(20, -4))
        assert np.array_equal(dataset.variables["refin_e_ctrl"][:], np.full(30, -8))
        assert np.array_equal(
            dataset.variables["refin_v_ctrl"][:], np.zeros(12, dtype=np.int32)
        )
        for name, values in grid.refinement.items():
            assert np.array_equal(dataset.variables[name][:], values)
        assert np.array_equal(
            dataset.variables["start_idx_c"][:],
            np.array([[21] * 4 + [1] * 10], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_c"][:],
            np.array([[20] * 5 + [0] * 9], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["start_idx_e"][:],
            np.array([[31] * 5 + [1] * 19], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_e"][:],
            np.array([[30] * 6 + [0] * 18], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["start_idx_v"][:],
            np.array([[13] * 7 + [1] * 6], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_v"][:],
            np.array([[12] * 8 + [0] * 5], dtype=np.int32),
        )


def test_to_netcdf_rejects_radius_mismatch(tmp_path):
    grid = generate_grid("R01B00", options={"sphere_radius": 2.0})
    with pytest.raises(ValueError, match="sphere_radius must match"):
        grid.to_netcdf(tmp_path / "wrong-radius.nc", sphere_radius=3.0)


def test_to_netcdf_reports_missing_netcdf4(monkeypatch, tmp_path):
    grid = generate_grid("R01B00")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "netCDF4":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="NetCDF export requires"):
        grid.to_netcdf(tmp_path / "grid.nc")


def test_netcdf_writer_releases_field_before_requesting_next(monkeypatch, tmp_path):
    grid = generate_grid("R01B00")

    class TrackingFields:
        def __init__(self):
            self.index = 0
            self.previous = None

        def __iter__(self):
            return self

        def __next__(self):
            if self.previous is not None:
                assert self.previous() is None
            if self.index == 2:
                raise StopIteration
            data = np.full(grid.dims["cell"], self.index, dtype=np.float64)
            self.previous = weakref.ref(data)
            self.index += 1
            return f"field_{self.index}", ("cell",), data, {}

    class Variable:
        def __setitem__(self, key, value):
            assert key == slice(None)
            assert value.shape == (grid.dims["cell"],)

        def setncattr(self, name, value):
            pass

    class Dataset:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def createDimension(self, name, size):
            pass

        def setncattr(self, name, value):
            pass

        def createVariable(self, name, dtype, dims):
            return Variable()

    monkeypatch.setitem(sys.modules, "netCDF4", types.SimpleNamespace(Dataset=Dataset))
    monkeypatch.setattr(
        _netcdf,
        "_icon_fields",
        lambda unused_grid, unused_fields: TrackingFields(),
    )

    _netcdf.write_icon_grid(grid, tmp_path / "lifetime.nc")
