"""ICON NetCDF field assembly and writing."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import getpass
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np

from ._grid_semantics import is_icon_open_boundary, is_planar_geometry

IconNetcdfFieldSelection = str | Iterable[str]
DEFAULT_NETCDF_CHUNK_SIZE = 1_000_000

ICON_NETCDF_FIELD_NAMES = (
    "clon",
    "clat",
    "clon_vertices",
    "clat_vertices",
    "vlon",
    "vlat",
    "elon",
    "elat",
    "elon_vertices",
    "elat_vertices",
    "lon_cell_centre",
    "lat_cell_centre",
    "longitude_vertices",
    "latitude_vertices",
    "lon_edge_centre",
    "lat_edge_centre",
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
    "cell_area_p",
    "dual_area_p",
    "edge_length",
    "dual_edge_length",
    "edge_cell_distance",
    "edge_vert_distance",
    "edgequad_area",
    "orientation_of_normal",
    "edge_system_orientation",
    "edge_orientation",
    "refin_c_ctrl",
    "refin_e_ctrl",
    "refin_v_ctrl",
    "start_idx_c",
    "end_idx_c",
    "start_idx_e",
    "end_idx_e",
    "start_idx_v",
    "end_idx_v",
    "cell_elevation",
    "edge_elevation",
    "cell_sea_land_mask",
    "edge_sea_land_mask",
    "cartesian_x_vertices",
    "cartesian_y_vertices",
    "cartesian_z_vertices",
    "cell_circumcenter_cartesian_x",
    "cell_circumcenter_cartesian_y",
    "cell_circumcenter_cartesian_z",
    "edge_middle_cartesian_x",
    "edge_middle_cartesian_y",
    "edge_middle_cartesian_z",
    "phys_cell_id",
    "phys_edge_id",
    "cell_index",
    "edge_index",
    "vertex_index",
    "edge_dual_middle_cartesian_x",
    "edge_dual_middle_cartesian_y",
    "edge_dual_middle_cartesian_z",
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
    "parent_cell_index",
    "parent_cell_type",
    "edge_parent_type",
    "parent_edge_index",
    "parent_vertex_index",
    "child_cell_index",
    "child_cell_id",
    "child_edge_index",
    "child_edge_id",
    "quadrilateral_area",
    "vlon_vertices",
    "vlat_vertices",
)

_ICON4PY_FIELDS = frozenset(
    {
        "clon",
        "clat",
        "vlon",
        "vlat",
        "elon",
        "elat",
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
        "orientation_of_normal",
        "edge_system_orientation",
        "edge_orientation",
        "refin_c_ctrl",
        "refin_e_ctrl",
        "refin_v_ctrl",
    }
)
_ICON_FIELDS = frozenset(
    {
        "lon_cell_centre",
        "lat_cell_centre",
        "longitude_vertices",
        "latitude_vertices",
        "lon_edge_centre",
        "lat_edge_centre",
        "edge_of_cell",
        "vertex_of_cell",
        "neighbor_cell_index",
        "adjacent_cell_of_edge",
        "edge_vertices",
        "cells_of_vertex",
        "edges_of_vertex",
        "vertices_of_vertex",
        "cell_area_p",
        "dual_area_p",
        "edge_length",
        "dual_edge_length",
        "edge_cell_distance",
        "edge_vert_distance",
        "orientation_of_normal",
        "edge_system_orientation",
        "edge_orientation",
        "zonal_normal_primal_edge",
        "meridional_normal_primal_edge",
        "zonal_normal_dual_edge",
        "meridional_normal_dual_edge",
        "refin_c_ctrl",
        "refin_e_ctrl",
        "refin_v_ctrl",
        "start_idx_c",
        "end_idx_c",
        "start_idx_e",
        "end_idx_e",
        "start_idx_v",
        "end_idx_v",
        "parent_cell_index",
        "parent_edge_index",
    }
)
ICON_NETCDF_FIELD_PROFILES = {
    "full": frozenset(ICON_NETCDF_FIELD_NAMES),
    "reduced": _ICON_FIELDS | _ICON4PY_FIELDS,
    "icon": _ICON_FIELDS,
    "icon4py": _ICON4PY_FIELDS,
}


def resolve_icon_netcdf_fields(
    fields: IconNetcdfFieldSelection = "full",
) -> frozenset[str]:
    """Resolve a named profile or exact field collection before generation."""
    if isinstance(fields, str):
        if fields not in ICON_NETCDF_FIELD_PROFILES:
            choices = ", ".join(sorted(ICON_NETCDF_FIELD_PROFILES))
            raise ValueError(
                f"unknown NetCDF field profile {fields!r}; choose {choices}"
            )
        return ICON_NETCDF_FIELD_PROFILES[fields]
    try:
        selected = frozenset(fields)
    except TypeError as exc:
        raise TypeError(
            "fields must be a profile name or an iterable of field names"
        ) from exc
    if any(not isinstance(name, str) for name in selected):
        raise TypeError("every NetCDF field name must be a string")
    unknown = selected.difference(ICON_NETCDF_FIELD_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown NetCDF field name(s): {names}")
    return selected


CELL_COORD_ATTRS = {
    "coordinates": "clon clat",
    "grid_type": "unstructured",
    "number_of_grid_in_reference": 1,
}
EDGE_COORD_ATTRS = {"coordinates": "elon elat"}
VERTEX_COORD_ATTRS = {"coordinates": "vlon vlat"}
ICON_VARIABLE_ATTRS: dict[str, dict[str, Any]] = {
    "clon": {
        "bounds": "clon_vertices",
        "long_name": "center longitude",
        "standard_name": "grid_longitude",
    },
    "clat": {
        "bounds": "clat_vertices",
        "long_name": "center latitude",
        "standard_name": "grid_latitude",
    },
    "vlon": {
        "bounds": "vlon_vertices",
        "long_name": "vertex longitude",
        "standard_name": "grid_longitude",
    },
    "vlat": {
        "bounds": "vlat_vertices",
        "long_name": "vertex latitude",
        "standard_name": "grid_latitude",
    },
    "elon": {
        "bounds": "elon_vertices",
        "long_name": "edge midpoint longitude",
        "standard_name": "grid_longitude",
    },
    "elat": {
        "bounds": "elat_vertices",
        "long_name": "edge midpoint latitude",
        "standard_name": "grid_latitude",
    },
    "lon_cell_centre": {**CELL_COORD_ATTRS, "long_name": "longitude of cell centre"},
    "lat_cell_centre": {**CELL_COORD_ATTRS, "long_name": "latitude of cell centre"},
    "longitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "longitude of vertices"},
    "latitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "latitude of vertices"},
    "lon_edge_centre": {
        **EDGE_COORD_ATTRS,
        "long_name": "longitudes of edge midpoints",
    },
    "lat_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "latitudes of edge midpoints"},
    "edge_of_cell": {"long_name": "edges of each cell"},
    "vertex_of_cell": {"long_name": "vertices of each cell"},
    "neighbor_cell_index": {"long_name": "cell neighbor index"},
    "adjacent_cell_of_edge": {"long_name": "cells adjacent to each edge"},
    "edge_vertices": {"long_name": "vertices at the end of each edge"},
    "cells_of_vertex": {"long_name": "cells around each vertex"},
    "edges_of_vertex": {"long_name": "edges around each vertex"},
    "vertices_of_vertex": {"long_name": "vertices around each vertex"},
    "cell_area": {
        **CELL_COORD_ATTRS,
        "long_name": "area of grid cell",
        "standard_name": "area",
    },
    "dual_area": {
        **VERTEX_COORD_ATTRS,
        "long_name": "areas of dual hexagonal/pentagonal cells",
        "standard_name": "area",
    },
    "cell_area_p": {**CELL_COORD_ATTRS, "long_name": "area of grid cell"},
    "dual_area_p": {"long_name": "areas of dual hexagonal/pentagonal cells"},
    "edge_length": {
        **EDGE_COORD_ATTRS,
        "long_name": "lengths of edges of triangular cells",
    },
    "dual_edge_length": {
        **EDGE_COORD_ATTRS,
        "long_name": "lengths of dual edges (distances between triangular cell circumcenters)",
    },
    "edge_cell_distance": {
        "long_name": "distances between edge midpoint and adjacent triangle midpoints",
    },
    "edge_vert_distance": {
        "long_name": "distances between edge midpoint and vertices of that edge",
    },
    "edgequad_area": {
        **EDGE_COORD_ATTRS,
        "long_name": "area around the edge formed by the two adjacent triangles",
    },
    "quadrilateral_area": {
        **EDGE_COORD_ATTRS,
        "long_name": "legacy quadrilateral area placeholder",
    },
    "vlon_vertices": {"long_name": "vertex-neighbor longitude bounds"},
    "vlat_vertices": {"long_name": "vertex-neighbor latitude bounds"},
    "orientation_of_normal": {
        "long_name": "orientations of normals to triangular cell edges"
    },
    "edge_system_orientation": {
        **EDGE_COORD_ATTRS,
        "long_name": "edge system orientation",
    },
    "edge_orientation": {"long_name": "edge orientation"},
    "refin_c_ctrl": {"long_name": "refinement control flag for cells"},
    "refin_e_ctrl": {"long_name": "refinement control flag for edges"},
    "refin_v_ctrl": {"long_name": "refinement control flag for vertices"},
    "start_idx_c": {
        "long_name": "list of start indices for each refinement control level for cells"
    },
    "end_idx_c": {
        "long_name": "list of end indices for each refinement control level for cells"
    },
    "start_idx_e": {
        "long_name": "list of start indices for each refinement control level for edges"
    },
    "end_idx_e": {
        "long_name": "list of end indices for each refinement control level for edges"
    },
    "start_idx_v": {
        "long_name": "list of start indices for each refinement control level for vertices"
    },
    "end_idx_v": {
        "long_name": "list of end indices for each refinement control level for vertices"
    },
    "cell_elevation": {
        **CELL_COORD_ATTRS,
        "long_name": "elevation at the cell centers",
    },
    "edge_elevation": {
        **EDGE_COORD_ATTRS,
        "long_name": "elevation at the edge centers",
    },
    "cell_sea_land_mask": {
        **CELL_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "edge_sea_land_mask": {
        **EDGE_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "cartesian_x_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate x on unit sp",
    },
    "cartesian_y_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate y on unit sp",
    },
    "cartesian_z_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate z on unit sp",
    },
    "cell_circumcenter_cartesian_x": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate x",
    },
    "cell_circumcenter_cartesian_y": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate y",
    },
    "cell_circumcenter_cartesian_z": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate z",
    },
    "edge_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate x on unit sphere",
    },
    "edge_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate y on unit sphere",
    },
    "edge_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate z on unit sphere",
    },
    "phys_cell_id": {**CELL_COORD_ATTRS, "long_name": "physical domain ID of cell"},
    "phys_edge_id": {**EDGE_COORD_ATTRS, "long_name": "physical domain ID of edge"},
    "cell_index": {"long_name": "cell index"},
    "edge_index": {"long_name": "edge index"},
    "vertex_index": {"long_name": "vertices index"},
    "edge_dual_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate x on unit sphere",
    },
    "edge_dual_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate y on unit sphere",
    },
    "edge_dual_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate z on unit sphere",
    },
    "edge_primal_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate x",
    },
    "edge_primal_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate y",
    },
    "edge_primal_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate z",
    },
    "edge_dual_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate x",
    },
    "edge_dual_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate y",
    },
    "edge_dual_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate z",
    },
    "zonal_normal_primal_edge": {
        "long_name": "zonal component of normal to primal edge"
    },
    "meridional_normal_primal_edge": {
        "long_name": "meridional component of normal to primal edge",
    },
    "zonal_normal_dual_edge": {"long_name": "zonal component of normal to dual edge"},
    "meridional_normal_dual_edge": {
        "long_name": "meridional component of normal to dual edge",
    },
    "parent_cell_index": {**CELL_COORD_ATTRS, "long_name": "parent cell index"},
    "parent_cell_type": {"long_name": "parent cell type"},
    "edge_parent_type": {"long_name": "edge parent type"},
    "parent_edge_index": {"long_name": "parent edge index"},
    "parent_vertex_index": {"long_name": "parent vertex index"},
    "child_cell_index": {"long_name": "child cell index"},
    "child_cell_id": {"long_name": "domain ID of child cell"},
    "child_edge_index": {"long_name": "child edge index"},
    "child_edge_id": {"long_name": "domain ID of child edge"},
}


def _icon_variable_definitions() -> list[
    tuple[str, tuple[str, ...], np.dtype[Any], dict[str, Any]]
]:
    """Return the canonical ordered schema shared by both NetCDF writers."""
    float64 = np.dtype(np.float64)
    int32 = np.dtype(np.int32)
    definitions: list[tuple[str, tuple[str, ...], np.dtype[Any], dict[str, Any]]] = []

    def add(
        name: str,
        dims: tuple[str, ...],
        dtype: np.dtype[Any],
        attrs: dict[str, Any] | None = None,
    ) -> None:
        definitions.append((name, dims, dtype, attrs or {}))

    radians = {"units": "radian"}
    for name, dims in (
        ("clon", ("cell",)),
        ("clat", ("cell",)),
        ("clon_vertices", ("cell", "nv")),
        ("clat_vertices", ("cell", "nv")),
        ("vlon", ("vertex",)),
        ("vlat", ("vertex",)),
        ("elon", ("edge",)),
        ("elat", ("edge",)),
        ("elon_vertices", ("edge", "no")),
        ("elat_vertices", ("edge", "no")),
        ("lon_cell_centre", ("cell",)),
        ("lat_cell_centre", ("cell",)),
        ("longitude_vertices", ("vertex",)),
        ("latitude_vertices", ("vertex",)),
        ("lon_edge_centre", ("edge",)),
        ("lat_edge_centre", ("edge",)),
    ):
        add(name, dims, float64, radians)

    for name, dims in (
        ("edge_of_cell", ("nv", "cell")),
        ("vertex_of_cell", ("nv", "cell")),
        ("neighbor_cell_index", ("nv", "cell")),
        ("adjacent_cell_of_edge", ("nc", "edge")),
        ("edge_vertices", ("nc", "edge")),
        ("cells_of_vertex", ("ne", "vertex")),
        ("edges_of_vertex", ("ne", "vertex")),
        ("vertices_of_vertex", ("ne", "vertex")),
    ):
        add(name, dims, int32)

    add("cell_area", ("cell",), float64, {"units": "m2"})
    add("dual_area", ("vertex",), float64, {"units": "m2"})
    add("cell_area_p", ("cell",), float64, {"units": "m2"})
    add("dual_area_p", ("vertex",), float64, {"units": "m2"})
    add("edge_length", ("edge",), float64, {"units": "m"})
    add("dual_edge_length", ("edge",), float64, {"units": "m"})
    add("edge_cell_distance", ("nc", "edge"), float64, {"units": "m"})
    add("edge_vert_distance", ("nc", "edge"), float64, {"units": "m"})
    add(
        "edgequad_area",
        ("edge",),
        float64,
        {
            "units": "1",
            "normalization": "physical edge quadrilateral area divided by sphere_radius squared",
        },
    )
    add("orientation_of_normal", ("nv", "cell"), int32)
    add("edge_system_orientation", ("edge",), int32)
    add("edge_orientation", ("ne", "vertex"), int32)

    for name, dims in (
        ("refin_c_ctrl", ("cell",)),
        ("refin_e_ctrl", ("edge",)),
        ("refin_v_ctrl", ("vertex",)),
        ("start_idx_c", ("max_chdom", "cell_grf")),
        ("end_idx_c", ("max_chdom", "cell_grf")),
        ("start_idx_e", ("max_chdom", "edge_grf")),
        ("end_idx_e", ("max_chdom", "edge_grf")),
        ("start_idx_v", ("max_chdom", "vert_grf")),
        ("end_idx_v", ("max_chdom", "vert_grf")),
    ):
        add(name, dims, int32)

    add("cell_elevation", ("cell",), float64, {"units": "m"})
    add("edge_elevation", ("edge",), float64, {"units": "m"})
    add("cell_sea_land_mask", ("cell",), int32)
    add("edge_sea_land_mask", ("edge",), int32)

    meters = {"units": "meters"}
    for name, dims, dtype, attrs in (
        ("cartesian_x_vertices", ("vertex",), float64, meters),
        ("cartesian_y_vertices", ("vertex",), float64, meters),
        ("cartesian_z_vertices", ("vertex",), float64, meters),
        ("cell_circumcenter_cartesian_x", ("cell",), float64, meters),
        ("cell_circumcenter_cartesian_y", ("cell",), float64, meters),
        ("cell_circumcenter_cartesian_z", ("cell",), float64, meters),
        ("edge_middle_cartesian_x", ("edge",), float64, meters),
        ("edge_middle_cartesian_y", ("edge",), float64, meters),
        ("edge_middle_cartesian_z", ("edge",), float64, meters),
        ("phys_cell_id", ("cell",), int32, {}),
        ("phys_edge_id", ("edge",), int32, {}),
        ("cell_index", ("cell",), int32, {}),
        ("edge_index", ("edge",), int32, {}),
        ("vertex_index", ("vertex",), int32, {}),
        ("edge_dual_middle_cartesian_x", ("edge",), float64, meters),
        ("edge_dual_middle_cartesian_y", ("edge",), float64, meters),
        ("edge_dual_middle_cartesian_z", ("edge",), float64, meters),
    ):
        add(name, dims, dtype, attrs)

    for name in (
        "edge_primal_normal_cartesian_x",
        "edge_primal_normal_cartesian_y",
        "edge_primal_normal_cartesian_z",
        "edge_dual_normal_cartesian_x",
        "edge_dual_normal_cartesian_y",
        "edge_dual_normal_cartesian_z",
    ):
        add(name, ("edge",), float64, meters)
    for name in (
        "zonal_normal_primal_edge",
        "meridional_normal_primal_edge",
        "zonal_normal_dual_edge",
        "meridional_normal_dual_edge",
    ):
        add(name, ("edge",), float64, radians)

    for name, dims in (
        ("parent_cell_index", ("cell",)),
        ("parent_cell_type", ("cell",)),
        ("edge_parent_type", ("edge",)),
        ("parent_edge_index", ("edge",)),
        ("parent_vertex_index", ("vertex",)),
        ("child_cell_index", ("no", "cell")),
        ("child_cell_id", ("cell",)),
        ("child_edge_index", ("no", "edge")),
        ("child_edge_id", ("edge",)),
    ):
        add(name, dims, int32)

    add("quadrilateral_area", ("edge",), float64)
    add("vlon_vertices", ("vertex", "ne"), float64, radians)
    add("vlat_vertices", ("vertex", "ne"), float64, radians)

    return definitions


def write_icon_grid(
    grid: Any,
    path: str | Path,
    *,
    sphere_radius: float | None = None,
    fields: IconNetcdfFieldSelection = "full",
    chunk_size: int | None = None,
) -> Path:
    """Write selected fields from an ICON-style NetCDF grid schema atomically."""
    selected_fields = resolve_icon_netcdf_fields(fields)
    _require_complete_icon_grid(grid)
    if sphere_radius is None:
        sphere_radius = grid.options.sphere_radius
    if not np.isclose(sphere_radius, grid.options.sphere_radius):
        raise ValueError(
            "sphere_radius must match the value used by generate_grid(); "
            "pass options={'sphere_radius': ...} when generating the grid"
        )

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size is not None and (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        partial.unlink()

    with nc.Dataset(partial, "w", format="NETCDF4") as dataset:
        _write_icon_dimensions(dataset, grid)
        _write_icon_attributes(dataset, grid, path)
        _write_icon_fields_chunked(
            dataset,
            grid,
            selected_fields,
            DEFAULT_NETCDF_CHUNK_SIZE if chunk_size is None else chunk_size,
        )
    # Lightweight test doubles may not create a physical file. Real netCDF4
    # datasets always do, and are published through one atomic replacement.
    if partial.exists():
        os.replace(partial, path)

    return path


def _write_icon_fields_chunked(
    dataset: Any,
    grid: Any,
    selected_fields: frozenset[str],
    chunk_size: int,
) -> None:
    """Create and fill established fields without full export-only arrays."""
    for name, dimensions, dtype, base_attrs in _icon_variable_definitions():
        if name not in selected_fields:
            continue
        attrs = dict(base_attrs)
        if name == "edgequad_area" and is_planar_geometry(grid.metadata):
            attrs = {"units": "m2"}
        attrs = _with_icon_variable_attrs(name, attrs, selected_fields)
        variable = dataset.createVariable(name, dtype, dimensions)
        for attr_name, attr_value in attrs.items():
            variable.setncattr(attr_name, attr_value)

        entity_axis = next(
            (
                axis
                for axis, dimension in enumerate(dimensions)
                if dimension in {"cell", "edge", "vertex"}
            ),
            None,
        )
        if entity_axis is None:
            variable[:] = _fixed_icon_field(grid, name)
            continue
        dimension = dimensions[entity_axis]
        size = grid.dims[dimension]
        for start in range(0, size, chunk_size):
            section = slice(start, min(start + chunk_size, size))
            target = [slice(None)] * len(dimensions)
            target[entity_axis] = section
            variable[tuple(target)] = _icon_field_chunk(grid, name, section)


def _fixed_icon_field(grid: Any, name: str) -> np.ndarray:
    if name in grid.refinement:
        return np.asarray(grid.refinement[name])
    raise RuntimeError(f"chunked NetCDF writer has no fixed field builder for {name}")


def _icon_field_chunk(grid: Any, name: str, section: slice) -> np.ndarray:
    """Build one entity slice for the canonical ICON schema."""
    for builder in (
        _coordinate_field_chunk,
        _connectivity_field_chunk,
        _metric_field_chunk,
        _surface_refinement_field_chunk,
        _cartesian_field_chunk,
        _normal_hierarchy_field_chunk,
    ):
        values = builder(grid, name, section)
        if values is not None:
            return values
    raise RuntimeError(f"chunked NetCDF writer has no field builder for {name}")


def _coordinate_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    coordinate_source = {
        "clon": grid.lon,
        "clat": grid.lat,
        "lon_cell_centre": grid.lon,
        "lat_cell_centre": grid.lat,
        "vlon": grid.vertex_lon,
        "vlat": grid.vertex_lat,
        "longitude_vertices": grid.vertex_lon,
        "latitude_vertices": grid.vertex_lat,
        "elon": grid.edge_lon,
        "elat": grid.edge_lat,
        "lon_edge_centre": grid.edge_lon,
        "lat_edge_centre": grid.edge_lat,
    }
    if name in coordinate_source:
        return np.radians(coordinate_source[name][section])
    if name == "clon_vertices":
        return np.radians(grid.vertex_lon[grid.cells[section]])
    if name == "clat_vertices":
        return np.radians(grid.vertex_lat[grid.cells[section]])
    if name == "elon_vertices":
        return _edge_coordinate_bounds(grid, "lon", section=section)
    if name == "elat_vertices":
        return _edge_coordinate_bounds(grid, "lat", section=section)
    return None


def _connectivity_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    open_grid = is_icon_open_boundary(grid.metadata)
    if name == "edge_of_cell":
        return grid.icon_connectivity["c2e"][section].T + 1
    if name == "vertex_of_cell":
        return grid.cells[section].T + 1
    if name == "neighbor_cell_index":
        values = grid.icon_connectivity["c2c"][section]
        return (np.where(values < 0, -1, values + 1) if open_grid else values + 1).T
    if name == "adjacent_cell_of_edge":
        values = grid.edge_cells[section]
        return (np.where(values < 0, -1, values + 1) if open_grid else values + 1).T
    if name == "edge_vertices":
        return grid.edges[section].T + 1
    connectivity_names = {
        "cells_of_vertex": "v2c",
        "edges_of_vertex": "v2e",
        "vertices_of_vertex": "v2v",
    }
    if name in connectivity_names:
        values = grid.icon_connectivity[connectivity_names[name]][section]
        if open_grid:
            values = np.where(values == 0, -1, values)
        return values.T
    return None


def _metric_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    geometry_names = {
        "cell_area": "cell_area",
        "cell_area_p": "cell_area",
        "dual_area": "dual_area",
        "dual_area_p": "dual_area",
        "edge_length": "edge_length",
        "dual_edge_length": "dual_edge_length",
        "edge_system_orientation": "edge_system_orientation",
    }
    if name in geometry_names:
        return np.asarray(grid.geometry[geometry_names[name]][section])
    if name == "edge_cell_distance":
        return grid.geometry[name][section].T
    if name == "edge_vert_distance":
        return grid.geometry[name][section].T
    if name == "edgequad_area":
        values = grid.geometry[name][section]
        return values if is_planar_geometry(grid.metadata) else values / grid.options.sphere_radius**2
    if name == "orientation_of_normal":
        return grid.geometry[name][section].T
    if name == "edge_orientation":
        return grid.geometry[name][section].T
    return None


def _surface_refinement_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    if name in grid.refinement:
        return np.asarray(grid.refinement[name][section])
    if name in {"cell_elevation", "edge_elevation", "quadrilateral_area"}:
        return np.zeros(section.stop - (section.start or 0), dtype=np.float64)
    if name in {"cell_sea_land_mask", "edge_sea_land_mask"}:
        return np.zeros(section.stop - (section.start or 0), dtype=np.int32)
    return None


def _cartesian_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    cartesian_sources = {
        "cartesian_x_vertices": (grid.vertices, 0),
        "cartesian_y_vertices": (grid.vertices, 1),
        "cartesian_z_vertices": (grid.vertices, 2),
        "cell_circumcenter_cartesian_x": (grid.cell_center_xyz, 0),
        "cell_circumcenter_cartesian_y": (grid.cell_center_xyz, 1),
        "cell_circumcenter_cartesian_z": (grid.cell_center_xyz, 2),
        "edge_middle_cartesian_x": (grid.edge_center_xyz, 0),
        "edge_middle_cartesian_y": (grid.edge_center_xyz, 1),
        "edge_middle_cartesian_z": (grid.edge_center_xyz, 2),
        "edge_dual_middle_cartesian_x": (grid.edge_center_xyz, 0),
        "edge_dual_middle_cartesian_y": (grid.edge_center_xyz, 1),
        "edge_dual_middle_cartesian_z": (grid.edge_center_xyz, 2),
    }
    if name in cartesian_sources:
        source, component = cartesian_sources[name]
        values = np.asarray(source[section])
        if not is_planar_geometry(grid.metadata):
            values = _gg()._normalize_rows(values)
        return values[:, component]
    if name in {"phys_cell_id", "cell_index", "phys_edge_id", "edge_index", "vertex_index"}:
        return np.arange((section.start or 0) + 1, section.stop + 1, dtype=np.int32)
    return None


def _normal_hierarchy_field_chunk(
    grid: Any,
    name: str,
    section: slice,
) -> np.ndarray | None:
    normal_components = {
        "edge_primal_normal_cartesian_x": ("edge_primal_normal_cartesian", 0),
        "edge_primal_normal_cartesian_y": ("edge_primal_normal_cartesian", 1),
        "edge_primal_normal_cartesian_z": ("edge_primal_normal_cartesian", 2),
        "edge_dual_normal_cartesian_x": ("edge_dual_normal_cartesian", 0),
        "edge_dual_normal_cartesian_y": ("edge_dual_normal_cartesian", 1),
        "edge_dual_normal_cartesian_z": ("edge_dual_normal_cartesian", 2),
    }
    if name in normal_components:
        source, component = normal_components[name]
        return grid.geometry[source][section, component]
    if name in {
        "zonal_normal_primal_edge",
        "meridional_normal_primal_edge",
        "zonal_normal_dual_edge",
        "meridional_normal_dual_edge",
    }:
        return grid.geometry[name][section]

    if name in {"child_cell_index", "child_edge_index"}:
        return np.zeros((4, section.stop - (section.start or 0)), dtype=np.int32)
    if name in {"child_cell_id", "child_edge_id"}:
        return np.zeros(section.stop - (section.start or 0), dtype=np.int32)
    if name in {"vlon_vertices", "vlat_vertices"}:
        neighbors = np.asarray(grid.icon_connectivity["v2v"][section])
        valid = neighbors > 0
        indices = np.maximum(neighbors - 1, 0)
        coordinates = grid.vertex_lon if name == "vlon_vertices" else grid.vertex_lat
        return np.where(valid, np.radians(coordinates[indices]), 0.0)
    return None


def _require_complete_icon_grid(grid: Any) -> None:
    for name, fields in {
        "icon_connectivity": grid.icon_connectivity,
        "geometry": grid.geometry,
        "refinement": grid.refinement,
    }.items():
        if not fields:
            raise ValueError(f"ICON NetCDF export requires populated {name}")
    if is_icon_open_boundary(grid.metadata):
        _require_valid_open_icon_grid(grid)


def _require_valid_open_icon_grid(grid: Any) -> None:
    """Reject regional files that ICON would only fail on after initialization."""
    if not np.any(grid.edge_cells[:, 1] < 0):
        raise ValueError("regional ICON grid must contain open boundary edges")
    if grid.metadata.get("boundary_ordering") != "icon":
        raise ValueError(
            "ICON NetCDF export requires boundary ordering 'icon'; "
            "source ordering is intended for in-memory analysis"
        )
    for name in (
        "cell_area",
        "dual_area",
        "edge_length",
        "dual_edge_length",
        "edge_cell_distance",
        "edge_vert_distance",
        "edgequad_area",
    ):
        values = np.asarray(grid.geometry[name])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"regional ICON metric {name} contains non-finite values")
    for name in ("cell_area", "dual_area", "edge_length", "dual_edge_length"):
        if np.any(np.asarray(grid.geometry[name]) <= 0.0):
            raise ValueError(f"regional ICON metric {name} must be positive")
    controls = {
        "c": np.asarray(grid.refinement["refin_c_ctrl"]),
        "e": np.asarray(grid.refinement["refin_e_ctrl"]),
        "v": np.asarray(grid.refinement["refin_v_ctrl"]),
    }
    maxima = {"c": 5, "e": 10, "v": 5}
    for key, values in controls.items():
        order = np.concatenate(
            [np.flatnonzero(values == level) for level in range(1, maxima[key] + 1)]
            + [np.flatnonzero((values == 0) | (values > maxima[key]))]
        )
        if not np.array_equal(order, np.arange(values.size)):
            raise ValueError(f"regional ICON refin_{key}_ctrl is not boundary ordered")
    from ._limited_area import _start_end

    dimensions = {"c": "cell_grf", "e": "edge_grf", "v": "vert_grf"}
    for key, values in controls.items():
        expected_start, expected_end = _start_end(values, dimensions[key])
        if not np.array_equal(grid.refinement[f"start_idx_{key}"], expected_start):
            raise ValueError(f"regional ICON start_idx_{key} is inconsistent")
        if not np.array_equal(grid.refinement[f"end_idx_{key}"], expected_end):
            raise ValueError(f"regional ICON end_idx_{key} is inconsistent")
    expected_shapes = {
        "parent_cell_index": (grid.dims["cell"],),
        "parent_cell_type": (grid.dims["cell"],),
        "parent_edge_index": (grid.dims["edge"],),
        "edge_parent_type": (grid.dims["edge"],),
        "parent_vertex_index": (grid.dims["vertex"],),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(grid.refinement[name]).shape != shape:
            raise ValueError(f"regional ICON hierarchy field {name} has wrong shape")


def _write_icon_dimensions(dataset: Any, grid: Any) -> None:
    dataset.createDimension("cell", grid.dims["cell"])
    dataset.createDimension("vertex", grid.dims["vertex"])
    dataset.createDimension("edge", grid.dims["edge"])
    for name, size in _gg().FIXED_DIMS.items():
        dataset.createDimension(name, size)


def _write_icon_attributes(dataset: Any, grid: Any, path: Path) -> None:
    external_attrs = {
        "revision": "pure-python",
        "history": f"grid.to_netcdf {path}",
        "date": datetime.now().strftime("%Y%m%d at %H%M%S"),
        "user_name": getpass.getuser(),
        "os_name": platform.platform(),
        "grid_ID": 1,
        "parent_grid_ID": 0,
        "no_of_subgrids": 1,
        "start_subgrid_id": 0,
        "max_childdom": 1,
        "boundary_depth_index": 0,
        "rotation_vector": np.zeros(3, dtype=np.float64),
        "domain_length": grid.metadata.get(
            "domain_length",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_height": grid.metadata.get(
            "domain_height",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_cartesian_center": np.zeros(3, dtype=np.float64),
    }
    attrs = {
        "title": f"Pure Python ICON grid {grid.name}",
        "institution": "grid_generator",
        "source": "grid_generator Python ICON grid generator",
        "ICON_grid_file_uri": str(path),
        **external_attrs,
        **grid.metadata,
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _with_icon_variable_attrs(
    name: str,
    attrs: dict[str, Any],
    selected_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    merged = dict(ICON_VARIABLE_ATTRS.get(name, {}))
    merged.update(attrs)
    if selected_fields is not None:
        bounds = merged.get("bounds")
        if isinstance(bounds, str) and bounds not in selected_fields:
            del merged["bounds"]
        coordinates = merged.get("coordinates")
        if isinstance(coordinates, str):
            retained = [
                coordinate
                for coordinate in coordinates.split()
                if coordinate in selected_fields
            ]
            if retained:
                merged["coordinates"] = " ".join(retained)
            else:
                del merged["coordinates"]
    return merged


def _edge_coordinate_bounds(
    grid: Any,
    coordinate: str,
    *,
    section: slice = slice(None),
) -> np.ndarray:
    edge_vertices = np.asarray(grid.edges[section], dtype=np.int32)
    edge_cells = np.asarray(grid.edge_cells[section], dtype=np.int32)
    vertex_values = grid.vertex_lon if coordinate == "lon" else grid.vertex_lat
    cell_values = grid.lon if coordinate == "lon" else grid.lat
    all_edge_values = grid.edge_lon if coordinate == "lon" else grid.edge_lat
    edge_values = all_edge_values[section]
    values = np.empty((edge_vertices.shape[0], 4), dtype=np.float64)
    values[:, 0] = vertex_values[edge_vertices[:, 0]]
    second_cell = edge_cells[:, 1]
    values[:, 1] = np.where(
        second_cell >= 0,
        cell_values[np.maximum(second_cell, 0)],
        edge_values,
    )
    values[:, 2] = vertex_values[edge_vertices[:, 1]]
    first_cell = edge_cells[:, 0]
    values[:, 3] = np.where(
        first_cell >= 0,
        cell_values[np.maximum(first_cell, 0)],
        edge_values,
    )
    if coordinate == "lon":
        pole_mask = np.empty_like(values, dtype=bool)
        pole_mask[:, 0] = np.isclose(
            np.abs(grid.vertex_lat[edge_vertices[:, 0]]),
            90.0,
        )
        pole_mask[:, 1] = np.isclose(
            np.abs(
                np.where(
                    second_cell >= 0,
                    grid.lat[np.maximum(second_cell, 0)],
                    grid.edge_lat[section],
                )
            ),
            90.0,
        )
        pole_mask[:, 2] = np.isclose(
            np.abs(grid.vertex_lat[edge_vertices[:, 1]]),
            90.0,
        )
        pole_mask[:, 3] = np.isclose(
            np.abs(
                np.where(
                    first_cell >= 0,
                    grid.lat[np.maximum(first_cell, 0)],
                    grid.edge_lat[section],
                )
            ),
            90.0,
        )
        pole_edges, pole_slots = np.nonzero(pole_mask)
        values[pole_edges, pole_slots] = edge_values[pole_edges]
    return np.radians(values)


def _gg() -> Any:
    from . import grid_generator as gg

    return gg
