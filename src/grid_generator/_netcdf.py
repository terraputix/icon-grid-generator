"""ICON NetCDF field assembly and writing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
import getpass
from pathlib import Path
import platform
from typing import Any

import numpy as np

IconNetcdfField = tuple[str, tuple[str, ...], Any, dict[str, Any]]
IconNetcdfFieldSelection = str | Iterable[str]

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
            raise ValueError(f"unknown NetCDF field profile {fields!r}; choose {choices}")
        return ICON_NETCDF_FIELD_PROFILES[fields]
    try:
        selected = frozenset(fields)
    except TypeError as exc:
        raise TypeError("fields must be a profile name or an iterable of field names") from exc
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
    "vlon": {"long_name": "vertex longitude", "standard_name": "grid_longitude"},
    "vlat": {"long_name": "vertex latitude", "standard_name": "grid_latitude"},
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
    "lon_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "longitudes of edge midpoints"},
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
    "edge_length": {**EDGE_COORD_ATTRS, "long_name": "lengths of edges of triangular cells"},
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
    "orientation_of_normal": {"long_name": "orientations of normals to triangular cell edges"},
    "edge_system_orientation": {**EDGE_COORD_ATTRS, "long_name": "edge system orientation"},
    "edge_orientation": {"long_name": "edge orientation"},
    "refin_c_ctrl": {"long_name": "refinement control flag for cells"},
    "refin_e_ctrl": {"long_name": "refinement control flag for edges"},
    "refin_v_ctrl": {"long_name": "refinement control flag for vertices"},
    "start_idx_c": {"long_name": "list of start indices for each refinement control level for cells"},
    "end_idx_c": {"long_name": "list of end indices for each refinement control level for cells"},
    "start_idx_e": {"long_name": "list of start indices for each refinement control level for edges"},
    "end_idx_e": {"long_name": "list of end indices for each refinement control level for edges"},
    "start_idx_v": {"long_name": "list of start indices for each refinement control level for vertices"},
    "end_idx_v": {"long_name": "list of end indices for each refinement control level for vertices"},
    "cell_elevation": {**CELL_COORD_ATTRS, "long_name": "elevation at the cell centers"},
    "edge_elevation": {**EDGE_COORD_ATTRS, "long_name": "elevation at the edge centers"},
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
    "zonal_normal_primal_edge": {"long_name": "zonal component of normal to primal edge"},
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



def write_icon_grid(
    grid: Any,
    path: str | Path,
    *,
    sphere_radius: float | None = None,
    fields: IconNetcdfFieldSelection = "full",
) -> Path:
    """Write selected fields from an ICON-style NetCDF grid schema."""
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

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        _write_icon_dimensions(dataset, grid)
        _write_icon_attributes(dataset, grid, path)
        for name, dims, data, attrs in _icon_fields(grid, selected_fields):
            variable = dataset.createVariable(name, np.asarray(data).dtype, dims)
            variable[:] = data
            for attr_name, attr_value in attrs.items():
                variable.setncattr(attr_name, attr_value)
            # Release generated field storage before advancing the iterator;
            # the next field can itself be an edge-by-four bounds array.
            del data

    return path


def _require_complete_icon_grid(grid: Any) -> None:
    for name, fields in {
        "icon_connectivity": grid.icon_connectivity,
        "geometry": grid.geometry,
        "refinement": grid.refinement,
    }.items():
        if not fields:
            raise ValueError(f"ICON NetCDF export requires populated {name}")
    if grid.metadata.get("grid_geometry") == 3:
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


def _icon_fields(
    grid: Any,
    selected_fields: frozenset[str] | None = None,
) -> Iterator[IconNetcdfField]:
    """Yield field groups incrementally to bound export-time memory."""
    if selected_fields is None:
        selected_fields = ICON_NETCDF_FIELD_PROFILES["full"]
    builders = (
        (
            lambda: _coordinate_fields(grid, selected_fields),
            frozenset(ICON_NETCDF_FIELD_NAMES[0:16]),
        ),
        (
            lambda: _connectivity_fields(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[16:24]),
        ),
        (
            lambda: _metric_fields(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[24:36]),
        ),
        (
            lambda: _refinement_fields_for_netcdf(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[36:45]),
        ),
        (
            lambda: _static_surface_fields(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[45:49]),
        ),
        (
            lambda: _cartesian_fields(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[49:66]),
        ),
        (
            lambda: _normal_vector_fields(grid),
            frozenset(ICON_NETCDF_FIELD_NAMES[66:76]),
        ),
        (
            lambda: _hierarchy_fields(grid, selected_fields),
            frozenset(ICON_NETCDF_FIELD_NAMES[76:85]),
        ),
    )
    for build, group in builders:
        if group.isdisjoint(selected_fields):
            continue
        for name, dims, data, attrs in build():
            if name in selected_fields:
                yield name, dims, data, _with_icon_variable_attrs(
                    name,
                    attrs,
                    selected_fields,
                )


def _coordinate_fields(
    grid: Any,
    selected_fields: frozenset[str] | None = None,
) -> Iterator[IconNetcdfField]:
    if selected_fields is None:
        selected_fields = ICON_NETCDF_FIELD_PROFILES["full"]
    attrs = {"units": "radian"}
    yield "clon", ("cell",), np.radians(grid.lon), attrs
    yield "clat", ("cell",), np.radians(grid.lat), attrs
    if "clon_vertices" in selected_fields:
        yield "clon_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lon), attrs
    if "clat_vertices" in selected_fields:
        yield "clat_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lat), attrs
    yield "vlon", ("vertex",), np.radians(grid.vertex_lon), attrs
    yield "vlat", ("vertex",), np.radians(grid.vertex_lat), attrs
    yield "elon", ("edge",), np.radians(grid.edge_lon), attrs
    yield "elat", ("edge",), np.radians(grid.edge_lat), attrs
    if "elon_vertices" in selected_fields:
        yield "elon_vertices", ("edge", "no"), _edge_coordinate_bounds(grid, "lon"), attrs
    if "elat_vertices" in selected_fields:
        yield "elat_vertices", ("edge", "no"), _edge_coordinate_bounds(grid, "lat"), attrs
    yield "lon_cell_centre", ("cell",), np.radians(grid.lon), attrs
    yield "lat_cell_centre", ("cell",), np.radians(grid.lat), attrs
    yield "longitude_vertices", ("vertex",), np.radians(grid.vertex_lon), attrs
    yield "latitude_vertices", ("vertex",), np.radians(grid.vertex_lat), attrs
    yield "lon_edge_centre", ("edge",), np.radians(grid.edge_lon), attrs
    yield "lat_edge_centre", ("edge",), np.radians(grid.edge_lat), attrs


def _connectivity_fields(grid: Any) -> Iterator[IconNetcdfField]:
    connectivity = grid.icon_connectivity
    yield "edge_of_cell", ("nv", "cell"), connectivity["c2e"].T + 1, {}
    yield "vertex_of_cell", ("nv", "cell"), grid.cells.T + 1, {}
    open_grid = grid.metadata.get("grid_geometry") == 3
    neighbor = (
        np.where(connectivity["c2c"] < 0, -1, connectivity["c2c"] + 1)
        if open_grid
        else connectivity["c2c"] + 1
    )
    adjacent = (
        np.where(grid.edge_cells < 0, -1, grid.edge_cells + 1)
        if open_grid
        else grid.edge_cells + 1
    )
    yield "neighbor_cell_index", ("nv", "cell"), neighbor.T, {}
    yield "adjacent_cell_of_edge", ("nc", "edge"), adjacent.T, {}
    yield "edge_vertices", ("nc", "edge"), grid.edges.T + 1, {}
    for name, key in (
        ("cells_of_vertex", "v2c"),
        ("edges_of_vertex", "v2e"),
        ("vertices_of_vertex", "v2v"),
    ):
        values = connectivity[key]
        if open_grid:
            values = np.where(values == 0, -1, values)
        yield name, ("ne", "vertex"), values.T, {}


def _metric_fields(grid: Any) -> list[IconNetcdfField]:
    geometry = grid.geometry
    is_planar = (
        grid.metadata.get("grid_geometry") == 2
        or grid.metadata.get("source_grid_geometry") == 2
    )
    edgequad_normalizer = 1.0 if is_planar else grid.options.sphere_radius**2
    edgequad_attrs = (
        {"units": "m2"}
        if is_planar
        else {
            "units": "1",
            "normalization": "physical edge quadrilateral area divided by sphere_radius squared",
        }
    )
    return [
        ("cell_area", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("cell_area_p", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area_p", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("edge_length", ("edge",), geometry["edge_length"], {"units": "m"}),
        ("dual_edge_length", ("edge",), geometry["dual_edge_length"], {"units": "m"}),
        ("edge_cell_distance", ("nc", "edge"), geometry["edge_cell_distance"].T, {"units": "m"}),
        ("edge_vert_distance", ("nc", "edge"), geometry["edge_vert_distance"].T, {"units": "m"}),
        (
            "edgequad_area",
            ("edge",),
            geometry["edgequad_area"] / edgequad_normalizer,
            edgequad_attrs,
        ),
        ("orientation_of_normal", ("nv", "cell"), geometry["orientation_of_normal"].T, {}),
        ("edge_system_orientation", ("edge",), geometry["edge_system_orientation"], {}),
        ("edge_orientation", ("ne", "vertex"), geometry["edge_orientation"].T, {}),
    ]


def _refinement_fields_for_netcdf(grid: Any) -> list[IconNetcdfField]:
    refinement = grid.refinement
    return [
        ("refin_c_ctrl", ("cell",), refinement["refin_c_ctrl"], {}),
        ("refin_e_ctrl", ("edge",), refinement["refin_e_ctrl"], {}),
        ("refin_v_ctrl", ("vertex",), refinement["refin_v_ctrl"], {}),
        ("start_idx_c", ("max_chdom", "cell_grf"), refinement["start_idx_c"], {}),
        ("end_idx_c", ("max_chdom", "cell_grf"), refinement["end_idx_c"], {}),
        ("start_idx_e", ("max_chdom", "edge_grf"), refinement["start_idx_e"], {}),
        ("end_idx_e", ("max_chdom", "edge_grf"), refinement["end_idx_e"], {}),
        ("start_idx_v", ("max_chdom", "vert_grf"), refinement["start_idx_v"], {}),
        ("end_idx_v", ("max_chdom", "vert_grf"), refinement["end_idx_v"], {}),
    ]


def _static_surface_fields(grid: Any) -> Iterator[IconNetcdfField]:
    yield "cell_elevation", ("cell",), np.zeros(grid.dims["cell"]), {"units": "m"}
    yield "edge_elevation", ("edge",), np.zeros(grid.dims["edge"]), {"units": "m"}
    yield "cell_sea_land_mask", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}
    yield "edge_sea_land_mask", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}


def _cartesian_fields(grid: Any) -> Iterator[IconNetcdfField]:
    if grid.metadata.get("grid_geometry") == 2:
        unit_vertices = grid.vertices
    else:
        unit_vertices = _gg()._normalize_rows(grid.vertices)
    attrs = {"units": "meters"}
    yield "cartesian_x_vertices", ("vertex",), unit_vertices[:, 0], attrs
    yield "cartesian_y_vertices", ("vertex",), unit_vertices[:, 1], attrs
    yield "cartesian_z_vertices", ("vertex",), unit_vertices[:, 2], attrs
    del unit_vertices

    unit_centers = (
        grid.cell_center_xyz
        if grid.metadata.get("grid_geometry") == 2
        else _gg()._normalize_rows(grid.cell_center_xyz)
    )
    yield "cell_circumcenter_cartesian_x", ("cell",), unit_centers[:, 0], attrs
    yield "cell_circumcenter_cartesian_y", ("cell",), unit_centers[:, 1], attrs
    yield "cell_circumcenter_cartesian_z", ("cell",), unit_centers[:, 2], attrs
    del unit_centers

    unit_edge_centers = (
        grid.edge_center_xyz
        if grid.metadata.get("grid_geometry") == 2
        else _gg()._normalize_rows(grid.edge_center_xyz)
    )
    yield "edge_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], attrs
    yield "edge_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], attrs
    yield "edge_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], attrs
    yield "phys_cell_id", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}
    yield "phys_edge_id", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}
    yield "cell_index", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}
    yield "edge_index", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}
    yield "vertex_index", ("vertex",), np.arange(1, grid.dims["vertex"] + 1, dtype=np.int32), {}
    yield "edge_dual_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], attrs
    yield "edge_dual_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], attrs
    yield "edge_dual_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], attrs


def _normal_vector_fields(grid: Any) -> list[IconNetcdfField]:
    geometry = grid.geometry
    return [
        (
            "edge_primal_normal_cartesian_x",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_y",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_z",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_x",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_y",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_z",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        ("zonal_normal_primal_edge", ("edge",), geometry["zonal_normal_primal_edge"], {"units": "radian"}),
        (
            "meridional_normal_primal_edge",
            ("edge",),
            geometry["meridional_normal_primal_edge"],
            {"units": "radian"},
        ),
        ("zonal_normal_dual_edge", ("edge",), geometry["zonal_normal_dual_edge"], {"units": "radian"}),
        (
            "meridional_normal_dual_edge",
            ("edge",),
            geometry["meridional_normal_dual_edge"],
            {"units": "radian"},
        ),
    ]


def _hierarchy_fields(
    grid: Any,
    selected_fields: frozenset[str] | None = None,
) -> Iterator[IconNetcdfField]:
    if selected_fields is None:
        selected_fields = ICON_NETCDF_FIELD_PROFILES["full"]
    refinement = grid.refinement
    for name, dimension in (
        ("parent_cell_index", "cell"),
        ("parent_cell_type", "cell"),
        ("edge_parent_type", "edge"),
        ("parent_edge_index", "edge"),
        ("parent_vertex_index", "vertex"),
    ):
        if name in selected_fields:
            yield name, (dimension,), refinement[name], {}
    if "child_cell_index" in selected_fields:
        yield "child_cell_index", ("no", "cell"), np.zeros(
            (4, grid.dims["cell"]),
            dtype=np.int32,
        ), {}
    if "child_cell_id" in selected_fields:
        yield "child_cell_id", ("cell",), np.zeros(
            grid.dims["cell"],
            dtype=np.int32,
        ), {}
    if "child_edge_index" in selected_fields:
        yield "child_edge_index", ("no", "edge"), np.zeros(
            (4, grid.dims["edge"]),
            dtype=np.int32,
        ), {}
    if "child_edge_id" in selected_fields:
        yield "child_edge_id", ("edge",), np.zeros(
            grid.dims["edge"],
            dtype=np.int32,
        ), {}


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


def _edge_lon_lat_bounds(grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ICON-style four-point edge bounds in radians.

    The upstream grid generator stores bounds for each edge as a quadrilateral:
    first edge vertex, second adjacent cell center, second edge vertex, first
    adjacent cell center.
    """
    return _edge_coordinate_bounds(grid, "lon"), _edge_coordinate_bounds(grid, "lat")


def _edge_coordinate_bounds(grid: Any, coordinate: str) -> np.ndarray:
    edge_vertices = np.asarray(grid.edges, dtype=np.int32)
    edge_cells = np.asarray(grid.edge_cells, dtype=np.int32)
    vertex_values = grid.vertex_lon if coordinate == "lon" else grid.vertex_lat
    cell_values = grid.lon if coordinate == "lon" else grid.lat
    edge_values = grid.edge_lon if coordinate == "lon" else grid.edge_lat
    values = np.empty((grid.dims["edge"], 4), dtype=np.float64)
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
                    grid.edge_lat,
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
                    grid.edge_lat,
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
