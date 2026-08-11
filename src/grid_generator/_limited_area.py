"""Limited-area grids extracted from generated global grids."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from math import cos, radians
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np

from ._grid_semantics import SPHERE_GEOMETRY, is_planar_geometry
from ._types import (
    BisectionProvenance,
    GeometryData,
    MetricsData,
    RefinementData,
    TopologyData,
)


class LimitedAreaExtractor:
    """Extract a compact limited-area grid from a generated global parent."""

    def build(
        self,
        spec: Any,
        options: Any,
    ) -> tuple[GeometryData, TopologyData, MetricsData, RefinementData, Any, str]:
        from . import grid_generator as gg

        construction = spec.construction
        if construction == "refine_last" and spec.parent.bisections > 0:
            generation_spec = gg.GlobalGridSpec(
                root=spec.parent.root,
                bisections=spec.parent.bisections - 1,
            )
        else:
            construction = "cut_final"
            generation_spec = spec.parent
        parent = gg.generate_grid(generation_spec, options=options)
        selected = _selected_cells(parent, spec)
        selected = _clean_selection(parent, selected, spec.selection.cleanup)
        selected = _expand_cells(parent, selected, _buffer_rings(spec))
        if selected.size == 0:
            raise ValueError("limited-area selection does not contain any cells")
        parent_cells = np.asarray(selected, dtype=np.int32)
        immediate_parent_uuid = parent.metadata["uuidOfHGrid"]
        if construction == "refine_last":
            immediate_parent_uuid = _compact_parent_uuid(
                immediate_parent_uuid,
                parent_cells,
            )
        geometry = _compact_geometry(parent, parent_cells)
        topology = _open_topology(parent, geometry, parent_cells, options)
        if construction == "refine_last":
            geometry, topology = _refine_open_grid(geometry, topology, options)
            geometry = _optimize_local_geometry(
                geometry,
                topology,
                options,
                spec.local_optimization_iterations,
            )
        geometry, topology, controls = _apply_boundary_ordering(
            geometry,
            topology,
            spec.boundary,
            options,
        )
        metrics = _limited_metrics(
            parent,
            geometry,
            topology,
            spec.boundary.metric_closure,
            options,
        )
        refinement = _limited_refinement(
            geometry,
            topology,
            controls,
        )
        return geometry, topology, metrics, refinement, parent, immediate_parent_uuid


def _compact_parent_uuid(source_uuid: str, source_cells: np.ndarray) -> str:
    payload = {
        "generator": "grid_generator",
        "kind": "compact_limited_area_parent",
        "source_uuid": source_uuid,
        "source_cell_index_sha256": hashlib.sha256(
            np.asarray(source_cells, dtype="<i4").tobytes()
        ).hexdigest(),
    }
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    )


def cut_existing_grid(parent: Any, spec: Any) -> tuple[GeometryData, TopologyData, MetricsData, RefinementData]:
    """Cut an existing grid using region predicates."""
    selected = _selected_cells_from_regions(parent, spec)
    if spec.mode == "remove":
        retained = np.ones(parent.dims["cell"], dtype=bool)
        retained[selected] = False
        selected = np.flatnonzero(retained).astype(np.int32)
    selected = _clean_selection(parent, selected, spec.selection.cleanup)
    selected = _expand_cells(parent, selected, _buffer_rings(spec))
    if selected.size == 0:
        raise ValueError("cut grid selection does not contain any cells")
    parent_cells = np.asarray(selected, dtype=np.int32)
    geometry = _compact_geometry(parent, parent_cells)
    topology = _open_topology(parent, geometry, parent_cells, parent.options)
    geometry, topology, controls = _apply_boundary_ordering(
        geometry,
        topology,
        spec.boundary,
        parent.options,
    )
    metrics = _limited_metrics(
        parent,
        geometry,
        topology,
        spec.boundary.metric_closure,
        parent.options,
    )
    refinement = _limited_refinement(geometry, topology, controls)
    refinement.fields["smooth_c_ctrl"] = np.full(
        geometry.cells.shape[0],
        spec.smoothing_depth,
        dtype=np.int32,
    )
    return geometry, topology, metrics, refinement


def _selected_cells(parent: Any, spec: Any) -> np.ndarray:
    return _selected_cells_with_policy(parent, (spec.region,), spec.selection.inclusion)


def _selected_cells_from_regions(parent: Any, spec: Any) -> np.ndarray:
    return _selected_cells_with_policy(parent, spec.regions, spec.selection.inclusion)


def _selected_cells_with_policy(
    parent: Any,
    regions: tuple[Any, ...],
    inclusion: str,
) -> np.ndarray:
    mask = np.zeros(parent.dims["cell"], dtype=bool)
    for region in regions:
        if inclusion == "circumradius":
            mask |= _circumradius_region_mask(parent, region)
            continue
        mask |= _region_mask(parent.lon, parent.lat, region, parent.cell_center_xyz)
        if inclusion == "overlap":
            vertex_mask = _region_mask(
                parent.vertex_lon,
                parent.vertex_lat,
                region,
                parent.vertices,
            )
            mask |= np.any(vertex_mask[parent.cells], axis=1)
    return np.flatnonzero(mask).astype(np.int32)


def _circumradius_region_mask(parent: Any, region: Any) -> np.ndarray:
    """Select cells through circumcenter/circumradius region intersection."""
    name = region.__class__.__name__.removeprefix("_")
    if name == "PolygonRegion":
        center_mask = _region_mask(
            parent.lon,
            parent.lat,
            region,
            parent.cell_center_xyz,
        )
        vertex_mask = _region_mask(
            parent.vertex_lon,
            parent.vertex_lat,
            region,
            parent.vertices,
        )
        return center_mask | np.any(vertex_mask[parent.cells], axis=1)

    centers = parent.cell_center_xyz
    first_vertices = parent.vertices[parent.cells[:, 0]]
    center_norm = np.linalg.norm(centers, axis=1)
    vertex_norm = np.linalg.norm(first_vertices, axis=1)
    cosine = np.sum(centers * first_vertices, axis=1) / (
        center_norm * vertex_norm
    )
    radius = np.degrees(
        np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )
    )

    if name == "CircleRegion":
        distance = _angular_distance_degrees(
            parent.lon,
            parent.lat,
            region.lon,
            region.lat,
        )
        return distance < radius + region.radius_degrees

    if name == "OrientedRectangleRegion":
        dx = _wrapped_lon_delta(parent.lon - region.center_lon) * cos(
            radians(region.center_lat)
        )
        dy = parent.lat - region.center_lat
        angle = radians(region.angle_degrees)
        x_rot = dx * np.cos(angle) + dy * np.sin(angle)
        y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
        return (
            (np.abs(x_rot) <= 0.5 * region.width_degrees + radius)
            & (np.abs(y_rot) <= 0.5 * region.height_degrees + radius)
        )

    if name == "LonLatBoxRegion":
        longitude_span = float(np.mod(region.lon_max - region.lon_min, 360.0))
        if np.isclose(longitude_span, 0.0) and region.lon_min != region.lon_max:
            longitude_span = 360.0
        half_width_lon = 0.5 * longitude_span
        center_lon = float(_wrapped_lon_delta(region.lon_min + half_width_lon))
        center_lat = 0.5 * (region.lat_min + region.lat_max)
        half_width_lat = 0.5 * (region.lat_max - region.lat_min)
        lon = parent.lon
        lat = parent.lat
    elif name == "RotatedLonLatBoxRegion":
        from . import grid_generator as gg

        rotated = gg._unrotate_latlon(
            centers,
            region.pole_lon,
            region.pole_lat,
        )
        lon, lat = gg._lon_lat(rotated)
        center_lon = region.center_lon
        center_lat = region.center_lat
        half_width_lon = region.half_width_lon
        half_width_lat = region.half_width_lat
    else:
        raise TypeError(f"unsupported circumradius region {name}")

    lon_delta = np.abs(_wrapped_lon_delta(lon - center_lon))
    return (
        (lon_delta <= half_width_lon + radius)
        & (lat >= center_lat - half_width_lat - radius)
        & (lat <= center_lat + half_width_lat + radius)
    )


def _region_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    region: Any,
    xyz: np.ndarray | None = None,
) -> np.ndarray:
    name = region.__class__.__name__.removeprefix("_")
    if name == "LonLatBoxRegion":
        if region.lon_min <= region.lon_max:
            lon_mask = (lon >= region.lon_min) & (lon <= region.lon_max)
        else:
            lon_mask = (lon >= region.lon_min) | (lon <= region.lon_max)
        return lon_mask & (lat >= region.lat_min) & (lat <= region.lat_max)
    if name == "CircleRegion":
        return _angular_distance_degrees(lon, lat, region.lon, region.lat) <= region.radius_degrees
    if name == "OrientedRectangleRegion":
        dx = _wrapped_lon_delta(lon - region.center_lon) * cos(radians(region.center_lat))
        dy = lat - region.center_lat
        angle = radians(region.angle_degrees)
        x_rot = dx * np.cos(angle) + dy * np.sin(angle)
        y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
        return (
            (np.abs(x_rot) <= 0.5 * region.width_degrees)
            & (np.abs(y_rot) <= 0.5 * region.height_degrees)
        )
    if name == "RotatedLonLatBoxRegion":
        if xyz is None:
            xyz = _xyz_from_lon_lat(lon, lat)
        from . import grid_generator as gg

        rotated = gg._unrotate_latlon(xyz, region.pole_lon, region.pole_lat)
        rotated_lon, rotated_lat = gg._lon_lat(rotated)
        dx = _wrapped_lon_delta(rotated_lon - region.center_lon)
        return (
            (np.abs(dx) <= region.half_width_lon)
            & (np.abs(rotated_lat - region.center_lat) <= region.half_width_lat)
        )
    if name == "PolygonRegion":
        return _polygon_mask(lon, lat, region.points)
    raise TypeError(f"unsupported cut region {name}")


def _xyz_from_lon_lat(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
    )


def _buffer_rings(spec: Any) -> int:
    return spec.boundary_depth or spec.selection.buffer_rings


def _clean_selection(
    parent: Any,
    selected: np.ndarray,
    policy: str,
) -> np.ndarray:
    if policy == "none" or selected.size < 4:
        return selected
    raw = np.zeros(parent.dims["cell"], dtype=bool)
    raw[selected] = True
    neighbors = np.asarray(parent.icon_connectivity["c2c"][selected], dtype=np.int64)
    valid = neighbors >= 0
    safe = np.maximum(neighbors, 0)
    neighbor_count = np.sum(raw[safe] & valid, axis=1)
    retained = selected[neighbor_count > 1]
    return retained if retained.size else selected


def _angular_distance_degrees(
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> np.ndarray:
    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lon2 = radians(center_lon)
    lat2 = radians(center_lat)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.sin(0.5 * dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    return np.degrees(2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))


def _polygon_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    points: tuple[tuple[float, float], ...],
) -> np.ndarray:
    reference = points[0][0]
    x = _wrapped_lon_delta(lon - reference)
    y = lat
    polygon = np.asarray(
        [(_wrapped_lon_delta(point_lon - reference), point_lat) for point_lon, point_lat in points],
        dtype=np.float64,
    )
    inside = np.zeros(lon.shape, dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-30) + xi
        )
        inside ^= crosses
        j = i
    return inside


def _wrapped_lon_delta(delta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(delta) + 180.0) % 360.0 - 180.0


def _expand_cells(parent: Any, selected: np.ndarray, depth: int) -> np.ndarray:
    n_cells = parent.dims["cell"]
    expanded = np.zeros(n_cells, dtype=bool)
    valid_selected = selected[(selected >= 0) & (selected < n_cells)]
    expanded[valid_selected] = True
    frontier = valid_selected
    for _ in range(depth):
        neighbors = np.asarray(parent.icon_connectivity["c2c"][frontier]).reshape(-1)
        neighbors = neighbors[(neighbors >= 0) & (neighbors < n_cells)]
        frontier = np.unique(neighbors[~expanded[neighbors]]).astype(np.int32)
        expanded[frontier] = True
        if frontier.size == 0:
            break
    return np.flatnonzero(expanded).astype(np.int32)


def _compact_geometry(parent: Any, parent_cells: np.ndarray) -> GeometryData:
    source_vertices = np.unique(parent.cells[parent_cells].reshape(-1)).astype(
        np.int32,
        copy=False,
    )
    cells = np.searchsorted(source_vertices, parent.cells[parent_cells]).astype(
        np.int32,
        copy=False,
    )
    return GeometryData(
        vertices=parent.vertices[source_vertices],
        cells=cells,
        lon=parent.lon[parent_cells],
        lat=parent.lat[parent_cells],
        vertex_lon=parent.vertex_lon[source_vertices],
        vertex_lat=parent.vertex_lat[source_vertices],
        cell_center_xyz=parent.cell_center_xyz[parent_cells],
        cell_vertex_lon=parent.cell_vertex_lon[parent_cells],
        cell_vertex_lat=parent.cell_vertex_lat[parent_cells],
        source_cell_index=parent_cells,
        source_vertex_index=source_vertices,
    )


def _open_topology(
    parent: Any,
    geometry: GeometryData,
    parent_cells: np.ndarray,
    options: Any,
) -> TopologyData:
    from . import grid_generator as gg

    source_edge_ids = np.unique(parent.cell_edges[parent_cells].reshape(-1)).astype(
        np.int32,
        copy=False,
    )
    cell_edges = np.searchsorted(
        source_edge_ids,
        parent.cell_edges[parent_cells],
    ).astype(np.int32, copy=False)
    edges_array = np.searchsorted(
        geometry.source_vertex_index,
        parent.edges[source_edge_ids],
    ).astype(np.int32, copy=False)
    source_adjacency = parent.edge_cells[source_edge_ids]
    positions = np.searchsorted(parent_cells, source_adjacency)
    clipped_positions = np.minimum(positions, parent_cells.size - 1)
    present = (
        (source_adjacency >= 0)
        & (positions < parent_cells.size)
        & (parent_cells[clipped_positions] == source_adjacency)
    )
    edge_cell_array = np.where(present, positions, -1).astype(np.int32)
    reversed_boundary = (edge_cell_array[:, 0] < 0) & (edge_cell_array[:, 1] >= 0)
    edge_cell_array[reversed_boundary] = edge_cell_array[reversed_boundary, ::-1]
    edge_center_xyz = parent.edge_center_xyz[source_edge_ids]
    edge_lon = parent.edge_lon[source_edge_ids]
    edge_lat = parent.edge_lat[source_edge_ids]
    icon_connectivity = _open_icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges_array,
        cell_edges,
        edge_cell_array,
        options.accelerator,
    )
    del gg, options, parent_cells
    return TopologyData(
        edges=edges_array,
        cell_edges=cell_edges,
        edge_cells=edge_cell_array,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon_connectivity,
        connectivity=_open_public_connectivity(geometry.cells, edges_array, edge_cell_array, icon_connectivity),
        neighbor_tables=_open_neighbor_tables(geometry.cells, edges_array, edge_cell_array, icon_connectivity),
        source_edge_index=source_edge_ids,
    )


def _geometry_from_mesh(
    vertices: np.ndarray,
    cells: np.ndarray,
    options: Any,
    *,
    provenance: BisectionProvenance | None = None,
    source_cell_index: np.ndarray | None = None,
    source_vertex_index: np.ndarray | None = None,
) -> GeometryData:
    from . import grid_generator as gg

    vertex_lon, vertex_lat = gg._lon_lat(vertices)
    cell_center_xyz = gg._cell_centers(
        vertices,
        cells,
        options.radius,
        options.accelerator,
    )
    lon, lat = gg._lon_lat(cell_center_xyz)
    return GeometryData(
        vertices=vertices,
        cells=cells,
        lon=lon,
        lat=lat,
        vertex_lon=vertex_lon,
        vertex_lat=vertex_lat,
        cell_center_xyz=cell_center_xyz,
        cell_vertex_lon=vertex_lon[cells],
        cell_vertex_lat=vertex_lat[cells],
        source_cell_index=source_cell_index,
        source_vertex_index=source_vertex_index,
        bisection_provenance=provenance,
    )


def _topology_from_mesh(
    geometry: GeometryData,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    options: Any,
    *,
    source_edge_index: np.ndarray | None = None,
) -> TopologyData:
    from . import grid_generator as gg

    edge_center_xyz = gg._edge_centers(
        geometry.vertices,
        edges,
        options.radius,
        options.accelerator,
    )
    edge_lon, edge_lat = gg._lon_lat(edge_center_xyz)
    icon_connectivity = _open_icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges,
        cell_edges,
        edge_cells,
        options.accelerator,
    )
    return TopologyData(
        edges=np.asarray(edges, dtype=np.int32),
        cell_edges=np.asarray(cell_edges, dtype=np.int32),
        edge_cells=np.asarray(edge_cells, dtype=np.int32),
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon_connectivity,
        connectivity=_open_public_connectivity(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        neighbor_tables=_open_neighbor_tables(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        source_edge_index=source_edge_index,
    )


def _refine_open_grid(
    geometry: GeometryData,
    topology: TopologyData,
    options: Any,
) -> tuple[GeometryData, TopologyData]:
    """Bisect an open triangular mesh once and retain structural provenance."""
    from . import _accelerated
    from . import grid_generator as gg

    old_vertex_count = geometry.vertices.shape[0]
    old_edge_count = topology.edges.shape[0]
    old_cell_count = geometry.cells.shape[0]
    midpoint_vertices = 0.5 * (
        geometry.vertices[topology.edges[:, 0]]
        + geometry.vertices[topology.edges[:, 1]]
    )
    vertices = gg._normalize_rows(
        np.vstack((geometry.vertices, midpoint_vertices)).astype(np.float64, copy=False)
    ) * options.radius
    edge_midpoint_index = old_vertex_count + np.arange(old_edge_count, dtype=np.int32)
    split_edge_index = (
        2 * np.arange(old_edge_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1], dtype=np.int32)
    )
    inner_edge_index = (
        2 * old_edge_count
        + 3 * np.arange(old_cell_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1, 2], dtype=np.int32)
    )
    if _accelerated.should_use_numba_ordering(
        options.accelerator,
        old_cell_count * 4,
    ):
        (
            cells,
            raw_cell_edges,
            _,
            parent_edge_index,
            edge_parent_type,
            failure_cell,
            failure_kind,
        ) = _accelerated.fill_bisection_children_numba(
            geometry.cells,
            topology.edges,
            topology.cell_edges,
            edge_midpoint_index,
            split_edge_index,
            inner_edge_index,
            gg.EDGE_CHILD_TYPE_FROM_VERTEX_0,
            gg.EDGE_CHILD_TYPE_FROM_VERTEX_1,
            gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0,
            gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1,
            gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2,
        )
        if failure_kind or failure_cell >= 0:
            raise RuntimeError(
                f"open regional bisection failed at cell {failure_cell} "
                f"(failure kind {failure_kind})"
            )
    else:
        cells, raw_cell_edges, parent_edge_index, edge_parent_type = (
            _fill_open_bisection_numpy(
                geometry.cells,
                topology.edges,
                topology.cell_edges,
                edge_midpoint_index,
                split_edge_index,
                inner_edge_index,
            )
        )

    edge_count = old_edge_count * 2 + old_cell_count * 3
    flat_edges = raw_cell_edges.reshape(-1)
    flat_cells = np.repeat(np.arange(cells.shape[0], dtype=np.int32), 3)
    counts = np.bincount(flat_edges, minlength=edge_count)
    first = np.full(edge_count, cells.shape[0], dtype=np.int32)
    second = np.full(edge_count, -1, dtype=np.int32)
    np.minimum.at(first, flat_edges, flat_cells)
    np.maximum.at(second, flat_edges, flat_cells)
    second[counts == 1] = -1
    edge_cells = np.column_stack((first, second)).astype(np.int32)
    edges = gg._edge_vertices_from_cell_edges(cells, raw_cell_edges, edge_cells)
    cells, cell_edges = _order_open_cells_by_edges(
        vertices,
        cells,
        edges,
        raw_cell_edges,
        edge_cells,
        options.accelerator,
    )

    parent_vertex_index = np.empty(vertices.shape[0], dtype=np.int32)
    parent_vertex_index[:old_vertex_count] = np.arange(
        1, old_vertex_count + 1, dtype=np.int32
    )
    parent_vertex_index[old_vertex_count:] = -np.arange(
        1, old_edge_count + 1, dtype=np.int32
    )
    provenance = BisectionProvenance(
        cells=geometry.cells,
        edges=topology.edges,
        cell_edges=topology.cell_edges,
        parent_vertex_index=parent_vertex_index,
        parent_cell_index=np.repeat(
            np.arange(1, old_cell_count + 1, dtype=np.int32),
            4,
        ),
        parent_cell_type=np.tile(
            np.array(
                [
                    gg.CHILD_CELL_TYPE_CENTER,
                    gg.CHILD_CELL_TYPE_AT_VERTEX_2,
                    gg.CHILD_CELL_TYPE_AT_VERTEX_0,
                    gg.CHILD_CELL_TYPE_AT_VERTEX_1,
                ],
                dtype=np.int32,
            ),
            old_cell_count,
        ),
        child_edges=edges,
        child_cell_edges=cell_edges,
        child_edge_cells=edge_cells,
        child_parent_edge_index=parent_edge_index,
        child_edge_parent_type=edge_parent_type,
    )
    refined_geometry = _geometry_from_mesh(
        vertices,
        cells,
        options,
        provenance=provenance,
    )
    refined_topology = _topology_from_mesh(
        refined_geometry,
        edges,
        cell_edges,
        edge_cells,
        options,
    )
    return refined_geometry, refined_topology


def _order_open_cells_by_edges(
    vertices: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    accelerator: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ICON's local cell ordering with a boundary-edge ghost center."""
    from . import _accelerated
    from . import grid_generator as gg

    cell_centers = gg._cell_centers(vertices, cells, 1.0, accelerator)
    edge_centers = gg._edge_centers(vertices, edges, 1.0, accelerator)
    metric_cells = edge_cells.copy()
    missing_edge, missing_side = np.nonzero(metric_cells < 0)
    metric_cells[missing_edge, missing_side] = (
        cell_centers.shape[0] + np.arange(missing_edge.size, dtype=np.int32)
    )
    augmented = np.vstack((cell_centers, edge_centers[missing_edge]))
    orientation = gg._edge_system_orientation(
        vertices,
        augmented,
        edges,
        metric_cells,
        edge_centers,
    )
    if _accelerated.should_use_numba_ordering(accelerator, cells.shape[0]):
        ordered_cells, ordered_edges, failure_cell, failure_kind = (
            _accelerated.order_cells_by_edges_numba(
                edges,
                cell_edges,
                edge_cells,
                orientation,
            )
        )
        if failure_kind or failure_cell >= 0:
            raise RuntimeError(
                f"open cell ordering failed at cell {failure_cell} "
                f"(failure kind {failure_kind})"
            )
        return ordered_cells, ordered_edges

    ordered_cells = np.empty_like(cells)
    ordered_edges = np.empty_like(cell_edges)
    for cell_index, edges_for_cell in enumerate(cell_edges):
        first_edge = int(edges_for_cell[0])
        start_vertex, next_vertex = map(int, edges[first_edge])
        cell_orientation = 1 if edge_cells[first_edge, 0] == cell_index else -1
        if cell_orientation * int(orientation[first_edge]) > 0:
            start_vertex, next_vertex = next_vertex, start_vertex
        ordered_edges[cell_index, 0] = first_edge
        ordered_cells[cell_index, 0] = start_vertex
        current_edge = first_edge
        current_vertex = next_vertex
        previous_edge = -1
        for output_index in range(1, 3):
            edge_index = -1
            following_vertex = -1
            for candidate in map(int, edges_for_cell):
                if candidate in {current_edge, previous_edge}:
                    continue
                first, second = map(int, edges[candidate])
                if first == current_vertex or second == current_vertex:
                    edge_index = candidate
                    following_vertex = second if first == current_vertex else first
                    break
            if edge_index < 0:
                raise RuntimeError("could not order open cell edges")
            ordered_edges[cell_index, output_index] = edge_index
            ordered_cells[cell_index, output_index] = current_vertex
            previous_edge = current_edge
            current_edge = edge_index
            current_vertex = following_vertex
        if current_vertex != start_vertex:
            raise RuntimeError("open cell edges do not form a closed triangle")
    return ordered_cells, ordered_edges


def _fill_open_bisection_numpy(
    cells: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_midpoint_index: np.ndarray,
    split_edge_index: np.ndarray,
    inner_edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy/Python fallback matching ICON's four-child bisection ordering."""
    from . import grid_generator as gg

    new_cells = np.empty((cells.shape[0] * 4, 3), dtype=np.int32)
    new_cell_edges = np.empty_like(new_cells)
    edge_pairs_by_vertex = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    child_slot_by_vertex = np.array([2, 3, 1], dtype=np.int32)
    for cell_index, cell in enumerate(cells):
        parent_edges = cell_edges[cell_index]
        midpoints = edge_midpoint_index[parent_edges]
        center_child = 4 * cell_index
        new_cells[center_child] = midpoints
        new_cell_edges[center_child] = inner_edge_index[cell_index]
        for first_pos, second_pos, _ in edge_pairs_by_vertex:
            first_edge = int(parent_edges[first_pos])
            second_edge = int(parent_edges[second_pos])
            vertex = gg._common_edge_vertex(edges[first_edge], edges[second_edge])
            vertex_pos = gg._local_vertex_position(cell, vertex)
            child = center_child + int(child_slot_by_vertex[vertex_pos])
            new_cells[child] = (
                edge_midpoint_index[first_edge],
                vertex,
                edge_midpoint_index[second_edge],
            )
            new_cell_edges[child] = (
                split_edge_index[first_edge, gg._edge_endpoint_slot(edges[first_edge], vertex)],
                split_edge_index[second_edge, gg._edge_endpoint_slot(edges[second_edge], vertex)],
                inner_edge_index[cell_index, vertex_pos],
            )
    edge_count = edges.shape[0] * 2 + cells.shape[0] * 3
    parent_edge_index = np.empty(edge_count, dtype=np.int32)
    edge_parent_type = np.empty(edge_count, dtype=np.int32)
    parent_ids = np.arange(1, edges.shape[0] + 1, dtype=np.int32)
    parent_edge_index[split_edge_index[:, 0]] = parent_ids
    parent_edge_index[split_edge_index[:, 1]] = parent_ids
    edge_parent_type[split_edge_index[:, 0]] = gg.EDGE_CHILD_TYPE_FROM_VERTEX_0
    edge_parent_type[split_edge_index[:, 1]] = gg.EDGE_CHILD_TYPE_FROM_VERTEX_1
    parent_cell_edges = cell_edges + 1
    parent_edge_index[inner_edge_index[:, 0]] = parent_cell_edges[:, 1]
    parent_edge_index[inner_edge_index[:, 1]] = parent_cell_edges[:, 2]
    parent_edge_index[inner_edge_index[:, 2]] = parent_cell_edges[:, 0]
    edge_parent_type[inner_edge_index[:, 0]] = gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0
    edge_parent_type[inner_edge_index[:, 1]] = gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1
    edge_parent_type[inner_edge_index[:, 2]] = gg.EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2
    return new_cells, new_cell_edges, parent_edge_index, edge_parent_type


def _open_icon_connectivity(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    accelerator: str = "numpy",
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg
    return gg._icon_connectivity(
        vertices,
        cells,
        cell_center_xyz,
        edges,
        cell_edges,
        edge_cells,
        accelerator,
    )


def _open_public_connectivity(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    return {
        "edge_of_cell": icon_connectivity["c2e"],
        "vertex_of_cell": cells,
        "neighbor_cell_index": icon_connectivity["c2c"],
        "adjacent_cell_of_edge": edge_cells,
        "edge_vertices": edges,
        "cells_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2c"]),
        "edges_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2e"]),
        "vertices_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2v"]),
    }


def _open_neighbor_tables(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    return {
        "c2e2c": icon_connectivity["c2c"],
        "c2e": icon_connectivity["c2e"],
        "e2c": np.asarray(edge_cells, dtype=np.int32),
        "v2e": gg._zero_based_with_skip(icon_connectivity["v2e"]),
        "v2c": gg._zero_based_with_skip(icon_connectivity["v2c"]),
        "c2v": np.asarray(cells, dtype=np.int32),
        "v2e2v": gg._zero_based_with_skip(icon_connectivity["v2v"]),
        "e2v": np.asarray(edges, dtype=np.int32),
    }


def _optimize_local_geometry(
    geometry: GeometryData,
    topology: TopologyData,
    options: Any,
    iterations: int,
) -> GeometryData:
    if iterations == 0:
        return geometry
    from ._optimization import _GlobalOptimizationOptions, _spring_relaxed_vertices

    proxy = SimpleNamespace(
        vertices=geometry.vertices,
        cells=geometry.cells,
        edges=topology.edges,
        edge_cells=topology.edge_cells,
        icon_connectivity=topology.icon_connectivity,
        options=options,
        dims={
            "cell": geometry.cells.shape[0],
            "edge": topology.edges.shape[0],
            "vertex": geometry.vertices.shape[0],
        },
        metadata={"grid_geometry": SPHERE_GEOMETRY, "open_boundary": 1},
        incident_edges_sorted=False,
    )
    vertices = _spring_relaxed_vertices(
        proxy,
        _GlobalOptimizationOptions(method="spring", iterations=iterations),
    )
    triangles = vertices[geometry.cells]
    orientation = np.einsum(
        "ij,ij->i",
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        triangles.sum(axis=1),
    )
    if not np.all(np.isfinite(vertices)) or np.any(orientation <= 0.0):
        return geometry
    return _geometry_from_mesh(
        vertices,
        geometry.cells,
        options,
        provenance=geometry.bisection_provenance,
        source_cell_index=geometry.source_cell_index,
        source_vertex_index=geometry.source_vertex_index,
    )


def _boundary_controls(
    geometry: GeometryData,
    topology: TopologyData,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_ctrl = np.zeros(geometry.vertices.shape[0], dtype=np.int32)
    boundary_edges = topology.edge_cells[:, 1] < 0
    vertex_ctrl[np.unique(topology.edges[boundary_edges])] = 1
    for level in range(1, depth):
        touched = np.any((vertex_ctrl == level)[geometry.cells], axis=1)
        candidates = np.unique(geometry.cells[touched])
        candidates = candidates[vertex_ctrl[candidates] == 0]
        vertex_ctrl[candidates] = level + 1

    per_cell = vertex_ctrl[geometry.cells]
    positive = np.where(per_cell > 0, per_cell, depth + 1)
    cell_ctrl = np.min(positive, axis=1).astype(np.int32)
    cell_ctrl[cell_ctrl == depth + 1] = 0

    edge_ctrl = np.zeros(topology.edges.shape[0], dtype=np.int32)
    valid0 = topology.edge_cells[:, 0] >= 0
    valid1 = topology.edge_cells[:, 1] >= 0
    side0 = np.where(
        valid0,
        cell_ctrl[np.maximum(topology.edge_cells[:, 0], 0)],
        0,
    )
    side1 = np.where(
        valid1,
        cell_ctrl[np.maximum(topology.edge_cells[:, 1], 0)],
        0,
    )
    edge_ctrl[:] = side0 + side1
    transition = (
        ((side0 == 0) & (side1 > 0) & valid0)
        | ((side1 == 0) & (side0 > 0) & valid1)
    )
    edge_ctrl[transition] = 0
    return cell_ctrl, edge_ctrl, vertex_ctrl


def _ordered_ids(control: np.ndarray, maximum_ordered_level: int) -> np.ndarray:
    parts = [
        np.flatnonzero(control == level)
        for level in range(1, maximum_ordered_level + 1)
    ]
    parts.append(
        np.flatnonzero((control == 0) | (control > maximum_ordered_level))
    )
    return np.concatenate(parts).astype(np.int64)


def _apply_boundary_ordering(
    geometry: GeometryData,
    topology: TopologyData,
    boundary: Any,
    options: Any,
) -> tuple[GeometryData, TopologyData, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    controls = _boundary_controls(geometry, topology, boundary.indexing_depth)
    if boundary.ordering == "source":
        return geometry, topology, controls

    cell_ctrl, edge_ctrl, vertex_ctrl = controls
    cell_order = _ordered_ids(cell_ctrl, 5)
    edge_order = _ordered_ids(edge_ctrl, 10)
    vertex_order = _ordered_ids(vertex_ctrl, 5)
    cell_inverse = np.empty(cell_order.size, dtype=np.int32)
    edge_inverse = np.empty(edge_order.size, dtype=np.int32)
    vertex_inverse = np.empty(vertex_order.size, dtype=np.int32)
    cell_inverse[cell_order] = np.arange(cell_order.size, dtype=np.int32)
    edge_inverse[edge_order] = np.arange(edge_order.size, dtype=np.int32)
    vertex_inverse[vertex_order] = np.arange(vertex_order.size, dtype=np.int32)

    cells = vertex_inverse[geometry.cells[cell_order]]
    edges = vertex_inverse[topology.edges[edge_order]]
    cell_edges = edge_inverse[topology.cell_edges[cell_order]]
    edge_cells = topology.edge_cells[edge_order].copy()
    present = edge_cells >= 0
    edge_cells[present] = cell_inverse[edge_cells[present]]

    provenance = geometry.bisection_provenance
    if provenance is not None:
        provenance = replace(
            provenance,
            parent_vertex_index=provenance.parent_vertex_index[vertex_order],
            parent_cell_index=provenance.parent_cell_index[cell_order],
            parent_cell_type=provenance.parent_cell_type[cell_order],
            child_edges=edges,
            child_cell_edges=cell_edges,
            child_edge_cells=edge_cells,
            child_parent_edge_index=(
                None
                if provenance.child_parent_edge_index is None
                else provenance.child_parent_edge_index[edge_order]
            ),
            child_edge_parent_type=(
                None
                if provenance.child_edge_parent_type is None
                else provenance.child_edge_parent_type[edge_order]
            ),
        )
    source_cells = (
        None
        if geometry.source_cell_index is None
        else geometry.source_cell_index[cell_order]
    )
    source_vertices = (
        None
        if geometry.source_vertex_index is None
        else geometry.source_vertex_index[vertex_order]
    )
    ordered_geometry = _geometry_from_mesh(
        geometry.vertices[vertex_order],
        cells,
        options,
        provenance=provenance,
        source_cell_index=source_cells,
        source_vertex_index=source_vertices,
    )
    ordered_topology = _topology_from_mesh(
        ordered_geometry,
        edges,
        cell_edges,
        edge_cells,
        options,
        source_edge_index=(
            None
            if topology.source_edge_index is None
            else topology.source_edge_index[edge_order]
        ),
    )
    return (
        ordered_geometry,
        ordered_topology,
        (cell_ctrl[cell_order], edge_ctrl[edge_order], vertex_ctrl[vertex_order]),
    )


def _limited_metrics(
    parent: Any,
    geometry: GeometryData,
    topology: TopologyData,
    closure: str,
    options: Any,
) -> MetricsData:
    if is_planar_geometry(parent.metadata):
        return _limited_planar_metrics(parent, geometry, topology, closure)
    return _spherical_open_metrics(geometry, topology, closure, options)


def _spherical_open_metrics(
    geometry: GeometryData,
    topology: TopologyData,
    closure: str,
    options: Any,
) -> MetricsData:
    from . import grid_generator as gg

    metric_cells = topology.edge_cells.copy()
    missing_edge, missing_side = np.nonzero(metric_cells < 0)
    ghost_ids = np.arange(
        geometry.cell_center_xyz.shape[0],
        geometry.cell_center_xyz.shape[0] + missing_edge.size,
        dtype=np.int32,
    )
    metric_cells[missing_edge, missing_side] = ghost_ids
    augmented_centers = np.vstack(
        (geometry.cell_center_xyz, topology.edge_center_xyz[missing_edge])
    )
    fields = gg._geometry_fields(
        geometry.vertices,
        geometry.cells,
        augmented_centers,
        topology.edges,
        metric_cells,
        topology.edge_center_xyz,
        topology.icon_connectivity,
        options.sphere_radius,
        options.accelerator,
    )
    fields["edge_cell_distance"][missing_edge, missing_side] = 0.0
    boundary_edges = topology.edge_cells[:, 1] < 0
    if closure == "mirrored":
        fields["dual_edge_length"][boundary_edges] *= 2.0
        fields["edge_cell_distance"][boundary_edges, 1] = fields[
            "edge_cell_distance"
        ][boundary_edges, 0]
        fields["edgequad_area"] = (
            0.5 * fields["edge_length"] * fields["dual_edge_length"]
        )
        fields["dual_area"] = gg._dual_areas_from_edges(
            geometry.vertices.shape[0],
            topology.icon_connectivity["v2e"],
            fields["edge_length"],
            fields["dual_edge_length"],
        )
    return MetricsData(fields=fields)


def _limited_planar_metrics(
    parent: Any,
    geometry: GeometryData,
    topology: TopologyData,
    closure: str,
) -> MetricsData:
    from . import grid_generator as gg

    source_edges = topology.source_edge_index
    edge_lengths = parent.geometry["edge_length"][source_edges]
    edge_cell_distance = np.zeros((topology.edges.shape[0], 2), dtype=np.float64)
    for edge_index, adjacent in enumerate(topology.edge_cells):
        source_edge = int(source_edges[edge_index])
        parent_adjacent = parent.edge_cells[source_edge]
        for side in range(2):
            if adjacent[side] < 0:
                continue
            source_cell = int(geometry.source_cell_index[adjacent[side]])
            match = np.flatnonzero(parent_adjacent == source_cell)
            if match.size != 1:
                raise RuntimeError("cut edge is inconsistent with its source grid")
            edge_cell_distance[edge_index, side] = parent.geometry[
                "edge_cell_distance"
            ][source_edge, int(match[0])]
    boundary_edges = topology.edge_cells[:, 1] < 0
    dual_edge_lengths = edge_cell_distance.sum(axis=1)
    if closure == "mirrored":
        edge_cell_distance[boundary_edges, 1] = edge_cell_distance[boundary_edges, 0]
        dual_edge_lengths[boundary_edges] *= 2.0

    normal_sign = _source_normal_sign(parent, geometry, topology)
    normals = {
        name: parent.geometry[name][source_edges] * normal_sign[:, np.newaxis]
        for name in (
            "edge_primal_normal_cartesian",
            "edge_dual_normal_cartesian",
        )
    }
    normals.update(
        {
            name: parent.geometry[name][source_edges] * normal_sign
            for name in (
                "zonal_normal_primal_edge",
                "meridional_normal_primal_edge",
                "zonal_normal_dual_edge",
                "meridional_normal_dual_edge",
            )
        }
    )
    cell_areas = parent.geometry["cell_area"][geometry.source_cell_index]
    return MetricsData(
        fields={
            "cell_area": cell_areas,
            "dual_area": gg._dual_areas(
                geometry.vertices.shape[0], geometry.cells, cell_areas
            ),
            "edge_length": edge_lengths,
            "dual_edge_length": dual_edge_lengths,
            "edge_cell_distance": edge_cell_distance,
            "edge_vert_distance": np.column_stack(
                (edge_lengths * 0.5, edge_lengths * 0.5)
            ),
            "orientation_of_normal": topology.icon_connectivity[
                "orientation_of_normal"
            ],
            "edge_system_orientation": _source_edge_system_orientation(
                parent, geometry, topology, normal_sign
            ),
            "edge_orientation": topology.icon_connectivity["edge_orientation"],
            "edgequad_area": 0.5 * edge_lengths * dual_edge_lengths,
            **normals,
        }
    )


def _source_normal_sign(
    parent: Any,
    geometry: GeometryData,
    topology: TopologyData,
) -> np.ndarray:
    signs = np.ones(topology.edges.shape[0], dtype=np.int32)
    for edge_index, adjacent in enumerate(topology.edge_cells):
        source_edge = int(topology.source_edge_index[edge_index])
        parent_adjacent = parent.edge_cells[source_edge]
        first_source_cell = int(geometry.source_cell_index[adjacent[0]])
        if first_source_cell == int(parent_adjacent[0]):
            signs[edge_index] = 1
        elif first_source_cell == int(parent_adjacent[1]):
            signs[edge_index] = -1
        else:
            raise RuntimeError(
                f"source cell {first_source_cell} is not adjacent to source edge "
                f"{source_edge}"
            )
    return signs


def _source_edge_system_orientation(
    parent: Any,
    geometry: GeometryData,
    topology: TopologyData,
    normal_sign: np.ndarray,
) -> np.ndarray:
    source_edges = topology.source_edge_index
    local_source_vertices = geometry.source_vertex_index[topology.edges]
    parent_edges = parent.edges[source_edges]
    same_direction = np.all(local_source_vertices == parent_edges, axis=1)
    reverse_direction = np.all(local_source_vertices == parent_edges[:, ::-1], axis=1)
    if not np.all(same_direction | reverse_direction):
        bad_edge = int(np.flatnonzero(~(same_direction | reverse_direction))[0])
        raise RuntimeError(f"local edge {bad_edge} does not match its source edge")
    endpoint_sign = np.where(same_direction, 1, -1).astype(np.int32)
    return (
        parent.geometry["edge_system_orientation"][source_edges]
        * endpoint_sign
        * normal_sign
    ).astype(np.int32)


def _limited_refinement(
    geometry: GeometryData,
    topology: TopologyData,
    controls: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> RefinementData:
    n_cells = geometry.cells.shape[0]
    n_edges = topology.edges.shape[0]
    n_vertices = geometry.vertices.shape[0]
    cell_ctrl, edge_ctrl, vertex_ctrl = controls
    provenance = geometry.bisection_provenance
    if provenance is None:
        parent_cell_index = (
            np.full(n_cells, -1, dtype=np.int32)
            if geometry.source_cell_index is None
            else geometry.source_cell_index.astype(np.int32) + 1
        )
        parent_cell_type = np.zeros(n_cells, dtype=np.int32)
        parent_edge_index = (
            np.full(n_edges, -1, dtype=np.int32)
            if topology.source_edge_index is None
            else topology.source_edge_index.astype(np.int32) + 1
        )
        edge_parent_type = np.zeros(n_edges, dtype=np.int32)
        parent_vertex_index = (
            np.full(n_vertices, -1, dtype=np.int32)
            if geometry.source_vertex_index is None
            else geometry.source_vertex_index.astype(np.int32) + 1
        )
    else:
        parent_cell_index = provenance.parent_cell_index
        parent_cell_type = provenance.parent_cell_type
        parent_edge_index = provenance.child_parent_edge_index
        edge_parent_type = provenance.child_edge_parent_type
        parent_vertex_index = provenance.parent_vertex_index
        if parent_edge_index is None or edge_parent_type is None:
            raise RuntimeError("refined regional grid is missing edge provenance")
    return RefinementData(
        fields={
            "refin_c_ctrl": cell_ctrl,
            "refin_e_ctrl": edge_ctrl,
            "refin_v_ctrl": vertex_ctrl,
            "start_idx_c": _start_end(cell_ctrl, "cell_grf")[0],
            "end_idx_c": _start_end(cell_ctrl, "cell_grf")[1],
            "start_idx_e": _start_end(edge_ctrl, "edge_grf")[0],
            "end_idx_e": _start_end(edge_ctrl, "edge_grf")[1],
            "start_idx_v": _start_end(vertex_ctrl, "vert_grf")[0],
            "end_idx_v": _start_end(vertex_ctrl, "vert_grf")[1],
            "parent_cell_index": np.asarray(parent_cell_index, dtype=np.int32),
            "parent_cell_type": np.asarray(parent_cell_type, dtype=np.int32),
            "edge_parent_type": np.asarray(edge_parent_type, dtype=np.int32),
            "parent_edge_index": np.asarray(parent_edge_index, dtype=np.int32),
            "parent_vertex_index": np.asarray(parent_vertex_index, dtype=np.int32),
        }
    )


def _start_end(control: np.ndarray, dimension: str) -> tuple[np.ndarray, np.ndarray]:
    from . import grid_generator as gg

    minimum = gg.MIN_REFINEMENT_CONTROL[dimension]
    size = gg.FIXED_DIMS[dimension]
    maximum = minimum + size - 1
    entity_count = control.size
    start = np.full(size, entity_count + 1, dtype=np.int32)
    end = np.full(size, entity_count, dtype=np.int32)
    for level in range(0, maximum + 1):
        indices = np.flatnonzero(control == level)
        if indices.size:
            start[level - minimum] = int(indices[0]) + 1
            end[level - minimum] = int(indices[-1]) + 1
    unordered = np.flatnonzero((control == 0) | (control > maximum))
    if unordered.size:
        start[-minimum] = int(unordered[0]) + 1
        end[-minimum] = entity_count
    return start[np.newaxis, :], end[np.newaxis, :]
