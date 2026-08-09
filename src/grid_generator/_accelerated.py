"""Optional acceleration helpers.

The package must remain usable without Numba. Keep this module importable in a
plain NumPy environment and import Numba only inside functions that need it.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util

import numpy as np


ACCELERATOR_AUTO = "auto"
ACCELERATOR_NUMBA = "numba"
ACCELERATOR_NUMPY = "numpy"
SUPPORTED_ACCELERATORS = {
    ACCELERATOR_AUTO,
    ACCELERATOR_NUMBA,
    ACCELERATOR_NUMPY,
}
AUTO_NUMBA_MIN_LOOKUP_ROWS = 1_000_000
AUTO_NUMBA_MIN_ORDER_CELLS = 100_000


def is_numba_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def should_use_numba(accelerator: str, work_items: int | None = None) -> bool:
    if accelerator == ACCELERATOR_NUMPY:
        return False
    if accelerator == ACCELERATOR_NUMBA:
        if not is_numba_available():
            raise ModuleNotFoundError(
                "Numba acceleration requires installing the 'accelerate' extra"
            )
        return True
    if work_items is None or work_items < AUTO_NUMBA_MIN_LOOKUP_ROWS:
        return False
    return is_numba_available()


def should_use_numba_large(accelerator: str, work_items: int) -> bool:
    """Select Numba only when work is large enough to amortize compilation."""
    if accelerator == ACCELERATOR_NUMPY:
        return False
    if accelerator == ACCELERATOR_NUMBA and not is_numba_available():
        raise ModuleNotFoundError(
            "Numba acceleration requires installing the 'accelerate' extra"
        )
    return work_items >= AUTO_NUMBA_MIN_LOOKUP_ROWS and is_numba_available()


def should_use_numba_ordering(accelerator: str, cell_count: int) -> bool:
    if accelerator == ACCELERATOR_NUMPY:
        return False
    if accelerator == ACCELERATOR_NUMBA:
        if not is_numba_available():
            raise ModuleNotFoundError(
                "Numba acceleration requires installing the 'accelerate' extra"
            )
        return True
    return cell_count >= AUTO_NUMBA_MIN_ORDER_CELLS and is_numba_available()


def order_cells_by_edges_numba(
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_system_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    return _compiled_order_cells_by_edges()(
        edges,
        cell_edges,
        edge_cells,
        edge_system_orientation,
    )


def orient_closed_cells_inplace_numba(
    vertices: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> tuple[int, int]:
    """Rotate validated spherical cells in place without geometry work arrays."""
    return _compiled_orient_closed_cells_inplace()(
        vertices,
        cells,
        edges,
        cell_edges,
        edge_cells,
    )


def lookup_width2_numba(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _compiled_lookup_width2()(
        signature_keys,
        parent_index_values,
        type_values,
        query_keys,
    )


def lookup_width3_numba(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _compiled_lookup_width3()(
        signature_keys,
        parent_index_values,
        type_values,
        query_keys,
    )


def fill_bisection_children_numba(
    cells: np.ndarray,
    edge_vertices: np.ndarray,
    cell_edges: np.ndarray,
    edge_midpoint_index: np.ndarray,
    split_edge_index: np.ndarray,
    inner_edge_index: np.ndarray,
    edge_child_type_from_vertex_0: int,
    edge_child_type_from_vertex_1: int,
    edge_child_type_in_cell_opposite_vertex_0: int,
    edge_child_type_in_cell_opposite_vertex_1: int,
    edge_child_type_in_cell_opposite_vertex_2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    return _compiled_fill_bisection_children()(
        cells,
        edge_vertices,
        cell_edges,
        edge_midpoint_index,
        split_edge_index,
        inner_edge_index,
        edge_child_type_from_vertex_0,
        edge_child_type_from_vertex_1,
        edge_child_type_in_cell_opposite_vertex_0,
        edge_child_type_in_cell_opposite_vertex_1,
        edge_child_type_in_cell_opposite_vertex_2,
    )


def refine_closed_bisection_numba(
    cells: np.ndarray,
    edge_vertices: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    old_vertex_count: int,
    edge_child_type_from_vertex_0: int,
    edge_child_type_from_vertex_1: int,
    edge_child_type_in_cell_opposite_vertex_0: int,
    edge_child_type_in_cell_opposite_vertex_1: int,
    edge_child_type_in_cell_opposite_vertex_2: int,
) -> tuple[np.ndarray, ...]:
    """Refine a validated closed topology without sorting child incidences."""
    return _compiled_refine_closed_bisection()(
        cells,
        edge_vertices,
        cell_edges,
        edge_cells,
        old_vertex_count,
        edge_child_type_from_vertex_0,
        edge_child_type_from_vertex_1,
        edge_child_type_in_cell_opposite_vertex_0,
        edge_child_type_in_cell_opposite_vertex_1,
        edge_child_type_in_cell_opposite_vertex_2,
    )


def spring_relaxation_numba(
    vertices: np.ndarray,
    edges: np.ndarray,
    incident_edges: np.ndarray,
    movable: np.ndarray,
    target_length: float,
    iterations: int,
) -> tuple[np.ndarray, int]:
    """Run deterministic vertex-parallel global spring relaxation.

    ``incident_edges`` contains one-based edge indices sorted by edge number.
    Each vertex owns its force reduction, avoiding the conflicting scatter in
    ``np.add.at`` while retaining the reference algorithm's contribution order.
    """
    return _compiled_spring_relaxation()(
        vertices,
        edges,
        incident_edges,
        movable,
        target_length,
        iterations,
    )


def mean_edge_angle_numba(vertices: np.ndarray, edges: np.ndarray) -> float:
    """Return the mean spherical edge angle without materializing edge gathers."""
    return float(_compiled_mean_edge_angle()(vertices, edges))


def dual_areas_from_edges_numba(
    edges_of_vertex: np.ndarray,
    edge_lengths: np.ndarray,
    dual_edge_lengths: np.ndarray,
) -> np.ndarray:
    """Compute one independent dual-area reduction per vertex."""
    return _compiled_dual_areas_from_edges()(
        edges_of_vertex,
        edge_lengths,
        dual_edge_lengths,
    )


def cell_areas_numba(
    vertices: np.ndarray,
    cells: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    """Compute independent spherical triangle areas in parallel."""
    return _compiled_cell_areas()(vertices, cells, sphere_radius)


def matching_edge_indices_numba(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    target_edges_of_vertex: np.ndarray,
) -> np.ndarray:
    """Match undirected source edges through a target vertex-incidence table."""
    return _compiled_matching_edge_indices()(
        source_edges,
        target_edges,
        target_edges_of_vertex,
    )


def build_closed_edges_numba(
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build closed triangular edge tables in deterministic scan order."""
    return _compiled_build_closed_edges()(cells)


def fixed_incidence_numba(
    owners: np.ndarray,
    values: np.ndarray,
    row_count: int,
    width: int,
) -> tuple[np.ndarray, int, int]:
    """Fill a bounded incidence table while preserving input order."""
    return _compiled_fixed_incidence()(owners, values, row_count, width)


def spherical_edge_metrics_numba(
    vertices: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    sphere_radius: float,
) -> tuple[np.ndarray, ...]:
    """Compute independent closed-sphere edge metrics without large temporaries."""
    result = _compiled_spherical_edge_metrics()(
        vertices,
        cell_center_xyz,
        edges,
        edge_cells,
        edge_center_xyz,
        sphere_radius,
    )
    if result[-1]:
        raise RuntimeError("edge system orientation is degenerate for at least one edge")
    return result[:-1]


def primal_normals_numba(
    vertices: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
) -> np.ndarray:
    """Compute only oriented primal edge normals for a closed spherical grid."""
    normals, degenerate_count = _compiled_primal_normals()(
        vertices,
        cells,
        edges,
        edge_cells,
    )
    if degenerate_count:
        raise RuntimeError("edge system orientation is degenerate for at least one edge")
    return normals


def cell_centers_numba(
    vertices: np.ndarray,
    cells: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Compute spherical cell centers independently in parallel."""
    return _compiled_cell_centers()(vertices, cells, radius)


def edge_centers_numba(
    vertices: np.ndarray,
    edges: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Compute spherical edge centers independently in parallel."""
    return _compiled_edge_centers()(vertices, edges, radius)


def sort_fixed_around_vertices_numba(
    vertices: np.ndarray,
    ids: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Sort bounded incidence rows geometrically and rotate to minimum ID."""
    return _compiled_sort_fixed_around_vertices()(vertices, ids, points)


def global_edge_orientation_states_numba(
    vertices: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    parent_normals: np.ndarray,
    parent_edge_index: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Classify global edges as retained, flipped, or geometrically degenerate."""
    return _compiled_global_edge_orientation_states()(
        vertices,
        cell_center_xyz,
        edges,
        edge_cells,
        edge_center_xyz,
        parent_normals,
        parent_edge_index,
    )


def apply_global_edge_flips_numba(
    edges: np.ndarray,
    edge_cells: np.ndarray,
    orientation_of_normal: np.ndarray,
    edge_orientation: np.ndarray,
    cell_edges: np.ndarray,
    edges_of_vertex: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply global edge flips and their two dependent sign updates."""
    return _compiled_apply_global_edge_flips()(
        edges,
        edge_cells,
        orientation_of_normal,
        edge_orientation,
        cell_edges,
        edges_of_vertex,
        states,
    )


def apply_edge_flips_numba(
    edges: np.ndarray,
    edge_cells: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply orientation flips without constructing dependent sign tables."""
    empty_cells = np.empty((0, 3), dtype=np.int32)
    empty_vertices = np.empty((0, 6), dtype=np.int32)
    adjusted_edges, adjusted_edge_cells, _, _ = _compiled_apply_global_edge_flips()(
        edges,
        edge_cells,
        empty_cells,
        empty_vertices,
        empty_cells,
        empty_vertices,
        states,
    )
    return adjusted_edges, adjusted_edge_cells


def reindex_closed_topology_for_refinement_numba(
    cells: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover deterministic first-seen edge order from existing topology."""
    edges, reordered_cell_edges, _, emitted_count = (
        _compiled_reindex_closed_topology_for_refinement()(
            cells,
            cell_edges,
            edge_cells,
        )
    )
    return edges, reordered_cell_edges, emitted_count


def reindex_closed_topology_with_adjacency_numba(
    cells: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Recover deterministic edge order together with aligned adjacency."""
    return _compiled_reindex_closed_topology_for_refinement()(
        cells,
        cell_edges,
        edge_cells,
    )


@lru_cache(maxsize=1)
def _compiled_reindex_closed_topology_for_refinement():
    from numba import njit, prange

    @njit(parallel=True)
    def recover(cells, cell_edges, edge_cells):
        scan_count = cells.shape[0] * 3
        block_size = 65536
        block_count = (scan_count + block_size - 1) // block_size
        counts = np.zeros(block_count, dtype=np.int64)
        for block in prange(block_count):
            start = block * block_size
            stop = min(start + block_size, scan_count)
            count = 0
            for position in range(start, stop):
                cell = position // 3
                slot = position - cell * 3
                edge = cell_edges[cell, slot]
                if cell == min(edge_cells[edge, 0], edge_cells[edge, 1]):
                    count += 1
            counts[block] = count

        offsets = np.empty(block_count, dtype=np.int64)
        emitted_count = 0
        for block in range(block_count):
            offsets[block] = emitted_count
            emitted_count += counts[block]

        edges = np.empty((edge_cells.shape[0], 2), dtype=np.int32)
        old_to_new = np.empty(edge_cells.shape[0], dtype=np.int32)
        for block in prange(block_count):
            start = block * block_size
            stop = min(start + block_size, scan_count)
            output_edge = offsets[block]
            for position in range(start, stop):
                cell = position // 3
                slot = position - cell * 3
                old_edge = cell_edges[cell, slot]
                if cell != min(edge_cells[old_edge, 0], edge_cells[old_edge, 1]):
                    continue
                old_to_new[old_edge] = output_edge
                if slot == 0:
                    edges[output_edge, 0] = cells[cell, 1]
                    edges[output_edge, 1] = cells[cell, 0]
                elif slot == 1:
                    edges[output_edge, 0] = cells[cell, 2]
                    edges[output_edge, 1] = cells[cell, 1]
                else:
                    edges[output_edge, 0] = cells[cell, 0]
                    edges[output_edge, 1] = cells[cell, 2]
                output_edge += 1

        reordered_cell_edges = np.empty_like(cell_edges)
        for cell in prange(cell_edges.shape[0]):
            for slot in range(3):
                reordered_cell_edges[cell, slot] = old_to_new[cell_edges[cell, slot]]
        reordered_edge_cells = np.empty_like(edge_cells)
        for old_edge in prange(edge_cells.shape[0]):
            new_edge = old_to_new[old_edge]
            reordered_edge_cells[new_edge, 0] = edge_cells[old_edge, 0]
            reordered_edge_cells[new_edge, 1] = edge_cells[old_edge, 1]
        return edges, reordered_cell_edges, reordered_edge_cells, emitted_count

    return recover


@lru_cache(maxsize=1)
def _compiled_global_edge_orientation_states():
    from numba import njit, prange

    @njit(parallel=True)
    def classify(
        vertices,
        cell_centers,
        edges,
        edge_cells,
        edge_centers,
        parent_normals,
        parent_edge_index,
    ):
        states = np.empty(edges.shape[0], dtype=np.int8)
        degenerate_count = 0
        flip_count = 0
        for edge in prange(edges.shape[0]):
            vertex_0 = edges[edge, 0]
            vertex_1 = edges[edge, 1]
            v0x = vertices[vertex_0, 0]
            v0y = vertices[vertex_0, 1]
            v0z = vertices[vertex_0, 2]
            v1x = vertices[vertex_1, 0]
            v1y = vertices[vertex_1, 1]
            v1z = vertices[vertex_1, 2]
            v0norm = np.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
            v1norm = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            v0x /= v0norm
            v0y /= v0norm
            v0z /= v0norm
            v1x /= v1norm
            v1y /= v1norm
            v1z /= v1norm

            adjacent_0 = edge_cells[edge, 0]
            adjacent_1 = edge_cells[edge, 1]
            c0x = cell_centers[adjacent_0, 0]
            c0y = cell_centers[adjacent_0, 1]
            c0z = cell_centers[adjacent_0, 2]
            c1x = cell_centers[adjacent_1, 0]
            c1y = cell_centers[adjacent_1, 1]
            c1z = cell_centers[adjacent_1, 2]
            c0norm = np.sqrt(c0x * c0x + c0y * c0y + c0z * c0z)
            c1norm = np.sqrt(c1x * c1x + c1y * c1y + c1z * c1z)
            c0x /= c0norm
            c0y /= c0norm
            c0z /= c0norm
            c1x /= c1norm
            c1y /= c1norm
            c1z /= c1norm

            edge_x = edge_centers[edge, 0]
            edge_y = edge_centers[edge, 1]
            edge_z = edge_centers[edge, 2]
            edge_norm = np.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
            edge_x /= edge_norm
            edge_y /= edge_norm
            edge_z /= edge_norm

            vertex_dx = v1x - v0x
            vertex_dy = v1y - v0y
            vertex_dz = v1z - v0z
            cell_dx = c1x - c0x
            cell_dy = c1y - c0y
            cell_dz = c1z - c0z
            cross_x = vertex_dy * cell_dz - vertex_dz * cell_dy
            cross_y = vertex_dz * cell_dx - vertex_dx * cell_dz
            cross_z = vertex_dx * cell_dy - vertex_dy * cell_dx
            outward = cross_x * edge_x + cross_y * edge_y + cross_z * edge_z
            direction_scale = np.sqrt(
                vertex_dx * vertex_dx
                + vertex_dy * vertex_dy
                + vertex_dz * vertex_dz
            ) * np.sqrt(
                cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz
            )
            if (
                not np.isfinite(outward)
                or np.abs(outward)
                <= 64.0 * np.finfo(np.float64).eps * direction_scale
            ):
                states[edge] = 0
                degenerate_count += 1
                continue
            orientation = 1 if outward > 0.0 else -1
            tangent_x = orientation * vertex_dx
            tangent_y = orientation * vertex_dy
            tangent_z = orientation * vertex_dz
            tangent_norm = np.sqrt(
                tangent_x * tangent_x
                + tangent_y * tangent_y
                + tangent_z * tangent_z
            )
            tangent_x /= tangent_norm
            tangent_y /= tangent_norm
            tangent_z /= tangent_norm
            normal_x = edge_y * tangent_z - edge_z * tangent_y
            normal_y = edge_z * tangent_x - edge_x * tangent_z
            normal_z = edge_x * tangent_y - edge_y * tangent_x
            normal_norm = np.sqrt(
                normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
            )
            normal_x /= normal_norm
            normal_y /= normal_norm
            normal_z /= normal_norm

            parent_edge = parent_edge_index[edge] - 1
            alignment = (
                normal_x * parent_normals[parent_edge, 0]
                + normal_y * parent_normals[parent_edge, 1]
                + normal_z * parent_normals[parent_edge, 2]
            )
            if alignment < 0.0:
                states[edge] = -1
                flip_count += 1
            else:
                states[edge] = 1
        return states, degenerate_count, flip_count

    return classify


@lru_cache(maxsize=1)
def _compiled_apply_global_edge_flips():
    from numba import njit, prange

    @njit(parallel=True)
    def apply(
        edges,
        edge_cells,
        orientation_of_normal,
        edge_orientation,
        cell_edges,
        edges_of_vertex,
        states,
    ):
        adjusted_edges = np.empty_like(edges)
        adjusted_edge_cells = np.empty_like(edge_cells)
        for edge in prange(edges.shape[0]):
            if states[edge] < 0:
                adjusted_edges[edge, 0] = edges[edge, 1]
                adjusted_edges[edge, 1] = edges[edge, 0]
                adjusted_edge_cells[edge, 0] = edge_cells[edge, 1]
                adjusted_edge_cells[edge, 1] = edge_cells[edge, 0]
            else:
                adjusted_edges[edge, 0] = edges[edge, 0]
                adjusted_edges[edge, 1] = edges[edge, 1]
                adjusted_edge_cells[edge, 0] = edge_cells[edge, 0]
                adjusted_edge_cells[edge, 1] = edge_cells[edge, 1]

        adjusted_orientation_of_normal = np.empty_like(orientation_of_normal)
        for cell in prange(cell_edges.shape[0]):
            for slot in range(cell_edges.shape[1]):
                value = orientation_of_normal[cell, slot]
                if states[cell_edges[cell, slot]] < 0:
                    value = -value
                adjusted_orientation_of_normal[cell, slot] = value

        adjusted_edge_orientation = np.empty_like(edge_orientation)
        for vertex in prange(edges_of_vertex.shape[0]):
            for slot in range(edges_of_vertex.shape[1]):
                edge_id = edges_of_vertex[vertex, slot]
                value = edge_orientation[vertex, slot]
                if edge_id > 0 and states[edge_id - 1] < 0:
                    value = -value
                adjusted_edge_orientation[vertex, slot] = value
        return (
            adjusted_edges,
            adjusted_edge_cells,
            adjusted_orientation_of_normal,
            adjusted_edge_orientation,
        )

    return apply


@lru_cache(maxsize=1)
def _compiled_sort_fixed_around_vertices():
    from numba import njit, prange

    @njit(parallel=True)
    def sort_rows(vertices, ids, points):
        sorted_ids = np.zeros_like(ids)
        row_angles = np.empty(ids.shape, dtype=np.float64)
        result = np.zeros_like(ids)
        width = ids.shape[1]
        for vertex in prange(vertices.shape[0]):
            origin_x = vertices[vertex, 0]
            origin_y = vertices[vertex, 1]
            origin_z = vertices[vertex, 2]
            origin_norm = np.sqrt(
                origin_x * origin_x
                + origin_y * origin_y
                + origin_z * origin_z
            )
            origin_x /= origin_norm
            origin_y /= origin_norm
            origin_z /= origin_norm
            if np.abs(origin_z) > 0.9:
                reference_x = 1.0
                reference_y = 0.0
                reference_z = 0.0
            else:
                reference_x = 0.0
                reference_y = 0.0
                reference_z = 1.0
            projection = (
                reference_x * origin_x
                + reference_y * origin_y
                + reference_z * origin_z
            )
            axis_1_x = reference_x - projection * origin_x
            axis_1_y = reference_y - projection * origin_y
            axis_1_z = reference_z - projection * origin_z
            axis_1_norm = np.sqrt(
                axis_1_x * axis_1_x
                + axis_1_y * axis_1_y
                + axis_1_z * axis_1_z
            )
            axis_1_x /= axis_1_norm
            axis_1_y /= axis_1_norm
            axis_1_z /= axis_1_norm
            axis_2_x = origin_y * axis_1_z - origin_z * axis_1_y
            axis_2_y = origin_z * axis_1_x - origin_x * axis_1_z
            axis_2_z = origin_x * axis_1_y - origin_y * axis_1_x

            count = 0
            for slot in range(width):
                one_based_id = ids[vertex, slot]
                if one_based_id <= 0:
                    continue
                point = one_based_id - 1
                point_x = points[point, 0]
                point_y = points[point, 1]
                point_z = points[point, 2]
                radial = (
                    point_x * origin_x
                    + point_y * origin_y
                    + point_z * origin_z
                )
                tangent_x = point_x - radial * origin_x
                tangent_y = point_y - radial * origin_y
                tangent_z = point_z - radial * origin_z
                angle = np.arctan2(
                    tangent_x * axis_2_x
                    + tangent_y * axis_2_y
                    + tangent_z * axis_2_z,
                    tangent_x * axis_1_x
                    + tangent_y * axis_1_y
                    + tangent_z * axis_1_z,
                )
                position = count
                while position > 0 and row_angles[vertex, position - 1] > angle:
                    row_angles[vertex, position] = row_angles[vertex, position - 1]
                    sorted_ids[vertex, position] = sorted_ids[vertex, position - 1]
                    position -= 1
                row_angles[vertex, position] = angle
                sorted_ids[vertex, position] = one_based_id
                count += 1
            if count == 0:
                continue
            minimum_position = 0
            for slot in range(1, count):
                if sorted_ids[vertex, slot] < sorted_ids[vertex, minimum_position]:
                    minimum_position = slot
            for slot in range(count):
                result[vertex, slot] = sorted_ids[
                    vertex,
                    (minimum_position + slot) % count,
                ]
        return result

    return sort_rows


@lru_cache(maxsize=1)
def _compiled_cell_centers():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(vertices, cells, radius):
        unit_vertices = np.empty_like(vertices)
        for vertex in prange(vertices.shape[0]):
            x = vertices[vertex, 0]
            y = vertices[vertex, 1]
            z = vertices[vertex, 2]
            norm = np.sqrt(x * x + y * y + z * z)
            unit_vertices[vertex, 0] = x / norm
            unit_vertices[vertex, 1] = y / norm
            unit_vertices[vertex, 2] = z / norm
        centers = np.empty((cells.shape[0], 3), dtype=np.float64)
        for cell in prange(cells.shape[0]):
            first = cells[cell, 0]
            second = cells[cell, 1]
            third = cells[cell, 2]
            ax = unit_vertices[first, 0]
            ay = unit_vertices[first, 1]
            az = unit_vertices[first, 2]
            bx = unit_vertices[second, 0]
            by = unit_vertices[second, 1]
            bz = unit_vertices[second, 2]
            cx = unit_vertices[third, 0]
            cy = unit_vertices[third, 1]
            cz = unit_vertices[third, 2]
            abx = ax - bx
            aby = ay - by
            abz = az - bz
            acx = ax - cx
            acy = ay - cy
            acz = az - cz
            center_x = aby * acz - abz * acy
            center_y = abz * acx - abx * acz
            center_z = abx * acy - aby * acx
            center_norm = np.sqrt(
                center_x * center_x + center_y * center_y + center_z * center_z
            )
            center_x /= center_norm
            center_y /= center_norm
            center_z /= center_norm
            reference_x = ax + bx + cx
            reference_y = ay + by + cy
            reference_z = az + bz + cz
            reference_norm = np.sqrt(
                reference_x * reference_x
                + reference_y * reference_y
                + reference_z * reference_z
            )
            reference_x /= reference_norm
            reference_y /= reference_norm
            reference_z /= reference_norm
            if (
                center_x * reference_x
                + center_y * reference_y
                + center_z * reference_z
                < 0.0
            ):
                center_x = -center_x
                center_y = -center_y
                center_z = -center_z
            centers[cell, 0] = center_x * radius
            centers[cell, 1] = center_y * radius
            centers[cell, 2] = center_z * radius
        return centers

    return compute


@lru_cache(maxsize=1)
def _compiled_edge_centers():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(vertices, edges, radius):
        unit_vertices = np.empty_like(vertices)
        for vertex in prange(vertices.shape[0]):
            x = vertices[vertex, 0]
            y = vertices[vertex, 1]
            z = vertices[vertex, 2]
            norm = np.sqrt(x * x + y * y + z * z)
            unit_vertices[vertex, 0] = x / norm
            unit_vertices[vertex, 1] = y / norm
            unit_vertices[vertex, 2] = z / norm
        centers = np.empty((edges.shape[0], 3), dtype=np.float64)
        for edge in prange(edges.shape[0]):
            first = edges[edge, 0]
            second = edges[edge, 1]
            x = 0.5 * (unit_vertices[first, 0] + unit_vertices[second, 0])
            y = 0.5 * (unit_vertices[first, 1] + unit_vertices[second, 1])
            z = 0.5 * (unit_vertices[first, 2] + unit_vertices[second, 2])
            norm = np.sqrt(x * x + y * y + z * z)
            centers[edge, 0] = x / norm * radius
            centers[edge, 1] = y / norm * radius
            centers[edge, 2] = z / norm * radius
        return centers

    return compute


@lru_cache(maxsize=1)
def _compiled_spherical_edge_metrics():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(vertices, cell_centers, edges, edge_cells, edge_centers, radius):
        edge_count = edges.shape[0]
        edge_lengths = np.empty(edge_count, dtype=np.float64)
        dual_edge_lengths = np.empty(edge_count, dtype=np.float64)
        edge_cell_distance = np.empty((edge_count, 2), dtype=np.float64)
        edge_system_orientation = np.empty(edge_count, dtype=np.int32)
        primal_normal = np.empty((edge_count, 3), dtype=np.float64)
        dual_normal = np.empty((edge_count, 3), dtype=np.float64)
        primal_u = np.empty(edge_count, dtype=np.float64)
        primal_v = np.empty(edge_count, dtype=np.float64)
        dual_u = np.empty(edge_count, dtype=np.float64)
        dual_v = np.empty(edge_count, dtype=np.float64)
        degenerate_count = 0

        for edge in prange(edge_count):
            vertex_0 = edges[edge, 0]
            vertex_1 = edges[edge, 1]
            v0x = vertices[vertex_0, 0]
            v0y = vertices[vertex_0, 1]
            v0z = vertices[vertex_0, 2]
            v1x = vertices[vertex_1, 0]
            v1y = vertices[vertex_1, 1]
            v1z = vertices[vertex_1, 2]
            v0norm = np.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
            v1norm = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            v0x /= v0norm
            v0y /= v0norm
            v0z /= v0norm
            v1x /= v1norm
            v1y /= v1norm
            v1z /= v1norm
            vertex_dot = v0x * v1x + v0y * v1y + v0z * v1z
            vertex_dot = min(1.0, max(-1.0, vertex_dot))
            edge_lengths[edge] = np.arccos(vertex_dot) * radius

            edge_x = edge_centers[edge, 0]
            edge_y = edge_centers[edge, 1]
            edge_z = edge_centers[edge, 2]
            edge_norm = np.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
            edge_x /= edge_norm
            edge_y /= edge_norm
            edge_z /= edge_norm

            adjacent_0 = edge_cells[edge, 0]
            adjacent_1 = edge_cells[edge, 1]
            c0x = cell_centers[adjacent_0, 0]
            c0y = cell_centers[adjacent_0, 1]
            c0z = cell_centers[adjacent_0, 2]
            c1x = cell_centers[adjacent_1, 0]
            c1y = cell_centers[adjacent_1, 1]
            c1z = cell_centers[adjacent_1, 2]
            c0norm = np.sqrt(c0x * c0x + c0y * c0y + c0z * c0z)
            c1norm = np.sqrt(c1x * c1x + c1y * c1y + c1z * c1z)
            c0x /= c0norm
            c0y /= c0norm
            c0z /= c0norm
            c1x /= c1norm
            c1y /= c1norm
            c1z /= c1norm
            cell_dot = c0x * c1x + c0y * c1y + c0z * c1z
            cell_dot = min(1.0, max(-1.0, cell_dot))
            dual_edge_lengths[edge] = np.arccos(cell_dot) * radius
            cell_dot_0 = c0x * edge_x + c0y * edge_y + c0z * edge_z
            cell_dot_1 = c1x * edge_x + c1y * edge_y + c1z * edge_z
            cell_dot_0 = min(1.0, max(-1.0, cell_dot_0))
            cell_dot_1 = min(1.0, max(-1.0, cell_dot_1))
            edge_cell_distance[edge, 0] = np.arccos(cell_dot_0) * radius
            edge_cell_distance[edge, 1] = np.arccos(cell_dot_1) * radius

            vertex_dx = v1x - v0x
            vertex_dy = v1y - v0y
            vertex_dz = v1z - v0z
            cell_dx = c1x - c0x
            cell_dy = c1y - c0y
            cell_dz = c1z - c0z
            cross_x = vertex_dy * cell_dz - vertex_dz * cell_dy
            cross_y = vertex_dz * cell_dx - vertex_dx * cell_dz
            cross_z = vertex_dx * cell_dy - vertex_dy * cell_dx
            outward = cross_x * edge_x + cross_y * edge_y + cross_z * edge_z
            direction_scale = np.sqrt(
                vertex_dx * vertex_dx
                + vertex_dy * vertex_dy
                + vertex_dz * vertex_dz
            ) * np.sqrt(
                cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz
            )
            if (
                not np.isfinite(outward)
                or np.abs(outward)
                <= 64.0 * np.finfo(np.float64).eps * direction_scale
            ):
                degenerate_count += 1
            orientation = 1 if outward > 0.0 else -1
            edge_system_orientation[edge] = orientation

            tangent_x = orientation * vertex_dx
            tangent_y = orientation * vertex_dy
            tangent_z = orientation * vertex_dz
            tangent_norm = np.sqrt(
                tangent_x * tangent_x
                + tangent_y * tangent_y
                + tangent_z * tangent_z
            )
            tangent_x /= tangent_norm
            tangent_y /= tangent_norm
            tangent_z /= tangent_norm
            normal_x = edge_y * tangent_z - edge_z * tangent_y
            normal_y = edge_z * tangent_x - edge_x * tangent_z
            normal_z = edge_x * tangent_y - edge_y * tangent_x
            normal_norm = np.sqrt(
                normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
            )
            normal_x /= normal_norm
            normal_y /= normal_norm
            normal_z /= normal_norm
            primal_normal[edge, 0] = normal_x
            primal_normal[edge, 1] = normal_y
            primal_normal[edge, 2] = normal_z
            dual_normal[edge, 0] = tangent_x
            dual_normal[edge, 1] = tangent_y
            dual_normal[edge, 2] = tangent_z

            lon = np.arctan2(edge_y, edge_x)
            lat = np.arcsin(min(1.0, max(-1.0, edge_z)))
            east_x = -np.sin(lon)
            east_y = np.cos(lon)
            north_x = -np.sin(lat) * np.cos(lon)
            north_y = -np.sin(lat) * np.sin(lon)
            north_z = np.cos(lat)
            primal_u[edge] = normal_x * east_x + normal_y * east_y
            primal_v[edge] = (
                normal_x * north_x + normal_y * north_y + normal_z * north_z
            )
            dual_u[edge] = tangent_x * east_x + tangent_y * east_y
            dual_v[edge] = (
                tangent_x * north_x + tangent_y * north_y + tangent_z * north_z
            )
        return (
            edge_lengths,
            dual_edge_lengths,
            edge_cell_distance,
            edge_system_orientation,
            primal_normal,
            dual_normal,
            primal_u,
            primal_v,
            dual_u,
            dual_v,
            degenerate_count,
        )

    return compute


@lru_cache(maxsize=1)
def _compiled_primal_normals():
    from numba import njit, prange

    @njit(inline="always")
    def unit_cell_center(vertices, cells, cell):
        x = 0.0
        y = 0.0
        z = 0.0
        for slot in range(3):
            vertex = cells[cell, slot]
            x += vertices[vertex, 0]
            y += vertices[vertex, 1]
            z += vertices[vertex, 2]
        norm = np.sqrt(x * x + y * y + z * z)
        return x / norm, y / norm, z / norm

    @njit(parallel=True)
    def compute(vertices, cells, edges, edge_cells):
        normals = np.empty((edges.shape[0], 3), dtype=np.float64)
        degenerate_count = 0
        for edge in prange(edges.shape[0]):
            vertex_0 = edges[edge, 0]
            vertex_1 = edges[edge, 1]
            v0x = vertices[vertex_0, 0]
            v0y = vertices[vertex_0, 1]
            v0z = vertices[vertex_0, 2]
            v1x = vertices[vertex_1, 0]
            v1y = vertices[vertex_1, 1]
            v1z = vertices[vertex_1, 2]
            v0norm = np.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
            v1norm = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            v0x /= v0norm
            v0y /= v0norm
            v0z /= v0norm
            v1x /= v1norm
            v1y /= v1norm
            v1z /= v1norm

            edge_x = v0x + v1x
            edge_y = v0y + v1y
            edge_z = v0z + v1z
            edge_norm = np.sqrt(
                edge_x * edge_x + edge_y * edge_y + edge_z * edge_z
            )
            edge_x /= edge_norm
            edge_y /= edge_norm
            edge_z /= edge_norm

            adjacent_0 = edge_cells[edge, 0]
            adjacent_1 = edge_cells[edge, 1]
            c0x, c0y, c0z = unit_cell_center(vertices, cells, adjacent_0)
            c1x, c1y, c1z = unit_cell_center(vertices, cells, adjacent_1)
            vertex_dx = v1x - v0x
            vertex_dy = v1y - v0y
            vertex_dz = v1z - v0z
            cell_dx = c1x - c0x
            cell_dy = c1y - c0y
            cell_dz = c1z - c0z
            cross_x = vertex_dy * cell_dz - vertex_dz * cell_dy
            cross_y = vertex_dz * cell_dx - vertex_dx * cell_dz
            cross_z = vertex_dx * cell_dy - vertex_dy * cell_dx
            outward = cross_x * edge_x + cross_y * edge_y + cross_z * edge_z
            direction_scale = np.sqrt(
                vertex_dx * vertex_dx
                + vertex_dy * vertex_dy
                + vertex_dz * vertex_dz
            ) * np.sqrt(
                cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz
            )
            if (
                not np.isfinite(outward)
                or np.abs(outward)
                <= 64.0 * np.finfo(np.float64).eps * direction_scale
            ):
                degenerate_count += 1
            orientation = 1.0 if outward > 0.0 else -1.0

            tangent_x = orientation * vertex_dx
            tangent_y = orientation * vertex_dy
            tangent_z = orientation * vertex_dz
            tangent_norm = np.sqrt(
                tangent_x * tangent_x
                + tangent_y * tangent_y
                + tangent_z * tangent_z
            )
            tangent_x /= tangent_norm
            tangent_y /= tangent_norm
            tangent_z /= tangent_norm
            normal_x = edge_y * tangent_z - edge_z * tangent_y
            normal_y = edge_z * tangent_x - edge_x * tangent_z
            normal_z = edge_x * tangent_y - edge_y * tangent_x
            normal_norm = np.sqrt(
                normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
            )
            normals[edge, 0] = normal_x / normal_norm
            normals[edge, 1] = normal_y / normal_norm
            normals[edge, 2] = normal_z / normal_norm
        return normals, degenerate_count

    return compute


@lru_cache(maxsize=1)
def _compiled_build_closed_edges():
    from numba import njit

    @njit
    def build(cells):
        cell_count = cells.shape[0]
        expected_edges = 3 * cell_count // 2
        edges = np.empty((expected_edges, 2), dtype=np.int32)
        cell_edges = np.empty((cell_count, 3), dtype=np.int32)
        edge_cells = np.full((expected_edges, 2), -1, dtype=np.int32)
        edge_lookup = {}
        stride = np.int64(np.max(cells)) + 1
        edge_count = 0
        for cell_index in range(cell_count):
            for local_index in range(3):
                if local_index == 0:
                    first = cells[cell_index, 1]
                    second = cells[cell_index, 0]
                elif local_index == 1:
                    first = cells[cell_index, 2]
                    second = cells[cell_index, 1]
                else:
                    first = cells[cell_index, 0]
                    second = cells[cell_index, 2]
                low = first if first < second else second
                high = second if first < second else first
                key = np.int64(low) * stride + np.int64(high)
                if key not in edge_lookup:
                    if edge_count >= expected_edges:
                        return edges, cell_edges, edge_cells, edge_count, 1
                    edge_index = edge_count
                    edge_lookup[key] = edge_index
                    edges[edge_index, 0] = first
                    edges[edge_index, 1] = second
                    edge_cells[edge_index, 0] = cell_index
                    edge_count += 1
                else:
                    edge_index = edge_lookup[key]
                    if edge_cells[edge_index, 1] < 0:
                        edge_cells[edge_index, 1] = cell_index
                    else:
                        return edges, cell_edges, edge_cells, edge_index, 2
                cell_edges[cell_index, local_index] = edge_index
        if edge_count != expected_edges:
            return edges, cell_edges, edge_cells, edge_count, 3
        for edge_index in range(edge_count):
            if edge_cells[edge_index, 1] < 0:
                return edges, cell_edges, edge_cells, edge_index, 3
        return edges, cell_edges, edge_cells, -1, 0

    return build


@lru_cache(maxsize=1)
def _compiled_fixed_incidence():
    from numba import njit

    @njit
    def fill(owners, values, row_count, width):
        incidence = np.zeros((row_count, width), dtype=np.int32)
        counts = np.zeros(row_count, dtype=np.int32)
        for item in range(owners.shape[0]):
            owner = owners[item]
            position = counts[owner]
            if position >= width:
                return incidence, owner, position + 1
            incidence[owner, position] = values[item]
            counts[owner] = position + 1
        return incidence, -1, 0

    return fill


@lru_cache(maxsize=1)
def _compiled_matching_edge_indices():
    from numba import njit, prange

    @njit(parallel=True)
    def match(source_edges, target_edges, target_edges_of_vertex):
        indices = np.full(source_edges.shape[0], -1, dtype=np.int64)
        for source_index in prange(source_edges.shape[0]):
            first = source_edges[source_index, 0]
            second = source_edges[source_index, 1]
            for slot in range(target_edges_of_vertex.shape[1]):
                target_index = target_edges_of_vertex[first, slot] - 1
                if target_index < 0:
                    continue
                target_first = target_edges[target_index, 0]
                target_second = target_edges[target_index, 1]
                if (
                    (target_first == first and target_second == second)
                    or (target_first == second and target_second == first)
                ):
                    indices[source_index] = target_index
                    break
        return indices

    return match


@lru_cache(maxsize=1)
def _compiled_cell_areas():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(vertices, cells, sphere_radius):
        unit_vertices = np.empty_like(vertices)
        for vertex in prange(vertices.shape[0]):
            x = vertices[vertex, 0]
            y = vertices[vertex, 1]
            z = vertices[vertex, 2]
            norm = np.sqrt(x * x + y * y + z * z)
            unit_vertices[vertex, 0] = x / norm
            unit_vertices[vertex, 1] = y / norm
            unit_vertices[vertex, 2] = z / norm

        areas = np.empty(cells.shape[0], dtype=np.float64)
        for cell in prange(cells.shape[0]):
            angle_sum = 0.0
            for index in range(3):
                a_index = cells[cell, index]
                b_index = cells[cell, (index + 1) % 3]
                c_index = cells[cell, (index + 2) % 3]
                ax = unit_vertices[a_index, 0]
                ay = unit_vertices[a_index, 1]
                az = unit_vertices[a_index, 2]
                bx = unit_vertices[b_index, 0]
                by = unit_vertices[b_index, 1]
                bz = unit_vertices[b_index, 2]
                cx = unit_vertices[c_index, 0]
                cy = unit_vertices[c_index, 1]
                cz = unit_vertices[c_index, 2]

                abx = ay * bz - az * by
                aby = az * bx - ax * bz
                abz = ax * by - ay * bx
                acx = ay * cz - az * cy
                acy = az * cx - ax * cz
                acz = ax * cy - ay * cx
                ab_norm = np.sqrt(abx * abx + aby * aby + abz * abz)
                ac_norm = np.sqrt(acx * acx + acy * acy + acz * acz)
                dot = (abx * acx + aby * acy + abz * acz) / (ab_norm * ac_norm)
                if dot < -1.0:
                    dot = -1.0
                elif dot > 1.0:
                    dot = 1.0
                angle_sum += np.arccos(dot)
            areas[cell] = (angle_sum - np.pi) * sphere_radius * sphere_radius
        return areas

    return compute


@lru_cache(maxsize=1)
def _compiled_dual_areas_from_edges():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(edges_of_vertex, edge_lengths, dual_edge_lengths):
        dual = np.zeros(edges_of_vertex.shape[0], dtype=np.float64)
        for vertex in prange(edges_of_vertex.shape[0]):
            total = 0.0
            for slot in range(edges_of_vertex.shape[1]):
                edge_index = edges_of_vertex[vertex, slot] - 1
                if edge_index >= 0:
                    total += 0.25 * edge_lengths[edge_index] * dual_edge_lengths[edge_index]
            dual[vertex] = total
        return dual

    return compute


@lru_cache(maxsize=1)
def _compiled_spring_relaxation():
    from numba import njit, prange

    @njit(parallel=True)
    def relax(vertices, edges, incident_edges, movable, len0, maxit):
        velocity = np.zeros_like(vertices)
        spring = np.zeros_like(vertices)
        all_movable = np.all(movable)
        fixed_vertices = (
            np.empty((0, vertices.shape[1]), dtype=vertices.dtype)
            if all_movable
            else vertices.copy()
        )
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        epsilon = np.finfo(np.float64).eps
        max_ekin = 0.0
        max_test = 0.0
        completed_iterations = 0

        for iteration in range(1, maxit + 1):
            if iteration <= 50:
                dt = 1.6e-2
            elif iteration <= 150:
                dt = 1.6e-2 * (1.0 + 0.04 * (iteration - 50))
            else:
                dt = 8.0e-2

            # Match the two np.add.at passes in the NumPy implementation:
            # all edge-start contributions first, then all edge-end ones.
            for vertex in prange(vertices.shape[0]):
                sx = 0.0
                sy = 0.0
                sz = 0.0
                if movable[vertex]:
                    for endpoint in range(2):
                        for slot in range(incident_edges.shape[1]):
                            edge_index = incident_edges[vertex, slot] - 1
                            if edge_index < 0 or edges[edge_index, endpoint] != vertex:
                                continue
                            start = edges[edge_index, 0]
                            end = edges[edge_index, 1]
                            dx = vertices[end, 0] - vertices[start, 0]
                            dy = vertices[end, 1] - vertices[start, 1]
                            dz = vertices[end, 2] - vertices[start, 2]
                            dot = (
                                vertices[start, 0] * dx
                                + vertices[start, 1] * dy
                                + vertices[start, 2] * dz
                                + 1.0
                            )
                            if dot < -1.0:
                                dot = -1.0
                            elif dot > 1.0:
                                dot = 1.0
                            denominator = 1.0 - dot
                            if denominator < epsilon:
                                denominator = epsilon
                            scale = (np.arccos(dot) - len0) / np.sqrt(denominator)
                            sign = 1.0 if endpoint == 0 else -1.0
                            sx += sign * dx * scale
                            sy += sign * dy * scale
                            sz += sign * dz * scale
                    spring[vertex, 0] = sx * inv_sqrt2
                    spring[vertex, 1] = sy * inv_sqrt2
                    spring[vertex, 2] = sz * inv_sqrt2
                else:
                    spring[vertex, 0] = 0.0
                    spring[vertex, 1] = 0.0
                    spring[vertex, 2] = 0.0

            ekin = 0.0
            test = 0.0
            for vertex in prange(vertices.shape[0]):
                if movable[vertex]:
                    x = vertices[vertex, 0] + dt * velocity[vertex, 0]
                    y = vertices[vertex, 1] + dt * velocity[vertex, 1]
                    z = vertices[vertex, 2] + dt * velocity[vertex, 2]
                    norm = np.sqrt(x * x + y * y + z * z)
                    x /= norm
                    y /= norm
                    z /= norm
                    vertices[vertex, 0] = x
                    vertices[vertex, 1] = y
                    vertices[vertex, 2] = z

                    vx = (1.0 - dt) * velocity[vertex, 0] + dt * spring[vertex, 0]
                    vy = (1.0 - dt) * velocity[vertex, 1] + dt * spring[vertex, 1]
                    vz = (1.0 - dt) * velocity[vertex, 2] + dt * spring[vertex, 2]
                    radial = vx * x + vy * y + vz * z
                    vx -= radial * x
                    vy -= radial * y
                    vz -= radial * z
                    velocity[vertex, 0] = vx
                    velocity[vertex, 1] = vy
                    velocity[vertex, 2] = vz
                    ekin += 0.5 * (vx * vx + vy * vy + vz * vz)
                    test += (
                        spring[vertex, 0] * spring[vertex, 0]
                        + spring[vertex, 1] * spring[vertex, 1]
                        + spring[vertex, 2] * spring[vertex, 2]
                    )
                elif not all_movable:
                    vertices[vertex, 0] = fixed_vertices[vertex, 0]
                    vertices[vertex, 1] = fixed_vertices[vertex, 1]
                    vertices[vertex, 2] = fixed_vertices[vertex, 2]
                    velocity[vertex, 0] = 0.0
                    velocity[vertex, 1] = 0.0
                    velocity[vertex, 2] = 0.0

            completed_iterations = iteration
            if ekin > max_ekin:
                max_ekin = ekin
            if test > max_test:
                max_test = test
            if iteration > 5 and test == max_test and max_test > 0.0:
                break
            if iteration > 5 and max_ekin > 0.0 and ekin < 0.001 * max_ekin:
                break
        return vertices, completed_iterations

    return relax


@lru_cache(maxsize=1)
def _compiled_mean_edge_angle():
    from numba import njit, prange

    @njit(parallel=True)
    def compute(vertices, edges):
        angle_sum = 0.0
        for edge in prange(edges.shape[0]):
            first = edges[edge, 0]
            second = edges[edge, 1]
            dot = (
                vertices[first, 0] * vertices[second, 0]
                + vertices[first, 1] * vertices[second, 1]
                + vertices[first, 2] * vertices[second, 2]
            )
            if dot < -1.0:
                dot = -1.0
            elif dot > 1.0:
                dot = 1.0
            angle_sum += np.arccos(dot)
        return angle_sum / edges.shape[0]

    return compute


@lru_cache(maxsize=1)
def _compiled_order_cells_by_edges():
    from numba import njit

    @njit
    def order(edges, cell_edges, edge_cells, edge_system_orientation):
        ordered_cells = np.empty_like(cell_edges)
        ordered_cell_edges = np.empty_like(cell_edges)
        for cell_index in range(cell_edges.shape[0]):
            first_edge = cell_edges[cell_index, 0]
            start_vertex = edges[first_edge, 0]
            next_vertex = edges[first_edge, 1]
            cell_orientation = 1
            if edge_cells[first_edge, 0] != cell_index:
                cell_orientation = -1
            if cell_orientation * edge_system_orientation[first_edge] > 0:
                swap = start_vertex
                start_vertex = next_vertex
                next_vertex = swap

            ordered_cell_edges[cell_index, 0] = first_edge
            ordered_cells[cell_index, 0] = start_vertex
            current_edge = first_edge
            current_vertex = next_vertex
            previous_edge = -1
            for output_index in range(1, 3):
                edge_index = -1
                following_vertex = -1
                for candidate_pos in range(3):
                    candidate = cell_edges[cell_index, candidate_pos]
                    if candidate == current_edge or candidate == previous_edge:
                        continue
                    first = edges[candidate, 0]
                    second = edges[candidate, 1]
                    if first == current_vertex:
                        edge_index = candidate
                        following_vertex = second
                        break
                    if second == current_vertex:
                        edge_index = candidate
                        following_vertex = first
                        break
                if edge_index < 0:
                    return ordered_cells, ordered_cell_edges, cell_index, 1
                ordered_cell_edges[cell_index, output_index] = edge_index
                ordered_cells[cell_index, output_index] = current_vertex
                previous_edge = current_edge
                current_edge = edge_index
                current_vertex = following_vertex
            if current_vertex != start_vertex or current_edge < 0:
                return ordered_cells, ordered_cell_edges, cell_index, 2
        return ordered_cells, ordered_cell_edges, -1, 0

    return order


@lru_cache(maxsize=1)
def _compiled_orient_closed_cells_inplace():
    from numba import njit, prange

    @njit(inline="always")
    def unit_vertex(vertices, vertex):
        x = vertices[vertex, 0]
        y = vertices[vertex, 1]
        z = vertices[vertex, 2]
        norm = np.sqrt(x * x + y * y + z * z)
        return x / norm, y / norm, z / norm

    @njit(inline="always")
    def unit_cell_center(vertices, cells, cell):
        x = 0.0
        y = 0.0
        z = 0.0
        for slot in range(3):
            vertex = cells[cell, slot]
            x += vertices[vertex, 0]
            y += vertices[vertex, 1]
            z += vertices[vertex, 2]
        norm = np.sqrt(x * x + y * y + z * z)
        return x / norm, y / norm, z / norm

    @njit(parallel=True)
    def orient(vertices, cells, edges, cell_edges, edge_cells):
        degenerate_count = 0
        missing_vertex_count = 0
        for cell in prange(cells.shape[0]):
            first_edge = cell_edges[cell, 0]
            first_vertex = edges[first_edge, 0]
            second_vertex = edges[first_edge, 1]
            v0x, v0y, v0z = unit_vertex(vertices, first_vertex)
            v1x, v1y, v1z = unit_vertex(vertices, second_vertex)

            adjacent_0 = edge_cells[first_edge, 0]
            adjacent_1 = edge_cells[first_edge, 1]
            c0x, c0y, c0z = unit_cell_center(vertices, cells, adjacent_0)
            c1x, c1y, c1z = unit_cell_center(vertices, cells, adjacent_1)

            edge_x = v0x + v1x
            edge_y = v0y + v1y
            edge_z = v0z + v1z
            edge_norm = np.sqrt(
                edge_x * edge_x + edge_y * edge_y + edge_z * edge_z
            )
            edge_x /= edge_norm
            edge_y /= edge_norm
            edge_z /= edge_norm

            vertex_dx = v1x - v0x
            vertex_dy = v1y - v0y
            vertex_dz = v1z - v0z
            cell_dx = c1x - c0x
            cell_dy = c1y - c0y
            cell_dz = c1z - c0z
            cross_x = vertex_dy * cell_dz - vertex_dz * cell_dy
            cross_y = vertex_dz * cell_dx - vertex_dx * cell_dz
            cross_z = vertex_dx * cell_dy - vertex_dy * cell_dx
            outward = cross_x * edge_x + cross_y * edge_y + cross_z * edge_z
            direction_scale = np.sqrt(
                vertex_dx * vertex_dx
                + vertex_dy * vertex_dy
                + vertex_dz * vertex_dz
            ) * np.sqrt(
                cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz
            )
            if (
                not np.isfinite(outward)
                or np.abs(outward)
                <= 64.0 * np.finfo(np.float64).eps * direction_scale
            ):
                degenerate_count += 1

            cell_orientation = 1 if adjacent_0 == cell else -1
            edge_orientation = 1 if outward > 0.0 else -1
            start_vertex = first_vertex
            if cell_orientation * edge_orientation > 0:
                start_vertex = second_vertex

            position = -1
            for slot in range(3):
                if cells[cell, slot] == start_vertex:
                    position = slot
                    break
            if position < 0:
                missing_vertex_count += 1
            elif position == 1:
                cell_0 = cells[cell, 0]
                cells[cell, 0] = cells[cell, 1]
                cells[cell, 1] = cells[cell, 2]
                cells[cell, 2] = cell_0
            elif position == 2:
                cell_0 = cells[cell, 0]
                cell_1 = cells[cell, 1]
                cells[cell, 0] = cells[cell, 2]
                cells[cell, 1] = cell_0
                cells[cell, 2] = cell_1
        return degenerate_count, missing_vertex_count

    return orient


@lru_cache(maxsize=1)
def _compiled_refine_closed_bisection():
    from numba import njit, prange

    @njit(inline="always")
    def local_vertex_position(cells, cell_index, vertex):
        if cells[cell_index, 0] == vertex:
            return 0
        if cells[cell_index, 1] == vertex:
            return 1
        return 2

    @njit(inline="always")
    def child_at_vertex(cells, cell_index, vertex):
        position = local_vertex_position(cells, cell_index, vertex)
        if position == 0:
            return 4 * cell_index + 2
        if position == 1:
            return 4 * cell_index + 3
        return 4 * cell_index + 1

    @njit(inline="always")
    def common_edge_vertex(edge_vertices, first_edge, second_edge):
        first_0 = edge_vertices[first_edge, 0]
        first_1 = edge_vertices[first_edge, 1]
        second_0 = edge_vertices[second_edge, 0]
        second_1 = edge_vertices[second_edge, 1]
        if first_0 == second_0 or first_0 == second_1:
            return first_0
        return first_1

    @njit(inline="always")
    def endpoint_slot(edge_vertices, edge_index, vertex):
        return 0 if edge_vertices[edge_index, 0] == vertex else 1

    @njit(parallel=True)
    def refine(
        cells,
        edge_vertices,
        cell_edges,
        edge_cells,
        old_vertex_count,
        edge_child_type_from_vertex_0,
        edge_child_type_from_vertex_1,
        edge_child_type_in_cell_opposite_vertex_0,
        edge_child_type_in_cell_opposite_vertex_1,
        edge_child_type_in_cell_opposite_vertex_2,
    ):
        old_edge_count = edge_vertices.shape[0]
        old_cell_count = cells.shape[0]
        new_cell_count = old_cell_count * 4
        new_edge_count = old_edge_count * 2 + old_cell_count * 3
        inner_edge_offset = old_edge_count * 2

        new_cells = np.empty((new_cell_count, 3), dtype=np.int32)
        new_cell_edges = np.empty((new_cell_count, 3), dtype=np.int32)
        new_edges = np.empty((new_edge_count, 2), dtype=np.int32)
        new_edge_cells = np.empty((new_edge_count, 2), dtype=np.int32)
        child_parent_edge_index = np.empty(new_edge_count, dtype=np.int32)
        child_edge_parent_type = np.empty(new_edge_count, dtype=np.int32)

        # Split parent edges. The two adjacent child cells are known from the
        # parent adjacency, so no global cell-edge incidence sort is needed.
        for edge_index in prange(old_edge_count):
            first = edge_vertices[edge_index, 0]
            second = edge_vertices[edge_index, 1]
            midpoint = old_vertex_count + edge_index
            first_split = 2 * edge_index
            second_split = first_split + 1
            new_edges[first_split, 0] = first
            new_edges[first_split, 1] = midpoint
            new_edges[second_split, 0] = midpoint
            new_edges[second_split, 1] = second

            first_cell = edge_cells[edge_index, 0]
            second_cell = edge_cells[edge_index, 1]
            first_child_a = child_at_vertex(cells, first_cell, first)
            first_child_b = child_at_vertex(cells, second_cell, first)
            second_child_a = child_at_vertex(cells, first_cell, second)
            second_child_b = child_at_vertex(cells, second_cell, second)
            new_edge_cells[first_split, 0] = min(first_child_a, first_child_b)
            new_edge_cells[first_split, 1] = max(first_child_a, first_child_b)
            new_edge_cells[second_split, 0] = min(second_child_a, second_child_b)
            new_edge_cells[second_split, 1] = max(second_child_a, second_child_b)

            child_parent_edge_index[first_split] = edge_index + 1
            child_parent_edge_index[second_split] = edge_index + 1
            child_edge_parent_type[first_split] = edge_child_type_from_vertex_0
            child_edge_parent_type[second_split] = edge_child_type_from_vertex_1

        # Fill the four child cells and three inner edges of each parent cell.
        # Parent-cell ranges are disjoint, making this loop race-free.
        for cell_index in prange(old_cell_count):
            parent_edge_0 = cell_edges[cell_index, 0]
            parent_edge_1 = cell_edges[cell_index, 1]
            parent_edge_2 = cell_edges[cell_index, 2]
            midpoint_0 = old_vertex_count + parent_edge_0
            midpoint_1 = old_vertex_count + parent_edge_1
            midpoint_2 = old_vertex_count + parent_edge_2
            center_cell = 4 * cell_index

            new_cells[center_cell, 0] = midpoint_0
            new_cells[center_cell, 1] = midpoint_1
            new_cells[center_cell, 2] = midpoint_2
            new_cell_edges[center_cell, 0] = inner_edge_offset + 3 * cell_index
            new_cell_edges[center_cell, 1] = inner_edge_offset + 3 * cell_index + 1
            new_cell_edges[center_cell, 2] = inner_edge_offset + 3 * cell_index + 2

            for pair_index in range(3):
                if pair_index == 0:
                    first_edge = parent_edge_0
                    second_edge = parent_edge_1
                elif pair_index == 1:
                    first_edge = parent_edge_1
                    second_edge = parent_edge_2
                else:
                    first_edge = parent_edge_2
                    second_edge = parent_edge_0

                common_vertex = common_edge_vertex(
                    edge_vertices,
                    first_edge,
                    second_edge,
                )
                vertex_pos = local_vertex_position(cells, cell_index, common_vertex)
                child_cell = child_at_vertex(cells, cell_index, common_vertex)
                first_midpoint = old_vertex_count + first_edge
                second_midpoint = old_vertex_count + second_edge
                inner_edge = inner_edge_offset + 3 * cell_index + vertex_pos

                new_cells[child_cell, 0] = first_midpoint
                new_cells[child_cell, 1] = common_vertex
                new_cells[child_cell, 2] = second_midpoint
                new_cell_edges[child_cell, 0] = (
                    2 * first_edge + endpoint_slot(edge_vertices, first_edge, common_vertex)
                )
                new_cell_edges[child_cell, 1] = (
                    2 * second_edge + endpoint_slot(edge_vertices, second_edge, common_vertex)
                )
                new_cell_edges[child_cell, 2] = inner_edge
                new_edges[inner_edge, 0] = first_midpoint
                new_edges[inner_edge, 1] = second_midpoint
                new_edge_cells[inner_edge, 0] = center_cell
                new_edge_cells[inner_edge, 1] = child_cell

                if vertex_pos == 0:
                    child_parent_edge_index[inner_edge] = parent_edge_1 + 1
                    child_edge_parent_type[inner_edge] = (
                        edge_child_type_in_cell_opposite_vertex_0
                    )
                elif vertex_pos == 1:
                    child_parent_edge_index[inner_edge] = parent_edge_2 + 1
                    child_edge_parent_type[inner_edge] = (
                        edge_child_type_in_cell_opposite_vertex_1
                    )
                else:
                    child_parent_edge_index[inner_edge] = parent_edge_0 + 1
                    child_edge_parent_type[inner_edge] = (
                        edge_child_type_in_cell_opposite_vertex_2
                    )

        # Match _edge_vertices_from_cell_edges exactly: orient every edge from
        # its occurrence in the second (larger) adjacent child cell.
        for edge_index in prange(new_edge_count):
            cell_index = new_edge_cells[edge_index, 1]
            local_position = 0
            for position in range(3):
                if new_cell_edges[cell_index, position] == edge_index:
                    local_position = position
                    break
            new_edges[edge_index, 0] = new_cells[cell_index, local_position]
            new_edges[edge_index, 1] = new_cells[cell_index, (local_position + 1) % 3]

        return (
            new_cells,
            new_cell_edges,
            new_edges,
            new_edge_cells,
            child_parent_edge_index,
            child_edge_parent_type,
        )

    return refine


@lru_cache(maxsize=1)
def _compiled_fill_bisection_children():
    from numba import njit

    @njit
    def common_edge_vertex(edge_vertices, first_edge, second_edge):
        first_0 = edge_vertices[first_edge, 0]
        first_1 = edge_vertices[first_edge, 1]
        second_0 = edge_vertices[second_edge, 0]
        second_1 = edge_vertices[second_edge, 1]
        if first_0 == second_0 or first_0 == second_1:
            return first_0, 0
        if first_1 == second_0 or first_1 == second_1:
            return first_1, 0
        return -1, 1

    @njit
    def local_vertex_position(cells, cell_index, vertex):
        if cells[cell_index, 0] == vertex:
            return 0, 0
        if cells[cell_index, 1] == vertex:
            return 1, 0
        if cells[cell_index, 2] == vertex:
            return 2, 0
        return -1, 2

    @njit
    def edge_endpoint_slot(edge_vertices, edge_index, vertex):
        if edge_vertices[edge_index, 0] == vertex:
            return 0, 0
        if edge_vertices[edge_index, 1] == vertex:
            return 1, 0
        return -1, 3

    @njit
    def fill(
        cells,
        edge_vertices,
        cell_edges,
        edge_midpoint_index,
        split_edge_index,
        inner_edge_index,
        edge_child_type_from_vertex_0,
        edge_child_type_from_vertex_1,
        edge_child_type_in_cell_opposite_vertex_0,
        edge_child_type_in_cell_opposite_vertex_1,
        edge_child_type_in_cell_opposite_vertex_2,
    ):
        old_edge_count = edge_vertices.shape[0]
        old_cell_count = cells.shape[0]
        new_cell_count = old_cell_count * 4
        new_edge_count = old_edge_count * 2 + old_cell_count * 3
        new_cells = np.empty((new_cell_count, 3), dtype=np.int32)
        raw_cell_edges = np.empty((new_cell_count, 3), dtype=np.int32)
        new_edges = np.empty((new_edge_count, 2), dtype=np.int32)
        child_parent_edge_index = np.empty(new_edge_count, dtype=np.int32)
        child_edge_parent_type = np.empty(new_edge_count, dtype=np.int32)

        for edge_index in range(old_edge_count):
            first = edge_vertices[edge_index, 0]
            second = edge_vertices[edge_index, 1]
            midpoint = edge_midpoint_index[edge_index]
            first_split = split_edge_index[edge_index, 0]
            second_split = split_edge_index[edge_index, 1]
            new_edges[first_split, 0] = first
            new_edges[first_split, 1] = midpoint
            new_edges[second_split, 0] = midpoint
            new_edges[second_split, 1] = second
            child_parent_edge_index[first_split] = edge_index + 1
            child_parent_edge_index[second_split] = edge_index + 1
            child_edge_parent_type[first_split] = edge_child_type_from_vertex_0
            child_edge_parent_type[second_split] = edge_child_type_from_vertex_1

        for cell_index in range(old_cell_count):
            parent_edge_0 = cell_edges[cell_index, 0]
            parent_edge_1 = cell_edges[cell_index, 1]
            parent_edge_2 = cell_edges[cell_index, 2]
            midpoint_0 = edge_midpoint_index[parent_edge_0]
            midpoint_1 = edge_midpoint_index[parent_edge_1]
            midpoint_2 = edge_midpoint_index[parent_edge_2]
            center_cell = 4 * cell_index
            new_cells[center_cell, 0] = midpoint_0
            new_cells[center_cell, 1] = midpoint_1
            new_cells[center_cell, 2] = midpoint_2
            raw_cell_edges[center_cell, 0] = inner_edge_index[cell_index, 0]
            raw_cell_edges[center_cell, 1] = inner_edge_index[cell_index, 1]
            raw_cell_edges[center_cell, 2] = inner_edge_index[cell_index, 2]

            for pair_index in range(3):
                if pair_index == 0:
                    first_edge = parent_edge_0
                    second_edge = parent_edge_1
                elif pair_index == 1:
                    first_edge = parent_edge_1
                    second_edge = parent_edge_2
                else:
                    first_edge = parent_edge_2
                    second_edge = parent_edge_0

                common_vertex, failure = common_edge_vertex(edge_vertices, first_edge, second_edge)
                if failure != 0:
                    return (
                        new_cells,
                        raw_cell_edges,
                        new_edges,
                        child_parent_edge_index,
                        child_edge_parent_type,
                        cell_index,
                        failure,
                    )
                vertex_pos, failure = local_vertex_position(cells, cell_index, common_vertex)
                if failure != 0:
                    return (
                        new_cells,
                        raw_cell_edges,
                        new_edges,
                        child_parent_edge_index,
                        child_edge_parent_type,
                        cell_index,
                        failure,
                    )
                if vertex_pos == 0:
                    child_cell = center_cell + 2
                elif vertex_pos == 1:
                    child_cell = center_cell + 3
                else:
                    child_cell = center_cell + 1
                first_split_slot, failure = edge_endpoint_slot(edge_vertices, first_edge, common_vertex)
                if failure != 0:
                    return (
                        new_cells,
                        raw_cell_edges,
                        new_edges,
                        child_parent_edge_index,
                        child_edge_parent_type,
                        cell_index,
                        failure,
                    )
                second_split_slot, failure = edge_endpoint_slot(edge_vertices, second_edge, common_vertex)
                if failure != 0:
                    return (
                        new_cells,
                        raw_cell_edges,
                        new_edges,
                        child_parent_edge_index,
                        child_edge_parent_type,
                        cell_index,
                        failure,
                    )
                first_midpoint = edge_midpoint_index[first_edge]
                second_midpoint = edge_midpoint_index[second_edge]
                new_cells[child_cell, 0] = first_midpoint
                new_cells[child_cell, 1] = common_vertex
                new_cells[child_cell, 2] = second_midpoint
                raw_cell_edges[child_cell, 0] = split_edge_index[first_edge, first_split_slot]
                raw_cell_edges[child_cell, 1] = split_edge_index[second_edge, second_split_slot]
                raw_cell_edges[child_cell, 2] = inner_edge_index[cell_index, vertex_pos]
                new_edges[inner_edge_index[cell_index, vertex_pos], 0] = first_midpoint
                new_edges[inner_edge_index[cell_index, vertex_pos], 1] = second_midpoint

                if vertex_pos == 0:
                    child_parent_edge_index[inner_edge_index[cell_index, vertex_pos]] = parent_edge_1 + 1
                    child_edge_parent_type[inner_edge_index[cell_index, vertex_pos]] = (
                        edge_child_type_in_cell_opposite_vertex_0
                    )
                elif vertex_pos == 1:
                    child_parent_edge_index[inner_edge_index[cell_index, vertex_pos]] = parent_edge_2 + 1
                    child_edge_parent_type[inner_edge_index[cell_index, vertex_pos]] = (
                        edge_child_type_in_cell_opposite_vertex_1
                    )
                else:
                    child_parent_edge_index[inner_edge_index[cell_index, vertex_pos]] = parent_edge_0 + 1
                    child_edge_parent_type[inner_edge_index[cell_index, vertex_pos]] = (
                        edge_child_type_in_cell_opposite_vertex_2
                    )

        return (
            new_cells,
            raw_cell_edges,
            new_edges,
            child_parent_edge_index,
            child_edge_parent_type,
            -1,
            0,
        )

    return fill


@lru_cache(maxsize=1)
def _compiled_lookup_width2():
    from numba import njit

    @njit
    def lookup(signature_keys, parent_index_values, type_values, query_keys):
        out_parent = np.empty(query_keys.shape[0], dtype=np.int32)
        out_type = np.empty(query_keys.shape[0], dtype=np.int32)
        for row in range(query_keys.shape[0]):
            q0 = query_keys[row, 0]
            q1 = query_keys[row, 1]
            low = 0
            high = signature_keys.shape[0]
            found = -1
            while low < high:
                mid = (low + high) // 2
                k0 = signature_keys[mid, 0]
                k1 = signature_keys[mid, 1]
                if k0 < q0 or (k0 == q0 and k1 < q1):
                    low = mid + 1
                elif k0 > q0 or (k0 == q0 and k1 > q1):
                    high = mid
                else:
                    found = mid
                    break
            if found < 0:
                out_parent[row] = 0
                out_type[row] = 0
            else:
                out_parent[row] = parent_index_values[found]
                out_type[row] = type_values[found]
        return out_parent, out_type

    return lookup


@lru_cache(maxsize=1)
def _compiled_lookup_width3():
    from numba import njit

    @njit
    def lookup(signature_keys, parent_index_values, type_values, query_keys):
        out_parent = np.empty(query_keys.shape[0], dtype=np.int32)
        out_type = np.empty(query_keys.shape[0], dtype=np.int32)
        for row in range(query_keys.shape[0]):
            q0 = query_keys[row, 0]
            q1 = query_keys[row, 1]
            q2 = query_keys[row, 2]
            low = 0
            high = signature_keys.shape[0]
            found = -1
            while low < high:
                mid = (low + high) // 2
                k0 = signature_keys[mid, 0]
                k1 = signature_keys[mid, 1]
                k2 = signature_keys[mid, 2]
                if k0 < q0 or (k0 == q0 and (k1 < q1 or (k1 == q1 and k2 < q2))):
                    low = mid + 1
                elif k0 > q0 or (k0 == q0 and (k1 > q1 or (k1 == q1 and k2 > q2))):
                    high = mid
                else:
                    found = mid
                    break
            if found < 0:
                out_parent[row] = 0
                out_type[row] = 0
            else:
                out_parent[row] = parent_index_values[found]
                out_type[row] = type_values[found]
        return out_parent, out_type

    return lookup
