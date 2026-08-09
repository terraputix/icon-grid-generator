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


@lru_cache(maxsize=1)
def _compiled_spring_relaxation():
    from numba import njit, prange

    @njit(parallel=True)
    def relax(vertices, edges, incident_edges, movable, len0, maxit):
        velocity = np.zeros_like(vertices)
        spring = np.zeros_like(vertices)
        fixed_vertices = vertices.copy()
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
                else:
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
