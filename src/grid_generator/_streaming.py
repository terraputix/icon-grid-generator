"""Memory-bounded generation and NetCDF export for very large global grids."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np

from . import _accelerated
from ._optimization import _GlobalOptimizationOptions, _spring_relaxed_vertices
from ._types import BisectionProvenance


DEFAULT_IN_MEMORY_BASE_CELLS = 1_310_720
DEFAULT_STREAM_CHUNK_SIZE = 1_000_000

_EDGE_METRIC_FIELDS = frozenset(
    {
        "edge_length",
        "dual_edge_length",
        "edge_cell_distance",
        "edge_vert_distance",
        "edgequad_area",
        "edge_system_orientation",
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
        "dual_area",
        "dual_area_p",
    }
)


@dataclass
class _CompactGlobalGrid:
    """Only the global arrays required by refinement and spring relaxation."""

    spec: Any
    options: Any
    vertices: np.ndarray
    cells: np.ndarray
    edges: np.ndarray
    cell_edges: np.ndarray
    edge_cells: np.ndarray
    edges_of_vertex: np.ndarray
    edge_primal_normal_cartesian: np.ndarray | None
    provenance: BisectionProvenance | None = None

    @property
    def dims(self) -> dict[str, int]:
        return {
            "cell": int(self.cells.shape[0]),
            "edge": int(self.edges.shape[0]),
            "vertex": int(self.vertices.shape[0]),
        }

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def metadata(self) -> dict[str, int]:
        return {"grid_geometry": 1}

    @property
    def icon_connectivity(self) -> dict[str, np.ndarray]:
        return {"v2e": self.edges_of_vertex}

    @property
    def incident_edges_sorted(self) -> bool:
        return True


@dataclass(frozen=True)
class _CheckpointStore:
    root: Path
    configuration_uuid: str

    def load(self, spec: Any, options: Any) -> _CompactGlobalGrid | None:
        stage = self.root / spec.name
        manifest_path = stage / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        checkpoint_format = manifest.get("format")
        if (
            checkpoint_format not in (1, 2)
            or manifest.get("configuration_uuid") != self.configuration_uuid
            or manifest.get("stage") != spec.name
        ):
            return None
        snapshot = None
        if checkpoint_format == 2:
            snapshot = manifest.get("snapshot")
            if (
                not isinstance(snapshot, str)
                or len(snapshot) != 32
                or any(character not in "0123456789abcdef" for character in snapshot)
            ):
                return None

        def array(name: str) -> np.ndarray:
            filename = f"{snapshot}.{name}.npy" if snapshot is not None else f"{name}.npy"
            return np.load(stage / filename, mmap_mode="r", allow_pickle=False)

        try:
            provenance = None
            if manifest.get("has_provenance"):
                provenance = BisectionProvenance(
                    cells=np.empty((0, 3), dtype=np.int32),
                    edges=np.empty((0, 2), dtype=np.int32),
                    cell_edges=np.empty((0, 3), dtype=np.int32),
                    parent_vertex_index=array("parent_vertex_index"),
                    parent_cell_index=array("parent_cell_index"),
                    parent_cell_type=array("parent_cell_type"),
                    child_parent_edge_index=array("child_parent_edge_index"),
                    child_edge_parent_type=array("child_edge_parent_type"),
                )
            normal = (
                array("edge_primal_normal_cartesian")
                if manifest.get("has_parent_normals")
                else None
            )
            compact = _CompactGlobalGrid(
                spec=spec,
                options=options,
                vertices=array("vertices"),
                cells=array("cells"),
                edges=array("edges"),
                cell_edges=array("cell_edges"),
                edge_cells=array("edge_cells"),
                edges_of_vertex=array("edges_of_vertex"),
                edge_primal_normal_cartesian=normal,
                provenance=provenance,
            )
        except (OSError, ValueError):
            return None
        if compact.dims != {
            "cell": spec.expected_cells,
            "edge": spec.expected_edges,
            "vertex": spec.expected_vertices,
        }:
            raise RuntimeError(f"checkpoint {stage} has inconsistent dimensions")
        return compact

    def save(self, grid: _CompactGlobalGrid) -> None:
        stage = self.root / grid.name
        stage.mkdir(parents=True, exist_ok=True)
        snapshot = uuid.uuid4().hex
        arrays = {
            "vertices": grid.vertices,
            "cells": grid.cells,
            "edges": grid.edges,
            "cell_edges": grid.cell_edges,
            "edge_cells": grid.edge_cells,
            "edges_of_vertex": grid.edges_of_vertex,
        }
        if grid.edge_primal_normal_cartesian is not None:
            arrays["edge_primal_normal_cartesian"] = grid.edge_primal_normal_cartesian
        if grid.provenance is not None:
            provenance_arrays = {
                "parent_vertex_index": grid.provenance.parent_vertex_index,
                "parent_cell_index": grid.provenance.parent_cell_index,
                "parent_cell_type": grid.provenance.parent_cell_type,
                "child_parent_edge_index": grid.provenance.child_parent_edge_index,
                "child_edge_parent_type": grid.provenance.child_edge_parent_type,
            }
            for name, value in provenance_arrays.items():
                if value is None:
                    raise RuntimeError("cannot checkpoint incomplete final provenance")
                arrays[name] = value
        for name, value in arrays.items():
            temporary = stage / f".{snapshot}.{name}.npy.partial"
            with temporary.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            os.replace(temporary, stage / f"{snapshot}.{name}.npy")
        manifest = {
            "format": 2,
            "configuration_uuid": self.configuration_uuid,
            "stage": grid.name,
            "snapshot": snapshot,
            "has_parent_normals": grid.edge_primal_normal_cartesian is not None,
            "has_provenance": grid.provenance is not None,
        }
        temporary_manifest = stage / f".manifest.{snapshot}.json.partial"
        temporary_manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        os.replace(temporary_manifest, stage / "manifest.json")
        self._remove_unreferenced_snapshot_files(stage, snapshot)

    @staticmethod
    def _remove_unreferenced_snapshot_files(stage: Path, snapshot: str) -> None:
        retained_prefix = f"{snapshot}."
        for path in (*stage.glob("*.npy"), *stage.glob(".*.partial")):
            if path.name.startswith(retained_prefix):
                continue
            try:
                path.unlink()
            except OSError:
                # Cleanup is best effort; the manifest already names the only
                # snapshot that can be loaded.
                pass


def generate_global_grid_to_netcdf(
    spec: Any,
    options: Any,
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
    work_dir: str | Path | None = None,
    resume: bool = True,
    selected_fields: frozenset[str],
) -> Path:
    """Generate a global grid and write it without retaining export-only fields."""
    output_path = Path(path)
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    base_bisection = _in_memory_base_bisection(spec)
    if spec.expected_cells <= DEFAULT_IN_MEMORY_BASE_CELLS:
        from . import grid_generator as gg

        return gg.generate_grid(spec, options).to_netcdf(
            path,
            fields=selected_fields,
        )
    if base_bisection is None:
        raise ValueError(
            "export-first generation cannot bound the initial root stage; "
            "use a smaller-root decomposition only if a different grid identity, "
            "refinement hierarchy, and potentially optimized geometry are "
            "acceptable, or use the in-memory generate_grid workflow with "
            "sufficient memory"
        )
    if not _accelerated.should_use_numba(options.accelerator, spec.expected_cells):
        raise ModuleNotFoundError(
            "streamed high-resolution generation requires installing the "
            "'accelerate' extra and using accelerator='auto' or 'numba'"
        )
    from . import grid_generator as gg

    checkpoint_root = (
        Path(work_dir)
        if work_dir is not None
        else output_path.parent / f".{output_path.name}.work"
    )
    configuration_spec = gg.GlobalGridSpec(root=spec.root, bisections=0)
    checkpoint_store = _CheckpointStore(
        checkpoint_root,
        gg._spec_uuid(configuration_spec, options),
    )
    compact = _generate_compact_global_grid(
        spec,
        options,
        checkpoint_store=checkpoint_store,
        resume=resume,
    )
    return _write_compact_global_grid(
        compact,
        output_path,
        chunk_size=chunk_size,
        selected_fields=selected_fields,
    )


def _in_memory_base_bisection(spec: Any) -> int | None:
    """Return the largest stage whose complete grid fits the base-cell budget."""
    for bisection in range(spec.bisections, -1, -1):
        frequency = spec.root * 2**bisection
        if 20 * frequency**2 <= DEFAULT_IN_MEMORY_BASE_CELLS:
            return bisection
    return None


def _generate_compact_global_grid(
    spec: Any,
    options: Any,
    *,
    checkpoint_store: _CheckpointStore | None = None,
    resume: bool = True,
) -> _CompactGlobalGrid:
    """Generate staged global topology while shedding derived parent fields."""
    from . import grid_generator as gg

    base_bisection = _in_memory_base_bisection(spec)
    if base_bisection is None:
        raise ValueError("global root stage exceeds the in-memory base-cell budget")
    compact = None
    if checkpoint_store is not None and resume:
        for bisection in range(spec.bisections, base_bisection - 1, -1):
            candidate_spec = gg.GlobalGridSpec(root=spec.root, bisections=bisection)
            compact = checkpoint_store.load(candidate_spec, options)
            if (
                compact is not None
                and bisection == spec.bisections
                and compact.provenance is None
            ):
                # Intermediate checkpoints omit export-only parent mappings. They
                # remain valid refinement inputs but cannot serve as a final grid.
                compact = None
            if compact is not None:
                break
    if compact is None:
        base_spec = gg.GlobalGridSpec(root=spec.root, bisections=base_bisection)
        base = gg.generate_grid(base_spec, options)
        compact = _compact_from_icon_grid(base)
        del base
        if checkpoint_store is not None:
            checkpoint_store.save(compact)
    for bisection in range(compact.spec.bisections + 1, spec.bisections + 1):
        child_spec = gg.GlobalGridSpec(root=spec.root, bisections=bisection)
        compact = _refine_compact_global_grid(
            compact,
            child_spec,
            terminal=bisection == spec.bisections,
        )
        if checkpoint_store is not None:
            checkpoint_store.save(compact)
    return compact


def _compact_from_icon_grid(grid: Any) -> _CompactGlobalGrid:
    return _CompactGlobalGrid(
        spec=grid.spec,
        options=grid.options,
        vertices=grid.vertices,
        cells=grid.cells,
        edges=grid.edges,
        cell_edges=grid.cell_edges,
        edge_cells=grid.edge_cells,
        edges_of_vertex=grid.icon_connectivity["v2e"],
        edge_primal_normal_cartesian=grid.geometry["edge_primal_normal_cartesian"],
    )


def _refine_compact_global_grid(
    parent: _CompactGlobalGrid,
    spec: Any,
    *,
    terminal: bool,
) -> _CompactGlobalGrid:
    from . import grid_generator as gg
    from ._global import (
        GLOBAL_RELAXATION_LONG_ITER_CELL_THRESHOLD,
        _matching_edge_indices_by_vertices,
    )

    options = parent.options
    parent_cell_count = parent.cells.shape[0]
    stage_iterations = options.global_optimization.iterations
    if parent_cell_count < GLOBAL_RELAXATION_LONG_ITER_CELL_THRESHOLD:
        stage_iterations *= 10
    vertices, cells, provenance = gg._refine_triangles_bisection_with_provenance(
        parent.vertices,
        parent.cells,
        options.accelerator,
        edge_vertices=parent.edges,
        cell_edges=parent.cell_edges,
        edge_cells=parent.edge_cells,
    )
    vertices *= options.radius
    cell_centers = gg._cell_centers(
        vertices,
        cells,
        options.radius,
        options.accelerator,
    )
    if (
        provenance.child_edges is None
        or provenance.child_cell_edges is None
        or provenance.child_edge_cells is None
        or provenance.child_parent_edge_index is None
    ):
        raise RuntimeError(
            "compact global generation requires complete bisection provenance"
        )
    edges = provenance.child_edges
    cell_edges = provenance.child_cell_edges
    edge_cells = provenance.child_edge_cells
    edge_centers = gg._edge_centers(
        vertices,
        edges,
        options.radius,
        options.accelerator,
    )

    if parent.edge_primal_normal_cartesian is None:
        raise RuntimeError("compact parent is missing edge-normal geometry")
    parent_edge_map = _matching_edge_indices_by_vertices(
        provenance.edges,
        parent.edges,
        accelerator=options.accelerator,
        target_edges_of_vertex=parent.edges_of_vertex,
    )
    parent_normals = parent.edge_primal_normal_cartesian[parent_edge_map]
    states, degenerate_count, flip_count = (
        _accelerated.global_edge_orientation_states_numba(
            vertices,
            cell_centers,
            edges,
            edge_cells,
            edge_centers,
            parent_normals,
            provenance.child_parent_edge_index,
        )
    )
    del parent_normals, parent_edge_map
    if degenerate_count:
        raise RuntimeError(
            "edge system orientation is degenerate for at least one edge"
        )
    if flip_count:
        edges, edge_cells = _accelerated.apply_edge_flips_numba(
            edges,
            edge_cells,
            states,
        )
    del states

    export_provenance = None
    if terminal:
        export_provenance = BisectionProvenance(
            cells=np.empty((0, 3), dtype=np.int32),
            edges=np.empty((0, 2), dtype=np.int32),
            cell_edges=np.empty((0, 3), dtype=np.int32),
            parent_vertex_index=provenance.parent_vertex_index,
            parent_cell_index=provenance.parent_cell_index,
            parent_cell_type=provenance.parent_cell_type,
            child_parent_edge_index=provenance.child_parent_edge_index,
            child_edge_parent_type=provenance.child_edge_parent_type,
        )
    del cell_centers, edge_centers, provenance
    _release_compact_arrays(parent)
    del parent

    edges_of_vertex = _compact_edges_of_vertex(
        edges, vertices.shape[0], options.accelerator
    )
    spring_grid = _CompactGlobalGrid(
        spec=spec,
        options=options,
        vertices=vertices,
        cells=cells,
        edges=edges,
        cell_edges=cell_edges,
        edge_cells=edge_cells,
        edges_of_vertex=edges_of_vertex,
        edge_primal_normal_cartesian=None,
        provenance=export_provenance,
    )
    if stage_iterations:
        vertices = _spring_relaxed_vertices(
            spring_grid,
            _GlobalOptimizationOptions(method="spring", iterations=stage_iterations),
        )
        spring_grid.vertices = vertices

    # Keep parent normals when another int32-addressable bisection is possible,
    # including terminal checkpoints that may be reused by a later request.
    needs_parent_normals = (
        not terminal or spec.expected_edges <= np.iinfo(np.int32).max // 4
    )
    if not needs_parent_normals:
        return spring_grid

    spring_grid.edge_primal_normal_cartesian = _compute_parent_primal_normals(
        spring_grid
    )
    return spring_grid


def _release_compact_arrays(grid: _CompactGlobalGrid) -> None:
    """Release an owned parent after its child no longer needs its arrays."""
    grid.vertices = np.empty((0, 3), dtype=np.float64)
    grid.cells = np.empty((0, 3), dtype=np.int32)
    grid.edges = np.empty((0, 2), dtype=np.int32)
    grid.cell_edges = np.empty((0, 3), dtype=np.int32)
    grid.edge_cells = np.empty((0, 2), dtype=np.int32)
    grid.edges_of_vertex = np.empty((0, 6), dtype=np.int32)
    grid.edge_primal_normal_cartesian = None
    grid.provenance = None


def _compute_parent_primal_normals(grid: _CompactGlobalGrid) -> np.ndarray:
    """Compute only the geometry retained for orienting the next stage."""
    return _accelerated.primal_normals_numba(
        grid.vertices,
        grid.cells,
        grid.edges,
        grid.edge_cells,
    )


def _compact_edges_of_vertex(
    edges: np.ndarray,
    vertex_count: int,
    accelerator: str,
) -> np.ndarray:
    """Build one-based incident edge IDs without other connectivity tables."""
    if _accelerated.should_use_numba(accelerator, edges.shape[0]):
        incidence, oversized_vertex, oversized_count = (
            _accelerated.edge_incidence_numba(edges, vertex_count, 6)
        )
        if oversized_vertex >= 0:
            raise RuntimeError(
                f"vertex {oversized_vertex} has {oversized_count} incident "
                "edges, expected at most 6"
            )
        incidence.sort(axis=1)
        return incidence

    from . import grid_generator as gg

    edge_ids = np.arange(1, edges.shape[0] + 1, dtype=np.int32)
    owners = np.concatenate((edges[:, 0], edges[:, 1]))
    values = np.concatenate((edge_ids, edge_ids))
    incidence = gg._fixed_incidence(owners, values, vertex_count, 6, accelerator)
    incidence.sort(axis=1)
    return incidence


def _write_compact_global_grid(
    grid: _CompactGlobalGrid,
    path: Path,
    *,
    chunk_size: int,
    selected_fields: frozenset[str],
) -> Path:
    from . import _netcdf
    from . import grid_generator as gg

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        partial.unlink()

    metadata = gg._metadata(grid.spec, grid.options)
    facade = SimpleNamespace(
        dims=grid.dims,
        metadata=metadata,
        name=grid.name,
        options=grid.options,
    )
    try:
        with nc.Dataset(partial, "w", format="NETCDF4") as dataset:
            _netcdf._write_icon_dimensions(dataset, facade)
            _netcdf._write_icon_attributes(dataset, facade, path)
            variables = _create_icon_variables(dataset, selected_fields)

            needs_cell_centers = bool(
                selected_fields
                & (
                    _EDGE_METRIC_FIELDS
                    | {
                        "clon",
                        "clat",
                        "lon_cell_centre",
                        "lat_cell_centre",
                        "elon_vertices",
                        "elat_vertices",
                        "cells_of_vertex",
                        "cell_circumcenter_cartesian_x",
                        "cell_circumcenter_cartesian_y",
                        "cell_circumcenter_cartesian_z",
                    }
                )
            )
            needs_edge_centers = bool(
                selected_fields
                & (
                    _EDGE_METRIC_FIELDS
                    | {
                        "elon",
                        "elat",
                        "lon_edge_centre",
                        "lat_edge_centre",
                        "elon_vertices",
                        "elat_vertices",
                        "edges_of_vertex",
                        "edge_orientation",
                        "edge_middle_cartesian_x",
                        "edge_middle_cartesian_y",
                        "edge_middle_cartesian_z",
                        "edge_dual_middle_cartesian_x",
                        "edge_dual_middle_cartesian_y",
                        "edge_dual_middle_cartesian_z",
                    }
                )
            )
            cell_centers = (
                gg._cell_centers(
                    grid.vertices,
                    grid.cells,
                    grid.options.radius,
                    grid.options.accelerator,
                )
                if needs_cell_centers
                else None
            )
            edge_centers = (
                gg._edge_centers(
                    grid.vertices,
                    grid.edges,
                    grid.options.radius,
                    grid.options.accelerator,
                )
                if needs_edge_centers
                else None
            )
            _write_coordinate_fields(
                variables,
                grid,
                cell_centers,
                edge_centers,
                chunk_size,
            )
            edges_of_vertex = _write_memory_bounded_icon_connectivity(
                variables,
                grid,
                cell_centers,
                edge_centers,
                chunk_size,
            )
            summary = _write_metric_fields(
                variables,
                grid,
                edges_of_vertex,
                cell_centers,
                edge_centers,
                chunk_size,
            )
            for name, value in summary.items():
                dataset.setncattr(name, value)
            _write_refinement_fields(variables, grid, chunk_size)
            _write_static_fields(variables, grid, chunk_size)
            _write_cartesian_fields(
                variables,
                grid,
                cell_centers,
                edge_centers,
                chunk_size,
            )
            _write_hierarchy_fields(variables, grid, chunk_size)
        os.replace(partial, path)
    except BaseException:
        # Preserve the partial file for post-mortem inspection. It is never
        # mistaken for a completed output because publication is one rename.
        raise
    return path


def _icon_variable_definitions() -> list[
    tuple[str, tuple[str, ...], np.dtype[Any], dict[str, Any]]
]:
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

    return definitions


def _create_icon_variables(
    dataset: Any,
    selected_fields: frozenset[str],
) -> dict[str, Any]:
    """Create the established ICON schema before filling fields out of order."""
    from . import _netcdf

    variables: dict[str, Any] = {}
    for name, dims, dtype, attrs in _icon_variable_definitions():
        if name not in selected_fields:
            continue
        variable = dataset.createVariable(name, dtype, dims)
        resolved_attrs = _netcdf._with_icon_variable_attrs(
            name,
            attrs,
            selected_fields,
        )
        for attr_name, value in resolved_attrs.items():
            variable.setncattr(attr_name, value)
        variables[name] = variable
    return variables


def _chunk_slices(size: int, chunk_size: int):
    for start in range(0, size, chunk_size):
        yield slice(start, min(start + chunk_size, size))


def _write_coordinate_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_centers: np.ndarray | None,
    edge_centers: np.ndarray | None,
    chunk_size: int,
) -> None:
    from . import grid_generator as gg

    vertex_fields = {
        "vlon",
        "vlat",
        "longitude_vertices",
        "latitude_vertices",
        "clon_vertices",
        "clat_vertices",
        "elon_vertices",
        "elat_vertices",
    }
    if vertex_fields.isdisjoint(variables):
        vertex_lon = vertex_lat = None
    else:
        vertex_lon, vertex_lat = gg._lon_lat(grid.vertices)
    cell_fields = {
        "clon",
        "clat",
        "lon_cell_centre",
        "lat_cell_centre",
        "elon_vertices",
        "elat_vertices",
    }
    if cell_fields.isdisjoint(variables):
        cell_lon = cell_lat = None
    else:
        if cell_centers is None:
            raise RuntimeError("selected coordinate fields require cell centers")
        cell_lon, cell_lat = gg._lon_lat(cell_centers)
    edge_fields = {
        "elon",
        "elat",
        "lon_edge_centre",
        "lat_edge_centre",
        "elon_vertices",
        "elat_vertices",
    }
    if edge_fields.isdisjoint(variables):
        edge_lon = edge_lat = None
    else:
        if edge_centers is None:
            raise RuntimeError("selected coordinate fields require edge centers")
        edge_lon, edge_lat = gg._lon_lat(edge_centers)

    _write_cell_coordinate_fields(
        variables,
        grid,
        cell_lon,
        cell_lat,
        vertex_lon,
        vertex_lat,
        chunk_size,
    )
    _write_vertex_coordinate_fields(
        variables,
        grid.dims["vertex"],
        vertex_lon,
        vertex_lat,
        chunk_size,
    )
    _write_edge_coordinate_fields(
        variables,
        grid,
        vertex_lon,
        vertex_lat,
        cell_lon,
        cell_lat,
        edge_lon,
        edge_lat,
        chunk_size,
    )


def _write_cell_coordinate_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_lon: np.ndarray | None,
    cell_lat: np.ndarray | None,
    vertex_lon: np.ndarray | None,
    vertex_lat: np.ndarray | None,
    chunk_size: int,
) -> None:
    names = {
        "clon",
        "clat",
        "lon_cell_centre",
        "lat_cell_centre",
        "clon_vertices",
        "clat_vertices",
    }
    if names.isdisjoint(variables):
        return
    for section in _chunk_slices(grid.dims["cell"], chunk_size):
        lon = np.radians(cell_lon[section]) if cell_lon is not None else None
        lat = np.radians(cell_lat[section]) if cell_lat is not None else None
        for name, values in (
            ("clon", lon),
            ("clat", lat),
            ("lon_cell_centre", lon),
            ("lat_cell_centre", lat),
        ):
            if name in variables:
                if values is None:
                    raise RuntimeError(f"selected field {name} is missing coordinates")
                variables[name][section] = values
        if "clon_vertices" in variables or "clat_vertices" in variables:
            if vertex_lon is None or vertex_lat is None:
                raise RuntimeError("cell coordinate bounds require vertex coordinates")
            cells = grid.cells[section]
            if "clon_vertices" in variables:
                variables["clon_vertices"][section, :] = np.radians(vertex_lon[cells])
            if "clat_vertices" in variables:
                variables["clat_vertices"][section, :] = np.radians(vertex_lat[cells])


def _write_vertex_coordinate_fields(
    variables: dict[str, Any],
    vertex_count: int,
    vertex_lon: np.ndarray | None,
    vertex_lat: np.ndarray | None,
    chunk_size: int,
) -> None:
    names = {"vlon", "vlat", "longitude_vertices", "latitude_vertices"}
    if names.isdisjoint(variables):
        return
    if vertex_lon is None or vertex_lat is None:
        raise RuntimeError("selected vertex fields require vertex coordinates")
    for section in _chunk_slices(vertex_count, chunk_size):
        lon = np.radians(vertex_lon[section])
        lat = np.radians(vertex_lat[section])
        for name, values in (
            ("vlon", lon),
            ("vlat", lat),
            ("longitude_vertices", lon),
            ("latitude_vertices", lat),
        ):
            if name in variables:
                variables[name][section] = values


def _write_edge_coordinate_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    vertex_lon: np.ndarray | None,
    vertex_lat: np.ndarray | None,
    cell_lon: np.ndarray | None,
    cell_lat: np.ndarray | None,
    edge_lon: np.ndarray | None,
    edge_lat: np.ndarray | None,
    chunk_size: int,
) -> None:
    names = {
        "elon",
        "elat",
        "lon_edge_centre",
        "lat_edge_centre",
        "elon_vertices",
        "elat_vertices",
    }
    if names.isdisjoint(variables):
        return
    if edge_lon is None or edge_lat is None:
        raise RuntimeError("selected edge fields require edge coordinates")
    for section in _chunk_slices(grid.dims["edge"], chunk_size):
        lon = np.radians(edge_lon[section])
        lat = np.radians(edge_lat[section])
        for name, values in (
            ("elon", lon),
            ("elat", lat),
            ("lon_edge_centre", lon),
            ("lat_edge_centre", lat),
        ):
            if name in variables:
                variables[name][section] = values
        if "elon_vertices" in variables or "elat_vertices" in variables:
            if any(
                values is None
                for values in (vertex_lon, vertex_lat, cell_lon, cell_lat)
            ):
                raise RuntimeError(
                    "edge coordinate bounds require vertex and cell coordinates"
                )
            lon_bounds, lat_bounds = _edge_coordinate_bounds_chunk(
                grid,
                section,
                vertex_lon,
                vertex_lat,
                cell_lon,
                cell_lat,
                edge_lon,
                edge_lat,
            )
            if "elon_vertices" in variables:
                variables["elon_vertices"][section, :] = lon_bounds
            if "elat_vertices" in variables:
                variables["elat_vertices"][section, :] = lat_bounds


def _edge_coordinate_bounds_chunk(
    grid: _CompactGlobalGrid,
    section: slice,
    vertex_lon: np.ndarray,
    vertex_lat: np.ndarray,
    cell_lon: np.ndarray,
    cell_lat: np.ndarray,
    edge_lon: np.ndarray,
    edge_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = grid.edges[section]
    edge_cells = grid.edge_cells[section]
    edge_indices = np.arange(section.start or 0, section.stop, dtype=np.int64)

    def bounds(
        vertex_values: np.ndarray,
        cell_values: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        values = np.empty((edges.shape[0], 4), dtype=np.float64)
        values[:, 0] = vertex_values[edges[:, 0]]
        second_cell = edge_cells[:, 1]
        values[:, 1] = np.where(
            second_cell >= 0,
            cell_values[np.maximum(second_cell, 0)],
            edge_values[edge_indices],
        )
        values[:, 2] = vertex_values[edges[:, 1]]
        first_cell = edge_cells[:, 0]
        values[:, 3] = np.where(
            first_cell >= 0,
            cell_values[np.maximum(first_cell, 0)],
            edge_values[edge_indices],
        )
        return values

    lon_values = bounds(vertex_lon, cell_lon, edge_lon)
    lat_values = bounds(vertex_lat, cell_lat, edge_lat)
    pole_mask = np.isclose(np.abs(lat_values), 90.0)
    pole_edges, pole_slots = np.nonzero(pole_mask)
    lon_values[pole_edges, pole_slots] = edge_lon[edge_indices[pole_edges]]
    return np.radians(lon_values), np.radians(lat_values)


def _write_memory_bounded_icon_connectivity(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_centers: np.ndarray | None,
    edge_centers: np.ndarray | None,
    chunk_size: int,
) -> np.ndarray | None:
    """Build, write, and release connectivity tables one at a time."""
    _write_cell_and_edge_connectivity(variables, grid, chunk_size)
    v2e = _write_edges_of_vertex(
        variables,
        grid,
        edge_centers,
        chunk_size,
    )
    _write_vertices_of_vertex(variables, grid, chunk_size)
    _write_cells_of_vertex(variables, grid, cell_centers, chunk_size)
    _write_edge_orientation(variables, grid, v2e, chunk_size)
    return v2e


def _write_cell_and_edge_connectivity(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    _write_cell_connectivity_fields(variables, grid, chunk_size)
    _write_edge_connectivity_fields(variables, grid, chunk_size)


def _write_cell_connectivity_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    cell_fields = {
        "edge_of_cell",
        "vertex_of_cell",
        "neighbor_cell_index",
        "orientation_of_normal",
    }
    if not cell_fields.isdisjoint(variables):
        for section in _chunk_slices(grid.dims["cell"], chunk_size):
            cell_edges = grid.cell_edges[section]
            adjacent = grid.edge_cells[cell_edges]
            cell_ids = np.arange(
                section.start or 0,
                section.stop,
                dtype=np.int32,
            )[:, None]
            first = adjacent[:, :, 0] == cell_ids
            if "edge_of_cell" in variables:
                variables["edge_of_cell"][:, section] = cell_edges.T + 1
            if "vertex_of_cell" in variables:
                variables["vertex_of_cell"][:, section] = grid.cells[section].T + 1
            if "neighbor_cell_index" in variables:
                variables["neighbor_cell_index"][:, section] = (
                    np.where(
                        first,
                        adjacent[:, :, 1],
                        adjacent[:, :, 0],
                    ).T
                    + 1
                )
            if "orientation_of_normal" in variables:
                variables["orientation_of_normal"][:, section] = np.where(
                    first,
                    1,
                    -1,
                ).T


def _write_edge_connectivity_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    if "adjacent_cell_of_edge" in variables or "edge_vertices" in variables:
        for section in _chunk_slices(grid.dims["edge"], chunk_size):
            if "adjacent_cell_of_edge" in variables:
                variables["adjacent_cell_of_edge"][:, section] = (
                    grid.edge_cells[section].T + 1
                )
            if "edge_vertices" in variables:
                variables["edge_vertices"][:, section] = grid.edges[section].T + 1


def _write_edges_of_vertex(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    edge_centers: np.ndarray | None,
    chunk_size: int,
) -> np.ndarray | None:
    from . import grid_generator as gg

    vertex_count = grid.dims["vertex"]
    needs_v2e = any(
        name in variables
        for name in (
            "edges_of_vertex",
            "edge_orientation",
            "dual_area",
            "dual_area_p",
        )
    )
    if not needs_v2e:
        return None
    if edge_centers is None:
        raise RuntimeError("selected connectivity requires edge centers")
    v2e = gg._sort_fixed_around_vertices(
        grid.vertices,
        grid.edges_of_vertex,
        points=edge_centers,
        accelerator=grid.options.accelerator,
    )
    if "edges_of_vertex" in variables:
        for section in _chunk_slices(vertex_count, chunk_size):
            variables["edges_of_vertex"][:, section] = v2e[section].T
    return v2e


def _write_vertices_of_vertex(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    if "vertices_of_vertex" not in variables:
        return
    from . import grid_generator as gg

    edge_owners = np.concatenate((grid.edges[:, 0], grid.edges[:, 1]))
    incident_vertices = np.concatenate(
        (grid.edges[:, 1] + 1, grid.edges[:, 0] + 1)
    ).astype(np.int32, copy=False)
    v2v = gg._fixed_incidence(
        edge_owners,
        incident_vertices,
        grid.dims["vertex"],
        6,
        grid.options.accelerator,
    )
    v2v = gg._sort_fixed_around_vertices(
        grid.vertices,
        v2v,
        accelerator=grid.options.accelerator,
    )
    for section in _chunk_slices(grid.dims["vertex"], chunk_size):
        variables["vertices_of_vertex"][:, section] = v2v[section].T


def _write_cells_of_vertex(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_centers: np.ndarray | None,
    chunk_size: int,
) -> None:
    if "cells_of_vertex" not in variables:
        return
    if cell_centers is None:
        raise RuntimeError("cells_of_vertex requires cell centers")
    from . import grid_generator as gg

    cell_values = np.repeat(
        np.arange(1, grid.dims["cell"] + 1, dtype=np.int32),
        3,
    )
    v2c = gg._fixed_incidence(
        grid.cells.reshape(-1),
        cell_values,
        grid.dims["vertex"],
        6,
        grid.options.accelerator,
    )
    unit_centers = gg._normalize_rows(cell_centers)
    v2c = gg._sort_fixed_around_vertices(
        grid.vertices,
        v2c,
        points=unit_centers,
        accelerator=grid.options.accelerator,
    )
    for section in _chunk_slices(grid.dims["vertex"], chunk_size):
        variables["cells_of_vertex"][:, section] = v2c[section].T


def _write_edge_orientation(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    edges_of_vertex: np.ndarray | None,
    chunk_size: int,
) -> None:
    if "edge_orientation" not in variables:
        return
    if edges_of_vertex is None:
        raise RuntimeError("edge orientation requires vertex-edge incidence")
    for section in _chunk_slices(grid.dims["vertex"], chunk_size):
        incident = edges_of_vertex[section]
        start_vertices = grid.edges[np.maximum(incident - 1, 0), 0]
        vertex_ids = np.arange(
            section.start or 0,
            section.stop,
            dtype=np.int32,
        )[:, None]
        values = np.where(start_vertices == vertex_ids, 1, -1).astype(np.int32)
        variables["edge_orientation"][:, section] = np.where(
            incident > 0,
            values,
            0,
        ).T


def _write_metric_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    edges_of_vertex: np.ndarray | None,
    cell_centers: np.ndarray | None,
    edge_centers: np.ndarray | None,
    chunk_size: int,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    mean_cell_area = _write_cell_area_fields(variables, grid, chunk_size)
    if mean_cell_area is not None:
        summary["mean_cell_area"] = mean_cell_area
    if _EDGE_METRIC_FIELDS.isdisjoint(variables):
        return summary
    if cell_centers is None or edge_centers is None:
        raise RuntimeError("selected metric fields require cell and edge centers")

    needs_dual_area = "dual_area" in variables or "dual_area_p" in variables
    edge_lengths, dual_edge_lengths, mean_edge, mean_dual_edge = (
        _write_edge_metric_fields(
            variables,
            grid,
            cell_centers,
            edge_centers,
            chunk_size,
            retain_lengths=needs_dual_area,
        )
    )
    summary["mean_edge_length"] = mean_edge
    summary["mean_dual_edge_length"] = mean_dual_edge
    mean_dual_area = _write_dual_area_fields(
        variables,
        grid,
        edges_of_vertex,
        edge_lengths,
        dual_edge_lengths,
        chunk_size,
    )
    if mean_dual_area is not None:
        summary["mean_dual_cell_area"] = mean_dual_area
    return summary


def _write_cell_area_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> float | None:
    selected = tuple(name for name in ("cell_area", "cell_area_p") if name in variables)
    if not selected:
        return None
    cell_areas = _accelerated.cell_areas_numba(
        grid.vertices,
        grid.cells,
        grid.options.sphere_radius,
    )
    for section in _chunk_slices(grid.dims["cell"], chunk_size):
        for name in selected:
            variables[name][section] = cell_areas[section]
    return float(np.mean(cell_areas))


def _write_edge_metric_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_centers: np.ndarray,
    edge_centers: np.ndarray,
    chunk_size: int,
    *,
    retain_lengths: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    edge_lengths = _optional_edge_array(grid, retain_lengths)
    dual_edge_lengths = _optional_edge_array(grid, retain_lengths)
    edge_length_sum = 0.0
    dual_edge_length_sum = 0.0
    radius_squared = grid.options.sphere_radius**2
    for section in _chunk_slices(grid.dims["edge"], chunk_size):
        metrics = _accelerated.spherical_edge_metrics_numba(
            grid.vertices,
            cell_centers,
            grid.edges[section],
            grid.edge_cells[section],
            edge_centers[section],
            grid.options.sphere_radius,
        )
        (
            edge_length,
            dual_edge_length,
            edge_cell_distance,
            edge_system_orientation,
            primal_normal,
            dual_normal,
            primal_u,
            primal_v,
            dual_u,
            dual_v,
        ) = metrics
        edge_length_sum += float(np.sum(edge_length))
        dual_edge_length_sum += float(np.sum(dual_edge_length))
        if edge_lengths is not None and dual_edge_lengths is not None:
            edge_lengths[section] = edge_length
            dual_edge_lengths[section] = dual_edge_length
        _write_edge_metric_chunk(
            variables,
            section,
            metrics,
            radius_squared,
        )

    if edge_lengths is not None and dual_edge_lengths is not None:
        means = float(np.mean(edge_lengths)), float(np.mean(dual_edge_lengths))
    else:
        edge_count = grid.dims["edge"]
        means = edge_length_sum / edge_count, dual_edge_length_sum / edge_count
    return edge_lengths, dual_edge_lengths, *means


def _optional_edge_array(
    grid: _CompactGlobalGrid,
    enabled: bool,
) -> np.ndarray | None:
    return np.empty(grid.dims["edge"], dtype=np.float64) if enabled else None


def _write_edge_metric_chunk(
    variables: dict[str, Any],
    section: slice,
    metrics: tuple[np.ndarray, ...],
    radius_squared: float,
) -> None:
    (
        edge_length,
        dual_edge_length,
        edge_cell_distance,
        edge_system_orientation,
        primal_normal,
        dual_normal,
        primal_u,
        primal_v,
        dual_u,
        dual_v,
    ) = metrics
    one_dimensional = {
        "edge_length": edge_length,
        "dual_edge_length": dual_edge_length,
        "edgequad_area": 0.5 * edge_length * dual_edge_length / radius_squared,
        "edge_system_orientation": edge_system_orientation,
        "zonal_normal_primal_edge": primal_u,
        "meridional_normal_primal_edge": primal_v,
        "zonal_normal_dual_edge": dual_u,
        "meridional_normal_dual_edge": dual_v,
    }
    for axis, suffix in enumerate(("x", "y", "z")):
        one_dimensional[f"edge_primal_normal_cartesian_{suffix}"] = primal_normal[
            :, axis
        ]
        one_dimensional[f"edge_dual_normal_cartesian_{suffix}"] = dual_normal[:, axis]
    for name, values in one_dimensional.items():
        if name in variables:
            variables[name][section] = values
    two_dimensional = {
        "edge_cell_distance": edge_cell_distance.T,
        "edge_vert_distance": np.stack((edge_length * 0.5, edge_length * 0.5)),
    }
    for name, values in two_dimensional.items():
        if name in variables:
            variables[name][:, section] = values


def _write_dual_area_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    edges_of_vertex: np.ndarray | None,
    edge_lengths: np.ndarray | None,
    dual_edge_lengths: np.ndarray | None,
    chunk_size: int,
) -> float | None:
    selected = tuple(name for name in ("dual_area", "dual_area_p") if name in variables)
    if not selected:
        return None
    if edges_of_vertex is None or edge_lengths is None or dual_edge_lengths is None:
        raise RuntimeError("dual areas require vertex-edge incidence")
    dual_areas = _accelerated.dual_areas_from_edges_numba(
        edges_of_vertex,
        edge_lengths,
        dual_edge_lengths,
    )
    for section in _chunk_slices(grid.dims["vertex"], chunk_size):
        for name in selected:
            variables[name][section] = dual_areas[section]
    return float(np.mean(dual_areas))


def _write_refinement_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    from . import grid_generator as gg

    for name, dimension, value in (
        ("refin_c_ctrl", "cell", -4),
        ("refin_e_ctrl", "edge", -8),
        ("refin_v_ctrl", "vertex", 0),
    ):
        if name in variables:
            for section in _chunk_slices(grid.dims[dimension], chunk_size):
                variables[name][section] = value
    for suffix, fixed_dimension, size, refinement_control in (
        ("c", "cell_grf", grid.dims["cell"], -4),
        ("e", "edge_grf", grid.dims["edge"], -8),
        ("v", "vert_grf", grid.dims["vertex"], 0),
    ):
        start_name = f"start_idx_{suffix}"
        end_name = f"end_idx_{suffix}"
        if start_name in variables:
            variables[start_name][:] = gg._start_index_fixed(
                fixed_dimension,
                size,
                refinement_control,
            )
        if end_name in variables:
            variables[end_name][:] = gg._end_index_fixed(
                fixed_dimension,
                size,
                refinement_control,
            )


def _write_static_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    for name, dimension in (
        ("cell_elevation", "cell"),
        ("edge_elevation", "edge"),
        ("cell_sea_land_mask", "cell"),
        ("edge_sea_land_mask", "edge"),
    ):
        if name in variables:
            for section in _chunk_slices(grid.dims[dimension], chunk_size):
                variables[name][section] = 0


def _write_cartesian_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    cell_centers: np.ndarray | None,
    edge_centers: np.ndarray | None,
    chunk_size: int,
) -> None:
    _write_cartesian_components(
        variables,
        grid.vertices,
        (
            ("cartesian_x_vertices", 0),
            ("cartesian_y_vertices", 1),
            ("cartesian_z_vertices", 2),
        ),
        grid.dims["vertex"],
        chunk_size,
    )
    _write_index_fields(
        variables,
        ("vertex_index",),
        grid.dims["vertex"],
        chunk_size,
    )
    _write_cartesian_components(
        variables,
        cell_centers,
        (
            ("cell_circumcenter_cartesian_x", 0),
            ("cell_circumcenter_cartesian_y", 1),
            ("cell_circumcenter_cartesian_z", 2),
        ),
        grid.dims["cell"],
        chunk_size,
    )
    _write_index_fields(
        variables,
        ("phys_cell_id", "cell_index"),
        grid.dims["cell"],
        chunk_size,
    )
    _write_cartesian_components(
        variables,
        edge_centers,
        tuple(
            (f"{prefix}_{suffix}", axis)
            for prefix in (
                "edge_middle_cartesian",
                "edge_dual_middle_cartesian",
            )
            for axis, suffix in enumerate(("x", "y", "z"))
        ),
        grid.dims["edge"],
        chunk_size,
    )
    _write_index_fields(
        variables,
        ("phys_edge_id", "edge_index"),
        grid.dims["edge"],
        chunk_size,
    )


def _write_cartesian_components(
    variables: dict[str, Any],
    points: np.ndarray | None,
    components: tuple[tuple[str, int], ...],
    size: int,
    chunk_size: int,
) -> None:
    selected = [(name, axis) for name, axis in components if name in variables]
    if not selected:
        return
    if points is None:
        raise RuntimeError("selected Cartesian fields require center coordinates")
    from . import grid_generator as gg

    for section in _chunk_slices(size, chunk_size):
        unit = gg._normalize_rows(points[section])
        for name, axis in selected:
            variables[name][section] = unit[:, axis]


def _write_index_fields(
    variables: dict[str, Any],
    names: tuple[str, ...],
    size: int,
    chunk_size: int,
) -> None:
    selected = [name for name in names if name in variables]
    if not selected:
        return
    for section in _chunk_slices(size, chunk_size):
        indices = np.arange(
            (section.start or 0) + 1,
            section.stop + 1,
            dtype=np.int32,
        )
        for name in selected:
            variables[name][section] = indices


def _write_hierarchy_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    hierarchy_names = {
        "parent_cell_index",
        "parent_cell_type",
        "edge_parent_type",
        "parent_edge_index",
        "parent_vertex_index",
        "child_cell_index",
        "child_cell_id",
        "child_edge_index",
        "child_edge_id",
    }
    if hierarchy_names.isdisjoint(variables):
        return
    provenance = grid.provenance
    if provenance is None:
        raise RuntimeError("streamed global export requires final-stage provenance")
    _write_parent_hierarchy_fields(variables, provenance, chunk_size)
    _write_child_hierarchy_fields(variables, grid, chunk_size)


def _write_parent_hierarchy_fields(
    variables: dict[str, Any],
    provenance: BisectionProvenance,
    chunk_size: int,
) -> None:
    arrays = {
        "parent_cell_index": provenance.parent_cell_index,
        "parent_cell_type": provenance.parent_cell_type,
        "edge_parent_type": provenance.child_edge_parent_type,
        "parent_edge_index": provenance.child_parent_edge_index,
        "parent_vertex_index": provenance.parent_vertex_index,
    }
    for name, values in arrays.items():
        if name not in variables:
            continue
        if values is None:
            raise RuntimeError(f"streamed global export is missing {name}")
        for section in _chunk_slices(values.shape[0], chunk_size):
            variables[name][section] = values[section]


def _write_child_hierarchy_fields(
    variables: dict[str, Any],
    grid: _CompactGlobalGrid,
    chunk_size: int,
) -> None:
    for name, dimension, leading in (
        ("child_cell_index", "cell", True),
        ("child_cell_id", "cell", False),
        ("child_edge_index", "edge", True),
        ("child_edge_id", "edge", False),
    ):
        if name in variables:
            for section in _chunk_slices(grid.dims[dimension], chunk_size):
                if leading:
                    variables[name][:, section] = 0
                else:
                    variables[name][section] = 0
