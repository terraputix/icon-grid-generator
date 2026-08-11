from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import math

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
)
from grid_generator.cutting import CutGridSpec, cut_grid
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
from grid_generator.transforms import DiffusionOptions, OptimizationOptions, diffuse_grid, optimize_grid

from grid_assertions import (
    assert_netcdf_grid_contract,
    assert_same_topology,
    assert_valid_grid_math,
)


@dataclass(frozen=True)
class GridCase:
    id: str
    factory: Callable[[], Any]
    geometry: str
    boundary: str
    spec_class: type | None = None


@dataclass(frozen=True)
class ScaleSeries:
    id: str
    factories: tuple[Callable[[], Any], ...]
    geometry: str
    boundary: str


def _global(parent: str = "R02B01") -> Any:
    return generate_grid(parent, spring_iterations=5)


def _limited(region: Any) -> Any:
    return generate_grid(
        LimitedAreaGridSpec(parent="R02B01", region=region, boundary_depth=1),
        spring_iterations=5,
    )


def _cut(region_or_spec: Any) -> Any:
    parent = _global()
    return cut_grid(parent, region_or_spec)


def _limited_at(parent: str) -> Any:
    return generate_grid(
        LimitedAreaGridSpec(
            parent=parent,
            region=Region.circle(lon=5.0, lat=15.0, radius_degrees=70.0),
            boundary_depth=1,
            local_optimization_iterations=0,
        ),
        optimize_global=False,
        max_cells=None,
    )


def _cut_at(parent: str) -> Any:
    return cut_grid(
        generate_grid(parent, optimize_global=False, max_cells=None),
        Region.circle(lon=5.0, lat=15.0, radius_degrees=70.0),
    )


GRID_CASES = [
    GridCase(
        "global-spec-optimized",
        lambda: generate_grid(GlobalGridSpec(root=1, bisections=1), spring_iterations=5),
        "global",
        "closed",
        GlobalGridSpec,
    ),
    GridCase(
        "global-string-raw",
        lambda: generate_grid("R01B01", optimize_global=False),
        "global",
        "closed",
        GlobalGridSpec,
    ),
    GridCase(
        "global-rotated",
        lambda: generate_grid(
            "R01B01",
            spring_iterations=5,
            north_pole_lon=15.0,
            north_pole_lat=80.0,
            rotation_angle_degrees=12.5,
        ),
        "global",
        "closed",
        GlobalGridSpec,
    ),
    GridCase(
        "global-numpy-accelerator",
        lambda: generate_grid("R01B01", spring_iterations=5, accelerator="numpy"),
        "global",
        "closed",
        GlobalGridSpec,
    ),
    GridCase(
        "planar-torus",
        lambda: generate_grid(TorusGridSpec(nx=4, ny=4, edge_length=1_000.0)),
        "periodic_planar",
        "closed",
        TorusGridSpec,
    ),
    GridCase(
        "planar-torus-skew",
        lambda: generate_grid(
            TorusGridSpec(
                nx=5,
                ny=5,
                edge_length=1_000.0,
                periodic_layout="skew",
            )
        ),
        "periodic_planar",
        "closed",
        TorusGridSpec,
    ),
    GridCase(
        "planar-stretched-torus",
        lambda: generate_grid(
            StretchedTorusGridSpec(nx=4, ny=4, edge_length=1_000.0, stretch_x=1.3)
        ),
        "periodic_planar",
        "closed",
        StretchedTorusGridSpec,
    ),
    GridCase(
        "planar-stretched-torus-skew",
        lambda: generate_grid(
            StretchedTorusGridSpec(
                nx=5,
                ny=5,
                edge_length=1_000.0,
                stretch_x=1.3,
                stretch_y=0.8,
                periodic_layout="skew",
            )
        ),
        "periodic_planar",
        "closed",
        StretchedTorusGridSpec,
    ),
    GridCase(
        "planar-channel",
        lambda: generate_grid(ChannelGridSpec(nx=4, ny=3, edge_length=1_000.0)),
        "open_planar",
        "open",
        ChannelGridSpec,
    ),
    GridCase(
        "planar-parallelogram",
        lambda: generate_grid(
            ParallelogramGridSpec(nx=4, ny=3, edge_length=1_000.0, shear=0.25)
        ),
        "open_planar",
        "open",
        ParallelogramGridSpec,
    ),
    GridCase(
        "planar-ragged-orthogonal",
        lambda: generate_grid(RaggedOrthogonalGridSpec(nx=4, ny=3, dx=1_000.0, dy=900.0)),
        "open_planar",
        "open",
        RaggedOrthogonalGridSpec,
    ),
    GridCase(
        "limited-box",
        lambda: _limited(
            Region.lonlat_box(lon_min=-30.0, lon_max=30.0, lat_min=-20.0, lat_max=35.0)
        ),
        "regional",
        "open",
        LimitedAreaGridSpec,
    ),
    GridCase(
        "limited-circle",
        lambda: _limited(Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0)),
        "regional",
        "open",
        LimitedAreaGridSpec,
    ),
    GridCase(
        "limited-rectangle",
        lambda: _limited(
            Region.rectangle(
                center_lon=0.0,
                center_lat=0.0,
                width_degrees=40.0,
                height_degrees=30.0,
                angle_degrees=20.0,
            )
        ),
        "regional",
        "open",
        LimitedAreaGridSpec,
    ),
    GridCase(
        "limited-polygon",
        lambda: _limited(
            Region.polygon(points=((-35.0, -5.0), (0.0, 35.0), (35.0, -5.0)))
        ),
        "regional",
        "open",
        LimitedAreaGridSpec,
    ),
    GridCase(
        "cut-direct-circle",
        lambda: _cut(Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0)),
        "regional",
        "open",
        None,
    ),
    GridCase(
        "cut-spec-keep-multiple",
        lambda: _cut(
            CutGridSpec(
                regions=(
                    Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
                    Region.lonlat_box(
                        lon_min=-20.0,
                        lon_max=20.0,
                        lat_min=-15.0,
                        lat_max=15.0,
                    ),
                ),
                boundary_depth=1,
                smoothing_depth=1,
                name="CUT_KEEP_MULTI",
            )
        ),
        "regional",
        "open",
        None,
    ),
    GridCase(
        "cut-spec-remove",
        lambda: _cut(
            CutGridSpec(
                regions=Region.lonlat_box(
                    lon_min=-20.0,
                    lon_max=20.0,
                    lat_min=-15.0,
                    lat_max=15.0,
                ),
                mode="remove",
            )
        ),
        "regional",
        "open",
        None,
    ),
]


SCALE_SERIES = [
    ScaleSeries(
        "global",
        (
            lambda: generate_grid("R01B00", optimize_global=False),
            lambda: generate_grid("R01B02", optimize_global=False),
            lambda: generate_grid("R02B03", optimize_global=False, max_cells=None),
        ),
        "global",
        "closed",
    ),
    ScaleSeries(
        "torus",
        (
            lambda: generate_grid(TorusGridSpec(nx=4, ny=4, edge_length=1_000.0)),
            lambda: generate_grid(TorusGridSpec(nx=12, ny=8, edge_length=500.0)),
            lambda: generate_grid(TorusGridSpec(nx=32, ny=24, edge_length=250.0)),
        ),
        "periodic_planar",
        "closed",
    ),
    ScaleSeries(
        "stretched-torus",
        (
            lambda: generate_grid(
                StretchedTorusGridSpec(
                    nx=4,
                    ny=4,
                    edge_length=1_000.0,
                    stretch_x=1.2,
                    stretch_y=0.8,
                )
            ),
            lambda: generate_grid(
                StretchedTorusGridSpec(
                    nx=12,
                    ny=8,
                    edge_length=500.0,
                    stretch_x=1.2,
                    stretch_y=0.8,
                )
            ),
            lambda: generate_grid(
                StretchedTorusGridSpec(
                    nx=32,
                    ny=24,
                    edge_length=250.0,
                    stretch_x=1.2,
                    stretch_y=0.8,
                )
            ),
        ),
        "periodic_planar",
        "closed",
    ),
    ScaleSeries(
        "channel",
        (
            lambda: generate_grid(ChannelGridSpec(nx=4, ny=2, edge_length=1_000.0)),
            lambda: generate_grid(ChannelGridSpec(nx=12, ny=8, edge_length=500.0)),
            lambda: generate_grid(ChannelGridSpec(nx=32, ny=24, edge_length=250.0)),
        ),
        "open_planar",
        "open",
    ),
    ScaleSeries(
        "parallelogram",
        (
            lambda: generate_grid(
                ParallelogramGridSpec(
                    nx=2,
                    ny=2,
                    edge_length=1_000.0,
                    shear=-0.2,
                )
            ),
            lambda: generate_grid(
                ParallelogramGridSpec(
                    nx=12,
                    ny=8,
                    edge_length=500.0,
                    shear=0.25,
                )
            ),
            lambda: generate_grid(
                ParallelogramGridSpec(
                    nx=32,
                    ny=24,
                    edge_length=250.0,
                    shear=0.4,
                )
            ),
        ),
        "open_planar",
        "open",
    ),
    ScaleSeries(
        "ragged-orthogonal",
        (
            lambda: generate_grid(
                RaggedOrthogonalGridSpec(
                    nx=2,
                    ny=2,
                    dx=1_000.0,
                    dy=800.0,
                    raggedness=0.0,
                )
            ),
            lambda: generate_grid(
                RaggedOrthogonalGridSpec(
                    nx=12,
                    ny=8,
                    dx=500.0,
                    dy=400.0,
                    raggedness=0.2,
                )
            ),
            lambda: generate_grid(
                RaggedOrthogonalGridSpec(
                    nx=32,
                    ny=24,
                    dx=250.0,
                    dy=200.0,
                    raggedness=0.4,
                )
            ),
        ),
        "open_planar",
        "open",
    ),
    ScaleSeries(
        "limited-area",
        (
            lambda: _limited_at("R01B01"),
            lambda: _limited_at("R01B02"),
            lambda: _limited_at("R01B03"),
        ),
        "regional",
        "open",
    ),
    ScaleSeries(
        "cut",
        (
            lambda: _cut_at("R01B01"),
            lambda: _cut_at("R01B02"),
            lambda: _cut_at("R01B03"),
        ),
        "regional",
        "open",
    ),
]


@pytest.mark.parametrize("case", GRID_CASES, ids=[case.id for case in GRID_CASES])
def test_all_generation_modes_satisfy_shared_mathematical_invariants(case):
    grid = case.factory()

    assert_valid_grid_math(grid, geometry=case.geometry, boundary=case.boundary)


@pytest.mark.parametrize("case", GRID_CASES, ids=[case.id for case in GRID_CASES])
def test_generation_modes_are_deterministic(case):
    first = case.factory()
    second = case.factory()

    assert_same_topology(first, second)
    assert np.allclose(first.vertices, second.vertices)
    assert np.allclose(first.geometry["cell_area"], second.geometry["cell_area"])
    assert first.metadata["uuidOfHGrid"] == second.metadata["uuidOfHGrid"]


def test_correctness_matrix_covers_all_supported_grid_spec_classes():
    covered = {case.spec_class for case in GRID_CASES if case.spec_class is not None}

    assert covered == {
        GlobalGridSpec,
        TorusGridSpec,
        ChannelGridSpec,
        ParallelogramGridSpec,
        LimitedAreaGridSpec,
        StretchedTorusGridSpec,
        RaggedOrthogonalGridSpec,
    }


@pytest.mark.parametrize("series", SCALE_SERIES, ids=lambda series: series.id)
def test_increasing_size_series_match_in_memory_and_netcdf(series, tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    cell_counts = []

    for index, factory in enumerate(series.factories):
        grid = factory()
        assert_valid_grid_math(
            grid,
            geometry=series.geometry,
            boundary=series.boundary,
        )
        if all(
            hasattr(grid.spec, name)
            for name in ("expected_cells", "expected_edges", "expected_vertices")
        ) and grid.spec.expected_cells > 0:
            assert grid.dims == {
                "cell": grid.spec.expected_cells,
                "edge": grid.spec.expected_edges,
                "vertex": grid.spec.expected_vertices,
            }
        if hasattr(grid.spec, "domain_length"):
            assert grid.metadata["domain_length"] == pytest.approx(
                grid.spec.domain_length
            )
            assert grid.metadata["domain_height"] == pytest.approx(
                grid.spec.domain_height
            )

        path = grid.to_netcdf(tmp_path / f"{series.id}-{index}.nc")
        with netcdf4.Dataset(path) as dataset:
            assert_netcdf_grid_contract(dataset, grid)
        cell_counts.append(grid.dims["cell"])

    assert all(
        smaller < larger
        for smaller, larger in zip(cell_counts, cell_counts[1:])
    )


def test_planar_regular_modes_match_analytic_metrics():
    torus = generate_grid(TorusGridSpec(nx=4, ny=4, edge_length=1_000.0))
    channel = generate_grid(ChannelGridSpec(nx=4, ny=3, edge_length=1_000.0))
    expected_area = math.sqrt(3.0) * 0.25 * 1_000.0**2

    for grid in (torus, channel):
        assert np.allclose(grid.geometry["cell_area"], expected_area)
        assert np.allclose(grid.geometry["edge_length"], 1_000.0)
        assert np.allclose(grid.geometry["dual_edge_length"], 1_000.0 / math.sqrt(3.0))


def test_spherical_metric_scaling_is_independent_of_display_radius():
    base = generate_grid("R01B01", spring_iterations=5, sphere_radius=2.0, radius=1.0)
    physical_scaled = generate_grid(
        "R01B01",
        spring_iterations=5,
        sphere_radius=4.0,
        radius=1.0,
    )
    display_scaled = generate_grid(
        "R01B01",
        spring_iterations=5,
        sphere_radius=2.0,
        radius=3.0,
    )

    assert_same_topology(base, physical_scaled)
    assert np.allclose(physical_scaled.geometry["cell_area"], 4.0 * base.geometry["cell_area"])
    assert np.allclose(physical_scaled.geometry["edge_length"], 2.0 * base.geometry["edge_length"])
    assert np.allclose(display_scaled.geometry["cell_area"], base.geometry["cell_area"])
    assert np.allclose(display_scaled.geometry["edge_length"], base.geometry["edge_length"])
    assert np.allclose(np.linalg.norm(display_scaled.vertices, axis=1), 3.0)


@pytest.mark.parametrize(
    "factory,geometry,boundary",
    [
        (
            lambda: generate_grid("R01B01", spring_iterations=5),
            "global",
            "closed",
        ),
        (
            lambda: generate_grid(ChannelGridSpec(nx=4, ny=3, edge_length=1_000.0)),
            "open_planar",
            "open",
        ),
        (
            lambda: cut_grid(
                _global(),
                Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
            ),
            "regional",
            "open",
        ),
    ],
    ids=["global", "open-planar", "cut-grid"],
)
def test_geometry_transforms_preserve_topology_and_return_valid_grids(factory, geometry, boundary):
    grid = factory()
    optimized = optimize_grid(grid, OptimizationOptions(iterations=2, relaxation=0.1))
    diffused = diffuse_grid(grid, DiffusionOptions(iterations=2, diffusion_constant=0.02))

    for transformed in (optimized, diffused):
        assert transformed is not grid
        assert_same_topology(grid, transformed)
        assert_valid_grid_math(transformed, geometry=geometry, boundary=boundary)


@pytest.mark.parametrize(
    "case",
    GRID_CASES,
    ids=lambda case: case.id,
)
def test_all_generation_modes_export_valid_netcdf(case, tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = case.factory()
    path = grid.to_netcdf(tmp_path / f"{case.id}.nc")

    with netcdf4.Dataset(path) as dataset:
        assert_netcdf_grid_contract(dataset, grid)
