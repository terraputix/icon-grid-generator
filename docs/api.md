# API Overview

## Everyday API

Most users import only `generate_grid`:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
grid.to_netcdf("icon_grid_R02B04.nc")
```

The root package exports the primary grid specifications and generators:

```python
from grid_generator import (
    generate_grid,
    generate_grid_to_netcdf,
    IconGrid,
    IconGridOptions,
    GlobalGridSpec,
    TorusGridSpec,
    ChannelGridSpec,
    ParallelogramGridSpec,
    LimitedAreaGridSpec,
    Region,
)
```

For global grids whose complete in-memory `IconGrid` would exceed available
memory, use the export-first `generate_grid_to_netcdf` path:

```py
from grid_generator import generate_grid_to_netcdf

generate_grid_to_netcdf(
    "R2B8",
    "icon_grid_R02B08.nc",
    options={"max_cells": None, "accelerator": "numba"},
    chunk_size=1_000_000,
    work_dir="icon-grid-R2B08-work",
    fields="reduced",
)
```

This global-only path keeps compact staged topology, checkpoints completed
bisection levels, and writes export-only fields in bounded chunks. New output is
published atomically only after the NetCDF file closes successfully; an existing
output remains in place until then. High-resolution streaming requires the
`accelerate` extra. The
optional-dependency-free NumPy fallback remains available through
`generate_grid` for grids that fit in memory.

If `work_dir` is omitted, checkpoints are stored in a hidden directory beside
the requested output file. For large grids, put the output or an explicit
`work_dir` on disk-backed scratch storage rather than a memory-backed temporary
filesystem. Set `resume=False` to ignore existing checkpoints. `chunk_size`
limits export-only work arrays, but does not partition the compact core topology.
Checkpoint manifests atomically select immutable array snapshots; an interrupted
overwrite therefore leaves the preceding completed snapshot resumable.
If writing fails, the requested output is not published; the `.partial` file is
left in place for diagnosis.

Allow capacity beyond the completed sizes in the resource tables. Replacing a
checkpoint temporarily retains both snapshots, and replacing an existing output
temporarily retains both the old file and the new `.partial` file. Successful
checkpoint directories can be deleted unless they are needed to resume or extend
the same configuration. Use distinct output and work-directory paths for
concurrent jobs.

The in-memory base stage is selected by cell count, not by the spelling of the
`R<n>B<k>` decomposition. By default the largest stage with at most 1,310,720
cells is built as an `IconGrid`, then later stages use compact refinement. If the
unrefined `B0` root already exceeds that budget, export-first generation rejects
the request. A smaller-root decomposition with the same frequency is a different
grid: it changes the canonical spec and UUID, changes the refinement hierarchy,
and can change staged optimization geometry. Use that alternative only when
those differences are acceptable; otherwise use the exact spec through the
in-memory workflow with sufficient memory.

### NetCDF field profiles

Both `grid.to_netcdf(..., fields=...)` and
`generate_grid_to_netcdf(..., fields=...)` accept the same selector. The
default is the complete schema.

| Profile | Fields | Consumer scope | Approximate R2B12 payload |
| --- | ---: | --- | ---: |
| `"full"` | 85 | Complete schema for general exchange | 1,047.5 GiB |
| `"reduced"` | 46 | Union required by standard ICON and icon4py global-grid readers | 485 GiB |
| `"icon"` | 38 | Standard ICON triangular-grid importer | 410 GiB |
| `"icon4py"` | 26 | icon4py spherical `GridManager` | 337.5 GiB |

The reduced and ICON profiles omit the complete Cartesian bundle so ICON uses
its existing reconstruction fallback; a partial Cartesian bundle is not
emitted. These profiles cover the standard global readers, not every ocean,
coupling, output-copying, remapping, or institutional tool. Use `"full"` for
general exchange.

Expert callers can pass an iterable of exact variable names instead of a
profile. Unknown names and non-string members fail before expensive generation,
duplicates are collapsed, and variables retain the established schema order:

```python
grid.to_netcdf(
    "reconstructable_mesh.nc",
    fields=["vlon", "vlat", "vertex_of_cell"],
)
```

Such a custom file is only as compatible as the selected fields. Dimensions,
identity, and configuration metadata are retained, and dangling `bounds` or
`coordinates` attributes are removed. On the export-first path, metric-summary
attributes are emitted only when the selected fields require those metrics;
computing omitted metrics would defeat exact field selection. Reduced exports
also skip avoidable bounds, Cartesian, placeholder, hierarchy, and connectivity
work rather than materializing the complete schema first.

## Grid Specifications

- `GlobalGridSpec` describes spherical ICON `R<n>B<k>` grids. Strings such as
  `"R2B4"` are shorthand for this common path.
- `LimitedAreaGridSpec(parent=..., region=..., selection=..., boundary=...)`
  selects at the preceding global bisection level and refines the compact
  region once by default. `construction="cut_final"` selects directly from the
  requested final-resolution parent.
- `TorusGridSpec` describes planar doubly periodic triangular torus grids.
- `ChannelGridSpec` describes a planar triangular channel with open boundaries
  in one direction and periodic boundaries in the other.
- `ParallelogramGridSpec` describes a skewed planar triangular parallelogram.

Advanced but supported planar variants live in `grid_generator.planar`:

```python
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
```

### Specification Signatures

| Object | Parameters |
| --- | --- |
| `GlobalGridSpec` | `root`, `bisections`, optional `name` |
| `TorusGridSpec` | `nx`, `ny`, `edge_length`, optional `name` |
| `ChannelGridSpec` | `nx`, `ny`, `edge_length`, optional `name` |
| `ParallelogramGridSpec` | `nx`, `ny`, `edge_length`, optional `shear`, `name` |
| `LimitedAreaGridSpec` | `parent`, `region`, optional `boundary_depth`, `construction`, `selection`, `boundary`, `local_optimization_iterations`, `name` |

Use `generate_grid("R2B4")` for the common global-grid case. Use explicit spec
objects when the grid family has parameters beyond the standard `R<n>B<k>`
name.

Planar counts and boundary conditions differ by family:

| Spec | Minimum size | Boundary condition |
| --- | --- | --- |
| `TorusGridSpec` | `nx >= 3`, `ny >= 3` | Periodic in both coupled lattice directions |
| `ChannelGridSpec` | `nx >= 3`, `ny >= 2` | Periodic in x, open in y |
| `ParallelogramGridSpec` | `nx >= 1`, `ny >= 1` | Open |
| `StretchedTorusGridSpec` | `nx >= 3`, `ny >= 3` | Periodic in both directions; positive x/y stretch |
| `RaggedOrthogonalGridSpec` | `nx >= 1`, `ny >= 1` | Open; `0 <= raggedness < 0.45` |

The regular planar families interpret `nx` and `ny` as numbers of rectangular
lattice patches, each split into two triangles. `edge_length`, `dx`, and `dy`
are physical planar coordinate values; supply metres when metre-labelled export
is expected. The package does not convert another planar unit to metres.
`stretch_x` and `stretch_y` multiply the regular torus coordinates; `shear` is
a dimensionless extra x shift per row height; `raggedness` is a deterministic
fraction of `dx`/`dy` used to perturb interior vertices.

## Regions

Use `Region` constructors for limited-area extraction and cutting:

- `Region.lonlat_box(lon_min=..., lon_max=..., lat_min=..., lat_max=...)`
- `Region.circle(lon=..., lat=..., radius_degrees=...)`
- `Region.rectangle(center_lon=..., center_lat=..., width_degrees=..., height_degrees=..., angle_degrees=...)`
- `Region.rotated_lonlat_box(pole_lon=..., pole_lat=..., center_lon=..., center_lat=..., half_width_lon=..., half_width_lat=...)`

Regional policies are explicit public values:

- `RegionSelectionOptions(inclusion="overlap", cleanup="remove_ears", buffer_rings=0)`
  controls coverage, mask cleanup, and selection expansion. Use `"center"` or
  `"none"` for the narrower/raw alternatives.
- `OpenBoundaryOptions(metric_closure="clipped", indexing_depth=14, ordering="icon")`
  controls physical versus mirrored dual closure and ICON boundary indexing.
  Source ordering is intended for in-memory analysis and cannot be exported as
  a valid ICON regional NetCDF.
- `Region.polygon(((lon0, lat0), (lon1, lat1), ...))`

Longitudes and latitudes are in degrees. Region predicates select cells by cell
center. Circles use great-circle angular distance. Rectangles and polygons use
a wrapped equirectangular lon/lat projection, so they are intended for regional
selection rather than exact large-area or polar spherical polygons. For a
longitude box, `lon_min > lon_max` intentionally selects across the antimeridian.

## Options

Pass common options directly to `generate_grid()`:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4", sphere_radius=6_371_229.0)
raw_grid = generate_grid("R2B4", optimize_global=False)
large_grid = generate_grid("R2B4", max_cells=None)
```

Use `IconGridOptions` when the same configuration is reused:

```python
from grid_generator import IconGridOptions, generate_grid

options = IconGridOptions(sphere_radius=6_371_229.0, spring_iterations=2_000)
grid = generate_grid("R2B4", options=options)
```

`options` may be an `IconGridOptions` instance or a mapping. Explicit keyword
arguments take precedence over values in `options`; omitted keywords retain the
mapping, dataclass, or default value. Unknown option names fail immediately.

```python
options = IconGridOptions(max_cells=1_000, sphere_radius=6_000_000.0)
grid = generate_grid(
    "R1B1",
    options=options,
    max_cells=None,  # overrides only this field
)
```

Common options:

- `max_cells`: generation safety limit. Set to `None` for intentional large
  grids.
- `sphere_radius`: radius used for spherical metric fields.
- `optimize_global`: global grids are optimized by default. Limited-area grids
  use the same setting for their generated global parent. Planar grids do not
  support global optimization; omit the option or pass `False` for planar specs.
- `north_pole_lon`, `north_pole_lat`, and `rotation_angle_degrees`: spherical
  orientation controls.

Advanced options:

- `accelerator`: `"auto"`, `"numpy"`, or `"numba"`. `"auto"` uses NumPy for
  smaller work and uses Numba for selected larger kernels when it is installed;
  explicit `"numba"` raises if the `accelerate` extra is unavailable. Without
  Numba, `"auto"` falls back to the correct deterministic NumPy implementation,
  but that fallback is not performance-equivalent and is not practical for
  high-resolution global grids. Install `icon-grid-generator[accelerate]` for
  the measured large-grid performance.
- `spring_beta` and `spring_iterations`: global spring relaxation controls.
- `indexing`: accepts `"new"` or `"old"`. It is compatibility metadata and part
  of grid identity; both values use the same deterministic in-memory ordering.
- `centre`, `subcentre`, and `number_of_grid_used`: exported metadata fields.

Prefer `sphere_radius` for physical grid metrics. The lower-level `radius`
option controls the displayed Cartesian coordinate radius and is mainly useful
for tests and visualization.

Planar generation automatically disables the default global optimizer when no
value was supplied. Passing `optimize_global=True` explicitly for a planar spec
is rejected; omit it or pass `False`. A default `LimitedAreaGridSpec` generates
the preceding bisection-level global parent with the same options, compacts the
selection, and performs the final refinement locally. `max_cells`, orientation,
and global optimization therefore apply to that construction parent.

`IconGridOptions.fixed_boundary` belongs to the global spring configuration and
has no practical effect on a closed global mesh. It does not configure
post-generation `optimize_grid()` or `diffuse_grid()`; set `fixed_boundary` on
`OptimizationOptions` or `DiffusionOptions` for those transforms.

## Grid Optimization

“Optimization” refers to three different geometry updates. None of them changes
the grid topology: cell/edge/vertex connectivity and refinement relationships
remain fixed. Geometry-dependent arrays are rebuilt after vertices move.

### Default Global Spring Relaxation

`generate_grid("R2B4")` optimizes spherical global grids by default. Generation
refines the spherical triangular mesh in stages and spring-relaxes the grid
after refinement stages. Each edge acts as a spherical spring with target
angular length

```text
target_angle = 1.164 * spring_beta * initial_mean_edge_angle
```

The optimizer integrates a damped velocity and spring-force system,
projects vertices back to the configured Cartesian `radius` after every step,
and stops early when its force/kinetic-energy criteria are met. The intended
effect is a more uniform distribution of edge lengths and cell areas than raw
bisection; topology is identical.

`B0` grids have no bisection stage, so the staged generator has no relaxation
pass to apply even when `optimize_global=True`. The direct helper below relaxes
an already completed spherical grid.

- `optimize_global=True` enables the staged relaxation and is the default.
- `spring_beta` controls the spring rest length. The default `0.9` is the
  compatibility setting and is recommended unless results are independently
  validated.
- `spring_iterations` is a requested iteration cap, not a promise that exactly
  that many steps run. Smaller refinement stages may receive a larger internal
  cap, and convergence criteria can stop a stage early.
- `fixed_boundary` has no practical effect on a closed global grid because it
  has no open boundary vertices.

Use `optimize_global=False` for raw bisection geometry diagnostics. It is not
the normal grid-file path.

The direct helper applies one spring-relaxation pass to an already generated
spherical global grid:

```python
from grid_generator import generate_grid
from grid_generator.transforms import optimize_global_grid

raw = generate_grid("R1B1", optimize_global=False)
relaxed = optimize_global_grid(
    raw,
    {"method": "spring", "iterations": 20},
)
```

This single completed-grid pass is not numerically equivalent to staged
optimization during `generate_grid()`. The method strings `"spring"` and
`"none"` are accepted as shorthands. Passing `"none"`,
`{"method": "none"}`, or zero iterations returns the input object unchanged.

### General Laplacian and Target-Length Smoothing

`optimize_grid()` works on generated spherical, planar, limited-area, and cut
grids. With the default `target_edge_length=None`, every movable vertex uses the
simultaneous update

```text
neighbor_mean = vertex + mean(minimum_image(neighbor - vertex))
new_vertex = vertex + relaxation * (neighbor_mean - vertex)
```

This is Jacobi graph-Laplacian smoothing; it is not a global objective-function
solver. When `target_edge_length` is positive, each incident edge instead
proposes a point at that length along its current direction, and their mean is
the target. For planar grids the target length uses planar coordinate units. On
spherical grids it is a Cartesian chord length in the configured display
`radius`, not metres derived from `sphere_radius`.

```python
from grid_generator import ChannelGridSpec, generate_grid
from grid_generator.transforms import OptimizationOptions, optimize_grid

grid = generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1_000.0))
smoothed = optimize_grid(
    grid,
    OptimizationOptions(
        iterations=5,
        relaxation=0.15,
        fixed_boundary=True,
        target_edge_length=None,
    ),
)
```

`relaxation` lies in `[0, 1]`; smaller values move more cautiously. Zero
iterations or zero relaxation returns the input object unchanged.

### Explicit Diffusion

`diffuse_grid()` uses the same simultaneous neighbor mean but expresses the
fractional move as

```text
effective_step = diffusion_constant * dt * neighbor_weight
new_vertex = vertex + effective_step * (neighbor_mean - vertex)
```

The defaults give `effective_step = 0.1`. Values between zero and one interpolate
toward the neighbor mean. Values above one extrapolate beyond it and can degrade
or invert cells; no automatic stability or cell-inversion limiter is applied.

```python
from grid_generator.transforms import DiffusionOptions, diffuse_grid

diffused = diffuse_grid(
    grid,
    DiffusionOptions(
        iterations=5,
        diffusion_constant=0.05,
        dt=1.0,
        neighbor_weight=1.0,
        fixed_boundary=True,
    ),
)
```

### Boundaries, Projection, and Provenance

- `fixed_boundary=True` freezes every vertex belonging to an edge with only one
  adjacent cell. Set it to `False` only when boundary motion is intended.
- Spherical results are projected back to `radius`; planar results retain their
  z coordinate; doubly or horizontally periodic grids use minimum-image neighbor
  directions and wrap back into the source domain.
- Cuts retain their source geometry, so transforming a cut across a periodic
  seam does not create long seam edges.
- Nontrivial public transforms receive a deterministic new `uuidOfHGrid` based
  on the input UUID and normalized options. `uuidOfParHGrid` remains unchanged,
  and `geometry_transform_source_uuid` records the immediate input.
- Centers, metric fields, normal vectors, coordinate arrays, and metric-summary
  metadata are recomputed. Connectivity and refinement arrays are copied
  unchanged.
- The algorithms do not guarantee monotonic improvement, positive cell area, or
  application-specific numerical quality for arbitrary settings. Validate the
  returned grid.

### Choosing an Operation

| Need | Recommended operation |
| --- | --- |
| Normal global ICON-style generation | Keep `optimize_global=True` |
| Exact raw refinement geometry | `optimize_global=False` |
| Smooth an existing irregular grid | `optimize_grid()` with small relaxation |
| Encourage a planar target edge length | `optimize_grid(..., target_edge_length=...)` |
| Experiment with explicit diffusion | `diffuse_grid()` with effective step at most one |

## Resource Expectations

| Grid | Cells | Edges | Vertices |
| --- | ---: | ---: | ---: |
| `R1B0` | 20 | 30 | 12 |
| `R1B1` | 80 | 120 | 42 |
| `R2B3` | 5,120 | 7,680 | 2,562 |
| `R2B4` | 20,480 | 30,720 | 10,242 |
| `R2B6` | 327,680 | 491,520 | 163,842 |
| `R2B8` | 5,242,880 | 7,864,320 | 2,621,442 |
| `R2B9` | 20,971,520 | 31,457,280 | 10,485,762 |
| `R2B10` | 83,886,080 | 125,829,120 | 41,943,042 |
| `R2B11` | 335,544,320 | 503,316,480 | 167,772,162 |
| `R2B12` | 1,342,177,280 | 2,013,265,920 | 671,088,642 |

Each additional global bisection multiplies all leading counts and approximate
memory work by four. The default `max_cells=2_000_000` rejects larger requests
before allocation. `max_cells=None` removes that safety cap, but not the signed
32-bit index limit. See [Performance and Scaling](design.md#performance-and-scaling)
for measured time, memory, and storage requirements. Large global grids also
require the optional `accelerate` extra for practical runtime.

R2B12 is the last standard R2 grid whose zero-based and exported one-based
cell, edge, and vertex identifiers fit in signed 32-bit integers. R2B13 is
rejected before allocation.

## Coordinates, Units, and Indexing

The in-memory object deliberately carries both computational Cartesian
coordinates and ICON-compatible angular coordinates:

| Field group | Spherical grid | Planar grid |
| --- | --- | --- |
| `vertices`, `cell_center_xyz`, `edge_center_xyz` | Cartesian points on `radius` | Cartesian planar coordinates in spec length units |
| `lon`, `lat`, `vertex_lon`, `vertex_lat`, `edge_lon`, `edge_lat` | Geographic degrees | Values linearly scaled into degree-like plotting ranges; not geodetic |
| Length metrics | Metres derived from `sphere_radius` | Spec coordinate values, exported as metres |
| Area metrics | Square metres derived from `sphere_radius` | Squared spec coordinate values, exported as square metres |

Consequences that matter in downstream code:

- On spherical grids, changing `radius` only rescales Cartesian display
  coordinates. Changing `sphere_radius` rescales length and area metrics.
- Planar `lon`/`lat` exists for visualization and ICON field compatibility. Do
  not use it for geographic region interpretation or map projection.
- In-memory `cells`, `edges`, `cell_edges`, `edge_cells`, connectivity, and
  neighbor tables are zero-based. Missing open-boundary neighbors use `-1`.
- Parent-provenance arrays in `refinement` already use ICON's one-based
  convention and use `0` for no parent.
- NetCDF longitude/latitude variables are radians, and NetCDF connectivity is
  converted to ICON's one-based representation.
- Spherical NetCDF `edgequad_area` is divided by `sphere_radius**2` for ICON
  compatibility and is dimensionless. The in-memory and xarray value remains
  the physical square-metre value. Planar `edgequad_area` is not normalized.

## Grid Object

`IconGrid` is the in-memory object returned by `generate_grid`. It exposes:

- `dims`: cell, edge, and vertex counts.
- `vertices`, `cells`, `edges`, `cell_edges`, and `edge_cells`: core topology.
- `lon`, `lat`, `vertex_lon`, `vertex_lat`, `edge_lon`, and `edge_lat`:
  geographic or projected coordinates.
- `geometry`: metric fields such as cell area and edge length.
- `refinement`: ICON refinement and parent-index fields.
- `metadata`: scalar grid attributes used for export and provenance.
- `to_dict()`, `to_xarray()`, and `to_netcdf(path)`: conversion helpers.

`to_xarray()` includes topology, metric, connectivity, refinement, provenance,
metadata, and unit annotations. Its connectivity indices retain the in-memory
zero-based convention and boundary sentinel `-1`. Parent-provenance fields are
already one-based and use `0` for no parent. Index variables expose these
conventions through `start_index` and `missing_value` attributes.
`to_netcdf()` performs the ICON-specific one-based connectivity conversion and
accepts the field profiles described above.

```python
dataset = grid.to_xarray()
print(dataset[["cell_area", "edge_length", "parent_cell_index"]])
```

Install the optional xarray dependency with
`python -m pip install "icon-grid-generator[xarray]"`.

`to_dict()` returns references to the grid's existing NumPy arrays and nested
dictionaries; it is not a deep copy. Although `IconGrid` and its option/spec
dataclasses are frozen, NumPy buffers remain mutable. Treat completed grids as
immutable values, or copy an array explicitly before modifying it.

`to_netcdf(path, fields="full")` creates missing parent directories and returns
the resulting `Path`. Its optional `sphere_radius` argument must match the
radius used during generation; regenerate with the desired `sphere_radius`
rather than relabeling already computed metrics during export.

Grid identity is deterministic: equal canonical specs and options produce the
same UUID. A limited-area grid, cut, or geometry transform also incorporates its
source UUID, preventing unrelated parents from sharing a derived-grid identity.

## Cutting

Cutting an existing grid is an advanced workflow kept in a focused module:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import CutGridSpec, cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(
    parent,
    CutGridSpec(regions=Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0)),
)
```

For a single-region cut, pass the region directly:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(parent, Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0))
```

Selection is cell based and deterministic:

- The default overlap policy retains cells whose center or any vertex satisfies
  a region predicate. The center-only policy is available explicitly. Multiple
  regions are combined by union.
- `mode="keep"` retains the selected union. `mode="remove"` takes its
  complement.
- `RegionSelectionOptions.buffer_rings=N` then adds `N` cell-neighbor rings.
  `boundary_depth=N` is the backwards-compatible spelling. On a remove cut, this
  expands the retained complement, not the removed region.
- `smoothing_depth` does **not** move vertices or smooth geometry. It fills the
  ICON `smooth_c_ctrl` refinement field and records the requested depth as
  metadata for downstream consumers.
- The result is compacted and given ICON boundary-control ordering. Use its
  parent-index refinement fields to map values back to the source; do not assume
  that a source cell keeps the same local index.
- Boundary metrics are recomputed. Clipped duals are the default; the mirrored
  ghost closure is an explicit option.
- A selection containing no cells raises `ValueError`.

When passing a `CutGridSpec`, put all cut options in that object; supplying the
same options again as `cut_grid()` keywords is rejected. `LimitedAreaGridSpec`
supports one region and local final refinement; `cut_grid()` supports multiple
regions, keep/remove mode, and `smoothing_depth` on an existing grid.

## Diagnostics And Transforms

Diagnostics and postprocessing utilities are available from focused modules:

```python
from grid_generator.diagnostics import (
    check_grid,
    grid_statistics,
    triangle_properties,
    cell_divergence,
    cell_vorticity_fnorm,
)
from grid_generator.transforms import (
    diffuse_grid,
    optimize_global_grid,
    optimize_grid,
)
```

Their result and option dataclasses are exported from the same submodules. See
[Grid Optimization](#grid-optimization) for transform behavior and option
semantics.

The diagnostic helpers have intentionally narrow meanings:

- `check_grid()` checks core shapes, finite coordinates, index bounds,
  duplicate edges, boundary presence, and the expected closed-mesh Euler
  characteristic. `ok=True` is structural validation, not proof of positive
  areas, correct scientific metrics, or suitability for a numerical scheme.
- `grid_statistics()` summarizes counts plus extrema and means of existing area
  and edge-length fields.
- `triangle_properties()` applies the Euclidean cosine rule to stored edge
  lengths. Its angles are a practical shape diagnostic, not exact spherical
  interior angles on coarse global cells.
- `cell_divergence()` expects one edge-normal flux value per edge and applies
  the stored edge orientation, edge length, and cell area.
- `cell_vorticity_fnorm()` takes one value per vertex, arithmetically averages
  the three cell vertices, and divides by a nonzero scalar or per-cell Coriolis
  field. It is a lightweight postprocessor, not a complete dynamical operator.

A useful minimum validation pattern is:

```python
import numpy as np

from grid_generator.diagnostics import check_grid, triangle_properties

result = check_grid(grid)
if not result.ok:
    raise ValueError(result.errors)

triangles = triangle_properties(grid)
assert np.all(np.isfinite(triangles.area))
assert np.all(triangles.area > 0.0)
```

## Visualization

Use the lightweight SVG helper for quick dependency-free grid previews:

```python
from grid_generator import generate_grid
from grid_generator.visualization import write_svg

grid = generate_grid("R1B1", spring_iterations=20)
write_svg(grid, "global_r1b1.svg")
```

This is a dependency-free diagnostic edge plot, not a map projection or a
metric-preserving scientific visualization. It uses the grid's angular plotting
coordinates, omits seam-crossing segments, and deterministically subsamples
when the edge count exceeds `max_edges` (default 20,000). Create the destination
directory before calling it.

## Root Package Exports

The following names are available directly from `grid_generator`:

- `ChannelGridSpec`
- `GlobalGridSpec`
- `IconGrid`
- `IconGridOptions`
- `LimitedAreaGridSpec`
- `OpenBoundaryOptions`
- `ParallelogramGridSpec`
- `Region`
- `RegionSelectionOptions`
- `TorusGridSpec`
- `generate_grid`
- `generate_grid_to_netcdf`
