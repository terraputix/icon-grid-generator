# Generating Limited-Area ICON Grids

Use `LimitedAreaGridSpec` to generate an open spherical grid for a regional ICON
domain. A specification combines:

- the global `R<n>B<k>` grid that determines the target resolution and
  refinement topology;
- optional generation settings that orient that global grid;
- a geographic or rotated-pole region;
- optional rules for selecting cells and treating the open boundary; and
- an optional grid name.

For an operational-resolution grid, prefer `generate_grid_to_netcdf()`. It can
build a large global construction parent through resumable, disk-backed stages
and materializes only the selected regional mesh. Use `generate_grid()` when
you need the resulting `IconGrid` in memory for inspection or further Python
processing.

## Quick start

This example creates a modest regional grid in memory and writes the complete
ICON-style NetCDF schema:

```python
from grid_generator import LimitedAreaGridSpec, Region, generate_grid

spec = LimitedAreaGridSpec(
    parent="R2B4",
    region=Region.lonlat_box(
        lon_min=3.0,
        lon_max=17.0,
        lat_min=43.0,
        lat_max=50.0,
    ),
    name="central_europe_R02B04",
)

grid = generate_grid(spec)
print(grid.name, grid.dims)
grid.to_netcdf("central_europe_R02B04.nc", fields="full")
```

`parent="R2B4"` describes the requested final resolution. With the default
construction mode, the generator builds `R2B3`, selects and compacts the
region, and performs the final bisection locally. You do not need to specify
the construction parent yourself.

For a larger grid that is needed only as a file, install the acceleration and
NetCDF extras and use the file-oriented API:

```bash
python -m pip install "icon-grid-generator[accelerate,netcdf]"
```

```python
from grid_generator import LimitedAreaGridSpec, Region, generate_grid_to_netcdf

spec = LimitedAreaGridSpec(
    parent="R2B10",
    region=Region.lonlat_box(
        lon_min=3.0,
        lon_max=17.0,
        lat_min=43.0,
        lat_max=50.0,
    ),
    name="central_europe_R02B10",
)

generate_grid_to_netcdf(
    spec,
    "central_europe_R02B10.nc",
    max_cells=None,
    accelerator="numba",
    work_dir="central_europe_R02B10-work",
    fields="full",
)
```

Put both the output and `work_dir` on disk-backed storage. When staged global
construction is needed, the work directory contains resumable checkpoints and
can be removed after a successful export.

## Describing an operational domain

Operational centers describe their regional domains in different coordinate
systems. The API supports ordinary longitude/latitude extents, rotated-pole
boxes, circles, oriented rectangles, and polygons. Coordinates are in degrees.

The examples in this section demonstrate the corresponding construction
patterns. They do not reproduce or certify a currently operational grid. To
replace an existing MeteoSwiss, DWD, or other institutional grid, use that
system's authoritative parent resolution, orientation, region definition, and
boundary configuration.

### Rotated-pole domains

Use `Region.rotated_lonlat_box()` when a domain is defined in a rotated-pole
coordinate system, as in a MeteoSwiss ICON-CH-style setup:

```python
from grid_generator import LimitedAreaGridSpec, Region

ch_region = Region.rotated_lonlat_box(
    pole_lon=-170.0,
    pole_lat=43.0,
    center_lon=-1.01,
    center_lat=-0.53,
    half_width_lon=5.93,
    half_width_lat=4.01,
)

ch_spec = LimitedAreaGridSpec(
    parent="R2B8",
    region=ch_region,
    name="ch_domain_R02B08",
)
```

`center_lon`, `center_lat`, `half_width_lon`, and `half_width_lat` are expressed
in the rotated coordinate system. `pole_lon` and `pole_lat` define that system
in geographic coordinates. These values are independent of
`north_pole_lon`, `north_pole_lat`, and `rotation_angle_degrees`, which orient
the icosahedral global parent.

### Geographic and polygon domains

Use `Region.lonlat_box()` when a domain is given as a geographic extent, for
example in a DWD-style regional workflow. Replace the illustrative bounds with
the authoritative domain definition:

```python
from grid_generator import LimitedAreaGridSpec, Region

dwd_region = Region.lonlat_box(
    lon_min=-10.0,
    lon_max=30.0,
    lat_min=40.0,
    lat_max=60.0,
)

dwd_spec = LimitedAreaGridSpec(
    parent="R2B8",
    region=dwd_region,
    name="de_domain_R02B08",
)
```

For an irregular footprint, pass its geographic vertices to
`Region.polygon()`:

```python
region = Region.polygon(
    (
        (-10.0, 42.0),
        (22.0, 42.0),
        (30.0, 55.0),
        (5.0, 61.0),
        (-10.0, 52.0),
    )
)
```

The available region constructors are:

| Region | Use when the domain is defined by |
| --- | --- |
| `Region.lonlat_box(...)` | geographic longitude/latitude bounds |
| `Region.rotated_lonlat_box(...)` | bounds in a rotated-pole coordinate system |
| `Region.circle(...)` | a center and angular radius |
| `Region.rectangle(...)` | a locally oriented geographic rectangle |
| `Region.polygon(...)` | an irregular geographic footprint |

For `Region.lonlat_box()`, `lon_min > lon_max` intentionally selects a box that
crosses the antimeridian.

## Choosing the grid settings

The defaults are intended to produce an ICON-ordered regional grid with a
clean, covered boundary. Most users need to choose only `parent`, `region`, and
`name`.

### Parent grid and construction mode

`parent` is a global grid name such as `"R2B8"` or a `GlobalGridSpec`. It
controls the final regional resolution and refinement topology. Pass
`north_pole_lon`, `north_pole_lat`, and `rotation_angle_degrees` to the
generation call when the global parent needs a non-default orientation.

The default `construction="refine_last"` is recommended for a new regional
grid. It selects cells at bisection level `B-1`, compacts the result, and
refines it once locally. This preserves immediate-parent refinement information
without constructing the four-times-larger final global grid.

Use `construction="cut_final"` only when the regional grid must be an exact
cell subset of the final-resolution global parent:

```python
spec = LimitedAreaGridSpec(
    parent="R2B8",
    region=region,
    construction="cut_final",
    name="exact_R02B08_cut",
)
```

`cut_final` needs the final-resolution global construction parent and therefore
uses substantially more time, memory, and checkpoint storage. At bisection
level zero it is selected automatically because no preceding level exists.

### Cell selection and buffer rings

The default selection policy is equivalent to:

```python
from grid_generator import RegionSelectionOptions

selection = RegionSelectionOptions(
    inclusion="circumradius",
    cleanup="remove_ears",
    buffer_rings=0,
)
```

Choose an inclusion rule according to the domain contract:

| `inclusion` | Result |
| --- | --- |
| `"circumradius"` | Includes cells whose circumdisks intersect supported boxes or circles; recommended for boundary coverage. |
| `"overlap"` | Includes cells whose center or at least one vertex is inside the region. |
| `"center"` | Includes only cells whose centers are inside the region; produces the narrowest selection. |

Polygons always use center-or-vertex overlap because they have no corresponding
circumdisk predicate. `cleanup="remove_ears"` removes one-cell protrusions from
the selected mask. Use `cleanup="none"` only when the raw predicate result is
part of the required domain definition.

`buffer_rings` adds complete cell-neighbor rings after selection and cleanup.
This is useful when the supplied region describes the interior area of interest
and the grid must also include a halo for relaxation, interpolation, or boundary
forcing:

```python
spec = LimitedAreaGridSpec(
    parent="R2B8",
    region=region,
    selection=RegionSelectionOptions(buffer_rings=2),
)
```

Do not confuse selection buffer rings with ICON boundary indexing depth. The
older `boundary_depth` argument is retained only as an alias for
`selection.buffer_rings`; new code should use `buffer_rings`.

### Open-boundary policy

The default boundary policy is:

```python
from grid_generator import OpenBoundaryOptions

boundary = OpenBoundaryOptions(
    metric_closure="clipped",
    indexing_depth=14,
    ordering="icon",
)
```

Keep `ordering="icon"` for NetCDF export. It groups boundary-control levels and
creates matching ICON start/end tables. `ordering="source"` preserves the
parent entity order for in-memory analysis, but the strict regional writer
rejects it.

`indexing_depth=14` supports the usual ICON boundary and nudging-zone layout.
Change it only when the consuming model configuration requires a different
depth. It does not add cells to the selected domain.

`metric_closure="clipped"` terminates each dual edge at the physical boundary
and is the normal finite-volume choice. `"mirrored"` reflects the interior half
of the dual across the boundary for consumers that explicitly require a
ghost-cell convention.

### Local optimization

With `construction="refine_last"`, the compact child grid is spring-relaxed
with boundary vertices fixed. The default cap is 2,000 iterations. Set
`local_optimization_iterations=0` when unrelaxed local refinement geometry is
required, or set a smaller positive cap for exploratory and lower-cost runs.

This setting is separate from `spring_iterations`, which controls optimization
of the generated global construction parent.

## Producing and checking the result

Use `fields="full"` for general exchange unless the downstream reader's exact
field requirements are known. The `"icon"`, `"icon4py"`, and `"reduced"`
profiles produce smaller files for their documented consumer scopes; see
[NetCDF field profiles](api.md#netcdf-field-profiles).

The writer checks that the result has an open boundary, ICON-contiguous
boundary prefixes, finite physical metrics, and correctly shaped hierarchy
fields before publishing the file. A spherical regional file uses
`grid_geometry = 1` and carries the independent `open_boundary = 1` attribute.
Missing neighbors are encoded as `-1` in exported connectivity.

For an in-memory result, basic inspection can include:

```python
import numpy as np

grid = generate_grid(spec)
boundary_edges = np.any(grid.edge_cells < 0, axis=1)

print(grid.dims)
print("boundary edges:", np.count_nonzero(boundary_edges))
print("grid UUID:", grid.metadata["uuidOfHGrid"])
print("construction parent:", grid.metadata["construction_parent_grid_name"])
```

Generation creates a new horizontal-grid UUID. It does not recreate a
historical operational file bit for bit or reuse that file's UUID. Before a new
grid can drive an operational ICON run, generate matching EXTPAR, vertical-grid,
initial-condition, and lateral-boundary products and validate the complete
model configuration.

## Large-grid workflow

Operational resolutions can have very large global construction parents even
when the selected regional result is much smaller. For these cases:

1. use `generate_grid_to_netcdf()` rather than creating an in-memory
   `IconGrid`;
2. install the `accelerate` extra and select `accelerator="numba"`;
3. set `max_cells=None` only after confirming the intended resolution;
4. place the output and `work_dir` on disk-backed storage with enough space for
   checkpoints and atomic replacement files; and
5. leave `resume=True` to reuse completed construction stages after an
   interruption.

`chunk_size` bounds regional predicate evaluation and NetCDF-only work arrays.
It does not partition the compact core topology or the final selected regional
mesh. Use distinct output and work-directory paths for concurrent jobs.

See [Resource Expectations](api.md#resource-expectations) for global parent
sizes and [Performance and Scaling](design.md#performance-and-scaling) for
measured requirements.

## Implementation details

The preceding sections are sufficient for normal grid generation. The details
below explain the boundary and provenance fields that advanced users may need
when integrating a generated grid with ICON tooling.

### Refine-last pipeline and provenance

For a requested `R<n>B<k>` grid with `k > 0`, the default pipeline:

1. generates the global `R<n>B<k-1>` construction parent;
2. evaluates the region, cleans the mask, and adds requested buffer rings;
3. compacts the selected parent cells into an open mesh;
4. bisects that compact parent once;
5. locally optimizes the child while holding boundary vertices fixed;
6. creates boundary-control fields and applies stable ICON permutations; and
7. recomputes connectivity, coordinates, metrics, index tables, metadata, and
   grid identity.

The locally refined result records four one-based `parent_cell_index` entries
per compact coarse parent, ICON child-cell type codes, one-based parent-edge
indices and child types, and parent-vertex provenance. In
`parent_vertex_index`, a positive value refers to a parent vertex and a
negative value refers to a parent edge whose midpoint created the child vertex;
the negative values are provenance, not missing indices.

`uuidOfParHGrid` identifies the compact immediate parent against which these
structural indices are defined. `construction_parent_uuid` retains the UUID of
the full global source grid.

### Boundary indexing

The boundary indexer starts with vertices on edges that have one real adjacent
cell, then propagates levels inward through incident cells up to
`indexing_depth`. Cell and edge controls are derived from those vertex levels.
Stable permutations move the active control levels to entity prefixes, and
fixed-width `start_idx_*` and `end_idx_*` tables describe those prefixes and
the unordered interior.

Internal connectivity is zero-based and uses `-1` for a missing neighbor.
Regional NetCDF connectivity uses one-based real indices and retains `-1` for
missing neighbors, adjacent cells, and vertex incidence.

### Boundary metrics

With `metric_closure="clipped"`, the real side of a boundary edge stores its
cell-center-to-edge-center distance, the missing side stores zero, and the dual
edge ends at the physical boundary. `edgequad_area` and dual-area contributions
use that clipped dual.

With `metric_closure="mirrored"`, the generator reflects the real half of the
dual across the boundary. The boundary dual length is doubled and the exterior
side distance equals the real-side distance. Geometry transforms rebuild open
metrics using the policy recorded on the grid.

### Metadata and NetCDF contract

Regional metadata records the requested `grid_root` and `grid_level`, the
construction-parent name and UUID, the boundary indexing depth, resolved
selection and boundary policies, center/subcenter fields, and
`number_of_grid_used`.

Spherical limited-area and spherical cut files remain ICON geometry type `1`.
Geometry type `3` denotes a planar channel, not a generic limited-area grid.
The separate `open_boundary = 1` attribute activates regional validation and
missing-neighbor encoding without disabling spherical behavior.

The implementation maintains reciprocal manifold topology, requires exactly
one real cell on each open boundary edge, recomputes metrics after final
geometry changes, and discards a local-optimization update if it is non-finite
or inverts a cell. Tests cover region-selection policies, refinement
provenance, boundary controls and permutations, clipped and mirrored metrics,
open NetCDF connectivity, and unchanged global and periodic contracts.
