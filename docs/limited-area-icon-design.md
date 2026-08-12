# Valid Limited-Area and Open-Boundary ICON Grids

Status: implemented for spherical limited-area generation and direct cuts.

The implementation is informed by ICON-NWP's grid-reader and numerical-operator
contracts. The defaults retain deterministic Python implementation, finite
metrics, structural provenance, and valid ICON ordering. Bit-for-bit files and
UUID reuse are not goals.

## Requirements and choices

Some properties are required for a functional grid and are never compatibility
options:

- reciprocal, manifold topology with exactly one real cell on each boundary
  edge;
- metrics recomputed for the retained and finally optimized geometry;
- finite missing-side distances;
- ICON boundary-control ordering and matching start/end tables for export; and
- structural parent fields when the regional mesh is locally refined.

Other properties are scientific or numerical choices:

| Choice | Default | Alternative |
| --- | --- | --- |
| Construction | select at B-1 and refine once | cut the complete final grid |
| Cell inclusion | circumdisk intersection | center or center/vertex overlap |
| Mask cleanup | remove one-neighbor ears | preserve the raw predicate result |
| Boundary dual | clipped at the physical edge | mirrored ghost closure |
| Entity ordering | ICON boundary prefixes | source order for in-memory analysis |
| Local optimization | fixed-boundary spring, 2,000 iteration cap | explicit lower cap or zero iterations |

Historical count reconciliation, opaque rectangle scaling, old UUID reuse,
omission of useful metadata, and reproduction of non-finite values are not
implemented.

## Public policy objects

The public policy signatures are:

```text
RegionSelectionOptions(
    inclusion="circumradius",     # or "overlap" / "center"
    cleanup="remove_ears",        # or "none"
    buffer_rings=0,
)

OpenBoundaryOptions(
    metric_closure="clipped",     # or "mirrored"
    indexing_depth=14,
    ordering="icon",              # or "source"
)
```

`LimitedAreaGridSpec` accepts both objects, plus
`construction="refine_last" | "cut_final"` and
`local_optimization_iterations`. The old `boundary_depth` argument remains as
an alias for selection buffer rings; it is deliberately not the ICON boundary
indexing depth. Supplying both buffer spellings is rejected.

`CutGridSpec` and the direct `cut_grid()` form accept the same selection and
boundary policies. A source-ordered grid is useful for analysis, but strict
NetCDF export rejects it because ICON's start/end ranges require grouped
boundary prefixes.

## Rotated-pole regions

`Region.rotated_lonlat_box(...)` defines the region in an explicit rotated-pole
coordinate system. For the CH domains the parameters can be written as:

```text
Region.rotated_lonlat_box(
    pole_lon=-170.0,
    pole_lat=43.0,
    center_lon=-1.01,
    center_lat=-0.53,
    half_width_lon=5.93,
    half_width_lat=4.01,
)
```

This transform is independent of the icosahedral-pole options used to orient
the global parent and is not CH-specific.

## Native refine-last pipeline

For a requested `R<n>B<k>` grid with `k > 0`, the default path:

1. generates the global `R<n>B<k-1>` parent with the requested global options;
2. evaluates the region using cell circumcenters and circumradii;
3. applies deterministic cleanup and optional neighbor-ring buffering;
4. compacts the selected parent with typed NumPy index arrays;
5. bisects the open parent once;
6. spring-relaxes only the local child while keeping boundary vertices fixed;
7. rejects the optimized vertices if they are non-finite or invert a cell;
8. builds open-boundary controls and applies stable ICON permutations;
9. recomputes coordinates, connectivity, metrics, and index tables; and
10. records the requested resolution, actual construction parent, and resolved
    policies in metadata and grid identity.

At bisection level zero the implementation falls back to direct cutting
because no preceding bisection level exists. `construction="cut_final"` is
also available when an exact subset of an existing final-resolution parent is
the desired scientific object.

The local refinement records:

- four one-based `parent_cell_index` entries per compact coarse parent;
- ICON child-cell type codes;
- one-based parent edge indices and edge child types; and
- positive parent-vertex or negative parent-edge values in
  `parent_vertex_index`.

The negative midpoint encoding is intentional ICON provenance, not a missing
index.

## Region selection

`inclusion="circumradius"` is the default. For circles and longitude/latitude
boxes it applies a cell-circumcenter test expanded by the cell circumradius.
The same rule, followed by default ear cleanup, gives stable entity sets across
the tested R2B2-R2B5 circular, unrotated-box, and rotated-pole-box grids.
`inclusion="overlap"` retains a cell when its center or
any vertex is inside the region; `inclusion="center"` preserves the narrower
center-defined interpretation. Package-specific polygons use overlap semantics
because they do not have a directly corresponding circumdisk predicate.

The default cleanup removes raw cells with at most one selected edge neighbor.
This prevents isolated one-cell protrusions and stabilizes domain boundaries.
`cleanup="none"`
preserves the exact predicate result.

Selection and expansion use boolean masks and typed arrays rather than Python
sets, so their memory cost remains linear for large parents.

## Boundary metrics

For a spherical boundary edge under `metric_closure="clipped"`:

- the real side stores its center-to-edge-center distance;
- the missing side stores exactly zero;
- the dual edge ends at the physical boundary edge center; and
- `edgequad_area` and dual-area contributions use that clipped dual.

The implementation evaluates the established spherical metric kernels with a
temporary edge-center ghost, then removes the exterior distance. This keeps
normal and orientation conventions identical to the closed-grid kernels while
producing finite open-grid values.

`metric_closure="mirrored"` reflects the real half of the dual across the
boundary. It doubles boundary dual length and fills the exterior side distance
with the real-side distance. This is a useful ghost-cell convention and the
former open-planar package behavior, but it is not the physical default for
regional grids.

Direct planar cuts apply the same explicit choice. Standalone planar grid specs
retain their established metric convention because their public API does not
yet carry an open-boundary policy.

Geometry transforms rebuild spherical open metrics with the recorded closure,
so optimization and diffusion cannot silently revert a clipped grid to a
mirrored one.

## ICON boundary indexing

The boundary indexer follows the established ICON grid convention:

1. vertices on one-cell edges receive level one;
2. vertex levels propagate inward through incident cells up to the configured
   depth;
3. cells receive the minimum positive vertex level, or zero;
4. edge controls are formed from adjacent cell controls, with the controlled
   to uncontrolled transition mapped to zero;
5. cells at levels 1 through 5, edges at 1 through 10, and vertices at 1
   through 5 are stably moved to prefixes; and
6. fixed-width `start_idx_*` and `end_idx_*` tables describe those prefixes and
   the unordered interior.

Depth 14 is the default. It supports ICON's usual nudging-width assumptions and
is independent of selection buffer rings.

## NetCDF boundary contract

Internal connectivity remains zero-based with `-1` missing. Regional NetCDF
connectivity uses one-based real indices and `-1` missing consistently for
neighbors, adjacent cells, and vertex incidence. Global and periodic output is
unchanged.

Before writing a regional grid, the writer verifies:

- at least one open boundary edge exists;
- ICON ordering was selected and the boundary prefixes are contiguous;
- required area, length, and distance metrics are finite and physical; and
- hierarchy fields have the correct entity dimensions.

Metadata includes `grid_root`, `grid_level`, `boundary_depth_index`, the full
construction-parent name and UUID, center/subcenter fields,
`number_of_grid_used`, and all resolved policies. `uuidOfParHGrid` identifies
the compact immediate parent against which the structural indices are defined;
`construction_parent_uuid` retains the full global source identity.

Spherical limited-area and spherical cut files retain ICON
`grid_geometry = 1`. ICON defines geometry value `3` as a planar channel, not
as a generic limited-area grid. A separate `open_boundary = 1` attribute drives
the package's regional validation and missing-neighbor encoding without causing
ICON to disable spherical behavior. This distinction also lets the
Cartesian-free `"icon"` field profile select ICON's spherical reconstruction
path.

## Scaling and remaining work

The default avoids constructing the four-times-larger final-resolution global
grid. Compaction, incidence construction, bisection, and metric calculation use
typed arrays and the existing accelerated kernels where available.

Limited-area specs can be passed to either `generate_grid()` or the file-oriented
`generate_grid_to_netcdf()` API. The in-memory call deliberately returns a full
`IconGrid` and therefore retains its generated construction parent during
extraction. The file-oriented call instead uses the compact global engine,
including resumable bisection checkpoints above the base-stage budget, and
evaluates region predicates in bounded cell chunks. It materializes only the
selected regional mesh for boundary ordering, regional metrics, and export.
High-resolution parents require `max_cells=None`, the `accelerate` extra, and
disk-backed output/checkpoint storage.

Standalone non-periodic planar specs also remain on their existing mirrored
metric and zero-control conventions. Applying the shared policy/indexing layer
to those specs requires a deliberate public-API migration rather than silently
changing their numerical boundary definition.

## Validation evidence

Maintained tests cover:

- rotated-pole selection and overlap versus center inclusion;
- four-child structural provenance and midpoint encoding;
- boundary controls, permutations, and start/end tables;
- clipped versus mirrored lengths and distances;
- finite metric and topology invariants after generation and transforms;
- `-1` missing-index NetCDF encoding;
- ICON geometry-enum behavior for spherical regional and planar grids; and
- unchanged global and periodic contracts.

Large operational domains remain manual acceptance cases and are not repository
fixtures. A newly generated horizontal-grid UUID requires matching EXTPAR,
vertical-grid, initial-condition, and lateral-boundary products before an
operational ICON run.
