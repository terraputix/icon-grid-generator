# Design Notes

ICON Grid Generator builds in-memory `IconGrid` objects through a small,
deterministic pipeline:

1. Parse and validate a grid spec plus `IconGridOptions`.
2. Build geometry: vertices, cells, centers, and lon/lat coordinates.
3. Build topology: edges, cell-edge relations, edge-cell relations, and ICON
   connectivity tables.
4. Build metrics: cell areas, edge lengths, dual quantities, and normal vectors.
5. Build refinement/provenance fields.
6. Assemble metadata, UUIDs, conversion helpers, and optional NetCDF output.

## Compatibility Contracts

- Public grid specs and `generate_grid()` are the main API.
- `IconGrid.dims` and array shapes must remain predictable from the spec.
- Internal topology arrays are zero-based; exported NetCDF index fields are
  one-based where ICON expects that convention.
- Metadata keys used by UUIDs, NetCDF export, and examples should not drift
  accidentally.
- Grid UUIDs must stay stable for unchanged canonical inputs.

## Feature Boundaries

- The package is Python API first. Keep command wrappers and workflow glue out
  unless they support an existing public API use case.
- Global, planar, limited-area, optimization, diffusion, diagnostics, and
  NetCDF export features should share the `IconGrid` data model.
- `grid_generator.py` is the public facade. Keep large implementation concerns
  in focused private modules such as `_global.py`, `_netcdf.py`, `_planar.py`,
  and `_limited_area.py`; preserve thin private aliases only where internal
  builders/tests still rely on them.
- Triangular grids are the supported cell family. Add other cell families only
  with explicit public API, NetCDF, and diagnostic contracts.
- Ragged planar grids are deterministic Python variants; test structural
  validity and exported contracts rather than assuming metric identity with
  regular planar grids.
- Parent/provenance indices belong in `IconGrid.refinement`; metadata should
  carry descriptive scalar attributes only.

## Architectural Decisions

- Global grid generation uses staged spring relaxation by default; raw
  bisection remains available with `optimize_global=False` for diagnostics and
  topology checks.
- Planar triangular variants share one builder pipeline. The spec object
  carries variant flags such as `periodic` and `periodic_x`; `_planar.py`
  dispatches geometry, topology, and metric behavior from those flags instead
  of adding separate public generation entry points.
- Doubly periodic triangular grids use the two coupled lattice vectors of the
  skew fundamental domain. Crossing the periodic y boundary therefore applies
  both a y wrap and the corresponding x shift. Geometry transforms use the same
  minimum-image convention when averaging neighboring vertices.
- Planar dual-edge lengths are computed from generated cell centers through
  edge centers rather than assumed equilateral formulas. Planar vertex dual
  areas use one third of each incident triangle, so open-grid dual areas
  partition the total primal area without assigning full interior control
  volumes to boundary vertices. Planar `edgequad_area` uses the cross product
  of primal and dual edge vectors, so sheared grids include their actual
  intersection angle rather than assuming orthogonality.
- Limited-area and cut grids are compacted views of an `IconGrid` selected by
  region predicates plus optional boundary expansion. They deliberately reuse
  the open-mesh topology, metrics, and refinement reconstruction path so
  regional extraction does not fork the grid contract.
- Geometry optimization and diffusion are post-generation transforms that
  preserve topology and rebuild geometry-derived fields. Cut grids retain the
  source spec needed for periodic coordinate reconstruction. A nontrivial
  public transform receives a deterministic UUID derived from the input grid
  UUID and canonical transform options while preserving `uuidOfParHGrid`;
  `geometry_transform_source_uuid` records that immediate input and zero-step
  transforms return the input grid unchanged. Global spring
  relaxation shares the same module but remains part of global generation when
  `optimize_global=True`.
- Spherical `dual_area` follows the ICON grid-file contract: each vertex area
  is the sum of `0.25 * edge_length * dual_edge_length` over incident edges.
  It is an edge-quadrilateral metric field and is not forced to sum exactly to
  the spherical cell-area total.
- Optional Numba acceleration is an implementation detail selected through
  `IconGridOptions.accelerator`; NumPy remains the required baseline.
- UUIDs use deterministic UUIDv5 payloads derived from canonical specs and
  options. Limited-area and cut-grid payloads include the source parent UUID,
  and `uuidOfParHGrid` records that source. Any payload change is a
  compatibility change.
- NetCDF export is an internal module boundary. Public users should call
  `IconGrid.to_netcdf(path)`.
- For compatibility with established ICON grid files, spherical NetCDF
  `edgequad_area` values are normalized by `sphere_radius**2`. The exported
  variable is therefore dimensionless (`units = "1"`) and carries a
  `normalization` attribute. In-memory `IconGrid.geometry` retains physical
  square-metre values. This compatibility convention is not applied to planar
  grids.
- Xarray public connectivity is zero-based with `-1` for missing neighbors;
  parent-provenance arrays remain one-based with `0` for no parent. Variables
  carry explicit `start_index` and `missing_value` attributes.
- Pipeline stage results use frozen dataclasses to keep builder boundaries
  explicit. Arrays remain mutable NumPy buffers during construction; callers
  should treat completed `IconGrid` objects as immutable values.
- `grid_generator.py` owns public specs, validation-facing helpers, metadata,
  UUID payloads, and the `generate_grid()` facade. Large implementation
  concerns should stay in focused private modules rather than growing the
  facade again.
- Performance checks live behind `make perf-check` and are intentionally
  separate from default CI-style checks because runtime varies with local load.

## Optimization Algorithms

### Staged Global Spring System

Default global generation operates recursively across bisection levels. A
parent stage is generated and relaxed before its child is refined, and the child
is then relaxed as another stage. The spring kernel normalizes coordinates to
the unit sphere, computes the initial mean angular edge length, and uses
`1.164 * beta_spring` times that mean as its rest angle. Incident edge forces
are accumulated per vertex and integrated with a damped velocity. Every
position and velocity update is projected into the sphere tangent geometry.

The time step is `0.016` for the first 50 steps, increases through step 150, and
is `0.08` thereafter. Iteration is bounded by the stage cap and may terminate
early when the force statistic reaches its observed maximum or kinetic energy
falls below `0.001` of its observed maximum. For stages whose parent has fewer
than 100,000 cells, the internal cap is ten times the requested
`spring_iterations`; the public metadata records the requested setting rather
than the realized number of integration steps.

A `B0` specification returns the completed root geometry directly because no
bisection stage exists; consequently the staged relaxation path is not entered.

This path aims to reduce edge-length and cell-area variation while retaining
the exact refined topology. It is a compatibility-oriented heuristic, not an
optimization with a reported scalar objective or formal convergence proof.

### General Smoothing and Diffusion

`optimize_grid()` and `diffuse_grid()` build undirected vertex adjacency from
the immutable edge table. Each iteration is Jacobi-style: all targets are
computed from the previous vertex array and applied simultaneously.

Without `target_edge_length`, `optimize_grid()` moves a fraction `relaxation`
toward the mean neighbor displacement. With a target length, each active edge
direction is normalized, scaled to the requested length, and translated back to
the current vertex before the incident proposals are averaged. `diffuse_grid()`
uses the same mean displacement multiplied by
`diffusion_constant * dt * neighbor_weight`.

Periodic displacements use the same coupled lattice minimum-image operation as
metric generation. Spherical vertices are renormalized to `radius`; planar z
coordinates are restored; periodic planar coordinates are wrapped to the
fundamental domain. Open-boundary vertices are excluded from updates when
`fixed_boundary=True`.

After movement, the grid rebuilds centers, projected coordinates, metric fields,
normal vectors, and summary metadata. Topology and refinement arrays are
unchanged. Public transforms derive a new UUID from the source UUID, operation,
and canonical option payload. No inversion detector or line search is part of
the update, so callers remain responsible for checking scientific quality.

## Limitations

- Connectivity and NetCDF index fields use signed 32-bit integer arrays. Global
  grids up to current large operational scales such as `R02B11` are within that
  range; generation fails early when cells, edges, or vertices would exceed the
  int32 index limit.
- Global bisection parent/provenance fields are tracked structurally during
  refinement. Some defensive fallback paths can still use rounded coordinate
  matching when geometry is constructed outside the normal global pipeline.
- Spherical metrics use double-precision trigonometric formulas. They are
  appropriate for supported resolutions, but extremely small triangles can make
  angle-sum area formulas and `arccos`-based distances more sensitive to
  floating-point cancellation.
- The implementation assumes closed global triangular meshes have vertex
  valence at most six. Limited-area and planar grids use separate open-mesh
  paths where boundary sentinels are expected.
- Longitude/latitude rectangles and polygons are center-selection predicates
  evaluated in a wrapped equirectangular lon/lat plane. Circles use great-circle
  angular distance. Use small regional rectangles/polygons away from the poles,
  or provide an explicit scientific comparison for polar and very large areas.
- Planar `lon`/`lat` fields are normalized compatibility and visualization
  coordinates, not a geographic CRS. Scientific planar calculations should use
  Cartesian coordinates and metric arrays.
- `smoothing_depth` on a cut is an ICON control-field value written uniformly
  to `smooth_c_ctrl`; it does not invoke the geometry smoothing algorithms.
- `check_grid()` is a structural check. It does not independently rederive all
  metric fields, test cell inversion, or certify a grid for a numerical model.
- `write_svg()` is an equirectangular-style diagnostic preview. It omits
  periodic seam segments and can subsample edges, so it must not be used to
  assess metric distances, areas, or complete seam topology.

## Performance and Scaling

For large global `R<n>B<k>` grids, the useful scaling variable is the effective
refinement frequency

```text
f = n * 2^k
```

The main asymptotic behavior follows directly from `f`:

```text
cells    = 20 * f^2      = 20 * n^2 * 4^k
edges    = 30 * f^2      = 30 * n^2 * 4^k
vertices = 10 * f^2 + 2  = 10 * n^2 * 4^k + 2
```

Raw topology/metric generation time, peak memory, and NetCDF file size are
therefore all expected to scale approximately as `O(n^2 * 4^k)` for
sufficiently large global grids. Equivalently, each additional bisection level
roughly multiplies work and output size by four.

The optimized measurements below used default staged spring relaxation on an
exclusive dual-socket AMD EPYC 7713 node with 128 physical cores (256 hardware
threads) and about 446 GiB of scheduler-visible memory. The software stack was
Python 3.11, NumPy 2.4.6, Numba 0.66, and 128 Numba threads. These are
single-run results, not service-level guarantees.

| Grid | Resolution | Cells | Generation | Peak RSS | Retained arrays | NetCDF storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `R02B08` | 9.86 km | 5,242,880 | 56.5 s | 4.35 GiB | 3.01 GiB | 4.09 GiB |
| `R02B09` | 4.93 km | 20,971,520 | 2.02 min | 16.57 GiB | 12.03 GiB | 16.37 GiB |
| `R02B10` | 2.47 km | 83,886,080 | 6.13 min | 64.85 GiB | 48.13 GiB | ~65.5 GiB |
| `R02B11` | 1.23 km | 335,544,320 | 37.10 min | 255.87 GiB | 192.50 GiB | 261.88 GiB |

Generation time excludes NetCDF export. The measured R2B11 export took 8.52
minutes and peaked at 238.12 GiB, giving a 45.83-minute end-to-end run. The file
was reopened with all 85 variables, exact dimensions, and the grid UUID before
being removed. File size is stable for this schema, but write time depends on
shared-filesystem load, striping, page cache, and writeback behavior.

The performance improvement comes from five coordinated changes:

1. deterministic vertex-owned parallel spring reductions instead of serial
   scatter-heavy NumPy force accumulation;
2. compiled parallel geometry, metric, topology, connectivity, edge-matching,
   and orientation kernels;
3. reuse of staged parent topology and immutable post-relaxation arrays instead
   of rebuilding or copying them;
4. early release of compact parent data and temporary arrays; and
5. incremental NetCDF field construction with each converted field released
   immediately after writing.

Numba remains optional so the base package still depends only on NumPy.
`accelerator="auto"` uses Numba for sufficiently large kernels when the
`accelerate` extra is installed. Without it, generation falls back to the
correct deterministic NumPy reference path. That fallback is intentionally
supported but is not performance-equivalent: the original NumPy R2B9 run took
36.11 minutes rather than 2.02 minutes, and larger optimized grids are not
practical on it. Install `icon-grid-generator[accelerate]` for high-resolution
global generation; explicit `accelerator="numba"` fails early if Numba is not
available.

At R2B11, refinement and large-array assembly consume 52.6% of generation and
spring relaxation consumes 33.8%. The remaining refinement cost is dominated
by strided gathers, copies, allocation, and memory bandwidth rather than edge
lookup. A direct R2B12 projection is roughly 770 GiB of retained arrays, about
1 TiB peak RSS, and about 1 TiB of NetCDF storage, beyond this single-node
in-memory design. Further scaling therefore needs lower-copy refinement plus a
partitioned or out-of-core representation, not merely another compiled kernel.

## Testing Expectations

Changes to geometry, topology, metrics, refinement, limited-area extraction, or
NetCDF output should include tests for the relevant contract:

- expected cell, edge, and vertex counts
- index bounds and missing-neighbor sentinels
- finite numeric geometry and positive areas/lengths where applicable
- parent/provenance index validity
- exported NetCDF dimensions, variables, and metadata

Use the smallest grid that proves the behavior. Larger grids are useful only for
representative sanity checks.

Private helper tests may exercise defensive branches for coverage when the
branch protects a public contract. These tests are regression guards, not
scientific validation or additional public API.
