# Design Notes

User guidance for fully functional limited-area and general open-boundary ICON
grids is documented separately in
[Generating Limited-Area ICON Grids](limited-area-icon-design.md). Its
implementation-details section distinguishes correctness and current ICON-NWP
requirements from explicit alternative scientific policies.

ICON Grid Generator builds in-memory `IconGrid` objects through a small,
deterministic pipeline:

1. Parse and validate a grid spec plus `IconGridOptions`.
2. Build geometry: vertices, cells, centers, and lon/lat coordinates.
3. Build topology: edges, cell-edge relations, edge-cell relations, and ICON
   connectivity tables.
4. Build metrics: cell areas, edge lengths, dual quantities, and normal vectors.
5. Build refinement/provenance fields.
6. Assemble metadata, UUIDs, conversion helpers, and optional NetCDF output.

File-oriented generation is coordinated by one pipeline for every grid family.
It writes NetCDF-only transforms in entity chunks and returns the output `Path`
rather than a complete `IconGrid`. Very large global grids generate the largest
base stage with at most 1,310,720 cells (R2B7 for the standard R2 family), refine
compact topology stage by stage, and checkpoint each completed stage on disk.
Limited-area grids use that same compact engine for their construction parent
and perform chunked regional selection before materializing the selected mesh.
Planar grids have no recursive stages; preallocated geometry and array-based
edge assembly avoid the Python object graphs that previously dominated their
construction overhead.

## Compatibility Contracts

- Public grid specs, `generate_grid()`, and the file-oriented
  `generate_grid_to_netcdf()` are the main generation API.
- `IconGrid.dims` and array shapes derive deterministically from the spec.
- Internal topology arrays are zero-based; exported NetCDF index fields are
  one-based where ICON expects that convention.
- ICON `grid_geometry` describes only coordinate geometry. Spherical global,
  limited-area, and cut grids use `1`; planar torus, channel, and general-plane
  grids use `2`, `3`, and `4`. The independent `open_boundary` flag selects
  strict regional connectivity and boundary validation.
- Metadata and grid UUIDs are stable for unchanged canonical inputs.

## Feature Boundaries

- The package exposes a Python API and does not provide workflow-specific command
  wrappers.
- Global, planar, limited-area, optimization, diffusion, diagnostics, and
  normal NetCDF export share the `IconGrid` data model. File-oriented generation
  is coordinated in `_pipeline.py`; global grids and limited-area parents share
  the compact private representation in `_streaming.py`.
- `grid_generator.py` is the public facade; focused private modules contain the
  spherical, planar, regional, NetCDF, and streaming implementations.
- Triangles are the only supported cell family.
- Ragged planar grids are deterministic variants with their own geometry rather
  than metric-equivalent versions of regular planar grids.
- Parent/provenance indices live in `IconGrid.refinement`; metadata contains
  descriptive scalar attributes.

## Architectural Decisions

- Global grid generation uses staged spring relaxation by default; raw
  bisection remains available with `optimize_global=False` for diagnostics and
  topology checks.
- Root grids subdivide every icosahedron edge and face row by equal great-circle
  arc length for every root.
- Planar triangular variants share one builder pipeline. The spec object
  carries variant flags such as `periodic` and `periodic_x`; `_planar.py`
  dispatches geometry, topology, and metric behavior from those flags instead
  of adding separate public generation entry points.
- Doubly periodic triangular grids use a rectangular fundamental domain by
  default: x and y wrap independently, while the row offset is carried by the
  y-boundary vertex identification. This requires an even number of rows.
  `periodic_layout="skew"` retains the coupled lattice vectors and permits odd
  row counts. Generation and geometry transforms use the selected convention
  consistently.
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
- NetCDF export is an internal module boundary. Public users call
  `IconGrid.to_netcdf(path)` when they already have an in-memory grid, or
  `generate_grid_to_netcdf(...)` when the file is the desired result. The latter
  adds generation dispatch; both calls share the canonical chunked atomic
  writer. File-oriented generation selects compact export-first stages for
  large global grids and limited-area construction parents.
- For compatibility with established ICON grid files, spherical NetCDF
  `edgequad_area` values are normalized by `sphere_radius**2`. The exported
  variable is therefore dimensionless (`units = "1"`) and carries a
  `normalization` attribute. In-memory `IconGrid.geometry` retains physical
  square-metre values. This compatibility convention is not applied to planar
  grids.
- The default full schema includes the established `quadrilateral_area`,
  `vlon_vertices`, and `vlat_vertices` variables.
  The legacy quadrilateral field remains an all-zero placeholder; ICON uses
  `edgequad_area` for the actual edge-quadrilateral metric.
- Xarray public connectivity is zero-based with `-1` for missing neighbors;
  parent-provenance arrays remain one-based with `0` for no parent. Variables
  carry explicit `start_index` and `missing_value` attributes.
- Pipeline stage results use frozen dataclasses to keep builder boundaries
  explicit. Arrays remain mutable NumPy buffers during construction; callers
  should treat completed `IconGrid` objects as immutable values.

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

Periodic displacements use the same configured rectangular or coupled-lattice
minimum-image operation as metric generation. Spherical vertices are
renormalized to `radius`; planar z coordinates are restored; periodic planar
coordinates are wrapped to the selected fundamental domain. Open-boundary
vertices are excluded from updates when `fixed_boundary=True`.

After movement, the grid rebuilds centers, projected coordinates, metric fields,
normal vectors, and summary metadata. Topology and refinement arrays are
unchanged. Public transforms derive a new UUID from the source UUID, operation,
and canonical option payload. No inversion detector or line search is part of
the update, so callers remain responsible for checking scientific quality.

## Limitations

- Connectivity and NetCDF index fields use signed 32-bit integer arrays.
  `R02B12` is the final standard R2 grid within that range; `R02B13` and larger
  levels fail before allocation because their cell, edge, or vertex identifiers
  would exceed the int32 index limit.
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
- Regional circles and longitude/latitude boxes default to circumdisk
  intersection using cell circumcenters and circumradii. Explicit center and
  center-or-vertex policies remain available. Package-specific polygons use
  overlap semantics in a wrapped lon/lat plane; use small polygons away from
  the poles or provide an explicit scientific comparison for polar domains.
- Planar `lon`/`lat` fields are normalized compatibility and visualization
  coordinates, not a geographic CRS. Scientific planar calculations should use
  Cartesian coordinates and metric arrays.
- Planar NetCDF geometry attributes use the physical domain extents from the
  specification. Channel `domain_length` is the x period; open parallelogram
  and ragged values describe their nominal unperturbed extents.
- Correct geometry metadata does not guarantee model-operator support. ICON
  2024.10's standard NWP least-squares and tangent-plane interpolation dispatch
  supports spherical and planar-torus geometry, but rejects planar-channel and
  general-planar geometry. Those families require a consumer with matching
  operators and are not claimed as standard ICON-NWP simulation grids.
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
Python 3.11, NumPy 2.4.6, Numba 0.66, netCDF4 1.7.4, and 128 Numba threads.
Each row is an independent process with a profile-specific checkpoint directory.

`Generation` includes all checkpoint writes. `NetCDF export` includes
profile-specific field computation, HDF5 serialization, and file close.
`Total` is their sum and excludes the separately timed reopen validation.
Checkpoint and NetCDF columns are logical sizes at completion.

### Full output

The timings below were measured for the former 85-field full schema. The
current 88-field schema adds 75 GiB at R2B12, for an estimated 1,122.50 GiB
file. Export time has not yet been remeasured and the historical timings are
retained only as a scaling baseline.

#### Runtime

| Grid | Resolution | Cells | Generation | NetCDF export | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `R02B08` | 9.86 km | 5,242,880 | 54.8 s | 21.6 s | 76.4 s |
| `R02B09` | 4.93 km | 20,971,520 | 72.3 s | 46.9 s | 1.99 min |
| `R02B10` | 2.47 km | 83,886,080 | 3.45 min | 3.69 min | 7.14 min |
| `R02B11` | 1.23 km | 335,544,320 | 11.78 min | 17.49 min | 29.27 min |
| `R02B12` | 0.616 km | 1,342,177,280 | 43.51 min | 91.24 min | 134.74 min |

#### Memory and storage

| Grid | Peak RSS | Checkpoint storage | NetCDF storage |
| --- | ---: | ---: | ---: |
| `R02B08` | 1.94 GiB | 0.77 GiB | 4.09 GiB |
| `R02B09` | 6.48 GiB | 3.20 GiB | 16.37 GiB |
| `R02B10` | 24.00 GiB | 12.92 GiB | 65.47 GiB |
| `R02B11` | 93.78 GiB | 51.83 GiB | 261.88 GiB |
| `R02B12` | 328.18 GiB | 162.46 GiB | about 1,122.50 GiB |

### Reduced output

#### Runtime

| Grid | Resolution | Cells | Generation | NetCDF export | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `R02B08` | 9.86 km | 5,242,880 | 37.3 s | 10.8 s | 48.1 s |
| `R02B09` | 4.93 km | 20,971,520 | 72.8 s | 26.4 s | 1.65 min |
| `R02B10` | 2.47 km | 83,886,080 | 3.83 min | 1.57 min | 5.40 min |
| `R02B11` | 1.23 km | 335,544,320 | 12.92 min | 7.54 min | 20.46 min |
| `R02B12` | 0.616 km | 1,342,177,280 | 42.03 min | 42.05 min | 84.08 min |

#### Memory and storage

| Grid | Peak RSS | Checkpoint storage | NetCDF storage |
| --- | ---: | ---: | ---: |
| `R02B08` | 2.10 GiB | 0.77 GiB | 1.89 GiB |
| `R02B09` | 6.65 GiB | 3.20 GiB | 7.58 GiB |
| `R02B10` | 24.44 GiB | 12.92 GiB | 30.31 GiB |
| `R02B11` | 93.78 GiB | 51.83 GiB | 121.25 GiB |
| `R02B12` | 328.18 GiB | 162.46 GiB | 485.00 GiB |

The field profile does not change core topology generation, so differences in
that column are run-to-run variability. Reduced output omits unselected
coordinate bounds, connectivity, metrics, Cartesian, placeholder, and hierarchy
computations. At R2B12, reduced output uses 54% less export time, 38% less total
time, and 54% less final storage than full output.

### R2B12 component breakdown

The full-output run above recorded the following major components. Small
components are grouped so the table remains operational rather than specific
to the profiling tool.

| Phase | Component | Time | Share of phase |
| --- | --- | ---: | ---: |
| Generation | Spring relaxation | 37.79 min | 86.9% |
| Generation | All other generation and checkpoint work | 5.72 min | 13.1% |
| Export | Coordinates and bounds | 29.42 min | 32.2% |
| Export | Metrics | 22.79 min | 25.0% |
| Export | Connectivity | 15.86 min | 17.4% |
| Export | Cartesian fields | 14.48 min | 15.9% |
| Export | Hierarchy, refinement, placeholders, and overhead | 8.69 min | 9.5% |

Compact topology establishes a roughly 302 GiB generation memory floor at
R2B12. Constructing export connectivity raises peak RSS to 328.18 GiB. Spring
relaxation therefore dominates generation time, while coordinates, metrics,
connectivity, and Cartesian fields dominate full export. Field selection is the
strongest practical runtime and storage lever because it skips whole export
groups; it does not change the compact topology floor. Signed 32-bit identifiers
make R2B12 the hard ceiling for the standard R2 family.

### Storage by field profile

These representative cases compare measured uncompressed files. Full and
reduced include complete generation; ICON and icon4py reuse the same generated
topology because field selection affects export only.

| Grid | Full | Reduced | ICON | icon4py |
| --- | ---: | ---: | ---: | ---: |
| `R02B08` | 4.09 GiB | 1.89 GiB | 1.60 GiB | 1.32 GiB |
| `R02B10` | 65.47 GiB | 30.31 GiB | 25.63 GiB | 21.09 GiB |

All measurements used the same 8-way, 16 MiB filesystem striping. Each full and
reduced file was reopened to verify dimensions, variable count, UUID, finite
positive metrics, and spherical area consistency. R2B12 checkpoints grow by
less than four times from R2B11 because the final addressable grid does not
retain parent normals for an impossible R2B13 stage.

Logical output size is stable for a schema, but shared-filesystem write time is
highly dependent on load, striping, page cache, and writeback. These I/O timings
are representative measurements, not throughput guarantees.

### Scalability mechanisms

Large-grid generation uses:

1. deterministic vertex-owned parallel spring reductions instead of serial
   scatter-heavy NumPy force accumulation;
2. allocation-bounded spring-target and primal-normal kernels that avoid
   materializing whole-grid vertex gathers or unrelated metric fields;
3. sort-free closed-grid bisection whose child adjacency is derived directly
   from validated parent topology;
4. in-place child-cell orientation, avoiding full cell-center, edge-center, and
   orientation work arrays during refinement;
5. compact staged parents containing only refinement topology, vertex
   incidence, and the primal normals required by the next stage;
6. immutable disk-backed checkpoint snapshots selected by an atomic manifest
   update, preserving the preceding stage if an overwrite is interrupted;
7. export-first NetCDF construction that computes connectivity conversions and
   derived fields in entity chunks for every grid family; and
8. scale-relative orientation checks that remain valid when geometric
   determinants shrink at sub-kilometre resolution.

Numba is optional, so the base package depends only on NumPy.
`accelerator="auto"` uses Numba for sufficiently large kernels when the
`accelerate` extra is installed. Without it, generation falls back to the
correct deterministic NumPy path, which is suitable for modest grids but not
high-resolution global generation. Explicit `accelerator="numba"` fails if
Numba is unavailable.

Compact multilevel stages require Numba above the in-memory base stage and are
used for direct global output and limited-area construction parents.
`generate_grid()` retains the NumPy path for in-memory work. Standalone planar
grids use the shared chunked writer but need no bisection checkpoints. By
default, checkpoints live beside the output; large jobs require disk-backed
output and checkpoint locations rather than a memory-backed temporary
filesystem. Atomic replacement can temporarily retain both checkpoint
snapshots or both the old and new NetCDF files, so provision more than the
completed logical sizes. A successful work directory can be removed when no
later resume or extension is needed.
