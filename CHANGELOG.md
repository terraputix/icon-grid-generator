# Changelog

## Unreleased

- Make accelerated spring reductions deterministic across Numba thread counts
  through production-default convergence, preserving byte-identical geometry
  for one grid identity.
- Select export-first base stages by cell count so root-heavy global specs do not
  unexpectedly fall back to complete in-memory export, and clarify that changing
  the root/bisection decomposition changes grid identity and hierarchy.
- Clarify custom streamed metadata, transient disk headroom, checkpoint cleanup,
  safe large-grid examples, and component-level R2B12 performance behavior.
- Keep performance scratch data on workspace or explicitly configured disk-backed
  storage and exclude temporary-file cleanup from measured generation time.

## 0.6.1 - 2026-08-09

- Consolidate API and performance documentation, correct benchmark rounding,
  and align agent guidance with disk-backed checkpoints.

## 0.6.0 - 2026-08-09

- Release completed parent and center arrays before constructing child incidence,
  and build large vertex-edge incidence directly without concatenated endpoint
  arrays.
- Publish checkpoint arrays as immutable snapshots selected by one atomic
  manifest update, so interrupted overwrites leave the previous checkpoint
  resumable.
- Extend the optional performance regression suite to cover optimized spring
  relaxation and end-to-end export-first NetCDF generation.
- Add export-first global NetCDF generation with compact staged topology,
  disk-backed resumable checkpoints, bounded field serialization, and atomic
  output publication for grids too large to represent as a complete `IconGrid`.
- Remove the global incidence sort and geometry-sized ordering temporaries from
  validated staged bisection, and compute spring targets and reusable primal
  normals with allocation-bounded Numba kernels.
- Build and write large connectivity tables sequentially so export no longer
  retains all adjacency tables at once.
- Treat R2B12 as the final signed-32-bit-addressable global grid and reject the
  export-first high-resolution path early when Numba acceleration is absent.
- Scale edge-orientation degeneracy checks by the local direction magnitudes so
  valid sub-kilometre geometry is not rejected by an absolute tolerance.
- Add `full`, `reduced`, `icon`, and `icon4py` NetCDF field profiles plus an
  exact custom-field selector to in-memory and export-first writing.
- Align global start/end index tables with their refinement-control values so
  the standard ICON subdivision reader receives one non-empty global interval.
- Document fresh end-to-end `full` and `reduced` scaling measurements through
  R2B12, profile-specific storage, component timings, and scratch-I/O caveats.

## 0.5.0 - 2026-08-09

- Accelerate default global-grid generation with deterministic parallel Numba
  kernels for staged spring relaxation, spherical geometry and metrics,
  topology/connectivity construction, edge matching, and orientation updates.
- Reuse staged parent topology, compact and release parent data earlier, and
  reuse immutable topology during post-relaxation rebuilds to avoid large
  serial hash-map work and unnecessary copies.
- Stream ICON NetCDF field assembly and release each converted field after it is
  written, keeping export memory below the generation peak on large grids.
- Preserve the NumPy-only execution path as a correct reference and
  fallback; document that the optional `accelerate` extra is required for the
  measured high-resolution performance.
- Add deterministic accelerator-equivalence, topology-reuse, degenerate-edge,
  writer-lifetime, and performance regression coverage.
- Document single-node optimized performance through R2B11, including runtime,
  peak and retained memory, NetCDF storage, hardware context, and the remaining
  refinement/memory-bandwidth scalability limit.

## 0.4.1 - 2026-08-05

- Streamline the README to the essential project description, quick start,
  capabilities, installation, documentation, development, citation, and license
  sections.
- Remove repository-configuration recommendations from the published
  documentation and present the root import surface as a user-facing reference.

## 0.4.0 - 2026-08-05

- Correct skew-periodic torus seam geometry so coordinates, metric fields,
  normal vectors, divergence operators, optimization, and diffusion use the
  same minimum-image convention.
- Orient planar and extracted-grid normals consistently with local adjacent-cell
  ordering, while preserving source-parent metric provenance.
- Include parent UUIDs in limited-area and cut-grid identity payloads and export
  the actual parent as `uuidOfParHGrid`. Existing derived-grid UUIDs therefore
  change to remove collisions between distinct parents or parent options.
- Include geometry, connectivity, refinement, metadata, and units in
  `IconGrid.to_xarray()` datasets.
- Populate planar `edgequad_area` from the primal/dual vector cross product,
  including the actual intersection angle on sheared grids.
- Derive stretched, sheared, periodic, and open planar dual metrics from the
  generated coordinates, including boundary-aware dual areas that partition
  the primal cell area.
- Preserve source periodic geometry and parent provenance when transforming
  cut grids. Public optimization and diffusion transforms now assign a
  deterministic UUID derived from the input UUID and transform options.
- Inherit cut-grid CRS metadata from the source grid instead of labeling every
  cut as Cartesian.
- Mark normalized spherical NetCDF `edgequad_area` values as dimensionless and
  annotate xarray connectivity and parent-index conventions.
- Recognize toroidal Euler characteristic in diagnostics and omit periodic seam
  artifacts from SVG plots.
- Document the global spring system, Laplacian/target-length smoothing,
  diffusion update, boundary and projection rules, stability limitations, and
  UUID behavior. Direct `optimize_global_grid()` calls now record transform
  provenance consistently with the other public transforms.
- Document installation extras, option precedence, grid-family constraints,
  coordinate and index conventions, regional selection semantics, object
  mutability, export behavior, and the scientific scope of diagnostics.
- Refresh citation metadata for release 0.4.0.

## 0.3.2 - 2026-07-07

- Restore documentation grid figures to the stable 2D SVG edge-plot style.
- Remove the transient 3D visualization option from `write_svg()`.

## 0.3.1 - 2026-07-07

- Display generated SVG grid figures directly in the documentation examples.
- Add checked documentation figure generation with CI and release-time freshness
  checks.
- Publish GitHub Releases automatically from the release workflow.

## 0.3.0 - 2026-07-07

- Keep the root import surface focused while moving advanced cutting helpers to
  `grid_generator.cutting`.
- Allow common `generate_grid()` options to be passed directly as keyword
  arguments.
- Add lightweight SVG grid visualization via `grid_generator.visualization`.
- Expand copy-pasteable examples for global, raw, planar, limited-area,
  cutting, diagnostics, transforms, NetCDF export, and visualization workflows.
- Clarify and test `optimize_global` behavior across global, limited-area, and
  planar grid specs.
- Preserve planar geometry when optimizing or diffusing cut grids from planar
  parents.
- Add fast mathematical correctness matrix coverage across all supported grid
  families, region predicates, cut modes, transforms, metric scaling, and
  representative NetCDF exports.
- Add example scripts, type-marker packaging, and documentation updates for the
  refactored public API.

## 0.2.1 - 2026-07-05

Patch release for relaxed global grid generation.

- Add optional spring relaxation for global spherical grids with unchanged
  topology and recomputed metrics.
- Expose `GlobalOptimizationOptions` and `optimize_global_grid()` in the public
  API.

## 0.2.0 - 2026-07-05

Expanded grid generation, validation, and release automation.

- Add triangular planar variants for stretched periodic, channel, parallelogram,
  and ragged orthogonal grids.
- Add geometry optimization and diffusion transforms.
- Add region-based local-area cutting with parent-index metadata.
- Add grid diagnostics, statistics, triangle properties, divergence, and
  normalized vorticity helpers.
- Improve ICON-style NetCDF metadata, refinement fields, ordering, and
  large-grid safety checks.
- Add optional Numba acceleration support and CI coverage for accelerated and
  non-accelerated execution paths.
- Add documentation publishing, contributor guidance, and drift checks for
  documentation, badges, API exports, and the Python test matrix.

## 0.1.0 - 2026-07-04

Initial public release.

- Generate global spherical ICON `R<n>B<k>` grids.
- Generate planar doubly periodic torus grids.
- Extract limited-area grids from generated global parent grids.
- Export ICON-style NetCDF grid files with optional `netCDF4` support.
- Provide the public grid-spec API: `GlobalGridSpec`, `LimitedAreaGridSpec`,
  `TorusGridSpec`, and `generate_grid()`.
