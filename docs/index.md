# ICON Grid Generator

ICON Grid Generator is a pure Python package for creating ICON-style triangular
grids without depending on ICON model runtimes or stencil frameworks.

![Global ICON grid resolutions](assets/global-icon-grid-series.png)

## What It Provides

- Global spherical ICON `R<n>B<k>` grids.
- Rectangular-periodic planar tori and open planar triangular grids for local
  experiments; coupled skew tori remain an explicit option.
- Limited-area grids extracted from generated global parent grids.
- ICON-style NetCDF export when the optional `netCDF4` dependency is installed.
- Export-first global NetCDF generation with resumable disk checkpoints for
  grids too large to retain as a complete in-memory object.
- `full`, `reduced`, `icon`, and `icon4py` NetCDF field profiles plus exact
  custom field selection.
- In-memory geometry, topology, connectivity, metric, and refinement arrays for
  plotting, diagnostics, and downstream conversion.

## Basic Usage

Install `icon-grid-generator[netcdf]` before running this NetCDF example.

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
```

Global grids are optimized by default. Pass `optimize_global=False` only for raw
topology diagnostics.

For high-resolution global grids, install the optional Numba acceleration path:

```bash
python -m pip install "icon-grid-generator[accelerate,netcdf]"
```

Without `accelerate`, `generate_grid()` with `accelerator="auto"` uses the
correct NumPy fallback, but large-grid runtime is substantially higher. The
export-first high-resolution path fails early instead of selecting that
impractical fallback. See
[Performance and Scaling](design.md#performance-and-scaling) for measured time,
memory, and storage requirements.

## Which Grid Should I Use?

| Goal | Use |
| --- | --- |
| Standard spherical grid file | `generate_grid("R2B4")` |
| Large global grid file | `generate_grid_to_netcdf("R2B8", ...)` |
| Raw topology checks | `generate_grid("R2B4", optimize_global=False)` |
| Periodic planar experiment | `TorusGridSpec(...)` |
| Regional extract from a global parent | `LimitedAreaGridSpec(...)` |
| Cut an existing grid | `grid_generator.cutting.cut_grid(...)` |

## Project Links

- [Examples](examples.md)
- [API overview](api.md)
- [Coordinates, units, and indexing](api.md#coordinates-units-and-indexing)
- [Regional cutting semantics](api.md#cutting)
- [Grid optimization guide](api.md#grid-optimization)
- [Performance and scaling](design.md#performance-and-scaling)
- [Design notes and limitations](design.md)
- [Changelog](https://github.com/ofuhrer/icon-grid-generator/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/ofuhrer/icon-grid-generator/blob/main/CITATION.cff)
