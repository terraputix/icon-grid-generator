# ICON Grid Generator

[![Tests](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml/badge.svg)](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml)
[![Docs](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/docs.yml/badge.svg)](https://ofuhrer.github.io/icon-grid-generator/)
[![PyPI](https://img.shields.io/pypi/v/icon-grid-generator.svg)](https://pypi.org/project/icon-grid-generator/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](.github/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Pure Python generation of deterministic ICON-style triangular grids.

![Global ICON grid resolutions](docs/assets/global-icon-grid-series.png)

The package provides spherical `R<n>B<k>` grids, planar triangular grids,
limited-area extraction, geometry diagnostics and transforms, xarray conversion,
and ICON-compatible NetCDF export. The same file-oriented API handles every grid
family; large global grids automatically use bounded derived-field memory and
resumable disk checkpoints.

## Installation

The base package requires Python 3.10 or newer and NumPy:

```bash
python -m pip install icon-grid-generator
```

The NetCDF calls in the quick start require the `netcdf` extra. Install
acceleration and common output integrations for high-resolution work with:

```bash
python -m pip install "icon-grid-generator[accelerate,netcdf,xarray]"
```

Numba acceleration is optional for in-memory grids and required for the
high-resolution export-first path.

## Quick Start

Generate an in-memory grid and write the complete NetCDF schema:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
print(grid.name, grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
```

NetCDF `grid_geometry` follows ICON's geometry enum: spherical global and
limited-area grids use `1`, planar tori use `2`, planar channels use `3`, and
general planar grids use `4`. Regional spherical files carry a separate
`open_boundary=1` attribute; openness is not a coordinate-geometry type.
ICON 2024.10's standard NWP interpolation path supports spherical grids and
planar tori, but not its channel or general-plane enum values; those additional
planar families remain useful for diagnostics and consumers with matching
operators rather than as standard ICON-NWP simulation grids.
Planar files write `domain_length` and `domain_height` from the physical spec
extents; they never inherit spherical-Earth dimensions.

Planar tori use rectangular, independently wrapped x/y periods by default and
therefore require an even number of rows. The former coupled skew lattice
remains available explicitly:

```python
from grid_generator import TorusGridSpec, generate_grid

torus = generate_grid(TorusGridSpec(nx=12, ny=6, edge_length=1_000.0))
skew_torus = generate_grid(
    TorusGridSpec(nx=12, ny=5, edge_length=1_000.0, periodic_layout="skew")
)
```

Generate any grid directly to NetCDF when the in-memory object is not needed:

```python
from grid_generator import TorusGridSpec, generate_grid_to_netcdf

generate_grid_to_netcdf(
    TorusGridSpec(nx=12, ny=6, edge_length=1_000.0),
    "torus.nc",
)
```

The same call automatically selects the export-first implementation for a large
global grid:

```py
from grid_generator import generate_grid_to_netcdf

generate_grid_to_netcdf(
    "R2B8",
    "icon_grid_R02B08.nc",
    max_cells=None,
    accelerator="numba",
    work_dir="icon-grid-R2B08-work",
    fields="reduced",
)
```

Checkpointed, bounded-memory generation is currently specific to global grids;
other grid families use their normal in-memory generator before writing. `R2B8`
is a practical first large-grid example at about 9.86 km resolution; check the
resource tables before requesting finer grids.

The default `full` profile contains 88 fields, including the established
`quadrilateral_area`, `vlon_vertices`, and `vlat_vertices` fields.
`reduced` contains the 46-field union required by the standard ICON and
icon4py global-grid readers. Dedicated
`icon` and `icon4py` profiles and exact custom field lists are also available.
Place large outputs and checkpoint directories on disk-backed storage. Each
checkpoint manifest atomically selects a complete array snapshot, so an
interrupted overwrite leaves the preceding completed checkpoint resumable. The
final NetCDF file is also published atomically after it closes successfully.
Allow extra disk headroom when replacing checkpoints or an existing output:
old and new snapshots/files can coexist temporarily. After a successful export,
the work directory can be deleted unless it is being kept for a later resume.

## Performance

Measurements used an exclusive dual-socket AMD EPYC 7713 node with 128 physical
cores and about 446 GiB of available memory. Times cover independent generation
and uncompressed NetCDF export; shared-filesystem I/O varies with storage load.
`R2B12` has approximately 0.616 km resolution and is the largest standard R2
grid whose one-based exported identifiers fit signed 32-bit integers.

| R2B12 output | Generation | NetCDF export | Total | Peak RSS | Checkpoints | NetCDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full (85-field timing) | 43.51 min | 91.24 min | 134.74 min | 328.18 GiB | 162.46 GiB | 1,047.50 GiB |
| Reduced | 42.03 min | 42.05 min | 84.08 min | 328.18 GiB | 162.46 GiB | 485.00 GiB |

The full timing predates the three added grid-description fields. The
current 88-field R2B12 payload is approximately 1,122.5 GiB; it requires a new
scaling run before quoting an updated export time.

See [Performance and Scaling](https://ofuhrer.github.io/icon-grid-generator/design/#performance-and-scaling)
for R2B8–R2B12 measurements, component timings, validation details, and all
field-profile storage sizes.

## Documentation

- [Documentation home](https://ofuhrer.github.io/icon-grid-generator/)
- [API and usage](https://ofuhrer.github.io/icon-grid-generator/api/)
- [Examples](https://ofuhrer.github.io/icon-grid-generator/examples/)
- [Design, limits, and performance](https://ofuhrer.github.io/icon-grid-generator/design/)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

Citation metadata is provided in [CITATION.cff](CITATION.cff). The package is
distributed under the [BSD 3-Clause License](LICENSE).
