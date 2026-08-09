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
and ICON-compatible NetCDF export. Large global grids use export-first generation
with bounded derived-field memory and resumable disk checkpoints.

## Installation

The base package requires Python 3.10 or newer and NumPy:

```bash
python -m pip install icon-grid-generator
```

Install acceleration and common output integrations for high-resolution work:

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

Generate a large global grid directly to NetCDF:

```py
from grid_generator import generate_grid_to_netcdf

generate_grid_to_netcdf(
    "R2B12",
    "icon_grid_R02B12.nc",
    options={"max_cells": None, "accelerator": "numba"},
    work_dir="icon-grid-R2B12-work",
    fields="reduced",
)
```

The default `full` profile contains 85 fields. `reduced` contains the 46-field
union required by the standard ICON and icon4py global-grid readers. Dedicated
`icon` and `icon4py` profiles and exact custom field lists are also available.
Place large outputs and checkpoint directories on disk-backed storage. Each
checkpoint manifest atomically selects a complete array snapshot, so an
interrupted overwrite leaves the preceding completed checkpoint resumable.

## Performance

Measurements used an exclusive dual-socket AMD EPYC 7713 node with 128 physical
cores and about 446 GiB of available memory. Times cover independent generation
and uncompressed NetCDF export; shared-filesystem I/O varies with storage load.

| R2B12 output | Generation | NetCDF export | Total | Peak RSS | Checkpoints | NetCDF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 43.51 min | 91.24 min | 134.74 min | 328.18 GiB | 162.46 GiB | 1,047.50 GiB |
| Reduced | 42.03 min | 42.05 min | 84.08 min | 328.18 GiB | 162.46 GiB | 485.00 GiB |

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
