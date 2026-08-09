# ICON Grid Generator

[![Tests](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml/badge.svg)](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml)
[![Docs](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/docs.yml/badge.svg)](https://ofuhrer.github.io/icon-grid-generator/)
[![PyPI](https://img.shields.io/pypi/v/icon-grid-generator.svg)](https://pypi.org/project/icon-grid-generator/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](.github/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Pure Python generation of ICON-style triangular grids.

![Global ICON grid resolutions](docs/assets/global-icon-grid-series.png)

ICON Grid Generator creates spherical ICON `R<n>B<k>` grids, planar triangular
grids, limited-area extracts, and ICON-style NetCDF files without depending on
ICON model runtimes or stencil frameworks.

## Quick Start

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
```

For global grids too large to retain as a complete `IconGrid`, stream directly
to NetCDF with bounded derived-field memory and optional stage checkpoints:

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

When `work_dir` is omitted, resumable checkpoints are kept in a hidden directory
beside the requested NetCDF file. For large runs, place either the output or an
explicit `work_dir` on disk-backed scratch storage; memory-backed temporary
filesystems are not suitable for these checkpoints.

NetCDF export writes the complete 85-field schema by default. Select
`fields="reduced"` for the audited 46-field union consumed by current ICON and
icon4py global-grid readers, or use `"icon"`, `"icon4py"`, or an explicit list
of field names for a narrower consumer contract.

```text
R02B04
{'cell': 20480, 'vertex': 10242, 'edge': 30720}
```

Global grids are optimized by default. Pass `optimize_global=False` only when
raw bisection geometry is required for diagnostics or comparisons.

## What You Can Generate

- Global spherical ICON grids from standard `R<n>B<k>` names.
- Planar triangular torus, channel, parallelogram, stretched, and ragged grids.
- Limited-area grids extracted from generated global parents.
- Region-based cuts of existing spherical or planar grids.
- ICON-style NetCDF files and complete in-memory xarray datasets.
- Geometry, topology, metric, refinement, diagnostic, and visualization data.

## Installation

Python 3.10 or newer is required. Install acceleration plus the common NetCDF
and xarray support from PyPI with:

```bash
python -m pip install "icon-grid-generator[accelerate,netcdf,xarray]"
```

The base package still depends only on NumPy and can instead be installed with
`python -m pip install icon-grid-generator`. That path is correct and
deterministic, but its NumPy fallback is not practical for high-resolution
global grids. Install the `accelerate` extra to enable the measured Numba path;
`accelerator="auto"` selects it automatically for sufficiently large work.

## Performance

Optimized global grids were measured on an exclusive dual-socket AMD EPYC 7713
node with 128 physical cores (256 hardware threads) and about 446 GiB of
scheduler-visible memory. Python 3.11, NumPy 2.4.6, Numba 0.66, and 128 Numba
threads were used. Each row is a fresh end-to-end process. `Generation`
includes checkpoint writes; `NetCDF export` includes selected-field computation,
serialization, and close; `Total` is their sum and excludes reopen validation.

Full 85-field output:

| Grid | Resolution | Cells | Generation | NetCDF export | Total | Peak RSS | Checkpoint storage | NetCDF storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R2B8` | 9.86 km | 5,242,880 | 54.8 s | 21.6 s | 76.4 s | 1.94 GiB | 0.77 GiB | 4.09 GiB |
| `R2B9` | 4.93 km | 20,971,520 | 72.3 s | 46.9 s | 1.99 min | 6.48 GiB | 3.20 GiB | 16.37 GiB |
| `R2B10` | 2.47 km | 83,886,080 | 3.45 min | 3.69 min | 7.14 min | 24.00 GiB | 12.92 GiB | 65.47 GiB |
| `R2B11` | 1.23 km | 335,544,320 | 11.78 min | 17.49 min | 29.27 min | 93.78 GiB | 51.83 GiB | 261.88 GiB |
| `R2B12` | 0.616 km | 1,342,177,280 | 43.51 min | 91.24 min | 134.74 min | 328.18 GiB | 162.46 GiB | 1,047.50 GiB |

Reduced 46-field output:

| Grid | Resolution | Cells | Generation | NetCDF export | Total | Peak RSS | Checkpoint storage | NetCDF storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R2B8` | 9.86 km | 5,242,880 | 37.3 s | 10.8 s | 48.1 s | 2.10 GiB | 0.77 GiB | 1.90 GiB |
| `R2B9` | 4.93 km | 20,971,520 | 72.8 s | 26.4 s | 1.65 min | 6.65 GiB | 3.20 GiB | 7.58 GiB |
| `R2B10` | 2.47 km | 83,886,080 | 3.83 min | 1.57 min | 5.40 min | 24.44 GiB | 12.92 GiB | 30.31 GiB |
| `R2B11` | 1.23 km | 335,544,320 | 12.92 min | 7.54 min | 20.46 min | 93.79 GiB | 51.83 GiB | 121.25 GiB |
| `R2B12` | 0.616 km | 1,342,177,280 | 42.03 min | 42.05 min | 84.08 min | 328.18 GiB | 162.46 GiB | 485.00 GiB |

Actual uncompressed storage for all named profiles on representative grids:

| Grid | Full | Reduced | ICON | icon4py |
| --- | ---: | ---: | ---: | ---: |
| `R2B8` | 4.09 GiB | 1.90 GiB | 1.60 GiB | 1.32 GiB |
| `R2B10` | 65.47 GiB | 30.31 GiB | 25.63 GiB | 21.09 GiB |

The profile does not change core topology generation, so differences in that
column are independent-run variability. Reduced output saves work by never
computing omitted derived fields and cut R2B12 export time by 54% and total time
by 38% in this campaign. R2B12 full and reduced both completed in three-hour
allocations; reduced finished within the original two-hour target.

Files were uncompressed and used identical 8-way, 16 MiB filesystem striping.
Each was reopened to check dimensions, field count, UUID, and chunked metric
values, then deleted along with its fresh checkpoints. Storage is predictable
for a schema; shared-filesystem I/O time is highly load- and layout-dependent,
so the export timings should be taken with a grain of salt. Install
`accelerate`, set `max_cells=None`, and leave `accelerator="auto"` or select
`"numba"` explicitly for this scaling regime. See
[Performance and Scaling](https://ofuhrer.github.io/icon-grid-generator/design/#performance-and-scaling)
for component timings, profile-specific storage, and remaining bottlenecks.

## Documentation

The complete documentation is published at
[ofuhrer.github.io/icon-grid-generator](https://ofuhrer.github.io/icon-grid-generator/):

- [Examples](https://ofuhrer.github.io/icon-grid-generator/examples/)
- [API and usage guide](https://ofuhrer.github.io/icon-grid-generator/api/)
- [Design notes and limitations](https://ofuhrer.github.io/icon-grid-generator/design/)
- [Changelog](CHANGELOG.md)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution guidance.
Run the complete local check with:

```bash
python -m pip install -e ".[accelerate,test,docs,netcdf,xarray]"
make check
make perf-check
```

## Citation

If you use ICON Grid Generator in published work, cite it using
[CITATION.cff](CITATION.cff).

## License

ICON Grid Generator is distributed under the [BSD 3-Clause License](LICENSE).
