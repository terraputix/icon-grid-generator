from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.environ.get("GRID_GENERATOR_PERF_TESTS") != "1",
        reason="set GRID_GENERATOR_PERF_TESTS=1 or run make perf-check",
    ),
]


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PerformanceCase:
    workflow: str
    grid: str
    accelerator: str
    attempts: int
    max_best_seconds: float
    max_best_rss_mib: float


@dataclass(frozen=True)
class FileFamilyPerformanceCase:
    family: str
    attempts: int
    max_best_seconds: float
    max_best_rss_mib: float


def _run_generation_once(case: PerformanceCase) -> dict[str, float | int | str]:
    code = f"""
import json
import resource
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

from grid_generator import generate_grid, generate_grid_to_netcdf
from grid_generator import _streaming
from grid_generator.grid_generator import parse_grid_spec

if {case.workflow!r} == "streamed":
    spec = parse_grid_spec({case.grid!r})
    scratch_root = Path(
        os.environ.get("GRID_GENERATOR_PERF_WORK_DIR", Path.cwd() / "profiling")
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Exercise export-first staging on a modest fixture that normally fits the
    # in-memory base budget.
    _streaming.DEFAULT_IN_MEMORY_BASE_CELLS = spec.expected_cells // 4
    with TemporaryDirectory(dir=scratch_root) as directory:
        start = time.perf_counter()
        generate_grid_to_netcdf(
            spec,
            f"{{directory}}/grid.nc",
            options={{"max_cells": None, "accelerator": {case.accelerator!r}}},
            work_dir=f"{{directory}}/checkpoints",
            fields="icon4py",
        )
        elapsed = time.perf_counter() - start
    dimensions = {{
        "cell": spec.expected_cells,
        "edge": spec.expected_edges,
        "vertex": spec.expected_vertices,
    }}
else:
    options = {{"max_cells": None, "accelerator": {case.accelerator!r}}}
    if {case.workflow!r} == "raw":
        options["optimize_global"] = False
    start = time.perf_counter()
    grid = generate_grid({case.grid!r}, options=options)
    elapsed = time.perf_counter() - start
    dimensions = grid.dims
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mib = rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)
print(json.dumps({{
    "grid": {case.grid!r},
    "workflow": {case.workflow!r},
    "accelerator": {case.accelerator!r},
    "seconds": elapsed,
    "rss_mib": rss_mib,
    "cells": dimensions["cell"],
    "edges": dimensions["edge"],
    "vertices": dimensions["vertex"],
}}))
"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _best_of(case: PerformanceCase) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    results = [_run_generation_once(case) for _ in range(case.attempts)]
    return min(results, key=lambda row: float(row["seconds"])), results


@pytest.mark.parametrize(
    "case",
    [
        PerformanceCase(
            workflow="raw",
            grid="R02B04",
            accelerator="auto",
            attempts=5,
            max_best_seconds=1.25,
            max_best_rss_mib=180.0,
        ),
        PerformanceCase(
            workflow="raw",
            grid="R02B05",
            accelerator="auto",
            attempts=5,
            max_best_seconds=3.25,
            max_best_rss_mib=360.0,
        ),
        PerformanceCase(
            workflow="raw",
            grid="R01B07",
            accelerator="auto",
            attempts=4,
            max_best_seconds=11.0,
            max_best_rss_mib=1_350.0,
        ),
        PerformanceCase(
            workflow="optimized",
            grid="R02B05",
            accelerator="numba",
            attempts=3,
            max_best_seconds=12.0,
            max_best_rss_mib=600.0,
        ),
        PerformanceCase(
            workflow="streamed",
            grid="R01B08",
            accelerator="numba",
            attempts=2,
            max_best_seconds=90.0,
            max_best_rss_mib=2_500.0,
        ),
    ],
    ids=lambda case: f"{case.workflow}-{case.grid}-{case.accelerator}",
)
def test_global_generation_performance_regression(case: PerformanceCase):
    if case.workflow == "streamed":
        pytest.importorskip("netCDF4")
    best, results = _best_of(case)

    assert best["seconds"] <= case.max_best_seconds, results
    assert best["rss_mib"] <= case.max_best_rss_mib, results


def _run_file_family_once(
    case: FileFamilyPerformanceCase,
) -> dict[str, float | str]:
    code = f"""
import json
import resource
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from grid_generator import (
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    generate_grid_to_netcdf,
)
from grid_generator import _streaming

scratch_root = Path.cwd() / "profiling"
scratch_root.mkdir(parents=True, exist_ok=True)
with TemporaryDirectory(dir=scratch_root) as directory:
    if {case.family!r} == "planar":
        spec = ParallelogramGridSpec(
            nx=300,
            ny=200,
            edge_length=1.0,
            shear=0.2,
        )
        options = {{}}
    else:
        _streaming.DEFAULT_IN_MEMORY_BASE_CELLS = 1_280
        spec = LimitedAreaGridSpec(
            parent="R01B08",
            region=Region.circle(
                lon=0.0,
                lat=0.0,
                radius_degrees=12.0,
            ),
            local_optimization_iterations=0,
        )
        options = {{
            "max_cells": None,
            "accelerator": "numba",
            "optimize_global": False,
        }}
    start = time.perf_counter()
    generate_grid_to_netcdf(
        spec,
        Path(directory) / "grid.nc",
        options=options,
        work_dir=Path(directory) / "work",
        fields="icon4py",
        chunk_size=1_000,
    )
    elapsed = time.perf_counter() - start
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mib = rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)
print(json.dumps({{
    "family": {case.family!r},
    "seconds": elapsed,
    "rss_mib": rss_mib,
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "case",
    [
        FileFamilyPerformanceCase(
            family="planar",
            attempts=2,
            max_best_seconds=5.0,
            max_best_rss_mib=400.0,
        ),
        FileFamilyPerformanceCase(
            family="limited-area",
            attempts=2,
            max_best_seconds=20.0,
            max_best_rss_mib=800.0,
        ),
    ],
    ids=lambda case: case.family,
)
def test_file_oriented_family_memory_regression(case: FileFamilyPerformanceCase):
    pytest.importorskip("netCDF4")
    if case.family == "limited-area":
        pytest.importorskip("numba")
    results = [_run_file_family_once(case) for _ in range(case.attempts)]
    best = min(results, key=lambda row: float(row["seconds"]))

    assert best["seconds"] <= case.max_best_seconds, results
    assert best["rss_mib"] <= case.max_best_rss_mib, results
