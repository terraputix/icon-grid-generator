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


def _run_generation_once(case: PerformanceCase) -> dict[str, float | int | str]:
    code = f"""
import json
import resource
import sys
from tempfile import TemporaryDirectory
import time

from grid_generator import generate_grid, generate_grid_to_netcdf
from grid_generator.grid_generator import parse_grid_spec

start = time.perf_counter()
if {case.workflow!r} == "streamed":
    spec = parse_grid_spec({case.grid!r})
    with TemporaryDirectory() as directory:
        generate_grid_to_netcdf(
            spec,
            f"{{directory}}/grid.nc",
            options={{"max_cells": None, "accelerator": {case.accelerator!r}}},
            work_dir=f"{{directory}}/checkpoints",
            fields="icon4py",
        )
    dimensions = {{
        "cell": spec.expected_cells,
        "edge": spec.expected_edges,
        "vertex": spec.expected_vertices,
    }}
else:
    options = {{"max_cells": None, "accelerator": {case.accelerator!r}}}
    if {case.workflow!r} == "raw":
        options["optimize_global"] = False
    grid = generate_grid({case.grid!r}, options=options)
    dimensions = grid.dims
elapsed = time.perf_counter() - start
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
            max_best_seconds=45.0,
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
