"""ICON NetCDF writing boundary."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


class IconNetcdfWriter:
    """Write complete ICON grid objects to NetCDF."""

    def write(
        self,
        grid: Any,
        path: str | Path,
        *,
        sphere_radius: float | None = None,
        fields: str | Iterable[str] = "full",
    ) -> Path:
        from ._netcdf import write_icon_grid

        return write_icon_grid(
            grid,
            path,
            sphere_radius=sphere_radius,
            fields=fields,
        )
