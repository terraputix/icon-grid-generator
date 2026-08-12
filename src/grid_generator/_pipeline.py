"""Shared file-oriented generation pipeline for every supported grid family."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_grid_to_netcdf(
    spec: Any,
    options: Any,
    path: str | Path,
    *,
    chunk_size: int,
    work_dir: str | Path | None,
    resume: bool,
    selected_fields: frozenset[str],
) -> Path:
    """Generate one validated spec through the common file-oriented path."""
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    from . import grid_generator as gg

    if isinstance(spec, gg.GlobalGridSpec):
        from ._streaming import generate_global_grid_to_netcdf

        return generate_global_grid_to_netcdf(
            spec,
            options,
            path,
            chunk_size=chunk_size,
            work_dir=work_dir,
            resume=resume,
            selected_fields=selected_fields,
        )

    if isinstance(spec, gg.LimitedAreaGridSpec):
        grid = _generate_limited_area_grid(
            spec,
            options,
            path,
            work_dir=work_dir,
            resume=resume,
            chunk_size=chunk_size,
        )
    else:
        grid = gg._generate_resolved_grid(spec, options)

    from ._netcdf import write_icon_grid

    return write_icon_grid(
        grid,
        path,
        fields=selected_fields,
        chunk_size=chunk_size,
    )


def _generate_limited_area_grid(
    spec: Any,
    options: Any,
    path: str | Path,
    *,
    work_dir: str | Path | None,
    resume: bool,
    chunk_size: int,
) -> Any:
    """Generate a regional grid from a compact, optionally staged parent."""
    from . import grid_generator as gg
    from ._limited_area import LimitedAreaExtractor
    from ._streaming import compact_global_grid_for_export

    construction = spec.construction
    if construction == "refine_last" and spec.parent.bisections > 0:
        parent_spec = gg.GlobalGridSpec(
            root=spec.parent.root,
            bisections=spec.parent.bisections - 1,
        )
    else:
        construction = "cut_final"
        parent_spec = spec.parent
    parent = compact_global_grid_for_export(
        parent_spec,
        options,
        path,
        work_dir=work_dir,
        resume=resume,
        final_provenance=False,
    )
    result = LimitedAreaExtractor().build_from_compact_parent(
        spec,
        options,
        parent,
        construction,
        chunk_size=chunk_size,
    )
    return gg._assemble_limited_area_grid(spec, options, result)
