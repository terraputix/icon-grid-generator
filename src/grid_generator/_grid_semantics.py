"""ICON geometry codes and internal grid-topology semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SPHERE_GEOMETRY = 1
PLANAR_TORUS_GEOMETRY = 2
PLANAR_CHANNEL_GEOMETRY = 3
PLANAR_GEOMETRY = 4

_PLANAR_GEOMETRIES = frozenset(
    {PLANAR_TORUS_GEOMETRY, PLANAR_CHANNEL_GEOMETRY, PLANAR_GEOMETRY}
)


def is_planar_geometry(metadata: Mapping[str, Any]) -> bool:
    """Return whether ICON classifies the grid geometry as planar."""
    return metadata.get("grid_geometry") in _PLANAR_GEOMETRIES


def is_icon_open_boundary(metadata: Mapping[str, Any]) -> bool:
    """Return whether strict regional ICON open-boundary rules apply."""
    return bool(metadata.get("open_boundary", False))
