"""Column-name conventions shared by the publisher, the grid derivation and the
voxel tools. Patterns mirror the platform's dataset validator: projected
metres first, then geographic degrees converted to approximate local metres.

numpy only at module level (used by the system-python publisher CLI too).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

EASTING_PATTERNS = [r"^easting", r"easting.*gda", r"^coord_x$", r"^x$"]
NORTHING_PATTERNS = [r"^northing", r"northing.*gda", r"^coord_y$", r"^y$"]
LON_PATTERNS = [r"^longitude", r"^lon$", r"longitude.*gda"]
LAT_PATTERNS = [r"^latitude", r"^lat$", r"latitude.*gda"]
HOLE_PATTERNS = [r"^drillhole", r"^hole_id", r"^drill_hole", r"^borehole", r"^site_no$"]
LITHOLOGY_PATTERNS = [r"^major_lithology$", r"lithology"]
DEPTH_COLUMN = "depth_m"
DEPTH_FROM_COLUMN = "depth_from_m"
DEPTH_TO_COLUMN = "depth_to_m"
METERS_PER_DEG_LAT = 111_320.0

CRS_PROJECTED = "projected_meters"
CRS_APPROX_LATLON = "approx_meters_from_latlon"


def find_column(columns, patterns) -> str | None:
    for pattern in patterns:
        for col in columns:
            if re.search(pattern, str(col).strip().lower()):
                return col
    return None


def coordinate_columns(columns) -> dict[str, Any] | None:
    """Pick the coordinate pair: projected easting/northing preferred, else lon/lat."""
    east, north = find_column(columns, EASTING_PATTERNS), find_column(columns, NORTHING_PATTERNS)
    if east and north:
        return {"x": east, "y": north, "kind": "projected"}
    lon, lat = find_column(columns, LON_PATTERNS), find_column(columns, LAT_PATTERNS)
    if lon and lat:
        return {"x": lon, "y": lat, "kind": "geographic"}
    return None


def latlon_to_local_meters(lon, lat, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular approximation, identical to the publisher's ``build_samples``."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = lon * METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    y = lat * METERS_PER_DEG_LAT
    return x, y


def to_local_meters(x, y, kind: str, lat0: float | None = None) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Return (x_m, y_m, crs, conversion) for a coordinate pair of ``kind``.

    ``conversion`` is stored in the voxel store meta so that later callers
    (geometry records given as lon/lat, the source cross-check) convert the
    same way.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if kind == "projected":
        return x, y, CRS_PROJECTED, {"kind": "identity"}
    if lat0 is None:
        finite = np.isfinite(y)
        lat0 = float(np.mean(y[finite])) if finite.any() else 0.0
    xm, ym = latlon_to_local_meters(x, y, lat0)
    return xm, ym, CRS_APPROX_LATLON, {
        "kind": CRS_APPROX_LATLON,
        "lat0": lat0,
        "meters_per_deg_lat": METERS_PER_DEG_LAT,
        "formula": "x = lon * 111320 * cos(radians(lat0)); y = lat * 111320",
    }


def depth_columns(columns) -> dict[str, str | None]:
    cols = {str(c).strip().lower(): c for c in columns}
    return {
        "depth": cols.get(DEPTH_COLUMN),
        "from": cols.get(DEPTH_FROM_COLUMN),
        "to": cols.get(DEPTH_TO_COLUMN),
    }
