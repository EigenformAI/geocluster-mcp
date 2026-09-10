"""Cross-check geometry records against the tabular file they cite.

A record that names a ``source_file`` must have coordinates that actually
occur in that file (within one cell). This turns "coordinates come from the
data" from a prompt rule into a tool guarantee: fabricated coordinates and
wrong-column joins are rejected deterministically.

pandas/scipy are imported lazily (heavy-import rule of the MCP server).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .columns import coordinate_columns, depth_columns, latlon_to_local_meters
from .spatial import (
    BOX_X0, BOX_X1, BOX_Y0, BOX_Y1, END_X, END_Y, LAT_ALIASES, LON_ALIASES, START_X, START_Y,
    X_ALIASES, Y_ALIASES, _lowered, SpatialVoxelStore,
)

DEFAULT_TOLERANCE_M = 1.0
DEFAULT_TOLERANCE_DEG = 1e-5


def _table_points(df, store: SpatialVoxelStore) -> np.ndarray | None:
    """(N, 2) array of the table's coordinates in grid units, or None if it has none."""
    cc = coordinate_columns(list(df.columns))
    if cc is None:
        return None
    x = np.asarray(df[cc["x"]], dtype=float)
    y = np.asarray(df[cc["y"]], dtype=float)
    if cc["kind"] == "geographic" and store.grid.units == "meters":
        conv = store.meta.get("coordinate_conversion") or {}
        if conv.get("kind") != "approx_meters_from_latlon":
            return None  # cannot compare: grid was not derived from lon/lat
        x, y = latlon_to_local_meters(x, y, float(conv["lat0"]))
    ok = np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[ok], y[ok]])


def _record_xy(store: SpatialVoxelStore, low: dict[str, Any], kind: str) -> list[tuple[float, float]]:
    """The (x, y) points a record must justify: point centre, line endpoints, box corners."""
    if kind == "point":
        return [store._point_xy(low, X_ALIASES, Y_ALIASES, LON_ALIASES, LAT_ALIASES)]
    if kind == "line":
        return [
            store._point_xy(low, START_X, START_Y, ("start_longitude", "start_lon"), ("start_latitude", "start_lat")),
            store._point_xy(low, END_X, END_Y, ("end_longitude", "end_lon"), ("end_latitude", "end_lat")),
        ]
    if kind == "box":
        return [
            store._point_xy(low, BOX_X0, BOX_Y0, ("lon_min", "min_longitude"), ("lat_min", "min_latitude")),
            store._point_xy(low, BOX_X1, BOX_Y1, ("lon_max", "max_longitude"), ("lat_max", "max_latitude")),
        ]
    raise ValueError(f"Unsupported geometry_kind: {kind!r}")


def coordinate_mismatches(
    store: SpatialVoxelStore,
    records: list[dict[str, Any]],
    load_table: Callable[[str], Any],
    *,
    tolerance_xy: float | None = None,
) -> dict[str, str]:
    """Return {record_id: reason} for records whose coordinates are not in their source file.

    ``load_table(source_file)`` returns a DataFrame (or raises). Records without
    a ``source_file`` are not checked. Points/line endpoints must lie within
    ``tolerance_xy`` of a table row — default 1 m (1e-5 degrees on a degree
    grid): a record is a *row*, not a neighbourhood. A box must contain at
    least one table row (expanded by the tolerance).
    """
    if tolerance_xy is None:
        tolerance_xy = DEFAULT_TOLERANCE_DEG if store.grid.units == "degrees" else DEFAULT_TOLERANCE_M
    mismatches: dict[str, str] = {}
    tables: dict[str, np.ndarray | None] = {}
    trees: dict[str, Any] = {}
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:  # pragma: no cover - scipy is a declared dependency
        cKDTree = None

    for idx, rec in enumerate(records or []):
        low = _lowered(rec or {})
        record_id = str(low.get("record_id") if low.get("record_id") is not None else idx)
        src = low.get("source_file")
        if not src:
            continue
        src = str(src)
        if src not in tables:
            try:
                df = load_table(src)
                tables[src] = _table_points(df, store)
            except Exception as exc:  # noqa: BLE001
                tables[src] = None
                mismatches[record_id] = f"source_file {src!r} could not be read: {exc}"
                continue
        pts = tables[src]
        if pts is None:
            mismatches[record_id] = f"source_file {src!r} has no recognised coordinate columns to check against"
            continue
        if pts.shape[0] == 0:
            mismatches[record_id] = f"source_file {src!r} has no finite coordinates"
            continue
        kind = str(low.get("geometry_kind") or low.get("geometry_type") or "point").strip().lower()
        try:
            targets = _record_xy(store, low, kind)
        except Exception as exc:  # noqa: BLE001
            mismatches[record_id] = f"coordinates unreadable: {exc}"
            continue

        if kind == "box":
            (x0, y0), (x1, y1) = targets
            lo_x, hi_x = min(x0, x1) - tolerance_xy, max(x0, x1) + tolerance_xy
            lo_y, hi_y = min(y0, y1) - tolerance_xy, max(y0, y1) + tolerance_xy
            inside = (pts[:, 0] >= lo_x) & (pts[:, 0] <= hi_x) & (pts[:, 1] >= lo_y) & (pts[:, 1] <= hi_y)
            if not inside.any():
                mismatches[record_id] = f"box contains no row of {src!r}"
            continue

        if cKDTree is not None:
            if src not in trees:
                trees[src] = cKDTree(pts)
            dists, _ = trees[src].query(np.asarray(targets, dtype=float), k=1)
            worst = float(np.max(dists))
        else:
            worst = 0.0
            for tx, ty in targets:
                d = np.sqrt((pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2)
                worst = max(worst, float(d.min()))
        if not math.isfinite(worst) or worst > tolerance_xy:
            mismatches[record_id] = (
                f"coordinates are {worst:.1f} grid units from the nearest row of {src!r} "
                f"(tolerance {tolerance_xy:.1f})"
            )
    return mismatches
