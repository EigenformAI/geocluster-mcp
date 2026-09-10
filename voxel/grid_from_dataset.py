"""Derive a voxel grid from a project's dataset (bounds in projected metres).

The grid is never hardcoded: it comes from the EASTING/NORTHING (or lon/lat,
converted to local metres) and depth columns of the dataset the analysis
used, so every layer — publisher bins and agent transcriptions — shares one
frame. pandas is imported lazily.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .columns import coordinate_columns, depth_columns, to_local_meters
from .store import GridSpec

DEFAULT_SHAPE = (64, 64, 16)
MAX_VOXELS = 4_000_000
MIN_EXTENT_M = 1.0


def derive_grid(
    dataset_path: str | Path,
    shape: tuple[int, int, int] | list[int] | None = None,
    cell_size_xy_m: float | None = None,
    cell_size_z_m: float | None = None,
    padding_m: float = 0.0,
    max_voxels: int = MAX_VOXELS,
) -> tuple[GridSpec, dict[str, Any]]:
    """Return ``(GridSpec, info)`` for ``dataset_path``.

    Depth extent comes from ``depth_from_m``/``depth_to_m``, else ``depth_m``,
    else ``[0, 1]`` (flagged ``depth_degenerate``). Zero horizontal/vertical
    extents are widened to 1 m. ``shape`` defaults to 64x64x16 (nz=1 when depth
    is degenerate); explicit cell sizes override ``shape``.
    """
    import pandas as pd

    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"dataset not found: {path}")
    columns = list(pd.read_csv(path, nrows=0).columns)
    cc = coordinate_columns(columns)
    if cc is None:
        raise ValueError(
            f"no coordinate columns in {path.name}: expected EASTING/NORTHING (or x/y, coord_x/coord_y) "
            "or LONGITUDE/LATITUDE"
        )
    dc = depth_columns(columns)
    usecols = [cc["x"], cc["y"]] + [c for c in (dc["depth"], dc["from"], dc["to"]) if c]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    n_rows = int(len(df))

    x = pd.to_numeric(df[cc["x"]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[cc["y"]], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        raise ValueError(f"{path.name}: coordinate columns {cc['x']}/{cc['y']} contain no numeric values")
    xm, ym, crs, conversion = to_local_meters(x[ok], y[ok], cc["kind"])

    depth_source = None
    depth_degenerate = False
    if dc["from"] and dc["to"]:
        d0 = pd.to_numeric(df[dc["from"]], errors="coerce").to_numpy(dtype=float)
        d1 = pd.to_numeric(df[dc["to"]], errors="coerce").to_numpy(dtype=float)
        zs = np.concatenate([d0[np.isfinite(d0)], d1[np.isfinite(d1)]])
        depth_source = f"{dc['from']}/{dc['to']}"
    elif dc["depth"]:
        d = pd.to_numeric(df[dc["depth"]], errors="coerce").to_numpy(dtype=float)
        zs = d[np.isfinite(d)]
        depth_source = dc["depth"]
    else:
        zs = np.array([], dtype=float)
    if zs.size == 0:
        z0, z1 = 0.0, 1.0
        depth_degenerate = True
        depth_source = depth_source or "none (flat grid)"
    else:
        z0, z1 = float(zs.min()), float(zs.max())

    pad = float(padding_m or 0.0)
    x0, x1 = float(xm.min()) - pad, float(xm.max()) + pad
    y0, y1 = float(ym.min()) - pad, float(ym.max()) + pad
    widened = []
    if x1 - x0 < MIN_EXTENT_M:
        x1 = x0 + MIN_EXTENT_M
        widened.append("x")
    if y1 - y0 < MIN_EXTENT_M:
        y1 = y0 + MIN_EXTENT_M
        widened.append("y")
    if z1 - z0 < MIN_EXTENT_M:
        z1 = z0 + MIN_EXTENT_M
        widened.append("depth")
        depth_degenerate = True

    if cell_size_xy_m or cell_size_z_m:
        cxy = float(cell_size_xy_m) if cell_size_xy_m else None
        cz = float(cell_size_z_m) if cell_size_z_m else None
        base = tuple(shape) if shape else DEFAULT_SHAPE
        nx = max(1, math.ceil((x1 - x0) / cxy)) if cxy else base[0]
        ny = max(1, math.ceil((y1 - y0) / cxy)) if cxy else base[1]
        nz = max(1, math.ceil((z1 - z0) / cz)) if cz else base[2]
        # snap the maximum so cells are exactly the requested size
        if cxy:
            x1, y1 = x0 + nx * cxy, y0 + ny * cxy
        if cz:
            z1 = z0 + nz * cz
        shape_t = (nx, ny, nz)
    elif shape:
        shape_t = tuple(int(v) for v in shape)
    else:
        shape_t = (DEFAULT_SHAPE[0], DEFAULT_SHAPE[1], 1 if depth_degenerate else DEFAULT_SHAPE[2])
    if depth_degenerate and not cell_size_z_m and shape is None:
        shape_t = (shape_t[0], shape_t[1], 1)

    n_vox = shape_t[0] * shape_t[1] * shape_t[2]
    if n_vox > max_voxels:
        ext = ((x1 - x0), (y1 - y0), (z1 - z0))
        min_cell = (ext[0] * ext[1] * ext[2] / max_voxels) ** (1 / 3)
        raise ValueError(
            f"grid {shape_t} has {n_vox:,} voxels (max {max_voxels:,}); use a coarser shape or "
            f"cell size >= ~{min_cell:.1f} m"
        )

    grid = GridSpec(origin=(x0, y0, z0), maximum=(x1, y1, z1), shape=shape_t, crs=crs)
    info = {
        "dataset_path": str(path),
        "columns": {"x": cc["x"], "y": cc["y"], "kind": cc["kind"], "depth": depth_source},
        "rows": n_rows,
        "rows_with_coordinates": int(ok.sum()),
        "coordinate_conversion": conversion,
        "depth_degenerate": depth_degenerate,
        "widened_axes": widened,
        "padding_m": pad,
    }
    return grid, info
