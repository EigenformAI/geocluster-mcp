"""Spatial voxel store: stamp point / line / box features and full arrays into layers.

Port of combo-geology-nsl ``voxel_features/spatial.py`` (commit fd3091f):
- generic ``(x, y, depth)`` coordinates in the grid's units; radii/widths are
  metres and map 1:1 on projected grids (degree grids use the local-metres
  approximation, as upstream);
- every region is a boolean mask over the grid (vectorised — no Python
  triple loops); line sampling steps at half the smallest cell size;
- geometry records accept projected aliases (``x``/``easting``, ``y``/
  ``northing``, ``depth_m``…) as well as the upstream ``lon``/``lat`` names;
  lon/lat records are converted with the conversion the grid was derived with;
- ``coordinate_source`` must be ``"artifact"`` (data-derived) — records that
  claim anything else are rejected, and coordinates are never inferred;
- provenance goes to ``operations.jsonl`` (see ``opslog``) instead of SQLite;
- categorical/boolean empty sentinel is ``-1`` so label 0 is a valid class.
"""

from __future__ import annotations

import math
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .opslog import OperationsLog
from .store import CATEGORICAL_EMPTY, GridSpec, VoxelStore, empty_value, validate_layer_name

CombinationRule = Literal["replace", "max", "add", "mean"]
BoundsPolicy = Literal["skip", "clip", "fail"]
COMBINATION_RULES = ("replace", "max", "add", "mean")
BOUNDS_POLICIES = ("skip", "clip", "fail")
ALLOWED_COORDINATE_SOURCE = "artifact"
DEFAULT_SOURCE = "artifact"

# Per-store-path locks serialise read-modify-write on layers: FastMCP may run
# tool calls on threads and each call may build a fresh store instance.
_RMW_LOCKS: dict[str, threading.RLock] = {}
_RMW_GUARD = threading.Lock()


def _rmw_lock(store_path: Path) -> threading.RLock:
    key = str(store_path)
    with _RMW_GUARD:
        lk = _RMW_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _RMW_LOCKS[key] = lk
        return lk


# Coordinate column aliases for geometry records (matched case-insensitively).
X_ALIASES = ("x", "easting", "east", "coord_x")
Y_ALIASES = ("y", "northing", "north", "coord_y")
LON_ALIASES = ("longitude", "lon", "long", "lng")
LAT_ALIASES = ("latitude", "lat")
DEPTH_ALIASES = ("depth_m", "depth", "depth_mid_m", "depth_meters", "z")
START_X = ("start_x", "start_easting", "start_longitude", "start_lon")
START_Y = ("start_y", "start_northing", "start_latitude", "start_lat")
START_Z = ("start_depth_m", "start_depth", "start_z")
END_X = ("end_x", "end_easting", "end_longitude", "end_lon")
END_Y = ("end_y", "end_northing", "end_latitude", "end_lat")
END_Z = ("end_depth_m", "end_depth", "end_z")
BOX_X0 = ("x_min", "easting_min", "lon_min", "min_x", "min_longitude")
BOX_Y0 = ("y_min", "northing_min", "lat_min", "min_y", "min_latitude")
BOX_Z0 = ("depth_min_m", "depth_min", "min_depth_m", "z_min")
BOX_X1 = ("x_max", "easting_max", "lon_max", "max_x", "max_longitude")
BOX_Y1 = ("y_max", "northing_max", "lat_max", "max_y", "max_latitude")
BOX_Z1 = ("depth_max_m", "depth_max", "max_depth_m", "z_max")
_GEOGRAPHIC_HINT = ("longitude", "lon", "long", "lng", "latitude", "lat")


def _lowered(record: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in record.items()}


def _pick(lowered: dict[str, Any], aliases: tuple[str, ...]) -> tuple[str | None, Any]:
    for a in aliases:
        v = lowered.get(a)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v).strip() != "":
            return a, v
    return None, None


class SpatialVoxelStore(VoxelStore):
    """``VoxelStore`` plus geometry stamping, array deposit, probing and provenance."""

    def __init__(self, store_path: Path | str, grid: GridSpec | None = None, *, meta: dict[str, Any] | None = None):
        super().__init__(store_path, grid, meta=meta)
        self.ops = OperationsLog(self.store_path / OperationsLog.FILENAME)

    # --- coordinates -----------------------------------------------------------
    def coord_to_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return self.grid.coord_to_index(x, y, z)

    def index_to_coord(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        return self.grid.index_to_coord(ix, iy, iz)

    def in_bounds(self, x: float, y: float, z: float) -> bool:
        return self.grid.in_bounds(x, y, z)

    def lonlat_to_grid(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert geographic coordinates to grid units using the store's conversion."""
        if self.grid.units == "degrees":
            return float(lon), float(lat)
        conv = self.meta.get("coordinate_conversion") or {}
        if conv.get("kind") == "approx_meters_from_latlon" and "lat0" in conv:
            from .columns import latlon_to_local_meters

            x, y = latlon_to_local_meters([lon], [lat], float(conv["lat0"]))
            return float(x[0]), float(y[0])
        raise ValueError(
            "record gives longitude/latitude but the grid is in projected metres with no "
            "lon/lat conversion recorded — supply x/y (easting/northing) instead"
        )

    def _radius_units(self, radius_m: float, y: float) -> tuple[float, float, float]:
        """Radius in (x, y, z) grid units. z is always metres."""
        r = float(radius_m)
        if self.grid.units == "degrees":
            lat_deg = r / 111_320.0
            lon_deg = r / (111_320.0 * max(1e-9, math.cos(math.radians(y))))
            return lon_deg, lat_deg, r
        return r, r, r

    # --- region masks ----------------------------------------------------------
    def sphere_mask(self, x: float, y: float, z: float, radius_m: float) -> np.ndarray:
        """Boolean mask of cells whose centre lies within ``radius_m`` of (x, y, z).

        The containing cell is always claimed, so a sub-cell radius still
        stamps one voxel (upstream lost ~all records before this rule).
        """
        grid = self.grid
        mask = np.zeros(grid.shape, dtype=bool)
        cx, cy, cz = grid.coord_to_index(x, y, z)
        mask[cx, cy, cz] = True
        if radius_m <= 0:
            return mask
        rx, ry, rz = self._radius_units(radius_m, y)
        dx, dy, dz = grid.cell_size
        nx, ny, nz = grid.shape
        x0, x1 = max(0, int((x - rx - grid.origin[0]) // dx)), min(nx - 1, int((x + rx - grid.origin[0]) // dx))
        y0, y1 = max(0, int((y - ry - grid.origin[1]) // dy)), min(ny - 1, int((y + ry - grid.origin[1]) // dy))
        z0, z1 = max(0, int((z - rz - grid.origin[2]) // dz)), min(nz - 1, int((z + rz - grid.origin[2]) // dz))
        if x1 < x0 or y1 < y0 or z1 < z0:
            return mask
        xs = grid.origin[0] + (np.arange(x0, x1 + 1) + 0.5) * dx
        ys = grid.origin[1] + (np.arange(y0, y1 + 1) + 0.5) * dy
        zs = grid.origin[2] + (np.arange(z0, z1 + 1) + 0.5) * dz
        # distances in metres (scale degree axes back to metres)
        sx = 1.0 if self.grid.units == "meters" else radius_m / rx
        sy = 1.0 if self.grid.units == "meters" else radius_m / ry
        ddx = ((xs - x) * sx)[:, None, None]
        ddy = ((ys - y) * sy)[None, :, None]
        ddz = (zs - z)[None, None, :]
        within = ddx**2 + ddy**2 + ddz**2 <= float(radius_m) ** 2
        mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] |= within
        return mask

    def box_mask(
        self,
        x_min: float, y_min: float, z_min: float,
        x_max: float, y_max: float, z_max: float,
        *,
        bounds_policy: BoundsPolicy = "clip",
    ) -> np.ndarray | None:
        grid = self.grid
        lo = [min(a, b) for a, b in zip((x_min, y_min, z_min), (x_max, y_max, z_max))]
        hi = [max(a, b) for a, b in zip((x_min, y_min, z_min), (x_max, y_max, z_max))]
        outside = [
            (label, v)
            for label, v, o, m in zip(("x", "y", "depth"), lo + hi, grid.origin * 2, grid.maximum * 2)
            if v < o or v > m
        ]
        if outside:
            if bounds_policy == "fail":
                label, v = outside[0]
                raise ValueError(f"{label} {v} outside grid bounds")
            if bounds_policy == "skip":
                return None
        for axis in range(3):
            if hi[axis] < grid.origin[axis] or lo[axis] > grid.maximum[axis]:
                return None  # no overlap at all
        lo = grid.clamp(*lo)
        hi = grid.clamp(*hi)
        i0 = grid.coord_to_index(*lo)
        i1 = grid.coord_to_index(*hi)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[i0[0] : i1[0] + 1, i0[1] : i1[1] + 1, i0[2] : i1[2] + 1] = True
        return mask

    def line_mask(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        width_m: float,
        *,
        bounds_policy: BoundsPolicy = "clip",
    ) -> np.ndarray | None:
        grid = self.grid
        s, e = tuple(map(float, start)), tuple(map(float, end))
        if not (grid.in_bounds(*s) and grid.in_bounds(*e)):
            if bounds_policy == "fail":
                raise ValueError(f"line endpoint outside grid bounds: {s} -> {e}")
            if bounds_policy == "skip":
                return None
            s, e = grid.clamp(*s), grid.clamp(*e)
        # sample every half-cell (in metres along the segment)
        sx = sy = 1.0
        if grid.units == "degrees":
            lon_m = 111_320.0 * max(1e-9, math.cos(math.radians(s[1])))
            sx, sy = lon_m, 111_320.0
        length_m = math.sqrt(((e[0] - s[0]) * sx) ** 2 + ((e[1] - s[1]) * sy) ** 2 + (e[2] - s[2]) ** 2)
        cell_m = min(grid.cell_size[0] * sx, grid.cell_size[1] * sy, grid.cell_size[2])
        n = int(min(20_000, max(2, math.ceil(length_m / max(cell_m / 2.0, 1e-6)) + 1)))
        mask = np.zeros(grid.shape, dtype=bool)
        radius = float(width_m) / 2.0
        for t in np.linspace(0.0, 1.0, n):
            p = (s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1]), s[2] + t * (e[2] - s[2]))
            mask |= self.sphere_mask(*p, radius)
        return mask

    # --- combination -----------------------------------------------------------
    @staticmethod
    def _apply(layer_values: np.ndarray, mask: np.ndarray, value: float, rule: str, dtype: str) -> int:
        if rule not in COMBINATION_RULES:
            raise ValueError(f"Unsupported combination_rule: {rule}")
        affected = int(mask.sum())
        if affected == 0:
            return 0
        current = layer_values[mask]
        empty = empty_value(dtype)
        if rule == "replace":
            layer_values[mask] = value
        elif rule == "max":
            layer_values[mask] = np.where(current == empty, value, np.maximum(current, value))
        elif rule == "add":
            layer_values[mask] = np.where(current == empty, value, current + value)
        else:  # mean
            layer_values[mask] = np.where(current == empty, value, (current + value) / 2.0)
        return affected

    def _rmw(
        self,
        name: str,
        dtype: str,
        mutate: Callable[[np.ndarray], int],
        *,
        fresh: bool = False,
        metadata: dict[str, Any] | None = None,
        hypothesis: str | None = None,
    ) -> int:
        """Locked, disk-truthful read-modify-write of layer ``name``."""
        validate_layer_name(name)
        with _rmw_lock(self.store_path):
            path = self.layer_path(name)
            if not fresh and path.exists():
                values = np.load(path).astype(float, copy=True)
            else:
                values = self.new_layer_values(dtype)
            affected = mutate(values)
            existing = self._layers.get(name)
            meta = dict(existing.metadata) if (existing and not fresh) else {}
            meta.pop("_content_hash", None)
            meta.update(metadata or {})
            self.put_layer(name, values, dtype, metadata=meta, hypothesis=hypothesis or (existing.hypothesis if existing else None))
            return affected

    # --- single features ---------------------------------------------------------
    def _log(self, op: str, name: str, coordinates: str, parameters: str, *, source_file, source_excerpt,
             coordinate_source, affected_voxels, record_id=None, group_id=None) -> None:
        self.ops.append({
            "op": op,
            "layer": name,
            "coordinates": coordinates,
            "parameters": parameters,
            "source_file": source_file,
            "source_excerpt": source_excerpt,
            "coordinate_source": coordinate_source,
            "record_id": record_id,
            "group_id": group_id,
            "affected_voxels": int(affected_voxels),
        })

    @staticmethod
    def _check_source(coordinate_source: str | None) -> str:
        cs = str(coordinate_source or DEFAULT_SOURCE).strip().lower()
        if cs != ALLOWED_COORDINATE_SOURCE:
            raise ValueError(
                f"coordinate_source must be '{ALLOWED_COORDINATE_SOURCE}' (data-derived); got {cs!r}. "
                "Coordinates are never inferred — take them from the dataset."
            )
        return cs

    def add_point_feature(
        self, name: str, x: float, y: float, depth_m: float, value: float, radius_m: float = 50.0,
        dtype: str = "float", combination_rule: str = "max", source_file: str | None = None,
        source_excerpt: str | None = None, coordinate_source: str = DEFAULT_SOURCE,
        hypothesis: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            cs = self._check_source(coordinate_source)
            x, y, depth_m, value = float(x), float(y), float(depth_m), float(value)
            if not self.in_bounds(x, y, depth_m):
                raise ValueError(f"point ({x}, {y}, {depth_m}) outside grid bounds")
            mask = self.sphere_mask(x, y, depth_m, float(radius_m))
            affected = self._rmw(name, dtype, lambda v: self._apply(v, mask, value, combination_rule, dtype),
                                 metadata=metadata, hypothesis=hypothesis)
            self._log("point", name, f"{x},{y},{depth_m}", f"radius_m={radius_m},value={value}",
                      source_file=source_file, source_excerpt=source_excerpt, coordinate_source=cs,
                      affected_voxels=affected)
            return {"success": True, "operation": "point_feature", "layer_name": name, "affected_voxels": affected}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "operation": "point_feature", "layer_name": name, "error": str(e)}

    def add_line_feature(
        self, name: str, start: tuple[float, float, float], end: tuple[float, float, float], value: float,
        width_m: float = 25.0, dtype: str = "float", combination_rule: str = "max",
        source_file: str | None = None, source_excerpt: str | None = None,
        coordinate_source: str = DEFAULT_SOURCE, hypothesis: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            cs = self._check_source(coordinate_source)
            value = float(value)
            mask = self.line_mask(start, end, float(width_m), bounds_policy="fail")
            affected = self._rmw(name, dtype, lambda v: self._apply(v, mask, value, combination_rule, dtype),
                                 metadata=metadata, hypothesis=hypothesis)
            coords = ";".join(",".join(str(float(c)) for c in p) for p in (start, end))
            self._log("line", name, coords, f"width_m={width_m},value={value}", source_file=source_file,
                      source_excerpt=source_excerpt, coordinate_source=cs, affected_voxels=affected)
            return {"success": True, "operation": "line_feature", "layer_name": name, "affected_voxels": affected}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "operation": "line_feature", "layer_name": name, "error": str(e)}

    def add_box_feature(
        self, name: str, x_min: float, y_min: float, depth_min_m: float, x_max: float, y_max: float,
        depth_max_m: float, value: float, dtype: str = "float", combination_rule: str = "max",
        source_file: str | None = None, source_excerpt: str | None = None,
        coordinate_source: str = DEFAULT_SOURCE, hypothesis: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            cs = self._check_source(coordinate_source)
            value = float(value)
            mask = self.box_mask(x_min, y_min, depth_min_m, x_max, y_max, depth_max_m, bounds_policy="clip")
            if mask is None:
                raise ValueError("box does not overlap grid bounds")
            affected = self._rmw(name, dtype, lambda v: self._apply(v, mask, value, combination_rule, dtype),
                                 metadata=metadata, hypothesis=hypothesis)
            coords = f"{x_min},{y_min},{depth_min_m};{x_max},{y_max},{depth_max_m}"
            self._log("box", name, coords, f"value={value}", source_file=source_file,
                      source_excerpt=source_excerpt, coordinate_source=cs, affected_voxels=affected)
            return {"success": True, "operation": "box_feature", "layer_name": name, "affected_voxels": affected}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "operation": "box_feature", "layer_name": name, "error": str(e)}

    # --- full array ------------------------------------------------------------------
    def set_layer_array(
        self, name: str, values: np.ndarray, dtype: str = "float", *, source_file: str | None = None,
        source_excerpt: str | None = None, coordinate_source: str = DEFAULT_SOURCE,
        hypothesis: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deposit a full per-voxel array verbatim as layer ``name`` (replaces it)."""
        cs = self._check_source(coordinate_source)
        arr = np.asarray(values, dtype=float)
        if arr.shape != self.grid.shape:
            raise ValueError(f"Array shape {tuple(arr.shape)} does not match grid shape {tuple(self.grid.shape)}")
        empty = empty_value(dtype)
        nonempty = np.isfinite(arr) & (arr != empty)
        with _rmw_lock(self.store_path):
            self.put_layer(name, arr, dtype, metadata=metadata, hypothesis=hypothesis)
            self.ops.reset_layer(name)
            self._log("array", name, f"grid_origin={self.grid.origin};shape={tuple(arr.shape)}",
                      f"dtype={dtype},nonempty_voxels={int(nonempty.sum())},"
                      f"value_min={float(arr[nonempty].min()) if nonempty.any() else None},"
                      f"value_max={float(arr[nonempty].max()) if nonempty.any() else None}",
                      source_file=source_file, source_excerpt=source_excerpt, coordinate_source=cs,
                      affected_voxels=int(nonempty.sum()))
        return {
            "success": True, "operation": "set_layer_array", "layer_name": name, "shape": list(arr.shape),
            "nonempty_voxels": int(nonempty.sum()),
            "value_min": float(arr[nonempty].min()) if nonempty.any() else None,
            "value_max": float(arr[nonempty].max()) if nonempty.any() else None,
            "distinct_nonempty_values": int(np.unique(arr[nonempty]).size) if nonempty.any() else 0,
        }

    # --- geometry records --------------------------------------------------------------
    def _point_xy(self, lowered: dict[str, Any], xa: tuple[str, ...], ya: tuple[str, ...],
                  lon_a: tuple[str, ...], lat_a: tuple[str, ...]) -> tuple[float, float]:
        kx, vx = _pick(lowered, xa)
        ky, vy = _pick(lowered, ya)
        if kx is not None and ky is not None and not (kx in _GEOGRAPHIC_HINT or ky in _GEOGRAPHIC_HINT):
            return float(vx), float(vy)
        klon, vlon = _pick(lowered, lon_a)
        klat, vlat = _pick(lowered, lat_a)
        if klon is not None and klat is not None:
            return self.lonlat_to_grid(float(vlon), float(vlat))
        if kx is not None and ky is not None:  # geographic-named start_/end_ aliases
            return self.lonlat_to_grid(float(vx), float(vy))
        raise KeyError(f"no coordinates: expected one of {xa[:2]} + {ya[:2]} or {lon_a[:1]} + {lat_a[:1]}")

    def _record_mask(self, rec: dict[str, Any], kind: str, bounds_policy: str) -> tuple[np.ndarray | None, str, str]:
        """Return (mask, coordinates_str, parameters_str) for one geometry record."""
        low = _lowered(rec)
        if kind == "point":
            x, y = self._point_xy(low, X_ALIASES, Y_ALIASES, LON_ALIASES, LAT_ALIASES)
            _, dv = _pick(low, DEPTH_ALIASES)
            z = float(dv) if dv is not None else float(self.grid.origin[2])
            _, rv = _pick(low, ("radius_m", "radius"))
            radius = float(rv) if rv is not None else 0.0
            if not self.in_bounds(x, y, z):
                if bounds_policy == "fail":
                    raise ValueError(f"point ({x}, {y}, {z}) outside grid bounds")
                if bounds_policy == "skip":
                    return None, f"{x},{y},{z}", f"radius_m={radius}"
                x, y, z = self.grid.clamp(x, y, z)
            return self.sphere_mask(x, y, z, radius), f"{x},{y},{z}", f"radius_m={radius}"
        if kind == "line":
            sx, sy = self._point_xy(low, START_X, START_Y, ("start_longitude", "start_lon"), ("start_latitude", "start_lat"))
            ex, ey = self._point_xy(low, END_X, END_Y, ("end_longitude", "end_lon"), ("end_latitude", "end_lat"))
            _, sz = _pick(low, START_Z)
            _, ez = _pick(low, END_Z)
            sz = float(sz) if sz is not None else float(self.grid.origin[2])
            ez = float(ez) if ez is not None else float(self.grid.origin[2])
            _, wv = _pick(low, ("width_m", "width"))
            width = float(wv) if wv is not None else 0.0
            mask = self.line_mask((sx, sy, sz), (ex, ey, ez), width, bounds_policy=bounds_policy)
            return mask, f"{sx},{sy},{sz};{ex},{ey},{ez}", f"width_m={width}"
        if kind == "box":
            x0, y0 = self._point_xy(low, BOX_X0, BOX_Y0, ("lon_min", "min_longitude"), ("lat_min", "min_latitude"))
            x1, y1 = self._point_xy(low, BOX_X1, BOX_Y1, ("lon_max", "max_longitude"), ("lat_max", "max_latitude"))
            _, z0 = _pick(low, BOX_Z0)
            _, z1 = _pick(low, BOX_Z1)
            z0 = float(z0) if z0 is not None else float(self.grid.origin[2])
            z1 = float(z1) if z1 is not None else float(self.grid.maximum[2])
            mask = self.box_mask(x0, y0, z0, x1, y1, z1, bounds_policy=bounds_policy)
            return mask, f"{x0},{y0},{z0};{x1},{y1},{z1}", ""
        raise ValueError(f"Unsupported geometry_kind: {kind!r} (point|line|box)")

    def add_geometry_batch(
        self,
        name: str,
        records: list[dict[str, Any]],
        *,
        mode: Literal["replace_layer", "accumulate_layer"] = "replace_layer",
        dtype: str = "float",
        combination_rule: str = "max",
        max_records: int = 5000,
        bounds_policy: str = "skip",
        metadata: dict[str, Any] | None = None,
        hypothesis: str | None = None,
        rejected: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Materialise point/line/box records into one layer in one locked write.

        ``rejected`` maps record_id -> reason for records the caller already
        refused (e.g. the source cross-check); they are skipped and reported.
        """
        group_id = uuid.uuid4().hex
        seen = len(records or [])
        warnings: list[str] = []
        if mode not in ("replace_layer", "accumulate_layer"):
            return {"success": False, "operation": "geometry_batch", "error": f"Unsupported mode: {mode}"}
        if bounds_policy not in BOUNDS_POLICIES:
            return {"success": False, "operation": "geometry_batch", "error": f"Unsupported bounds_policy: {bounds_policy}"}
        if combination_rule not in COMBINATION_RULES:
            return {"success": False, "operation": "geometry_batch", "error": f"Unsupported combination_rule: {combination_rule}"}
        rejected = rejected or {}
        capped = max(0, seen - int(max_records))
        batch = list(records or [])[: int(max_records)]
        if capped:
            warnings.append(f"Skipped {capped} records beyond max_records={max_records}")

        applied: list[dict[str, Any]] = []
        kind_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        values_applied: list[float] = []
        affected_total = 0
        skipped_source = 0

        def mutate(layer_values: np.ndarray) -> int:
            nonlocal affected_total, skipped_source
            for idx, record in enumerate(batch):
                rec = dict(record or {})
                low = _lowered(rec)
                kind = str(low.get("geometry_kind") or low.get("geometry_type") or "point").strip().lower()
                record_id = str(low.get("record_id") if low.get("record_id") is not None else idx)
                if record_id in rejected:
                    skipped_source += 1
                    warnings.append(f"Rejected record {record_id}: {rejected[record_id]}")
                    continue
                try:
                    cs = self._check_source(low.get("coordinate_source"))
                    raw_value = low.get("value", 1.0)
                    try:
                        value = float(raw_value)
                        if math.isnan(value):
                            raise ValueError
                    except (TypeError, ValueError):
                        value = 1.0
                        warnings.append(f"Record {record_id}: non-numeric value {raw_value!r} coerced to presence 1.0")
                    rule = str(low.get("combination_rule") or combination_rule)
                    mask, coords, params = self._record_mask(rec, kind, bounds_policy)
                    if mask is None:
                        warnings.append(f"Skipped record {record_id}: outside grid bounds")
                        continue
                    affected = self._apply(layer_values, mask, value, rule, dtype)
                    if affected == 0:
                        warnings.append(f"Skipped record {record_id}: geometry intersects no voxel")
                        continue
                    applied.append({
                        "op": kind, "layer": name, "coordinates": coords,
                        "parameters": f"{params},value={value}".strip(","),
                        "source_file": low.get("source_file") or None,
                        "source_excerpt": low.get("source_excerpt") or None,
                        "coordinate_source": cs, "record_id": record_id, "group_id": group_id,
                        "affected_voxels": affected,
                    })
                    kind_counts[kind] = kind_counts.get(kind, 0) + 1
                    source_counts[cs] = source_counts.get(cs, 0) + 1
                    affected_total += affected
                    values_applied.append(value)
                except Exception as exc:  # noqa: BLE001
                    if bounds_policy == "fail":
                        raise
                    warnings.append(f"Skipped record {record_id}: {exc}")
            return affected_total

        try:
            with _rmw_lock(self.store_path):
                self._rmw(name, dtype, mutate, fresh=(mode == "replace_layer"), metadata=metadata, hypothesis=hypothesis)
                if mode == "replace_layer":
                    self.ops.reset_layer(name)
                self.ops.append_many(applied)
            values = self.get_layer_values(name)
            nonempty = values[np.isfinite(values) & (values != empty_value(dtype))]
            return {
                "success": True,
                "operation": "geometry_batch",
                "group_id": group_id,
                "layer_name": name,
                "records_seen": seen,
                "records_applied": len(applied),
                "records_skipped": seen - len(applied),
                "records_rejected_source_mismatch": skipped_source,
                "affected_voxels": affected_total,
                "nonempty_voxels": int(nonempty.size),
                "geometry_kind_counts": dict(sorted(kind_counts.items())),
                "coordinate_source_counts": dict(sorted(source_counts.items())),
                "value_min": min(values_applied) if values_applied else None,
                "value_max": max(values_applied) if values_applied else None,
                "distinct_nonempty_values": int(np.unique(nonempty).size) if nonempty.size else 0,
                "warnings": warnings,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "success": False, "operation": "geometry_batch", "group_id": group_id, "layer_name": name,
                "records_seen": seen, "records_applied": 0, "records_skipped": seen,
                "records_rejected_source_mismatch": skipped_source, "affected_voxels": 0,
                "warnings": warnings, "error": str(e),
            }

    # --- probing ------------------------------------------------------------------------
    def probe_region(
        self, x: float, y: float, depth_m: float, radius_m: float, layers: list[str] | None = None,
        max_voxels: int = 50,
    ) -> dict[str, Any]:
        """Values of every (or the named) layer within a sphere — a sanity check tool."""
        x, y, depth_m = float(x), float(y), float(depth_m)
        mask = self.sphere_mask(x, y, depth_m, float(radius_m))
        idx = np.argwhere(mask)
        names = layers or self.layer_names
        out: dict[str, Any] = {
            "center": [x, y, depth_m], "center_index": list(self.coord_to_index(x, y, depth_m)),
            "radius_m": float(radius_m), "voxels_in_region": int(mask.sum()), "in_bounds": self.in_bounds(x, y, depth_m),
            "layers": {},
        }
        for name in names:
            if name not in self._layers:
                out["layers"][name] = {"error": "layer not found"}
                continue
            layer = self.get_layer(name)
            vals = layer.values[mask]
            empty = empty_value(layer.dtype)
            ne = np.isfinite(vals) & (vals != empty)
            entry: dict[str, Any] = {
                "dtype": layer.dtype, "nonempty_voxels": int(ne.sum()),
                "value_min": float(vals[ne].min()) if ne.any() else None,
                "value_max": float(vals[ne].max()) if ne.any() else None,
                "value_mean": float(vals[ne].mean()) if (ne.any() and layer.dtype == "float") else None,
            }
            sample = []
            for (ix, iy, iz) in idx[: int(max_voxels)]:
                v = float(layer.values[ix, iy, iz])
                if v == empty:
                    continue
                sample.append({"index": [int(ix), int(iy), int(iz)],
                               "center": [round(c, 3) for c in self.index_to_coord(int(ix), int(iy), int(iz))],
                               "value": v})
            entry["voxels"] = sample
            out["layers"][name] = entry
        return out
