"""Voxel store: a regular 3-D grid with named feature layers persisted as .npy.

Port of combo-geology-nsl ``voxel_features/store.py`` (commit fd3091f) with:
- the per-episode read-only overlay and BIC scoring fields removed,
- generic ``(x, y, depth)`` coordinates and ``GridSpec`` helpers
  (``coord_to_index`` / ``index_to_coord`` / ``units``),
- a ``-1`` empty sentinel for categorical/boolean layers so label 0 is a
  valid class (float layers keep ``0.0`` = empty, the viewer contract).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

LayerDtype = Literal["float", "categorical", "boolean"]
LAYER_DTYPES = ("float", "categorical", "boolean")

# Empty-cell sentinels. Float layers follow the viewer contract (0.0 = empty).
# Categorical/boolean layers use -1 so that class label 0 is representable.
CATEGORICAL_EMPTY = -1.0
FLOAT_EMPTY = 0.0

_LAYER_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


def validate_layer_name(name: str) -> str:
    """Layer names become file names and manifest ids: keep them plain."""
    if not isinstance(name, str) or not _LAYER_NAME_RE.match(name):
        raise ValueError(
            f"Invalid layer name {name!r}: use letters, digits, '_', '.', '-' (max 64 chars)"
        )
    return name


def empty_value(dtype: str) -> float:
    return FLOAT_EMPTY if dtype == "float" else CATEGORICAL_EMPTY


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class GridSpec:
    """Regular axis-aligned grid: ``origin`` (min corner) to ``maximum`` (max corner).

    ``crs`` is informational except for ``units``: an ``EPSG:4326`` grid is in
    degrees (radii are converted with the local-metres approximation); anything
    else is treated as projected metres, where radii map 1:1.
    """

    origin: tuple[float, float, float]
    maximum: tuple[float, float, float]
    shape: tuple[int, int, int]
    crs: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        object.__setattr__(self, "maximum", tuple(float(v) for v in self.maximum))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        if len(self.origin) != 3 or len(self.maximum) != 3 or len(self.shape) != 3:
            raise ValueError("GridSpec origin/maximum/shape must each have 3 entries")
        for axis, (o, m, n) in enumerate(zip(self.origin, self.maximum, self.shape)):
            if n < 1:
                raise ValueError(f"GridSpec shape[{axis}] must be >= 1, got {n}")
            if not (m > o):
                raise ValueError(f"GridSpec maximum[{axis}] ({m}) must exceed origin[{axis}] ({o})")

    # --- geometry --------------------------------------------------------------
    @property
    def units(self) -> str:
        return "degrees" if (self.crs or "").upper().startswith("EPSG:4326") else "meters"

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(m - o for o, m in zip(self.origin, self.maximum))

    @property
    def cell_size(self) -> tuple[float, float, float]:
        return tuple(e / n for e, n in zip(self.extent, self.shape))

    @property
    def n_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx, dy, dz = self.cell_size
        x = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * dx
        y = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * dy
        z = self.origin[2] + (np.arange(self.shape[2]) + 0.5) * dz
        return x, y, z

    def in_bounds(self, x: float, y: float, z: float) -> bool:
        return (
            self.origin[0] <= x <= self.maximum[0]
            and self.origin[1] <= y <= self.maximum[1]
            and self.origin[2] <= z <= self.maximum[2]
        )

    def coord_to_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Cell containing (x, y, z): floor((c - origin) / cell), clamped to the grid."""
        out = []
        for c, o, cell, n in zip((x, y, z), self.origin, self.cell_size, self.shape):
            i = int(math.floor((float(c) - o) / cell))
            out.append(max(0, min(i, n - 1)))
        return out[0], out[1], out[2]

    def index_to_coord(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        """Centre of cell (ix, iy, iz)."""
        dx, dy, dz = self.cell_size
        return (
            self.origin[0] + (ix + 0.5) * dx,
            self.origin[1] + (iy + 0.5) * dy,
            self.origin[2] + (iz + 0.5) * dz,
        )

    def clamp(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return tuple(max(o, min(float(c), m)) for c, o, m in zip((x, y, z), self.origin, self.maximum))

    # --- serialisation ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": list(self.origin),
            "maximum": list(self.maximum),
            "shape": list(self.shape),
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GridSpec":
        return cls(origin=tuple(d["origin"]), maximum=tuple(d["maximum"]), shape=tuple(d["shape"]), crs=d.get("crs"))


@dataclass
class FeatureLayer:
    """One named layer; ``values`` has the grid's shape (float64 on disk)."""

    name: str
    values: np.ndarray
    dtype: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hypothesis: str | None = None
    added_timestamp: str = field(default_factory=_utc_now)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(np.ascontiguousarray(self.values).tobytes())
        h.update(self.dtype.encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.values.shape),
            "metadata": self.metadata,
            "hypothesis": self.hypothesis,
            "added_timestamp": self.added_timestamp,
            "content_hash": self.content_hash,
        }


class VoxelStore:
    """Persistent store: ``index.json`` + ``layers/<name>.npy`` under ``store_path``."""

    INDEX_NAME = "index.json"

    def __init__(self, store_path: Path | str, grid: GridSpec | None = None, *, meta: dict[str, Any] | None = None):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.store_path / self.INDEX_NAME
        self._layers_dir = self.store_path / "layers"
        self._layers_dir.mkdir(exist_ok=True)
        self._layers: dict[str, FeatureLayer] = {}

        if self._index_path.exists():
            self._load_index()
            if meta:
                self._meta.update(meta)
                self._save_index()
        else:
            if grid is None:
                raise ValueError("grid must be provided when creating a new store")
            self._grid = grid
            self._meta: dict[str, Any] = {"created_at": _utc_now(), **(meta or {})}
            self._save_index()

    # --- introspection ---------------------------------------------------------
    @staticmethod
    def exists(store_path: Path | str) -> bool:
        return (Path(store_path) / VoxelStore.INDEX_NAME).is_file()

    @property
    def grid(self) -> GridSpec:
        return self._grid

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def layer_path(self, name: str) -> Path:
        return self._layers_dir / f"{name}.npy"

    # --- index persistence -----------------------------------------------------
    def _load_index(self) -> None:
        with open(self._index_path) as f:
            data = json.load(f)
        self._grid = GridSpec.from_dict(data["grid"])
        self._meta = dict(data.get("meta", {}))
        self._layers = {}
        for name, ld in data.get("layers", {}).items():
            self._layers[name] = FeatureLayer(
                name=name,
                values=np.array([]),  # loaded lazily
                dtype=ld["dtype"],
                metadata=ld.get("metadata", {}),
                hypothesis=ld.get("hypothesis"),
                added_timestamp=ld.get("added_timestamp", ""),
            )
            # keep the persisted hash so listing never has to load the array
            self._layers[name].metadata.setdefault("_content_hash", ld.get("content_hash"))

    def _save_index(self) -> None:
        """Atomic write (per-writer tmp + os.replace) so readers never see a torn index."""
        layers = {}
        for name, layer in self._layers.items():
            d = layer.to_dict() if layer.values.size else {
                "name": name,
                "dtype": layer.dtype,
                "shape": list(self._grid.shape),
                "metadata": {k: v for k, v in layer.metadata.items() if k != "_content_hash"},
                "hypothesis": layer.hypothesis,
                "added_timestamp": layer.added_timestamp,
                "content_hash": layer.metadata.get("_content_hash"),
            }
            d["metadata"] = {k: v for k, v in d["metadata"].items() if k != "_content_hash"}
            layers[name] = d
        data = {"grid": self._grid.to_dict(), "meta": self._meta, "layers": layers}
        tmp = self._index_path.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path)

    # --- layer CRUD ------------------------------------------------------------
    @staticmethod
    def _check_dtype(dtype: str) -> str:
        if dtype not in LAYER_DTYPES:
            raise ValueError(f"dtype must be one of {LAYER_DTYPES}, got {dtype!r}")
        return dtype

    def new_layer_values(self, dtype: str) -> np.ndarray:
        """A fresh all-empty array for ``dtype`` (float64, grid shape)."""
        return np.full(self._grid.shape, empty_value(self._check_dtype(dtype)), dtype=float)

    def _coerce(self, values: np.ndarray, dtype: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != self._grid.shape:
            raise ValueError(f"Layer shape {tuple(arr.shape)} does not match grid shape {tuple(self._grid.shape)}")
        if dtype in ("categorical", "boolean"):
            finite = np.isfinite(arr)
            if not np.all(np.equal(np.mod(arr[finite], 1), 0)):
                raise ValueError(f"{dtype} layer values must be integers (empty = {int(CATEGORICAL_EMPTY)})")
        return np.ascontiguousarray(arr)

    def add_layer(
        self,
        name: str,
        values: np.ndarray,
        dtype: str,
        metadata: dict[str, Any] | None = None,
        hypothesis: str | None = None,
    ) -> FeatureLayer:
        """Add a new layer; fails if the name exists (use ``put_layer`` to replace)."""
        validate_layer_name(name)
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already exists")
        return self.put_layer(name, values, dtype, metadata=metadata, hypothesis=hypothesis)

    def put_layer(
        self,
        name: str,
        values: np.ndarray,
        dtype: str,
        metadata: dict[str, Any] | None = None,
        hypothesis: str | None = None,
    ) -> FeatureLayer:
        """Create or replace a layer (array written before the index is updated)."""
        validate_layer_name(name)
        self._check_dtype(dtype)
        arr = self._coerce(values, dtype)
        layer = FeatureLayer(name=name, values=arr, dtype=dtype, metadata=dict(metadata or {}), hypothesis=hypothesis)
        np.save(self.layer_path(name), arr)
        self._layers[name] = layer
        self._save_index()
        return layer

    def get_layer(self, name: str) -> FeatureLayer:
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        layer = self._layers[name]
        if layer.values.size == 0:
            layer.values = np.load(self.layer_path(name))
        return layer

    def get_layer_values(self, name: str) -> np.ndarray:
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        return np.load(self.layer_path(name))

    def remove_layer(self, name: str) -> None:
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        del self._layers[name]
        p = self.layer_path(name)
        if p.exists():
            p.unlink()
        self._save_index()

    def layer_summary(self, name: str) -> dict[str, Any]:
        """Index entry plus non-empty voxel count and value range (loads the array)."""
        layer = self.get_layer(name)
        d = layer.to_dict()
        d["metadata"] = {k: v for k, v in d["metadata"].items() if k != "_content_hash"}
        vals = layer.values
        mask = np.isfinite(vals) & (vals != empty_value(layer.dtype))
        d["nonempty_voxels"] = int(mask.sum())
        d["value_min"] = float(vals[mask].min()) if mask.any() else None
        d["value_max"] = float(vals[mask].max()) if mask.any() else None
        return d

    def list_layers(self) -> list[dict[str, Any]]:
        return [self.layer_summary(n) for n in self._layers]
