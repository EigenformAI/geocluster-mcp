"""Voxel tools (Section L): a per-project 3-D voxel store the voxel specialist
fills from validated analysis results, plus the exporter that writes the
viewer bundle (viz/).

Every tool is deterministic numpy/pandas — the LLM decides WHAT to stamp, the
tools decide WHERE it lands (floor((coord - origin) / cell), clamped) and log
it. All names start with ``voxel_`` and avoid substrings other specialists
glob on (inspect, query, log, viz, stat, cluster, map, ...): the name is the ACL.

Store location: <WORKSPACE_ROOT>/voxel_store/{index.json, layers/*.npy, operations.jsonl}
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import tools.config as _cfg
from tools.config import read_tabular, resolve_path

STORE_DIRNAME = "voxel_store"
_INIT_HINT = "call voxel_init_grid(dataset_path=<dataset with EASTING/NORTHING or lon/lat>) first"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _workspace_root() -> str:
    return _cfg.WORKSPACE_ROOT


def _store_dir() -> str:
    return os.path.join(_workspace_root(), STORE_DIRNAME)


def _open_store():
    from voxel.spatial import SpatialVoxelStore
    from voxel.store import VoxelStore

    path = _store_dir()
    if not VoxelStore.exists(path):
        raise ValueError(f"no voxel grid in this workspace — {_INIT_HINT}")
    return SpatialVoxelStore(path)


def _error(tool: str, exc: Exception, next_step: str) -> dict[str, Any]:
    return {"success": False, "tool": tool, "error": str(exc), "next_step": next_step}


def _grid_dict(store) -> dict[str, Any]:
    g = store.grid
    return {
        "origin": list(g.origin),
        "maximum": list(g.maximum),
        "shape": list(g.shape),
        "cell_size": [round(c, 4) for c in g.cell_size],
        "n_voxels": g.n_voxels,
        "crs": g.crs,
        "units": g.units,
        "index_formula": "ix = floor((x - origin[0]) / cell_size[0]); iy, iz likewise; clamped to shape-1; "
                         "array order is [ix, iy, iz] (C-order, z fastest)",
    }


def _relative_to_workspace(path: str) -> str:
    try:
        return os.path.relpath(path, _workspace_root())
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def voxel_init_grid(
    dataset_path: str,
    shape: list[int] | None = None,
    cell_size_xy_m: float | None = None,
    cell_size_z_m: float | None = None,
    padding_m: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the project voxel grid from a dataset's coordinate/depth bounds (projected metres). Default 64x64x16; set shape or cell sizes to change it."""
    try:
        from voxel.grid_from_dataset import derive_grid
        from voxel.spatial import SpatialVoxelStore
        from voxel.store import VoxelStore

        store_dir = _store_dir()
        if VoxelStore.exists(store_dir):
            existing = SpatialVoxelStore(store_dir)
            if existing.layer_names and not overwrite:
                return _error(
                    "voxel_init_grid",
                    ValueError(f"a grid with layers {existing.layer_names} already exists"),
                    "reuse it via voxel_get_grid, or pass overwrite=True to discard all layers",
                )
            import shutil

            shutil.rmtree(store_dir)

        resolved = resolve_path(dataset_path)
        grid, info = derive_grid(resolved, shape=shape, cell_size_xy_m=cell_size_xy_m,
                                 cell_size_z_m=cell_size_z_m, padding_m=padding_m)
        info["dataset_path"] = _relative_to_workspace(resolved)
        store = SpatialVoxelStore(store_dir, grid, meta={"dataset": info, "coordinate_conversion": info["coordinate_conversion"]})
        return {
            "success": True,
            "tool": "voxel_init_grid",
            "store_dir": _relative_to_workspace(store_dir),
            "grid": _grid_dict(store),
            "dataset": info,
            "depth_degenerate": info["depth_degenerate"],
            "note": "coordinates for all voxel_* tools are in this grid's units (projected metres unless crs is EPSG:4326)",
        }
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_init_grid", exc,
                      "pass a dataset CSV with EASTING/NORTHING (or LONGITUDE/LATITUDE) and optionally depth_m / depth_from_m / depth_to_m")


def voxel_get_grid() -> dict[str, Any]:
    """Read the project voxel grid (bounds, cell size, units, index formula) and its layer summaries. Call this first."""
    try:
        store = _open_store()
        meta = dict(store.meta)
        return {
            "success": True,
            "tool": "voxel_get_grid",
            "store_dir": _relative_to_workspace(str(store.store_path)),
            "grid": _grid_dict(store),
            "dataset": meta.get("dataset"),
            "coordinate_conversion": meta.get("coordinate_conversion"),
            "depth_degenerate": bool((meta.get("dataset") or {}).get("depth_degenerate", False)),
            "layers": store.list_layers(),
        }
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_get_grid", exc, _INIT_HINT)


# ---------------------------------------------------------------------------
# single geometry features
# ---------------------------------------------------------------------------
def voxel_add_point(
    layer: str,
    x: float,
    y: float,
    depth_m: float,
    value: float,
    radius_m: float = 50.0,
    dtype: str = "float",
    combination_rule: str = "max",
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: str = "artifact",
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Stamp a value into every voxel within radius_m of a point (x, y, depth_m in grid units). combination_rule: replace|max|add|mean."""
    try:
        store = _open_store()
        result = store.add_point_feature(layer, x, y, depth_m, value, radius_m=radius_m, dtype=dtype,
                                         combination_rule=combination_rule, source_file=source_file,
                                         source_excerpt=source_excerpt, coordinate_source=coordinate_source,
                                         hypothesis=hypothesis)
        result["tool"] = "voxel_add_point"
        if not result["success"]:
            result["next_step"] = "check the point is inside voxel_get_grid bounds and coordinate_source='artifact'"
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_add_point", exc, _INIT_HINT)


def voxel_add_line(
    layer: str,
    start_x: float,
    start_y: float,
    start_depth_m: float,
    end_x: float,
    end_y: float,
    end_depth_m: float,
    value: float,
    width_m: float = 25.0,
    dtype: str = "float",
    combination_rule: str = "max",
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: str = "artifact",
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Stamp a value along a line segment (fault, vein, drill trace) of width_m between two (x, y, depth_m) points."""
    try:
        store = _open_store()
        result = store.add_line_feature(layer, (start_x, start_y, start_depth_m), (end_x, end_y, end_depth_m),
                                        value, width_m=width_m, dtype=dtype, combination_rule=combination_rule,
                                        source_file=source_file, source_excerpt=source_excerpt,
                                        coordinate_source=coordinate_source, hypothesis=hypothesis)
        result["tool"] = "voxel_add_line"
        if not result["success"]:
            result["next_step"] = "both endpoints must be inside voxel_get_grid bounds; coordinate_source must be 'artifact'"
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_add_line", exc, _INIT_HINT)


def voxel_add_box(
    layer: str,
    x_min: float,
    y_min: float,
    depth_min_m: float,
    x_max: float,
    y_max: float,
    depth_max_m: float,
    value: float,
    dtype: str = "float",
    combination_rule: str = "max",
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: str = "artifact",
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Stamp a value into an axis-aligned box (x/y/depth extents in grid units); partial overlap is clipped to the grid."""
    try:
        store = _open_store()
        result = store.add_box_feature(layer, x_min, y_min, depth_min_m, x_max, y_max, depth_max_m, value,
                                       dtype=dtype, combination_rule=combination_rule, source_file=source_file,
                                       source_excerpt=source_excerpt, coordinate_source=coordinate_source,
                                       hypothesis=hypothesis)
        result["tool"] = "voxel_add_box"
        if not result["success"]:
            result["next_step"] = "the box must overlap voxel_get_grid bounds; coordinate_source must be 'artifact'"
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_add_box", exc, _INIT_HINT)


# ---------------------------------------------------------------------------
# batch geometry
# ---------------------------------------------------------------------------
def _load_records(records_path: str, value_col: str, source_file: str | None) -> tuple[list[dict], str]:
    import math

    resolved = resolve_path(records_path)
    df = read_tabular(resolved)
    if value_col != "value" and value_col in df.columns:
        df = df.assign(value=df[value_col])
    records = []
    for row in df.to_dict("records"):
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        if source_file and not clean.get("source_file"):
            clean["source_file"] = source_file
        records.append(clean)
    return records, resolved


def _apply_label_map(records: list[dict], dtype: str, label_map: dict | None) -> tuple[dict[str, str], list[str]]:
    """For categorical layers map string labels to integer codes. Returns (code->label, warnings)."""
    warnings: list[str] = []
    if dtype not in ("categorical", "boolean"):
        return {}, warnings
    provided = {str(k): int(v) for k, v in (label_map or {}).items()}
    auto: dict[str, int] = {}
    for rec in records:
        v = rec.get("value")
        if v is None:
            continue
        try:
            float(v)
            continue  # already numeric
        except (TypeError, ValueError):
            pass
        key = str(v)
        if key in provided:
            rec["value"] = provided[key]
        else:
            if key not in auto:
                auto[key] = len(provided) + len(auto)
            rec["value"] = auto[key]
    mapping = {**provided, **auto}
    if auto:
        warnings.append(f"auto-assigned codes for labels {sorted(auto)}; pass label_map to control them")
    return {str(code): label for label, code in mapping.items()}, warnings


def voxel_upsert_geometry(
    layer: str,
    records_path: str,
    mode: str = "replace_layer",
    dtype: str = "float",
    combination_rule: str = "max",
    bounds_policy: str = "skip",
    max_records: int = 5000,
    value_col: str = "value",
    label_map: dict[str, int] | None = None,
    hypothesis: str | None = None,
    source_file: str | None = None,
    source_check: str = "reject",
    source_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Materialise a feature_geometry CSV (one row per point/line/box with value + coordinates) into one layer. Records are cross-checked against their source_file (within 1 m by default); mismatches are rejected."""
    try:
        from voxel.source_check import coordinate_mismatches

        store = _open_store()
        if source_check not in ("reject", "warn", "off"):
            raise ValueError("source_check must be reject|warn|off")
        records, resolved_records = _load_records(records_path, value_col, source_file)
        if not records:
            raise ValueError(f"{records_path} has no rows")
        code_labels, warnings = _apply_label_map(records, dtype, label_map)

        records_dir = os.path.dirname(resolved_records)

        def load_table(p: str):
            try:
                return read_tabular(resolve_path(p))
            except ValueError:
                return read_tabular(resolve_path(os.path.join(records_dir, p)))

        mismatches: dict[str, str] = {}
        if source_check != "off":
            mismatches = coordinate_mismatches(store, records, load_table, tolerance_xy=source_tolerance_m)
        metadata = {"records_path": _relative_to_workspace(resolved_records)}
        if code_labels:
            metadata["label_map"] = code_labels
        result = store.add_geometry_batch(
            layer, records, mode=mode, dtype=dtype, combination_rule=combination_rule, max_records=max_records,
            bounds_policy=bounds_policy, metadata=metadata, hypothesis=hypothesis,
            rejected=mismatches if source_check == "reject" else None,
        )
        result["tool"] = "voxel_upsert_geometry"
        result.setdefault("warnings", []).extend(warnings)
        if source_check == "warn" and mismatches:
            result["warnings"].extend(f"source mismatch (not rejected) {rid}: {why}" for rid, why in mismatches.items())
        result["source_check"] = source_check
        result["source_mismatches"] = mismatches
        if code_labels:
            result["label_map"] = code_labels
        if result.get("success") and result.get("records_applied", 0) == 0:
            result["next_step"] = ("no record landed in the grid: check coordinates are in grid units "
                                   "(voxel_get_grid), coordinate_source='artifact', and source_file paths")
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_upsert_geometry", exc,
                      "records_path must be a CSV with geometry_kind, value, coordinate columns "
                      "(x,y,depth_m | start_x.. | x_min..) and coordinate_source='artifact'")


# ---------------------------------------------------------------------------
# full array
# ---------------------------------------------------------------------------
def voxel_set_layer_array(
    layer: str,
    array_path: str,
    dtype: str = "float",
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: str = "artifact",
    hypothesis: str | None = None,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deposit a precomputed per-voxel array (.npy/.npz, shape == grid shape, index order [ix, iy, iz]) verbatim as a layer. Use for interpolated/continuous fields."""
    try:
        import numpy as np

        store = _open_store()
        resolved = resolve_path(array_path)
        loaded = np.load(resolved, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = "value_grid" if "value_grid" in loaded.files else loaded.files[0]
            arr = loaded[key]
        else:
            arr = loaded
        with open(resolved, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        metadata = {"array_path": _relative_to_workspace(resolved), "array_sha256": digest}
        if label_map:
            metadata["label_map"] = {str(k): str(v) for k, v in label_map.items()}
        result = store.set_layer_array(layer, arr, dtype=dtype, source_file=source_file, source_excerpt=source_excerpt,
                                       coordinate_source=coordinate_source, hypothesis=hypothesis, metadata=metadata)
        result["tool"] = "voxel_set_layer_array"
        result["array_sha256"] = digest
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_set_layer_array", exc,
                      "the array must have exactly the grid shape from voxel_get_grid; empty cells are 0.0 (float) or -1 (categorical)")


# ---------------------------------------------------------------------------
# read-only
# ---------------------------------------------------------------------------
def voxel_probe_region(
    x: float,
    y: float,
    depth_m: float,
    radius_m: float,
    layers: list[str] | None = None,
    max_voxels: int = 50,
) -> dict[str, Any]:
    """Read layer values within radius_m of a point — sanity-check a transcription at a known location."""
    try:
        store = _open_store()
        out = store.probe_region(x, y, depth_m, radius_m, layers=layers, max_voxels=max_voxels)
        out["success"] = True
        out["tool"] = "voxel_probe_region"
        return out
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_probe_region", exc, _INIT_HINT)


def voxel_list_layers() -> dict[str, Any]:
    """List the voxel store's layers with dtype, non-empty voxel counts and value ranges."""
    try:
        store = _open_store()
        return {"success": True, "tool": "voxel_list_layers", "grid": _grid_dict(store), "layers": store.list_layers()}
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_list_layers", exc, _INIT_HINT)


def voxel_history(layer: str | None = None, limit: int = 200) -> dict[str, Any]:
    """Provenance of every stamp (newest first): operation kind, coordinates, source_file/excerpt, record_id, affected voxels."""
    try:
        store = _open_store()
        rows = store.ops.read(layer=layer, limit=limit)
        out: dict[str, Any] = {"success": True, "tool": "voxel_history", "count": len(rows), "operations": rows}
        if layer:
            out["summary"] = store.ops.summary(layer)
        return out
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_history", exc, _INIT_HINT)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def voxel_export_bundle(
    layers: list[str] | None = None,
    finding: str | None = None,
    hypothesis: str | None = None,
    include_samples: bool = True,
    assignments_path: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Export voxel-store layers (and, when a clustering run exists, per-sample voxels) as the viewer bundle under viz/. MANDATORY final step of a transcription."""
    try:
        import viz_bundle

        result = viz_bundle.publish(
            _workspace_root(),
            None,
            resolve_path(assignments_path) if assignments_path else None,
            resolve_path(dataset_path) if dataset_path else None,
            include_voxel_store=True,
            layers=layers,
            finding=finding,
            hypothesis=hypothesis,
            include_samples=include_samples,
        )
        result["success"] = True
        result["tool"] = "voxel_export_bundle"
        result["manifest_path"] = _relative_to_workspace(result["manifest_path"])
        result["viz_dir"] = _relative_to_workspace(result["viz_dir"])
        result["viewer"] = "Open 3D Visualization in the dashboard (Layers mode) to explore: " + ", ".join(result["layer_ids"])
        return result
    except Exception as exc:  # noqa: BLE001
        return _error("voxel_export_bundle", exc,
                      "add at least one layer (voxel_upsert_geometry / voxel_set_layer_array) before exporting")
