"""Viewer bundle writer (manifest schema v1) shared by the publisher CLI and the
``voxel_export_bundle`` MCP tool.

Writes under ``<workspace>/viz/``:
  samples.json            one voxel per sample interval (when a clustering run exists)
  layers/*.f32 and *.u8   gridded layers (publisher bins + voxel-store layers)
  manifest.json           written LAST, atomically — the viewer's commit point

Grid rule: if ``<workspace>/voxel_store/index.json`` exists its grid is
authoritative — publisher layers are binned onto it and every store layer is
appended — so the CLI and the tool never clobber each other. Without a store,
the legacy behaviour (grid from the samples' bounds) is unchanged.

Module-level imports: numpy/pandas only. Runs under the system Python (3.11,
pandas 2.x) as ``/usr/local/bin/publish_viz.py`` and under the MCP venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voxel.columns import (
    HOLE_PATTERNS,
    LITHOLOGY_PATTERNS,
    coordinate_columns,
    find_column,
    to_local_meters,
)

# Same fixed-order palette as the viewer (contract data, duplicated by design).
CATEGORICAL_PALETTE = [
    "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
    "#008300", "#9085e9", "#e66767", "#5aa9a2", "#a8874f",
]
EMPTY_U8 = 255
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def discover_assignments(workspace: Path) -> Path | None:
    candidates = [
        p for p in workspace.rglob("cluster_assignments*.csv")
        if "viz" not in p.parts and "voxel_store" not in p.parts
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def discover_dataset(workspace: Path, near: Path | None) -> Path | None:
    """Prefer dataset_fixed.csv next to the assignments, else the largest CSV with coordinates."""
    if near:
        fixed = near.parent / "dataset_fixed.csv"
        if fixed.is_file():
            return fixed
    for csv in sorted(workspace.rglob("*.csv"), key=lambda p: -p.stat().st_size):
        if "viz" in csv.parts or "voxel_store" in csv.parts or csv.name.startswith("cluster_assignments"):
            continue
        try:
            head = pd.read_csv(csv, nrows=1)
        except Exception:
            continue
        cols = list(head.columns)
        if "filename" in cols and coordinate_columns(cols):
            return csv
    return None


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------
def build_samples(assignments_csv: Path, dataset_csv: Path) -> dict:
    """Join cluster labels with spatial columns -> columnar samples payload."""
    assignments = pd.read_csv(assignments_csv)
    if "filename" not in assignments.columns or "cluster_label" not in assignments.columns:
        raise ValueError(f"{assignments_csv} lacks filename/cluster_label columns")

    spatial = pd.read_csv(dataset_csv, low_memory=False).drop_duplicates(subset="filename", keep="first")
    df = assignments.merge(spatial, on="filename", how="inner")
    if df.empty:
        raise ValueError("no rows joined between assignments and dataset on `filename`")

    feature_csv = None
    m = re.search(r"_k(\d+)\.csv$", assignments_csv.name)
    k = int(m.group(1)) if m else None
    if k:
        cand = assignments_csv.parent / f"feature_matrix_k{k}.csv"
        if cand.is_file():
            feature_csv = cand
    element_cols: list[str] = []
    if feature_csv is not None:
        features = pd.read_csv(feature_csv)
        df = df.merge(features, on="filename", how="left", suffixes=("", "_feat"))
        element_cols = [c for c in features.columns if c != "filename" and c in df.columns]

    cols = list(df.columns)
    cc = coordinate_columns(cols)
    if cc is None:
        raise ValueError("no coordinate columns found in dataset")
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    x_raw, y_raw = num(df[cc["x"]]), num(df[cc["y"]])
    valid = x_raw.notna() & y_raw.notna()
    df, x_raw, y_raw = df[valid], x_raw[valid], y_raw[valid]
    x_m, y_m, crs, conversion = to_local_meters(x_raw.to_numpy(), y_raw.to_numpy(), cc["kind"])
    x = pd.Series(x_m, index=df.index)
    y = pd.Series(y_m, index=df.index)

    depth = num(df["depth_m"]).fillna(0.0) if "depth_m" in df.columns else pd.Series(0.0, index=df.index)
    d_from = (num(df["depth_from_m"]) if "depth_from_m" in df.columns else depth - 0.5).fillna(depth - 0.5)
    d_to = (num(df["depth_to_m"]) if "depth_to_m" in df.columns else depth + 0.5).fillna(depth + 0.5)

    hole_col = find_column(cols, HOLE_PATTERNS)
    lith_col = find_column(cols, LITHOLOGY_PATTERNS)
    origin = [round(float(x.mean()), 2), round(float(y.mean()), 2)]
    x_rel, y_rel = x - origin[0], y - origin[1]

    rnd = lambda s, nd=4: [None if pd.isna(v) else round(float(v), nd) for v in s]  # noqa: E731
    signatures = {}
    if "cluster_signature" in df.columns:
        for label, sig in df[["cluster_label", "cluster_signature"]].drop_duplicates("cluster_label").values:
            signatures[str(int(label))] = str(sig)

    return {
        "n_samples": int(len(df)),
        "k": k,
        "coordinate_space": {"crs": crs, "origin": origin, "units": "meters", "conversion": conversion},
        "bounds": {
            "x": [round(float(x_rel.min()), 2), round(float(x_rel.max()), 2)],
            "y": [round(float(y_rel.min()), 2), round(float(y_rel.max()), 2)],
            "depth": [round(float(d_from.min()), 2), round(float(d_to.max()), 2)],
        },
        "cluster_signatures": signatures,
        "elements": element_cols,
        "x": rnd(x_rel, 2),
        "y": rnd(y_rel, 2),
        "depth": rnd(depth, 3),
        "depth_from": rnd(d_from, 3),
        "depth_to": rnd(d_to, 3),
        "cluster": [int(v) for v in df["cluster_label"]],
        "hole": [None if pd.isna(v) else str(v) for v in df[hole_col]] if hole_col else None,
        "lithology": [None if pd.isna(v) else str(v) for v in df[lith_col]] if lith_col else None,
        "grades": {c: rnd(num(df[c])) for c in element_cols},
    }


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def normalize_log(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Log-scale positive values -> [0.05, 1.0] float32; 0.0 stays the empty sentinel."""
    out = np.zeros(arr.shape, dtype="<f4")
    nz = np.isfinite(arr) & (arr > 0)
    if not nz.any():
        return out, 0.0, 0.0
    vals = arr[nz].astype("f8")
    raw_min, raw_max = float(vals.min()), float(vals.max())
    lo, hi = np.log10(raw_min), np.log10(raw_max)
    if hi > lo:
        out[nz] = (0.05 + 0.95 * (np.log10(vals) - lo) / (hi - lo)).astype("<f4")
    else:
        out[nz] = np.float32(1.0)
    return out, raw_min, raw_max


def _hash(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def encode_continuous(layer_id: str, source: str, cell_values: np.ndarray, provenance: dict | None = None) -> tuple[dict, bytes]:
    """float64 cells (0 = empty) -> (layer def, float32 blob) with log10 min-max normalisation."""
    norm, raw_min, raw_max = normalize_log(np.asarray(cell_values, dtype="f8"))
    blob = np.ascontiguousarray(norm).tobytes()
    spec = {
        "id": layer_id,
        "kind": "continuous",
        "source": source,
        "dtype": "float32",
        "encoding": {"empty_sentinel": 0.0, "normalization": "log10_minmax", "raw_min": raw_min, "raw_max": raw_max},
        "path": f"layers/{layer_id}.f32",
        "bytes": len(blob),
        "content_hash": _hash(blob),
    }
    if provenance:
        spec["provenance"] = provenance
    return spec, blob


def encode_categorical(
    layer_id: str, source: str, codes: np.ndarray, categories: list[dict], provenance: dict | None = None
) -> tuple[dict, bytes]:
    """uint8 codes (255 = empty) -> (layer def, blob)."""
    codes = np.ascontiguousarray(np.asarray(codes, dtype="u1"))
    blob = codes.tobytes()
    spec = {
        "id": layer_id,
        "kind": "categorical",
        "source": source,
        "dtype": "uint8",
        "encoding": {"empty_sentinel": EMPTY_U8, "normalization": "none"},
        "path": f"layers/{layer_id}.u8",
        "bytes": len(blob),
        "content_hash": _hash(blob),
        "categories": categories,
    }
    if provenance:
        spec["provenance"] = provenance
    return spec, blob


def _palette(i: int) -> str:
    return CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]


# ---------------------------------------------------------------------------
# publisher bins
# ---------------------------------------------------------------------------
def _grid_spec_dict(origin, spacing, shape, crs, z0, z1) -> dict:
    return {
        "crs": crs,
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "spacing": [round(float(spacing[0]), 4), round(float(spacing[1]), 4), round(float(spacing[2]), 4)],
        "shape": [int(shape[0]), int(shape[1]), int(shape[2])],
        "axis_order": "x,y,z; z fastest (C-order)",
        "depth": {"min": round(float(z0), 2), "max": round(float(z1), 2), "units": "m", "positive": "down"},
    }


def build_grid_layers(samples: dict, shape: tuple[int, int, int], grid=None, provenance: dict | None = None) -> tuple[dict, dict]:
    """Bin samples into a regular grid — pure counting/averaging, NO interpolation.

    ``grid`` (a ``voxel.store.GridSpec``) makes that grid authoritative; otherwise
    the grid is the samples' bounding box at ``shape``. Returns
    ``(grid_spec_dict, {layer_id: (layer_def, blob)})``.
    """
    ox, oy = samples["coordinate_space"]["origin"]
    x = np.asarray(samples["x"], dtype="f8") + ox
    y = np.asarray(samples["y"], dtype="f8") + oy
    z = np.asarray([(f + t) / 2 for f, t in zip(samples["depth_from"], samples["depth_to"])], dtype="f8")

    if grid is not None:
        nx, ny, nz = grid.shape
        dx, dy, dz = grid.cell_size
        x0, y0, z0 = grid.origin
        z1 = grid.maximum[2]
        crs = grid.crs or samples["coordinate_space"]["crs"]
    else:
        nx, ny, nz = shape
        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        z0, z1 = float(min(samples["depth_from"])), float(max(samples["depth_to"]))
        dx = (x1 - x0) / nx or 1.0
        dy = (y1 - y0) / ny or 1.0
        dz = (z1 - z0) / nz or 1.0
        crs = samples["coordinate_space"]["crs"] + "; local meters from origin"

    ix = np.clip(np.floor((x - x0) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((y - y0) / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.floor((z - z0) / dz).astype(int), 0, nz - 1)
    flat = (ix * ny + iy) * nz + iz  # C-order, z fastest
    ncells = nx * ny * nz
    grid_spec = _grid_spec_dict((x0, y0, z0), (dx, dy, dz), (nx, ny, nz), crs, z0, z1)

    layers: dict[str, tuple[dict, bytes]] = {}
    counts = np.bincount(flat, minlength=ncells).astype("f8")
    layers["sample_density"] = encode_continuous("sample_density", "density", counts, provenance)

    for el in samples["elements"]:
        vals = np.asarray([v if v is not None else np.nan for v in samples["grades"][el]], dtype="f8")
        ok = np.isfinite(vals) & (vals > 0)
        if not ok.any():
            continue
        sums = np.bincount(flat[ok], weights=vals[ok], minlength=ncells)
        n = np.bincount(flat[ok], minlength=ncells)
        means = np.divide(sums, n, out=np.zeros(ncells), where=n > 0)
        lid = f"grade_{el.lower()}"
        layers[lid] = encode_continuous(lid, f"grade:{el}", means, provenance)

    labels = np.asarray(samples["cluster"], dtype=int)
    uniq = sorted(set(labels.tolist()))
    per_label = np.zeros((len(uniq), ncells), dtype="i4")
    for row, lab in enumerate(uniq):
        per_label[row] = np.bincount(flat[labels == lab], minlength=ncells)
    total = per_label.sum(axis=0)
    mode = np.full(ncells, EMPTY_U8, dtype="u1")
    occupied = total > 0
    mode[occupied] = np.asarray(uniq, dtype="u1")[per_label[:, occupied].argmax(axis=0)]
    k = samples.get("k")
    lid = f"clusters_k{k}" if k else "clusters"
    layers[lid] = encode_categorical(
        lid, "clusters", mode,
        [{"value": int(lab), "label": f"Cluster {lab}", "color": _palette(i)} for i, lab in enumerate(uniq)],
        provenance,
    )
    _assert_bytes(layers, ncells)
    return grid_spec, layers


def _assert_bytes(layers: dict, ncells: int) -> None:
    for layer_id, (spec, blob) in layers.items():
        expected = ncells * (1 if spec["dtype"] == "uint8" else 4)
        if len(blob) != expected:
            raise ValueError(f"{layer_id}: {len(blob)} bytes != {expected}")


# ---------------------------------------------------------------------------
# voxel store layers
# ---------------------------------------------------------------------------
def voxel_layers_to_bundle(store, layer_names: list[str] | None = None, hypothesis: str | None = None,
                           finding: str | None = None) -> tuple[dict, dict, list[str]]:
    """Encode voxel-store layers for the viewer. Returns (grid_spec, layers, warnings)."""
    from voxel.store import CATEGORICAL_EMPTY

    grid = store.grid
    names = layer_names or store.layer_names
    warnings: list[str] = []
    layers: dict[str, tuple[dict, bytes]] = {}
    grid_spec = _grid_spec_dict(grid.origin, grid.cell_size, grid.shape, grid.crs, grid.origin[2], grid.maximum[2])

    for name in names:
        if name not in store.layer_names:
            warnings.append(f"layer {name!r} not in voxel store — skipped")
            continue
        layer = store.get_layer(name)
        vals = np.asarray(layer.values, dtype="f8")
        ops = store.ops.summary(name)
        prov = {
            "generator": "voxel_store",
            "store_layer": name,
            "store_dtype": layer.dtype,
            "hypothesis": layer.hypothesis or hypothesis,
            "finding": finding,
            "source_files": ops["source_files"],
            "coordinate_source_counts": ops["coordinate_source_counts"],
            "operations": {"count": ops["operations"], "kinds": ops["kinds"]},
            "npy_content_hash": layer.content_hash,
            "created_at": layer.added_timestamp,
        }
        prov.update({k: v for k, v in layer.metadata.items() if not str(k).startswith("_") and k != "label_map"})
        if layer.dtype == "float":
            finite = np.isfinite(vals)
            dropped = int((finite & (vals < 0)).sum())
            if dropped:
                warnings.append(f"layer {name!r}: {dropped} negative voxels are exported as empty (viewer contract: value > 0)")
                prov["dropped_nonpositive_voxels"] = dropped
            spec, blob = encode_continuous(name, f"voxel:{name}", vals, prov)
        else:
            nonempty = np.isfinite(vals) & (vals != CATEGORICAL_EMPTY)
            ints = np.rint(vals[nonempty]).astype(int)
            if ints.size and (ints.min() < 0 or ints.max() > 254):
                raise ValueError(f"layer {name!r}: categorical values must be in 0..254 (got {ints.min()}..{ints.max()})")
            codes = np.full(vals.shape, EMPTY_U8, dtype="u1")
            codes[nonempty] = ints.astype("u1")
            uniq = sorted(set(ints.tolist()))
            label_map = layer.metadata.get("label_map") or {}
            cats = [
                {"value": int(v), "label": str(label_map.get(str(v), label_map.get(v, f"{name} {v}"))), "color": _palette(i)}
                for i, v in enumerate(uniq)
            ]
            spec, blob = encode_categorical(name, f"voxel:{name}", codes, cats, prov)
        layers[name] = (spec, blob)
    _assert_bytes(layers, grid.n_voxels)
    return grid_spec, layers, warnings


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_atomic_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; match the layer blobs so any reader can open it
    os.replace(tmp, path)


def write_bundle(viz_dir: Path, samples: dict | None, grid_spec: dict | None, layers: dict,
                 project_id: str, provenance: dict | None = None, samples_provenance: dict | None = None,
                 log=None) -> dict:
    """Write layers, then samples, then the manifest (last, atomically). Returns the manifest."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    if layers:
        (viz_dir / "layers").mkdir(exist_ok=True)
        defs = []
        for layer_id, (spec, blob) in layers.items():
            (viz_dir / spec["path"]).write_bytes(blob)
            defs.append(spec)
            if log:
                log(f"wrote viz/{spec['path']} ({spec['bytes']} bytes)")
    if samples:
        write_atomic_json(viz_dir / "samples.json", samples)
        if log:
            log(f"wrote viz/samples.json ({samples['n_samples']} samples, {len(samples['elements'])} elements)")
        artifacts.append({
            "kind": "samples", "path": "samples.json", "n_samples": samples["n_samples"],
            "provenance": samples_provenance or {"k": samples.get("k"), "created_at": _now()},
        })
    if layers:
        artifacts.append({"kind": "grid", "grid": grid_spec, "layers": defs})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "generated_at": _now(),
        "provenance": provenance or {},
        "artifacts": artifacts,
    }
    write_atomic_json(viz_dir / "manifest.json", manifest)  # commit point
    if log:
        log("wrote viz/manifest.json — publish complete (uploaded to R2 by the viz uploader within seconds)")
    return manifest


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------
def publish(
    workspace: str | Path,
    grid_shape: tuple[int, int, int] | None = None,
    assignments: str | Path | None = None,
    dataset: str | Path | None = None,
    *,
    include_voxel_store: bool = True,
    layers: list[str] | None = None,
    finding: str | None = None,
    hypothesis: str | None = None,
    include_samples: bool = True,
    log=None,
) -> dict:
    """Build and write the bundle for ``workspace``. Raises ValueError with a clear message."""
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace {workspace} does not exist")
    viz_dir = workspace / "viz"
    warnings: list[str] = []

    store = None
    if include_voxel_store:
        from voxel.store import VoxelStore

        if VoxelStore.exists(workspace / "voxel_store"):
            from voxel.spatial import SpatialVoxelStore

            store = SpatialVoxelStore(workspace / "voxel_store")

    samples = None
    samples_prov = None
    assignments_path = Path(assignments) if assignments else discover_assignments(workspace)
    if include_samples and assignments_path and assignments_path.is_file():
        dataset_path = Path(dataset) if dataset else discover_dataset(workspace, assignments_path)
        if dataset_path and dataset_path.is_file():
            if log:
                log(f"assignments: {assignments_path}")
                log(f"dataset:     {dataset_path}")
            try:
                samples = build_samples(assignments_path, dataset_path)
                samples_prov = {"assignments": assignments_path.name, "dataset": dataset_path.name,
                                "k": samples["k"], "created_at": _now()}
            except ValueError as exc:
                if store is None:
                    raise
                warnings.append(f"samples skipped: {exc}")
        elif store is None:
            raise ValueError("no dataset CSV with `filename` + coordinate columns found")
    elif include_samples and store is None:
        raise ValueError("no cluster_assignments*.csv found — run a clustering analysis first")

    grid_spec = None
    all_layers: dict[str, tuple[dict, bytes]] = {}
    if samples and (grid_shape or store is not None):
        shape = tuple(grid_shape) if grid_shape else store.grid.shape
        grid_spec, binned = build_grid_layers(samples, shape, grid=store.grid if store else None, provenance=samples_prov)
        all_layers.update(binned)
    if store is not None:
        vgrid, vlayers, vwarn = voxel_layers_to_bundle(store, layers, hypothesis=hypothesis, finding=finding)
        warnings.extend(vwarn)
        for lid in vlayers:
            if lid in all_layers:
                warnings.append(f"layer {lid!r}: voxel-store version replaces the publisher-binned one")
        all_layers.update(vlayers)
        grid_spec = vgrid

    if not all_layers and not samples:
        raise ValueError("nothing to publish: no samples and no voxel-store layers")

    provenance = {
        "generator": "viz_bundle",
        "grid_source": "voxel_store" if store is not None else ("samples_bounds" if grid_spec else None),
        "voxel_store": "voxel_store/index.json" if store is not None else None,
        "finding": finding,
        "hypothesis": hypothesis,
        "warnings": warnings,
    }
    manifest = write_bundle(viz_dir, samples, grid_spec, all_layers, workspace.name,
                            provenance=provenance, samples_provenance=samples_prov, log=log)
    return {
        "manifest_path": str(viz_dir / "manifest.json"),
        "viz_dir": str(viz_dir),
        "layer_ids": list(all_layers.keys()),
        "n_samples": samples["n_samples"] if samples else 0,
        "bytes_written": int(sum(len(b) for _, b in all_layers.values())),
        "grid": grid_spec,
        "generated_at": manifest["generated_at"],
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, help="workspace root (/workspace/<project_id>)")
    ap.add_argument("--assignments", help="cluster assignments CSV (auto-discovered if omitted)")
    ap.add_argument("--dataset", help="dataset CSV with coordinates (auto-discovered if omitted)")
    ap.add_argument("--grid", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="also emit gridded layers at this shape, e.g. --grid 64 64 16 "
                         "(ignored when a voxel_store/ grid exists — that grid is authoritative)")
    ap.add_argument("--no-voxel-store", action="store_true", help="ignore <workspace>/voxel_store/")
    args = ap.parse_args(argv)
    try:
        result = publish(
            args.workspace, tuple(args.grid) if args.grid else None, args.assignments, args.dataset,
            include_voxel_store=not args.no_voxel_store, log=print,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    for w in result["warnings"]:
        print(f"warning: {w}")


if __name__ == "__main__":
    main()
