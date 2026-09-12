"""CSV -> voxel layer, deterministic, no LLM involved.

Meant to be called directly (e.g. from a UI button), bypassing the MCP/chat
layer entirely. Every step is plain pandas/numpy: detect the coordinate and
depth columns (voxel/columns.py), clean the chosen value column, build a
feature_geometry records table, and materialise it via the existing
voxel_init_grid / voxel_upsert_geometry / voxel_export_bundle tools.

Usage (long-format CSV, one row per element per sample -- e.g. a chem_code +
value column pair):
    csv_to_voxel("ReSolveSA/segments_with_geochem.csv", value_col="value",
                 layer="au_geochem", filter_col="chem_code", filter_value="Au")

Usage (wide-format CSV, one column per measurement):
    csv_to_voxel("segments_voxel_ready.csv", value_col="Cu_ppm", layer="cu_grade_ppm")
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
from typing import Any

from direct._log import get_logger
from tools.config import WORKSPACE_ROOT, read_tabular, resolve_path, save_csv
from voxel.columns import (
    EASTING_PATTERNS,
    LAT_PATTERNS,
    LON_PATTERNS,
    NORTHING_PATTERNS,
    coordinate_columns,
    depth_columns,
)

_BDL_RE = re.compile(r"^\s*<\s*([0-9.]+)\s*$")  # "<0.01" below-detection-limit notation
_log = get_logger("csv_to_voxel")


def _current_layer_names() -> list[str] | str:
    """Best-effort layer list for log lines -- 'no store' if there isn't one yet."""
    try:
        from tools.voxel import _open_store

        return _open_store().layer_names
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


@contextlib.contextmanager
def _project_lock():
    """Exclusive, non-blocking lock over the whole project's voxel_store +
    viz output. voxel_init_grid/voxel_upsert_geometry/voxel_export_bundle
    each write those files safely on their own, but nothing stops two
    csv_to_voxel() calls from *interleaving* -- e.g. one process's export
    reading the store mid-write by another. Held for the whole convert, so
    a second click while one is still running fails fast and clearly
    instead of silently racing.
    """
    lock_path = os.path.join(WORKSPACE_ROOT, ".csv_to_voxel.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        _log.warning("lock BUSY, refusing to run (workspace=%s)", WORKSPACE_ROOT)
        raise RuntimeError(
            "Another \"Convert to Voxel\" is already running for this project. "
            "Wait for it to finish, then try again."
        ) from None
    _log.info("lock acquired (workspace=%s, layers_before=%s)", WORKSPACE_ROOT, _current_layer_names())
    try:
        yield
    finally:
        _log.info("lock released (layers_after=%s)", _current_layer_names())
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def clean_numeric_column(series) -> dict[str, Any]:
    """Coerce a value column to float.

    ``"<X"`` (below-detection-limit) is treated as ``X / 2`` -- the common
    geochem convention -- and counted separately so callers can see how many
    values were substituted rather than have it happen silently. Anything
    else that doesn't parse is dropped (not coerced to a placeholder).
    """
    import pandas as pd

    def parse_one(v):
        if pd.isna(v):
            return None, "missing"
        if isinstance(v, (int, float)):
            return float(v), "numeric"
        s = str(v).strip()
        m = _BDL_RE.match(s)
        if m:
            return float(m.group(1)) / 2.0, "below_detection"
        try:
            return float(s.replace(",", "")), "numeric"
        except ValueError:
            return None, "unparseable"

    parsed = series.map(parse_one)
    values = parsed.map(lambda t: t[0])
    kinds = parsed.map(lambda t: t[1])
    dropped_mask = kinds == "unparseable"

    return {
        "values": values,
        "n_parsed": int((kinds == "numeric").sum()),
        "n_below_detection": int((kinds == "below_detection").sum()),
        "n_missing": int((kinds == "missing").sum()),
        "n_dropped": int(dropped_mask.sum()),
        "dropped_examples": series[dropped_mask].astype(str).unique().tolist()[:10],
    }


def geometry_records(df, dataset_path: str, value) -> dict[str, Any]:
    """Shared by every "-> voxel" path (numeric cleaning here, LLM
    classification in classify_text_to_voxel.py): given a dataframe and an
    already-computed per-row ``value`` series (float, NaN = unusable), find
    the coordinate/depth columns and build a feature_geometry records table
    ready for ``voxel_upsert_geometry``.

    geometry_kind is chosen automatically: depth_from_m/depth_to_m -> "line"
    (one drillhole interval per row), depth_m alone -> "point", neither ->
    "point" at depth 0 (flat/2-D dataset).
    """
    import pandas as pd

    cc = coordinate_columns(list(df.columns))
    if cc is None:
        raise ValueError(
            f"no coordinate columns in {dataset_path}: expected EASTING/NORTHING "
            "(or x/y, coord_x/coord_y) or LONGITUDE/LATITUDE"
        )
    dc = depth_columns(list(df.columns))

    x = pd.to_numeric(df[cc["x"]], errors="coerce")
    y = pd.to_numeric(df[cc["y"]], errors="coerce")
    usable = value.notna() & x.notna() & y.notna()

    if dc["from"] and dc["to"]:
        kind = "line"
        d0 = pd.to_numeric(df[dc["from"]], errors="coerce")
        d1 = pd.to_numeric(df[dc["to"]], errors="coerce")
        usable &= d0.notna() & d1.notna()
        records = pd.DataFrame({
            "geometry_kind": "line",
            "start_x": x, "start_y": y, "start_depth_m": d0,
            "end_x": x, "end_y": y, "end_depth_m": d1,
            "value": value,
        })
    elif dc["depth"]:
        kind = "point"
        d = pd.to_numeric(df[dc["depth"]], errors="coerce")
        usable &= d.notna()
        records = pd.DataFrame({"geometry_kind": "point", "x": x, "y": y, "depth_m": d, "value": value})
    else:
        kind = "point"
        records = pd.DataFrame({"geometry_kind": "point", "x": x, "y": y, "depth_m": 0.0, "value": value})

    records = records[usable].reset_index(drop=True)
    if records.empty:
        raise ValueError("no usable rows -- check coordinate/depth columns and the value column")

    records["coordinate_source"] = "artifact"
    records["source_file"] = dataset_path
    return {"records": records, "geometry_kind": kind}


def build_records(
    dataset_path: str,
    value_col: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
) -> dict[str, Any]:
    """Read ``dataset_path``, optionally filter to one category, clean
    ``value_col``, and return a feature_geometry records table ready for
    ``voxel_upsert_geometry`` plus cleaning stats.
    """
    resolved = resolve_path(dataset_path)
    df = read_tabular(resolved)

    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"filter_col {filter_col!r} not found in {dataset_path}")
        df = df[df[filter_col].astype(str) == str(filter_value)].copy()
        if df.empty:
            raise ValueError(f"no rows where {filter_col} == {filter_value!r}")

    if value_col not in df.columns:
        raise ValueError(f"value_col {value_col!r} not found in {dataset_path}")

    cleaned = clean_numeric_column(df[value_col])
    geo = geometry_records(df, dataset_path, cleaned["values"])

    stats = {
        "rows_in": int(len(df)),
        "rows_usable": int(len(geo["records"])),
        "geometry_kind": geo["geometry_kind"],
        "n_below_detection": cleaned["n_below_detection"],
        "n_dropped_unparseable": cleaned["n_dropped"],
        "dropped_examples": cleaned["dropped_examples"],
    }
    return {"records": geo["records"], "stats": stats}


def csv_to_voxel(
    dataset_path: str,
    value_col: str,
    layer: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    grid_shape: list[int] | None = None,
    cell_size_xy_m: float | None = None,
    cell_size_z_m: float | None = None,
    combination_rule: str = "max",
) -> dict[str, Any]:
    """End to end: clean -> stamp into the project's voxel store -> export
    the viewer bundle (viz/manifest.json + viz/layers/*). Reuses the
    project's existing grid if one is already initialised (so multiple
    layers/elements from the same CSV share one grid); otherwise derives a
    new one from this dataset's coordinate/depth bounds.
    """
    from tools.voxel import voxel_export_bundle, voxel_get_grid, voxel_init_grid, voxel_upsert_geometry

    _log.info("csv_to_voxel start: dataset=%s value_col=%s layer=%s filter=%s=%s",
              dataset_path, value_col, layer, filter_col, filter_value)

    # Check the name up front rather than after all the parsing work --
    # voxel_upsert_geometry validates it too, but only at the very last step.
    from voxel.store import validate_layer_name

    validate_layer_name(layer)

    with _project_lock():
        built = build_records(dataset_path, value_col, filter_col=filter_col, filter_value=filter_value)
        records, stats = built["records"], built["stats"]
        records_path = save_csv(records, dataset_path, f"voxel_records_{layer}")

        grid_result = voxel_get_grid()
        if not grid_result.get("success"):
            _log.info("no existing grid (%s) -- calling voxel_init_grid", grid_result.get("error"))
            grid_result = voxel_init_grid(
                dataset_path=dataset_path, shape=grid_shape,
                cell_size_xy_m=cell_size_xy_m, cell_size_z_m=cell_size_z_m,
            )
            _log.info("voxel_init_grid -> success=%s layers_now=%s", grid_result.get("success"), _current_layer_names())
            if not grid_result.get("success"):
                return {"success": False, "step": "voxel_init_grid", "cleaning": stats, **grid_result}

        upsert_result = voxel_upsert_geometry(
            layer=layer, records_path=records_path, value_col="value",
            combination_rule=combination_rule, source_file=dataset_path, source_check="reject",
            max_records=len(records),  # default (5000) silently drops rows past it on bigger datasets
        )
        _log.info("voxel_upsert_geometry(%s) -> success=%s applied=%s layers_now=%s",
                  layer, upsert_result.get("success"), upsert_result.get("records_applied"), _current_layer_names())
        if not upsert_result.get("success"):
            return {"success": False, "step": "voxel_upsert_geometry", "cleaning": stats, **upsert_result}

        # No `layers=` filter: voxel_export_bundle defaults to every layer
        # currently in the store, so a second call (e.g. a different element)
        # doesn't drop earlier layers out of the manifest.
        export_result = voxel_export_bundle()
        _log.info("voxel_export_bundle -> success=%s layer_ids=%s",
                  export_result.get("success"), export_result.get("layer_ids"))

        return {
            "success": bool(export_result.get("success")),
            "cleaning": stats,
            "grid": grid_result.get("grid"),
            "upsert": {k: v for k, v in upsert_result.items() if k != "warnings"},
            "export": export_result,
        }


def remove_layer(layer: str) -> dict[str, Any]:
    """Delete one layer from the project's voxel_store and re-publish viz/
    so the manifest (and the exported .f32/.u8 blob) drop it too.

    Exists so deleting a layer's file in the IDE's Explorer actually deletes
    the layer -- otherwise voxel_store still has it and the next conversion
    (which always re-exports every stored layer) silently brings it back.
    """
    from tools.voxel import _open_store, _workspace_root, voxel_export_bundle

    _log.info("remove_layer start: layer=%s", layer)

    with _project_lock():
        store = _open_store()
        if layer not in store.layer_names:
            _log.info("remove_layer(%s): not in store, nothing to do", layer)
            return {"success": True, "removed": False, "note": f"layer {layer!r} was not in the store"}
        store.remove_layer(layer)
        _log.info("remove_layer(%s): removed, layers_now=%s", layer, _current_layer_names())
        export_result = voxel_export_bundle()

        # write_bundle() only writes the layers it exports -- it never
        # deletes a stale blob left over from a layer that's no longer in
        # the store, so do that ourselves (harmless if already gone, e.g.
        # the caller deleted the file first and *that's* what triggered
        # this removal).
        for ext in ("f32", "u8"):
            stale = os.path.join(_workspace_root(), "viz", "layers", f"{layer}.{ext}")
            if os.path.isfile(stale):
                os.remove(stale)

        return {"success": bool(export_result.get("success")), "removed": True, "export": export_result}


def inspect_csv(dataset_path: str, unique_values_of: str | None = None, max_unique: int = 300) -> dict[str, Any]:
    """Header + column types for the IDE's picker UI, and (optionally) the
    distinct values of one column -- e.g. the chem_code values to choose
    from for a long-format CSV. No LLM: just a pandas read.
    """
    import pandas as pd

    resolved = resolve_path(dataset_path)
    df = read_tabular(resolved, nrows=5000)  # enough to type-sniff without loading huge files fully

    # Fast line count for the *real* row total (df above is capped at 5000
    # for speed) -- the IDE uses this to warn before an AI classification
    # that would take a long time on a big file.
    with open(resolved, "rb") as f:
        total_rows = max(0, sum(1 for _ in f) - 1)

    import re

    dc = depth_columns(list(df.columns))
    geo_patterns = EASTING_PATTERNS + NORTHING_PATTERNS + LON_PATTERNS + LAT_PATTERNS
    geo_cols = {c for c in df.columns if any(re.search(p, str(c).strip().lower()) for p in geo_patterns)}
    structural = geo_cols | {dc.get("depth"), dc.get("from"), dc.get("to")} - {None}

    numeric_cols = [
        c for c in df.columns
        if c not in structural and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    result: dict[str, Any] = {
        "success": True,
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "structural_columns": sorted(structural),  # coordinate/depth columns -- not useful as a value or filter column
        "total_rows": total_rows,
    }

    if unique_values_of:
        if unique_values_of not in df.columns:
            return {"success": False, "error": f"column {unique_values_of!r} not found"}
        full = read_tabular(resolved, usecols=[unique_values_of])[unique_values_of]
        values = sorted({str(v) for v in full.dropna().unique()})
        result["unique_values"] = values[:max_unique]
        result["unique_values_truncated"] = len(values) > max_unique
    return result


def _cli() -> None:
    """Subprocess entrypoint for the IDE's CSV-viewer extension -- not an
    MCP tool, no LLM involved. Two modes, both print one JSON object to
    stdout:

      python3 csv_to_voxel.py inspect --dataset X [--unique-values-of COL]
      python3 csv_to_voxel.py convert --dataset X --value-col V --layer L
                                       [--filter-col C --filter-value F]
    """
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    insp = sub.add_parser("inspect")
    insp.add_argument("--dataset", required=True)
    insp.add_argument("--unique-values-of", default=None)

    conv = sub.add_parser("convert")
    conv.add_argument("--dataset", required=True)
    conv.add_argument("--value-col", required=True)
    conv.add_argument("--layer", required=True)
    conv.add_argument("--filter-col", default=None)
    conv.add_argument("--filter-value", default=None)

    rem = sub.add_parser("remove-layer")
    rem.add_argument("--layer", required=True)

    args = p.parse_args()

    try:
        if args.mode == "inspect":
            result = inspect_csv(args.dataset, unique_values_of=args.unique_values_of)
        elif args.mode == "remove-layer":
            result = remove_layer(args.layer)
        else:
            result = csv_to_voxel(
                args.dataset, value_col=args.value_col, layer=args.layer,
                filter_col=args.filter_col, filter_value=args.filter_value,
            )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)

    print(json.dumps(result))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    _cli()
