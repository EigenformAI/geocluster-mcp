# How the voxel MCP tools work

Section L of the GeoCluster MCP server: eleven `voxel_*` tools that turn an
**already-validated analysis result** into a 3-D layer the dashboard's viewer can render.
Code: `voxel/` (store, stamping, provenance), `tools/voxel.py` (the MCP surface),
`viz_bundle.py` (exporter shared with `publish_viz.py`). Tests: `tests/test_voxel_*.py`.

## The short answer

**Every voxel value comes from a row in a table in the project workspace. The tools never
invent, interpolate, extrapolate or randomly fill anything.**

- A voxel is written only when a record with real coordinates and a real value is stamped
  into it. Voxels nothing was stamped into stay *empty* and the viewer draws nothing there.
- Coordinates must be data-derived (`coordinate_source="artifact"` is the only accepted
  value) and, for batch uploads, are cross-checked against the cited source file: a record
  whose coordinates are not within 1 m of a row in that file is **rejected**, not warned.
- The one "spreading" that happens is geometric: a record can claim the voxels within a
  radius (point), along a corridor (line) or inside a box. That footprint is a modelling
  choice (e.g. "this assay interval represents ±1 m of core"), it is logged per record, and
  it writes the record's own value, never a new one.
- The LLM (the *voxel specialist*) decides **what** to stamp: which result table, which
  value column, which radius. The tools decide **where** it lands (pure arithmetic) and
  record it. Where the LLM could go wrong, a tool check catches it (next section).

## Where it sits in the agent system

```
user ─▶ L1 geology agent ─▶ L2 orchestrator ─▶ L3 dataops / transform / analytics …
                                       │  after the result is verified:
                                       └▶ L3 voxel specialist ──▶ voxel_* tools ──▶ voxel_store/ ──▶ viz/ ──▶ R2 ──▶ viewer
```

- The orchestrator judges "is this result voxel-able?" (rows carry or join to coordinates
  and have a value per row) and dispatches `voxel` once, after verification. Otherwise it
  says "no 3D layer: <reason>".
- The voxel specialist can call only `voxel_*`, `list_files` and `summarize_provenance`
  (invariant L3-6). No other specialist can call `voxel_*` — the tool names are the ACL, which
  is why they avoid substrings like `log`, `stat`, `cluster`, `map`, `viz`, `ratio`.
- It has a shell and can write files, but only under `scripts/voxel/` and `results/`; source
  data is never modified (tool paths go through `resolve_path()`, invariant X-1).

## The voxel store

Created per project at `<workspace>/voxel_store/`:

| file | content |
|---|---|
| `index.json` | grid spec, dataset provenance, one entry per layer (dtype, metadata, hypothesis, content hash) |
| `layers/<name>.npy` | float64 array with the grid's shape, C-order `[ix, iy, iz]` |
| `operations.jsonl` | append-only provenance: one line per stamp (see [Provenance](#provenance)) |

**Grid.** `voxel_init_grid(dataset_path)` derives the grid from the dataset the analysis
used, never from constants:

- x/y bounds from `EASTING`/`NORTHING` (or `x`/`y`, `coord_x`/`coord_y`); `LONGITUDE`/
  `LATITUDE` are converted to local metres with `x = lon·111320·cos(lat0)`, `y = lat·111320`
  and the conversion is stored so later lon/lat records use the same one.
- depth bounds from `depth_from_m`/`depth_to_m`, else `depth_m`, else `[0, 1]` and the grid
  is flagged `depth_degenerate` with `nz = 1`.
- default shape `64 × 64 × 16`; `shape` or `cell_size_xy_m`/`cell_size_z_m` override it;
  hard cap 4,000,000 voxels; extents under 1 m are widened to 1 m.
- cell lookup: `i = floor((coord − origin) / cell_size)`, clamped to the grid. That formula
  is returned by `voxel_get_grid` so the specialist can reproduce it in a script.

**Layers.** dtype `float` (empty = `0.0`), `categorical` or `boolean` (empty = `−1`, so class
`0` is a valid label). Names are validated (`[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}`) because they
become file names and manifest ids.

## Tool reference

| tool | what it does | guarantees |
|---|---|---|
| `voxel_init_grid(dataset_path, shape?, cell_size_xy_m?, cell_size_z_m?, padding_m?, overwrite?)` | derive + create the grid | refuses to discard existing layers unless `overwrite=True`; errors name the expected columns |
| `voxel_get_grid()` | grid, units, dataset provenance, index formula, layer summaries | read-only; "call this first" |
| `voxel_add_point(layer, x, y, depth_m, value, radius_m=50, …)` | stamp one sphere | point must be inside the grid; the containing cell is always claimed even for `radius_m=0` |
| `voxel_add_line(layer, start_*, end_*, value, width_m=25, …)` | stamp a corridor (fault, vein, drill trace) | both endpoints must be inside the grid; sampled every half-cell |
| `voxel_add_box(layer, x_min…depth_max_m, value, …)` | stamp an axis-aligned box | clipped to the grid; fails if no overlap |
| `voxel_upsert_geometry(layer, records_path, mode, dtype, combination_rule, bounds_policy, max_records=5000, value_col, label_map?, hypothesis?, source_file?, source_check="reject", source_tolerance_m?)` | materialise a CSV of point/line/box records into one layer, in one locked write | **source cross-check** (below); records without coordinates skipped and counted; string labels mapped to codes and the map stored |
| `voxel_set_layer_array(layer, array_path, dtype, …)` | deposit a precomputed `.npy`/`.npz` verbatim | shape must equal the grid; array path + sha256 recorded |
| `voxel_probe_region(x, y, depth_m, radius_m, layers?, max_voxels=50)` | read values around a point | read-only sanity check at a location known from the data |
| `voxel_list_layers()` | dtype, non-empty count, min/max per layer | read-only |
| `voxel_history(layer?, limit=200)` | the operations log, newest first, + per-layer summary | read-only |
| `voxel_export_bundle(layers?, finding?, hypothesis?, include_samples=True, assignments_path?, dataset_path?)` | write `viz/` (layers → samples → manifest last) | mandatory final step; returns the "Open 3D Visualization…" line |

All tools return a dict with `success`, `tool` and, on failure, `error` + `next_step`
(invariant "no silent failures"). All are deterministic numpy/pandas; there is no LLM call
inside any tool.

## How a value gets into a voxel

**Route "geometry" (discrete records).** The specialist writes a script that joins the result
table to the dataset on the shared key (file name / hole id), takes the coordinates *from the
dataset*, and emits `results/<layer>_feature_geometry.csv`:

```
record_id, geometry_kind, value, coordinate_source, source_file, source_excerpt,
x, y, depth_m, radius_m                      # point
start_x, start_y, start_depth_m, end_x, end_y, end_depth_m, width_m   # line
x_min, y_min, depth_min_m, x_max, y_max, depth_max_m                  # box
```

`voxel_upsert_geometry` then, per record: checks `coordinate_source == "artifact"`; reads the
value (`value_col`, default `value`); builds a boolean mask over the grid — sphere (cells
whose centre is within `radius_m`, plus the containing cell), line (union of spheres of
`width_m/2` sampled every half-cell along the segment) or box (index range); and applies the
value to the masked cells with the **combination rule**:

| rule | on an empty cell | on a filled cell |
|---|---|---|
| `replace` | value | value |
| `max` (default) | value | max(current, value) |
| `add` | value | current + value |
| `mean` | value | (current + value) / 2 |

`mode="replace_layer"` starts from an all-empty array and drops the layer's old provenance;
`accumulate_layer` builds on the existing array. `bounds_policy` `skip` (default) drops
out-of-grid records and counts them, `clip` snaps them to the edge, `fail` aborts.

**Route "value_grid" (continuous fields).** For things that are inherently per-cell —
kernel density, distance-to-contact, an interpolated grade model — the specialist's script
computes an array of exactly the grid's shape using the index formula from `voxel_get_grid`
and hands it to `voxel_set_layer_array`. The tool checks the shape, stores the array
verbatim, and records the array path and sha256. *The tool does no interpolation itself;* if
the script interpolated, that is the analysis result being visualised, and the provenance
says which script/array produced it.

**Manual stamps** (`voxel_add_point/line/box`) exist for single located features quoted from
a report ("fault from A to B", "ore zone box"). They require in-bounds coordinates and log
`source_file`/`source_excerpt`, but they do **not** run the source cross-check — the batch
tool is the audited path and the specialist prompt steers everything tabular through it.

## Does it make up data? — the checks, one by one

| risk | what stops it |
|---|---|
| invented or "distributed" coordinates | `coordinate_source` must be `"artifact"`, tool-enforced; the port's `geonames`/`web`/`creative_fallback` sources and the "be creative" prompt text were deliberately not ported |
| wrong join (coordinates from the wrong row/file) | `voxel_upsert_geometry` cross-checks every point / line endpoint / box against the rows of its `source_file` (KD-tree, 1 m default; 1e-5° on a degree grid; `source_tolerance_m` to relax) and rejects mismatches; the result reports `records_rejected_source_mismatch` |
| rows without coordinates | skipped and counted (`records_skipped`, warnings), never placed |
| filling gaps between samples | no interpolation, smoothing or kriging in any tool; unstamped voxels stay empty (`0.0` / `−1`) and are not rendered |
| a footprint that overstates the data | `radius_m`/`width_m` are per-record and logged; `radius_m=0` claims only the containing cell; the specialist prompt requires real extents (e.g. half the sample interval) and forbids placeholders |
| constant/placeholder values | the specialist prompt rejects "value = 1.0 for everything"; a non-numeric value is coerced to presence `1.0` **with a warning** in the result (this is the one place a value can differ from the table, and it is reported, not silent) |
| silently empty layer | `records_applied == 0` returns a `next_step`; the prompt requires `affected_voxels > 0` and a `voxel_probe_region` check before export |
| wrong array | `voxel_set_layer_array` refuses shape mismatches; categorical arrays must be integers in `0..254` |
| tampering with inputs | tools only read source files; outputs go to `voxel_store/`, `results/`, `viz/` |
| losing the trail | every stamp is in `operations.jsonl`; the exporter copies the summary into the manifest |

What the tools do **not** check: that the *value* in the record matches the source table (only
the coordinates are cross-checked), and the *choice* of radius. Both are visible in
`voxel_history` and in the manifest provenance, so they can be reviewed, and the orchestrator
passes the value column and dtype explicitly in the dispatch context.

## Provenance

Each stamp appends one JSON line:

```json
{"ts":"2026-09-10T07:57:01+00:00","op":"point","layer":"au_grade",
 "coordinates":"246980.9,6065109.6,12.5","parameters":"radius_m=0.5,value=0.42",
 "source_file":"segments_voxel_compatible.csv","source_excerpt":"row 812 …",
 "coordinate_source":"artifact","record_id":"812","group_id":"<batch uuid>","affected_voxels":1}
```

`voxel_export_bundle` writes, per layer, into `viz/manifest.json → artifacts[grid].layers[*].provenance`:
generator `voxel_store`, dtype, hypothesis, finding, `source_files`,
`coordinate_source_counts`, operation counts by kind, the `.npy` content hash, and for arrays
the path + sha256. So any voxel on screen can be traced: manifest → layer → operations log →
record id → row of the source table.

Display encoding (in the exporter, not the store): float layers are log10 min–max scaled to
`[0.05, 1.0]` with `raw_min`/`raw_max` kept in the manifest for the legend; values `≤ 0` are
exported as empty (viewer contract) and counted in `dropped_nonpositive_voxels`; categorical
layers become `uint8` codes with `255` = empty and carry their label/colour table.

## Worked example: the first production run (2026-09-10)

Request: "make `segments_with_geochem.csv` compatible with the voxel visualizer".
Dispatch chain: dataops (inspect) → transform (detection limits, duplicates, wide table) →
dataops (verify) → **voxel**. The voxel specialist called `voxel_get_grid`, `voxel_init_grid`,
wrote `results/au_grade_feature_geometry.csv` from the cleaned table, then
`voxel_upsert_geometry`, `voxel_list_layers`, `voxel_probe_region`, `voxel_export_bundle`.

| result field | value |
|---|---|
| `records_applied` | 1,183 |
| `records_rejected_source_mismatch` | 0 |
| `affected_voxels` | 1,183 (one cell per record: radius ≈ 0) |
| `nonempty_voxels` | 51 |
| value range (Au) | 0.000005 – 10.9 |
| grid cell size | 8,073 m × 15,031 m × 47 m |

Reading it: all 1,183 located rows were real (none rejected) and each claimed exactly its own
cell, but the dataset spans roughly 500 × 960 km, so the default 64 × 64 × 16 grid has
8–15 km cells and many holes fall into the same voxel — with the `max` rule the cell shows
the highest grade among them, hence only 51 non-empty voxels. Nothing was filled in between.
For a denser picture the specialist should pass `cell_size_xy_m` to `voxel_init_grid`
(within the 4 M-voxel cap) or the request should be scoped to one prospect.

## Limits and caveats

- **Grid resolution** is the main lever on how the layer looks; it is derived from data
  bounds, so a state-wide dataset gives kilometre cells by default.
- `max_records` default 5,000 per upsert (raise it explicitly for bigger tables).
- Degree grids (`EPSG:4326`) use the local-metres approximation for radii; projected grids
  are exact.
- One store per workspace; `voxel_init_grid(overwrite=True)` deletes all layers — there is no
  per-layer delete tool.
- `voxel_add_point/line/box` are not source-checked (see above).
- The export normalises for display; use `voxel_list_layers`/`voxel_probe_region` for raw
  values.
- The tools run inside the IDE machine; only `viz/` is uploaded to R2, so `voxel_store/`
  lives and dies with the machine's volume.

## Verifying it yourself

```bash
cd geocluster-ai-ide-BE/geocluster-mcp
uv run --with pytest pytest tests/test_voxel_store.py tests/test_voxel_spatial.py \
    tests/test_voxel_tools.py tests/test_viz_bundle.py tests/test_voxel_tool_names.py -q
```

Known-answer tests include: a point at known coordinates lands in the expected cell; a radius
on 100 m cells touches exactly the expected voxels; label `0` survives a round trip; publisher
binning and point-per-sample stamping give identical density counts on the same grid; the
source cross-check rejects a row with invented coordinates and accepts the same row with the
real ones; the manifest is written last and every blob has `n_voxels × itemsize` bytes.
