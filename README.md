# Geocluster MCP

Geocluster MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server with 45 tools for exploration data. Requests to clean drillhole geochemistry, cluster measurements, rank anomalies, or plot a magnetic survey run as pandas, rasterio, and scikit-learn operations against workspace files.

Paths must resolve inside one workspace. Derived outputs use workspace or conversation-specific `results/` folders. `verify_claims` compares reported numbers against source data.

- **Inputs:** CSV, Excel, and LAS (well logs) for tabular data; GeoTIFF for rasters.
- **Built with:** [FastMCP](https://github.com/jlowin/fastmcp), pandas, GeoPandas, rasterio, scikit-learn, UMAP.
- **Transport:** SSE by default; shared Streamable HTTP at `/mcp` for runtime conversations.

Native voxel creation and named revisions belong to the canonical project runtime. The legacy shared voxel store, stamping tools, direct CSV-to-voxel pipeline, text-classification publishing pipeline, and viewer-bundle exporter are retired.

## Tools

The default catalog contains 45 tools, grouped by pipeline stage. File arguments use workspace paths. The optional training tools are disabled by default.

### Inspect & profile

- `list_files(directory)`: list files in a workspace directory.
- `inspect_dataset(path)`: structural summary; also loads the dataset into the cache for `query_data`.
- `inspect_specific_columns(path, columns, ...)`: per-column stats, unique values, or value counts.
- `check_missing(path)`: columns with missing values.
- `inspect_raster(path, verbose)`: raster shape, CRS, bounds, transform, per-band stats.
- `profile_geochem(path, from_col, to_col, element_cols)`: depth-weighted mean, max, and grade-thickness for assay intervals.
- `query_data(path, operation, ...)`: filter, sort, threshold, or evaluate an expression against a cached dataset.

### Spatial

- `reproject(path, target_crs, ...)`: reproject a raster, or a CSV of x/y points, to a new CRS.
- `resample(path, scale_factor)`: change raster resolution (`0.5` halves, `2.0` doubles).
- `clip_to_extent(path, min_x, min_y, max_x, max_y, crs)`: clip a raster to a bounding box.
- `align_grids(source_path, reference_path)`: match a raster to a reference grid and CRS.

### Reshape & transform

- `normalize`, `standardize`, `log_transform`, `smooth`: per-column scaling and rolling-window smoothing.
- `pivot`, `melt`: reshape long-to-wide and back.
- `merge_datasets(path, right_path, on, how)`: join two datasets.
- `filter_rows(path, column, operator, value)`: row filter (`==`, `>`, `in`, `contains`, `not_null`, ...).
- `convert_dtype(path, columns, dtype, errors)`: cast columns to numeric, int, float, str, or datetime.

### Features

- `select_bands(path, indices)`: extract raster bands (1-based).
- `band_math(path, expression)`: raster algebra with `b1`, `b2`, ... for example `(b1-b2)/(b1+b2)`.
- `compute_gradient(path, method)`: edge detection (Sobel).
- `texture_features(path, method)`: texture rasters (entropy).
- `select_columns`, `compute_ratios(pairs)`: keep columns, or compute ratios like `Au_ppb/Cu_ppm`.
- `aggregate(path, spatial_op, statistic, ...)`: aggregate by grid cell (`grid:50`) or by category (`col:RockType`).

### Anomaly & ranking

- `compute_anomaly(path, columns, method)`: anomaly scores by z-score, MAD, or ratio.
- `threshold(path, column, value, mode)`: keep rows above or below a cutoff.
- `rank_by_metric(path, metric, top_n)`: top N rows by a metric.

### Clustering

- `reduce_dimensions(path, columns, method, n_components)`: PCA or UMAP.
- `cluster(path, columns, algorithm, ...)`: k-means or DBSCAN.

### Visualization

- `plot_map(path, x_col, y_col, color_col)`: map from a raster or an x/y CSV.
- `plot_scatter`, `plot_histogram`, `plot_clusters`: standard plots, saved as PNG.

### Data cleaning (geology-aware)

- `validate_geology(path)`: scan a dataset for common quality issues across hole ids, coordinates, geochem columns, and detection limits.
- `detect_cleaning_issues(path, columns)`: granular per-column diagnosis.
- `fix_decimals(path, columns, ...)`: convert comma-decimal values (`1,23`) to dot-decimal.
- `parse_detection_limits(path, columns, method)`: turn `<0.01` style strings into numbers.
- `remove_duplicates`, `standardize_terms(column, mapping)`: dedupe rows, normalize categorical terms.

### Provenance, export & verification

- `summarize_provenance(workspace_path)`: list every artifact in `results/` with metadata.
- `export_artifact(path, format, ...)`: bundle results as `zip`, `xlsx`, or `geojson`.
- `verify_claims(claims, path)`: check numeric claims against the source data.

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/EigenformAI/geocluster-mcp
cd geocluster-mcp
uv sync
```

## Running the server

Point `MCP_WORKSPACE_ROOT` at the directory that holds your data, then start the server:

```bash
MCP_WORKSPACE_ROOT=/path/to/your/project uv run python main.py
# Starting Geocluster MCP on http://0.0.0.0:7654/sse
```

If `MCP_WORKSPACE_ROOT` is unset it defaults to the current directory. Any path a tool receives is resolved inside the workspace, and a path that escapes it is rejected.

### Shared Streamable HTTP

The runtime uses one server for every conversation:

```bash
MCP_WORKSPACE_ROOT=/path/to/your/project MCP_TRANSPORT=http \
	MCP_HOST=127.0.0.1 MCP_PORT=7710 uv run python main.py
```

Connect a Streamable HTTP client to `http://127.0.0.1:7710/mcp`. Each tool call must supply `run_dir: "runs/<conversation-id>"`; the middleware selects that call's output folder before validating the ordinary tool arguments. The server requires a loopback host, no server-wide `GEOCLUSTER_RUN_DIR`, and disabled training in shared mode. It exposes `/healthz` with mode and in-flight work counts.

Synchronous tools run in a bounded worker pool. `MCP_TOOL_THREADS` and `MCP_TOOL_THREADS_PER_RUN` configure positive worker limits. Calls exceeding capacity wait for their run's permit; different conversations retain independent output contexts.

### stdio instead of SSE

For a stdio client, run the imported server directly:

```bash
MCP_WORKSPACE_ROOT=/path/to/your/project uv run python -c 'from main import mcp; mcp.run()'
```

## Connecting an MCP client

For an SSE-capable client, point it at the server URL:

```json
{
  "mcpServers": {
    "geocluster": {
      "url": "http://localhost:7654/sse"
    }
  }
}
```

## Outputs

Without a conversation folder, file-derived outputs use `results/` beside the input. SSE callers can select a server-wide `GEOCLUSTER_RUN_DIR=runs/<conversation-id>`. Shared HTTP callers select the folder per call with `run_dir`; inputs outside that folder produce `<run>/results/`. A chain whose input already lives in that run's `results/` folder keeps its outputs there.

Tools return their generated paths. Publication behavior is tool-specific: the shared CSV/raster/plot output helpers and ZIP export use atomic replacement, while other writers can write directly to their final paths. `summarize_provenance` lists generated evidence, and `export_artifact` bundles ordinary results.

## Example

`example/` holds a magnetic-survey raster (`mag_survey.tif`) and a sample geochemistry CSV (`my_csv.csv`). Run the server with the workspace pointed at the repo root, then ask an agent to, for example, inspect the raster, compute a texture or gradient layer, cluster the geochemistry, and plot the result.

## Design notes

- **Library loading:** heavy libraries are generally imported within tools; shared mode also warms the threadpool controller before serving concurrent work.
- **DataFrame cache:** datasets are cached by resolved path with an mtime check and a memory budget, so repeated tool calls on the same file do not re-read it.
- **Workspace isolation:** `resolve_path` refuses any path outside `MCP_WORKSPACE_ROOT`.

## FAQ

**Which MCP clients does this work with?**
Clients that support SSE, Streamable HTTP, or stdio through the corresponding startup command. Shared HTTP clients must include the per-call `run_dir` argument.

**Does the server need an API key?**
No. It runs locally and calls no external services. The model doing the reasoning lives on the client side, and that is where an API key (if any) is configured.

**Can it overwrite or damage my data?**
Paths outside the workspace are rejected. Derived results go to the selected output folder; repeating an operation can replace its previous generated result. Check a tool's arguments before requesting edits.

**Is my data sent anywhere?**
The tools execute locally on your machine. Only what the agent chooses to read back (tool outputs and summaries) reaches whatever model your MCP client is configured to use.

**Is it geology-specific, or does it work on any tabular and raster data?**
The spatial, transform, clustering, anomaly, and plotting tools are generic. The cleaning tools (`validate_geology`, `parse_detection_limits`, `profile_geochem`) assume drillhole and assay conventions.

**How is this different from just asking a model to write pandas code?**
The tools are pre-built, input-validated, and consistent between runs. The agent composes them instead of regenerating fragile scripts each time, and `verify_claims` audits the numbers afterwards.
