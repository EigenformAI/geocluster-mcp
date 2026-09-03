# Geocluster MCP

LLMs are good at reasoning about geoscience data and bad at actually touching it: they will happily describe a clustering they never ran or quote a grade that is not in the file. Geocluster MCP closes that gap. It is a [Model Context Protocol](https://modelcontextprotocol.io) server that hands an agent about 50 real tools for exploration data, so a request like *"clean this drillhole geochemistry, cluster it, rank the anomalies, and plot them over the magnetic survey"* runs as actual pandas, rasterio, and scikit-learn calls against your files.

Every operation is sandboxed to one workspace directory and writes its output to a `results/` folder next to the input rather than overwriting it. A `verify_claims` tool checks the numbers an agent reports back against the source data, so a hallucinated grade or coordinate is caught rather than trusted.

- **Inputs:** CSV, Excel, and LAS (well logs) for tabular data; GeoTIFF for rasters.
- **Built with:** [FastMCP](https://github.com/jlowin/fastmcp), pandas, GeoPandas, rasterio, scikit-learn, UMAP.
- **Transport:** SSE on `http://0.0.0.0:7654/sse` (stdio is one line away, see below).

## Tools

Around 50 tools, grouped by pipeline stage. Each takes a workspace-relative `path`.

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

### stdio instead of SSE

`main.py` ships with SSE enabled. For a stdio client such as Claude Desktop, change the bottom of `main.py` to call `mcp.run()` with no arguments.

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

Each tool writes to a `results/` folder next to the file it read, named `<input>_<operation>.<ext>`, and returns the path. Inputs are never modified, and `results/` is git-ignored. Use `summarize_provenance` to see what has been produced and `export_artifact` to bundle it.

## Example

`example/` holds a magnetic-survey raster (`mag_survey.tif`) and a sample geochemistry CSV (`my_csv.csv`). Run the server with the workspace pointed at the repo root, then ask an agent to, for example, inspect the raster, compute a texture or gradient layer, cluster the geochemistry, and plot the result.

## Design notes

- **Fast startup:** heavy libraries (pandas, rasterio, matplotlib) are imported inside the functions that use them, so the server is ready in under two seconds.
- **DataFrame cache:** datasets are cached by resolved path with an mtime check and a memory budget, so repeated tool calls on the same file do not re-read it.
- **Workspace isolation:** `resolve_path` refuses any path outside `MCP_WORKSPACE_ROOT`.

## FAQ

**Which MCP clients does this work with?**
Any client that speaks MCP over SSE, and any stdio client after the one-line change in `main.py`. That includes Claude Desktop, Cline, Cursor, and others.

**Does the server need an API key?**
No. It runs locally and calls no external services. The model doing the reasoning lives on the client side, and that is where an API key (if any) is configured.

**Can it overwrite or damage my data?**
No. Inputs are read-only, every result is written to a `results/` folder, and any path outside the workspace is rejected.

**Is my data sent anywhere?**
The tools execute locally on your machine. Only what the agent chooses to read back (tool outputs and summaries) reaches whatever model your MCP client is configured to use.

**Is it geology-specific, or does it work on any tabular and raster data?**
The spatial, transform, clustering, anomaly, and plotting tools are generic. The cleaning tools (`validate_geology`, `parse_detection_limits`, `profile_geochem`) assume drillhole and assay conventions.

**How is this different from just asking a model to write pandas code?**
The tools are pre-built, input-validated, and consistent between runs. The agent composes them instead of regenerating fragile scripts each time, and `verify_claims` audits the numbers afterwards.