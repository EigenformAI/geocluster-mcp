# NOTE: Heavy imports (pandas, geopandas) deferred to function bodies for fast MCP startup.

import os
import time
import json
import zipfile

from .config import atomic_output, default_results_dir, get_output_dir, resolve_path


def summarize_provenance(workspace_path: str = None):
    """List all artifacts in results folder with metadata."""
    if workspace_path:
        results_dir = get_output_dir(workspace_path)
    else:
        results_dir = default_results_dir()  # was "results", relative to the server cwd

    if not os.path.exists(results_dir):
        return f"No results folder found at {results_dir}. No analysis performed yet."

    artifacts = []
    for f in os.listdir(results_dir):
        path = os.path.join(results_dir, f)
        if os.path.isfile(path) and not f.startswith("."):
            stats = os.stat(path)

            ftype = "Unknown Artifact"
            name_lower = f.lower()
            if "kmeans" in name_lower or "dbscan" in name_lower:
                ftype = "Clustering Result"
            elif "hist" in name_lower:
                ftype = "Histogram Plot"
            elif "map" in name_lower:
                ftype = "Map Plot"
            elif "scatter" in name_lower:
                ftype = "Scatter Plot"
            elif "anom" in name_lower:
                ftype = "Anomaly Detection"
            elif "clean" in name_lower or "subset" in name_lower:
                ftype = "Cleaned Data"
            elif ".zip" in name_lower:
                ftype = "Exported Archive"

            artifacts.append(
                {
                    "filename": f,
                    "type": ftype,
                    "size_kb": round(stats.st_size / 1024, 2),
                    "created": time.ctime(stats.st_ctime),
                }
            )

    summary_path = os.path.join(results_dir, "provenance_log.json")
    with atomic_output(summary_path, keep_ext=False) as tmp, open(tmp, "w") as f:
        json.dump(artifacts, f, indent=2)

    return {
        "status": "success",
        "results_dir": results_dir,
        "total_artifacts": len(artifacts),
        "log_path": summary_path,
        "inventory": artifacts,
    }


def export_artifact(path: str = "all", format: str = "zip", workspace_path: str = None):
    """Export results. path: file or 'all'. format: 'zip', 'xlsx', 'geojson'."""
    if workspace_path:
        output_dir = get_output_dir(workspace_path)
    elif path != "all" and os.path.exists(path):
        output_dir = get_output_dir(path)
    else:
        output_dir = default_results_dir()  # was "results", relative to the server cwd

    # Validate path containment for non-"all" paths (X-1 invariant)
    resolved_file = None
    if path != "all":
        try:
            resolved_file = resolve_path(path)
        except ValueError as e:
            return f"Error: {str(e)}"

    if format == "zip":
        timestamp = int(time.time())
        zip_name = f"geocluster_export_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_name)

        if path == "all" and not os.path.exists(output_dir):
            return f"Error: Results folder not found at {output_dir}"
        if path != "all" and not os.path.exists(resolved_file):
            return f"Error: File {path} not found."
        # built under a hidden temp name: a failed export leaves no partial zip for the next 'all' export to pack
        with atomic_output(zip_path, keep_ext=False) as tmp, zipfile.ZipFile(tmp, "w") as zipf:
            if path == "all":
                for root, dirs, files in os.walk(output_dir):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for file in files:
                        if file == zip_name or file.startswith("."):  # temps of writes in progress
                            continue
                        try:
                            zipf.write(os.path.join(root, file), file)
                        except FileNotFoundError:  # replaced or removed while walking
                            continue
            else:
                zipf.write(resolved_file, os.path.basename(resolved_file))

        return {"status": "success", "output_path": zip_path}

    elif format == "xlsx":
        if not path.endswith(".csv"):
            return "Error: xlsx export only works for CSV files."

        try:
            import pandas as pd
            df = pd.read_csv(resolved_file)
            excel_path = resolved_file.replace(".csv", ".xlsx")
            df.to_excel(excel_path, index=False)
            return {"status": "success", "output_path": excel_path}
        except Exception as e:
            return f"Error converting to Excel: {str(e)}"

    elif format == "geojson":
        try:
            import pandas as pd
            import geopandas as gpd
            df = pd.read_csv(resolved_file)

            x_col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ["x", "lon", "longitude", "easting"]
                ),
                None,
            )
            y_col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ["y", "lat", "latitude", "northing"]
                ),
                None,
            )

            if not x_col or not y_col:
                return "Error: Could not auto-detect coordinate columns."

            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
            )
            geojson_path = resolved_file.replace(".csv", ".geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")
            return {"status": "success", "output_path": geojson_path}
        except Exception as e:
            return f"Error converting to GeoJSON: {str(e)}"

    return "Error: Format not supported. Use 'zip', 'xlsx', or 'geojson'."
