import os
import time
import json
import zipfile
import pandas as pd
import geopandas as gpd


def summarize_provenance():
    """
    Generate a summary log of all artifacts created in the 'results' folder.
    Helps track what analysis steps have been performed in this session.
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        return "No results folder found. No analysis performed yet."

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
    with open(summary_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    return {
        "status": "success",
        "total_artifacts": len(artifacts),
        "log_path": summary_path,
        "inventory": artifacts,
    }


def export_artifact(path: str = "all", format: str = "zip"):
    """
    Package results for export/download.

    Args:
        path: Specific file path, or "all" to package the entire 'results' folder.
        format:
            - 'zip': Create a compressed archive (Good for downloading everything).
            - 'xlsx': Convert a CSV to Excel format.
            - 'geojson': Convert a CSV with coords to GeoJSON.
    """
    output_dir = "results"

    if format == "zip":
        timestamp = int(time.time())
        zip_name = f"geocluster_export_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_name)

        with zipfile.ZipFile(zip_path, "w") as zipf:
            if path == "all":
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        if file != zip_name:
                            zipf.write(os.path.join(root, file), file)
            else:
                if os.path.exists(path):
                    zipf.write(path, os.path.basename(path))
                else:
                    return f"Error: File {path} not found."

        return {"status": "success", "output_path": zip_path}

    elif format == "xlsx":
        if not path.endswith(".csv"):
            return "Error: xlsx export only works for CSV files."

        try:
            df = pd.read_csv(path)
            excel_path = path.replace(".csv", ".xlsx")
            df.to_excel(excel_path, index=False)
            return {"status": "success", "output_path": excel_path}
        except Exception as e:
            return f"Error converting to Excel: {str(e)}"

    elif format == "geojson":
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]

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
                return "Error: Could not auto-detect coordinate columns (x/lon/easting, y/lat/northing)."

            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
            )
            geojson_path = path.replace(".csv", ".geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")
            return {"status": "success", "output_path": geojson_path}
        except Exception as e:
            return f"Error converting to GeoJSON: {str(e)}"

    return "Error: Format not supported. Use 'zip', 'xlsx', or 'geojson'."
