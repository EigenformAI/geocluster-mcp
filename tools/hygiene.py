import os
import pandas as pd
import rasterio
import numpy as np

from .config import resolve_path


def list_files(directory: str = "."):
    """List all available files in a directory to identify data sources."""
    try:
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory."

        files = os.listdir(directory)
        count = len(files)
        # Limit the number of files returned to prevent context overflow
        if count > 500:
            return {
                "files": files[:500],
                "count": count,
                "warning": f"Output truncated. Showing 500 of {count} files.",
            }

        return {"files": files, "count": count}
    except Exception as e:
        return f"Error listing files: {str(e)}"


def inspect_dataset(path: str, stats: bool = True, preview: bool = True):
    """
    Inspect a dataset to report structure, data types, and metadata.
    Does not modify data.
    """
    try:
        # Resolve the file path relative to the workspace root
        resolved_path = resolve_path(path)

        df = pd.read_csv(resolved_path)
        report = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "rows": len(df),
        }
        if stats:
            report["statistics"] = df.describe().to_dict()
        if preview:
            report["preview"] = df.head(5).to_dict(orient="records")
        return report
    except Exception as e:
        return f"Error: Could not read {path}. {str(e)}"


def check_missing(path: str):
    """Report the count and percentage of missing values per column."""
    try:
        # Resolve the file path relative to the workspace root
        resolved_path = resolve_path(path)

        df = pd.read_csv(resolved_path)
        missing_count = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        return {
            "missing_counts": missing_count.to_dict(),
            "missing_percentages": missing_pct.to_dict(),
        }
    except Exception as e:
        return f"Error: {str(e)}"


def inspect_raster(path: str, metadata: bool = True, histogram: bool = True):
    """
    Inspect a raster file (GeoTIFF) to report resolution, CRS, and value distribution.
    """
    try:
        # Resolve the file path relative to the workspace root
        resolved_path = resolve_path(path)

        with rasterio.open(resolved_path) as src:
            info = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": [x for x in src.transform],
                "nodata": src.nodata,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
            }
            if histogram:
                band1 = src.read(1)
                if src.nodata is not None:
                    data = band1[band1 != src.nodata]
                else:
                    data = band1
                info["statistics"] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std_dev": float(np.std(data)),
                }
            return info
    except Exception as e:
        return f"Error reading raster: {str(e)}"
