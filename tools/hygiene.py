import os
import pandas as pd
import rasterio
import numpy as np
from typing import List

from .config import resolve_path, read_tabular


def _resolve_columns(
    requested: List[str], actual_columns
) -> tuple[list[str], list[str]]:
    """Match requested column names case-insensitively. Returns (matched, not_found)."""
    lookup = {c.lower(): c for c in actual_columns}
    matched = []
    not_found = []
    for col in requested:
        real = lookup.get(col.lower())
        if real:
            matched.append(real)
        else:
            not_found.append(col)
    return matched, not_found


def list_files(directory: str = "."):
    """List files in a directory."""
    try:
        resolved = resolve_path(directory)
        if not os.path.isdir(resolved):
            return f"Error: '{directory}' is not a directory."

        files = os.listdir(resolved)
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


def inspect_dataset(path: str):
    """Read dataset structure: columns, row count, optional sample."""
    try:
        resolved_path = resolve_path(path)
        ext = os.path.splitext(resolved_path)[1].lower()

        df = read_tabular(resolved_path, nrows=2)

        if ext == ".csv":
            with open(resolved_path, "r") as f:
                row_count = sum(1 for _ in f) - 1
        else:
            row_count = len(read_tabular(resolved_path))

        report = {"columns": list(df.columns), "row_count": row_count}

        return report
    except Exception as e:
        return f"Error: {str(e)}"


def inspect_specific_columns(
    path: str,
    columns: List[str],
    get_stats: bool = False,
    get_unique_values: bool = False,
    get_value_counts: bool = False,
    max_unique: int = 30,
):
    """Column summary. get_stats for min/max, get_unique_values for value lists, get_value_counts for top values by frequency."""
    try:
        if not columns:
            return "Error: Specify columns. Run inspect_dataset() first."

        resolved_path = resolve_path(path)

        df_header = read_tabular(resolved_path, nrows=0)
        resolved_cols, missing = _resolve_columns(columns, df_header.columns)
        if missing:
            return f"Error: Columns {missing} not found."

        df = read_tabular(resolved_path, usecols=resolved_cols)

        report = {"row_count": len(df)}

        for col in resolved_cols:
            col_info = {
                "unique": int(df[col].nunique()),
                "nulls": int(df[col].isnull().sum()),
            }

            if pd.api.types.is_numeric_dtype(df[col]) and get_stats:
                col_info["min"] = round(float(df[col].min()), 2)
                col_info["max"] = round(float(df[col].max()), 2)

            if get_unique_values and not pd.api.types.is_numeric_dtype(df[col]):
                uniques = df[col].dropna().unique().tolist()
                col_info["total_unique"] = len(uniques)
                col_info["values"] = uniques[:max_unique]

            if get_value_counts and not pd.api.types.is_numeric_dtype(df[col]):
                counts = df[col].value_counts().head(max_unique)
                col_info["top_values"] = {str(k): int(v) for k, v in counts.items()}

            report[col] = col_info

        return report

    except Exception as e:
        return f"Error inspecting columns: {str(e)}"


def check_missing(path: str):
    """Report columns with missing values (skips complete columns)."""
    try:
        resolved_path = resolve_path(path)
        df = read_tabular(resolved_path)
        missing_pct = ((df.isnull().sum() / len(df)) * 100).round(1)
        has_missing = missing_pct[missing_pct > 0]
        if has_missing.empty:
            return {"status": "no_missing", "total_columns": len(df.columns)}
        return {
            "columns_with_missing": has_missing.to_dict(),
            "total_columns": len(df.columns),
        }
    except Exception as e:
        return f"Error: {str(e)}"


def inspect_raster(path: str, verbose: bool = False):
    """Inspect raster (GeoTIFF). Set verbose=true for bounds, transform, and band stats."""
    try:
        resolved_path = resolve_path(path)
        with rasterio.open(resolved_path) as src:
            info = {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
            }
            if verbose:
                info["transform"] = [x for x in src.transform]
                info["nodata"] = src.nodata
                info["bounds"] = {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                }
                band1 = src.read(1)
                if src.nodata is not None:
                    data = band1[band1 != src.nodata]
                else:
                    data = band1
                info["statistics"] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": round(float(np.mean(data)), 2),
                    "std": round(float(np.std(data)), 2),
                }
            return info
    except Exception as e:
        return f"Error reading raster: {str(e)}"


def profile_geochem(
    path: str,
    from_col: str,
    to_col: str,
    element_cols: List[str],
):
    """Depth-weighted stats for geochem columns. Returns weighted mean, max, and grade-thickness."""
    try:
        resolved_path = resolve_path(path)
        df = read_tabular(resolved_path)

        all_requested = [from_col, to_col] + element_cols
        resolved_cols, missing = _resolve_columns(all_requested, df.columns)
        if missing:
            return f"Error: Columns {missing} not found."

        # Map back to resolved names
        from_col = resolved_cols[0]
        to_col = resolved_cols[1]
        element_cols = resolved_cols[2:]

        df = df.dropna(subset=[from_col, to_col])
        thickness = df[to_col] - df[from_col]

        if (thickness <= 0).any():
            return "Error: Some intervals have zero or negative thickness. Check from/to columns."

        total_thickness = float(thickness.sum())
        report = {
            "total_intervals": len(df),
            "total_thickness_m": round(total_thickness, 2),
            "depth_range": [
                round(float(df[from_col].min()), 2),
                round(float(df[to_col].max()), 2),
            ],
        }

        for col in element_cols:
            if col not in df.columns:
                continue
            vals = df[col]
            valid = vals.notna() & thickness.notna()
            v = vals[valid]
            t = thickness[valid]

            if t.sum() == 0:
                continue

            weighted_mean = float((v * t).sum() / t.sum())
            max_val = float(v.max())
            max_gt = float((v * t).max())

            report[col] = {
                "weighted_mean": round(weighted_mean, 4),
                "max": round(max_val, 4),
                "max_grade_thickness": round(max_gt, 4),
                "nulls": int(vals.isnull().sum()),
            }

        return report

    except Exception as e:
        return f"Error: {str(e)}"
