import rasterio
import os
import numpy as np
import pandas as pd

from skimage.filters import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from .config import save_raster, save_csv, get_output_dir


def select_bands(path: str, indices: list[int]):
    """
    Create a new raster containing only the selected bands.
    Useful for extracting specific channels (e.g., Red/Near-Infrared) from satellite imagery.

    Args:
        path: Path to the input raster (GeoTIFF).
        indices: List of band numbers to keep (1-based index, e.g., [1, 3]).
    """
    try:
        with rasterio.open(path) as src:
            max_band = src.count
            if any(i < 1 or i > max_band for i in indices):
                return f"Error: Invalid indices {indices}. Input raster only has {max_band} bands."

            data = src.read(indices)

            meta = src.meta.copy()
            meta.update({"count": len(indices)})

            suffix = "bands_" + "-".join(map(str, indices))
            output_path = save_raster(data, meta, path, suffix)

            return {
                "status": "success",
                "output_path": output_path,
                "original_band_count": max_band,
                "selected_bands": indices,
            }

    except Exception as e:
        return f"Error selecting bands: {str(e)}"


def band_math(path: str, expression: str):
    """
    Apply mathematical expression to raster bands.
    Use 'b1', 'b2', etc. to refer to bands.
    Example: "(b1 - b2) / (b1 + b2)" or "np.log(b1)"
    """
    try:
        with rasterio.open(path) as src:
            meta = src.meta.copy()
            bands = {f"b{i}": src.read(i) for i in range(1, src.count + 1)}

            allowed_names = {"__builtins__": None, "np": np}
            allowed_names.update(bands)

            try:
                result = eval(expression, allowed_names)
            except Exception as math_err:
                return f"Error in math expression: {str(math_err)}"

            if isinstance(result, (int, float)):
                result = np.full(src.shape, result, dtype=np.float32)

            meta.update({"dtype": "float32", "count": 1})

            safe_expr = "".join(c for c in expression if c.isalnum())
            output_path = save_raster(
                result.astype(np.float32), meta, path, f"math_{safe_expr}"
            )

            return {
                "status": "success",
                "expression": expression,
                "output_path": output_path,
            }

    except Exception as e:
        return f"Error: {str(e)}"


def compute_gradient(path: str, method: str = "sobel"):
    """
    Detect edges and structural changes using gradient filters.
    Useful for finding fault lines/contacts in magnetic data.

    Args:
        path: Path to raster file.
        method: Currently supports 'sobel'.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            meta = src.meta.copy()

            if method == "sobel":
                edges = sobel(data)
            else:
                return "Error: Only 'sobel' method is supported currently."

            meta.update({"dtype": "float32"})

            output_path = save_raster(edges.astype(np.float32), meta, path, "gradient")

            return {"status": "success", "method": method, "output_path": output_path}

    except Exception as e:
        return f"Error computing gradient: {str(e)}"


def texture_features(path: str, method: str = "entropy"):
    """
    Compute texture features (Roughness/Complexity).
    Method: 'entropy' (complexity).
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            meta = src.meta.copy()

            valid_data = np.nan_to_num(data)

            d_min, d_max = valid_data.min(), valid_data.max()
            if d_max == d_min:
                return "Error: Data is flat (constant value), cannot compute texture."
            norm_data = (valid_data - d_min) / (d_max - d_min)

            image_ubyte = img_as_ubyte(norm_data)

            if method == "entropy":
                result = entropy(image_ubyte, disk(3))
            else:
                return "Error: Only 'entropy' method supported currently."

            meta.update({"dtype": "float32"})
            output_path = save_raster(
                result.astype(np.float32), meta, path, f"texture_{method}"
            )

            return {"status": "success", "method": method, "output_path": output_path}

    except Exception as e:
        return f"Error computing texture: {str(e)}"


def select_columns(path: str, columns: list[str]):
    """
    Create a new CSV containing ONLY the selected columns.
    Useful for removing noise/irrelevant data before Machine Learning.
    """
    try:
        df = pd.read_csv(path)

        missing = [c for c in columns if c not in df.columns]
        if missing:
            return f"Error: Columns not found: {missing}"

        df_new = df[columns].copy()
        output_path = save_csv(df_new, path, "subset")
        return {"status": "success", "output_path": output_path}

    except Exception as e:
        return f"Error: {str(e)}"


def compute_ratios(path: str, pairs: list[str]):
    """
    Compute geochemical ratios (e.g., 'Au_ppb/Cu_ppm').
    Input format: List of strings like ["Au_ppb/Cu_ppm"]
    """
    try:
        df = pd.read_csv(path)
        df_new = df.copy()
        created_cols = []

        for pair in pairs:
            if "/" not in pair:
                return f"Error: Invalid format '{pair}'. Use 'Numerator/Denominator' (e.g., 'Au/Cu')."

            num, den = pair.split("/")
            if num not in df.columns or den not in df.columns:
                return f"Error: Columns for pair {pair} not found."

            denominator = df[den].replace(0, float("nan"))
            col_name = f"{num}_over_{den}"
            df_new[col_name] = df[num] / denominator
            created_cols.append(col_name)

        output_path = save_csv(df_new, path, "ratios")

        return {
            "status": "success",
            "new_ratios": created_cols,
            "output_path": output_path,
        }
    except Exception as e:
        return f"Error: {str(e)}"


def aggregate(
    path: str,
    spatial_op: str,
    statistic: str = "mean",
    x_col: str = "Easting",
    y_col: str = "Northing",
):
    """
    Aggregate data spatially (grid binning) or by category.

    Args:
        spatial_op:
            - "grid:50" -> Bin coordinates into 50m blocks (Spatial).
            - "col:RockType" -> Group by a specific column (Categorical).
        statistic: 'mean', 'median', 'max', 'min', 'sum', 'count'.
        x_col/y_col: Used only if spatial_op is 'grid:...'.
    """
    try:
        df = pd.read_csv(path)

        if spatial_op.startswith("grid:"):
            try:
                resolution = float(spatial_op.split(":")[1])
            except:
                return "Error: Invalid grid format. Use 'grid:50' (number)."

            if x_col not in df.columns or y_col not in df.columns:
                return f"Error: Coordinate columns '{x_col}'/'{y_col}' not found."

            df["X_bin"] = (df[x_col] / resolution).round() * resolution
            df["Y_bin"] = (df[y_col] / resolution).round() * resolution

            group_cols = ["X_bin", "Y_bin"]

        elif spatial_op.startswith("col:"):
            col_name = spatial_op.split(":")[1]
            if col_name not in df.columns:
                return f"Error: Column '{col_name}' not found."
            group_cols = [col_name]

        else:
            return "Error: spatial_op must start with 'grid:' or 'col:'"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        numeric_cols = [
            c for c in numeric_cols if c not in group_cols and c not in [x_col, y_col]
        ]

        if statistic == "mean":
            df_agg = df.groupby(group_cols)[numeric_cols].mean().reset_index()
        elif statistic == "median":
            df_agg = df.groupby(group_cols)[numeric_cols].median().reset_index()
        elif statistic == "max":
            df_agg = df.groupby(group_cols)[numeric_cols].max().reset_index()
        elif statistic == "sum":
            df_agg = df.groupby(group_cols)[numeric_cols].sum().reset_index()
        elif statistic == "count":
            df_agg = (
                df.groupby(group_cols)[numeric_cols[0]]
                .count()
                .reset_index()
                .rename(columns={numeric_cols[0]: "count"})
            )
        else:
            return "Error: Statistic not supported. Use mean/median/max/sum."

        safe_op = spatial_op.replace(":", "_")
        output_path = save_csv(df_agg, path, f"agg_{safe_op}_{statistic}")

        return {
            "status": "success",
            "original_rows": len(df),
            "aggregated_rows": len(df_agg),
            "output_path": output_path,
        }

    except Exception as e:
        return f"Error aggregating: {str(e)}"
