import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject as rio_reproject,
    Resampling,
)
from rasterio.mask import mask
import numpy as np

from .config import get_output_dir, save_raster


def reproject(
    path: str,
    target_crs: str = "EPSG:4326",
    x_col: str = None,
    y_col: str = None,
    source_crs: str = None,
):
    """
    Reproject a dataset to a new Coordinate Reference System (CRS).
    Saves the output to a 'results/' directory.

    Args:
        path: File path to CSV or spatial file.
        target_crs: The desired output CRS (default: EPSG:4326/LatLon).
        x_col: Column name for Easting/Longitude (required for CSV).
        y_col: Column name for Northing/Latitude (required for CSV).
        source_crs: If the input has no CRS (like a CSV), specify it here (e.g., 'EPSG:32750').
    """
    try:
        if path.endswith(".csv"):
            if not x_col or not y_col:
                return "Error: CSV files require 'x_col' and 'y_col' arguments."

            df = pd.read_csv(path)
            geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)

            if source_crs:
                gdf.set_crs(source_crs, inplace=True)

            if gdf.crs is None:
                return "Error: Input CSV has no CRS. Please provide 'source_crs' (e.g., 'EPSG:32750')."

        else:
            gdf = gpd.read_file(path)
            if source_crs:
                gdf.to_crs(source_crs, inplace=True)

        gdf_transformed = gdf.to_crs(target_crs)

        output_dir = get_output_dir(path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        safe_target = target_crs.replace(":", "-").replace(" ", "")
        output_path = os.path.join(output_dir, f"{name}_{safe_target}{ext}")

        if path.endswith(".csv"):
            gdf_transformed[x_col] = gdf_transformed.geometry.x
            gdf_transformed[y_col] = gdf_transformed.geometry.y
            pd.DataFrame(gdf_transformed.drop(columns="geometry")).to_csv(
                output_path, index=False
            )
        else:
            gdf_transformed.to_file(output_path, driver="GeoJSON")

        return {
            "status": "success",
            "original_crs": str(gdf.crs),
            "target_crs": target_crs,
            "output_path": output_path,
            "preview": gdf_transformed.head(3).to_dict(orient="records"),
        }

    except Exception as e:
        return f"Error reprojecting data: {str(e)}"


def resample(path: str, scale_factor: float = 0.5):
    """
    Resample a raster to a new resolution (e.g., 0.5 = half size, 2.0 = double size).
    Useful for aligning different map layers.
    """
    try:
        with rasterio.open(path) as src:
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)

            dst_transform = src.transform * src.transform.scale(
                (src.width / new_width), (src.height / new_height)
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {"transform": dst_transform, "width": new_width, "height": new_height}
            )

            output_dir = get_output_dir(path)
            name, ext = os.path.splitext(os.path.basename(path))
            output_path = os.path.join(
                output_dir, f"{name}_resampled_{scale_factor}x{ext}"
            )

            with rasterio.open(output_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    rio_reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear,
                    )

            return {
                "status": "success",
                "original_size": [src.width, src.height],
                "new_size": [new_width, new_height],
                "output_path": output_path,
            }

    except Exception as e:
        return f"Error resampling: {str(e)}"


def clip_to_extent(
    path: str, min_x: float, min_y: float, max_x: float, max_y: float, crs: str = None
):
    """
    Clip a raster to a specific bounding box (Extent).
    Args:
        crs: Coordinate system of the box (e.g., 'EPSG:4326').
             If None, assumes the box uses the same CRS as the raster file.
    """
    try:
        with rasterio.open(path) as src:
            bbox = box(min_x, min_y, max_x, max_y)

            box_crs = crs if crs else src.crs

            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=box_crs)

            if str(box_crs) != str(src.crs):
                geo = geo.to_crs(src.crs)

            out_image, out_transform = mask(src, geo.geometry, crop=True)
            out_meta = src.meta.copy()

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            output_dir = get_output_dir(path)
            name, ext = os.path.splitext(os.path.basename(path))
            output_path = os.path.join(output_dir, f"{name}_clipped{ext}")

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

            return {
                "status": "success",
                "output_path": output_path,
                "bbox_used": [min_x, min_y, max_x, max_y],
            }

    except Exception as e:
        return f"Error clipping: {str(e)}"


def align_grids(source_path: str, reference_path: str):
    """
    Align (warp) the source raster to match the exact grid, resolution, and CRS of the reference raster.
    Essential for stacking layers before Machine Learning.
    """
    try:
        with rasterio.open(reference_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height
            dst_profile = ref.profile.copy()

        with rasterio.open(source_path) as src:
            dst_profile.update(
                {
                    "crs": dst_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "driver": "GTiff",
                }
            )

            output_dir = get_output_dir(source_path)
            name = os.path.splitext(os.path.basename(source_path))[0]
            ref_name = os.path.splitext(os.path.basename(reference_path))[0]
            output_path = os.path.join(output_dir, f"{name}_aligned_to_{ref_name}.tif")

            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for i in range(1, src.count + 1):
                    rio_reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )

            return {
                "status": "success",
                "output_path": output_path,
                "aligned_to": reference_path,
                "new_dimensions": [dst_width, dst_height],
            }

    except Exception as e:
        return f"Error aligning grids: {str(e)}"
