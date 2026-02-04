import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rasterio
from rasterio.plot import show
import os
import numpy as np

plt.switch_backend("Agg")


def _save_plot(path_original, suffix):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(path_original)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_{suffix}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def plot_histogram(path: str, column: str, bins: int = 30):
    """
    Generate a histogram to visualize data distribution.
    Useful for checking if data is Normal (Bell curve) or Log-Normal.
    """
    try:
        df = pd.read_csv(path)
        if column not in df.columns:
            return f"Error: Column {column} not found."

        plt.figure(figsize=(10, 6))
        sns.histplot(df[column].dropna(), bins=bins, kde=True, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        output_path = _save_plot(path, f"hist_{column}")
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        return f"Error plotting histogram: {str(e)}"


def plot_scatter(path: str, x_col: str, y_col: str, color_col: str = None):
    """
    Generate a Scatter Plot (X vs Y).
    Optional: 'color_col' to color points by value (e.g., Gold grade) or category (Cluster).
    """
    try:
        df = pd.read_csv(path)
        if x_col not in df.columns or y_col not in df.columns:
            return f"Error: Columns not found."

        plt.figure(figsize=(10, 8))

        if color_col:
            if color_col not in df.columns:
                return f"Error: Color column {color_col} not found."
            is_categorical = df[color_col].dtype == "object" or "cluster" in color_col
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=color_col,
                palette="viridis",
                style=color_col if is_categorical else None,
            )
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, color="blue", alpha=0.6)

        plt.title(f"{x_col} vs {y_col}")
        plt.grid(True, linestyle="--", alpha=0.5)

        suffix = f"scatter_{x_col}_{y_col}"
        output_path = _save_plot(path, suffix)
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        return f"Error plotting scatter: {str(e)}"


def plot_map(path: str, x_col: str = None, y_col: str = None, color_col: str = None):
    """
    Visualize Geospatial Data.
    - If path is Raster (.tif): Displays the image/grid.
    - If path is CSV: Plots X/Y coordinates as a map.
    """
    try:
        plt.figure(figsize=(10, 10))

        if path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                show(src, title=os.path.basename(path), cmap="magma")
                suffix = "map_raster"
        else:
            df = pd.read_csv(path)
            if not x_col or not y_col:
                return "Error: For CSV maps, you must specify x_col (Easting/Lon) and y_col (Northing/Lat)."

            if color_col:
                scatter = plt.scatter(
                    df[x_col],
                    df[y_col],
                    c=df[color_col],
                    cmap="viridis",
                    s=50,
                    alpha=0.8,
                )
                plt.colorbar(scatter, label=color_col)
            else:
                plt.scatter(df[x_col], df[y_col], c="red", s=50, alpha=0.8)

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Map: {os.path.basename(path)}")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            suffix = "map_vector"

        output_path = _save_plot(path, suffix)
        return {"status": "success", "output_path": output_path}

    except Exception as e:
        return f"Error plotting map: {str(e)}"


def plot_clusters(path: str, x_col: str, y_col: str, cluster_col: str):
    """
    Visualize clustering results (K-Means/DBSCAN) on a 2D Scatter Plot.
    Automatically treats the cluster column as categorical (distinct colors).
    """
    try:
        df = pd.read_csv(path)

        required = [x_col, y_col, cluster_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return f"Error: Columns not found: {missing}"

        plt.figure(figsize=(10, 8))

        df[cluster_col] = df[cluster_col].astype(str)

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=cluster_col,
            palette="tab10",
            s=80,
            edgecolor="black",
            alpha=0.7,
        )

        plt.title(f"Cluster Analysis: {cluster_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")

        output_path = _save_plot(path, f"cluster_viz_{cluster_col}")
        return {"status": "success", "output_path": output_path}

    except Exception as e:
        return f"Error plotting clusters: {str(e)}"
