from fastmcp import FastMCP

# Import tools from your modules
from tools.hygiene import list_files, inspect_dataset, check_missing, inspect_raster
from tools.spatial import reproject, resample, clip_to_extent, align_grids
from tools.transforms import normalize, standardize, log_transform, smooth
from tools.features import (
    select_bands,
    band_math,
    compute_gradient,
    texture_features,
    select_columns,
    compute_ratios,
    aggregate,
)
from tools.anomaly import compute_anomaly, threshold, rank_by_metric
from tools.clustering import reduce_dimensions, cluster
from tools.visualization import plot_scatter, plot_map, plot_histogram, plot_clusters
from tools.provenance import summarize_provenance, export_artifact

mcp = FastMCP("Geocluster MCP")

# --- Register Tools ---

# Section A: Hygiene
mcp.tool()(list_files)
mcp.tool()(inspect_dataset)
mcp.tool()(inspect_raster)
mcp.tool()(check_missing)


# Section B: Spatial
mcp.tool()(reproject)
mcp.tool()(resample)
mcp.tool()(clip_to_extent)
mcp.tool()(align_grids)


# Section C: Transforms
mcp.tool()(normalize)
mcp.tool()(standardize)
mcp.tool()(log_transform)
mcp.tool()(smooth)


# Section D: Features
mcp.tool()(select_bands)
mcp.tool()(band_math)
mcp.tool()(compute_gradient)
mcp.tool()(texture_features)
mcp.tool()(select_columns)
mcp.tool()(compute_ratios)
mcp.tool()(aggregate)


# Section E: Anomaly Detection
mcp.tool()(compute_anomaly)
mcp.tool()(threshold)
mcp.tool()(rank_by_metric)


# Section F: Clustering
mcp.tool()(cluster)
mcp.tool()(reduce_dimensions)


# Section G: Visualization
mcp.tool()(plot_map)
mcp.tool()(plot_scatter)
mcp.tool()(plot_histogram)
mcp.tool()(plot_clusters)


# Section H: Provenance & Export
mcp.tool()(summarize_provenance)
mcp.tool()(export_artifact)

if __name__ == "__main__":
    mcp.run()
