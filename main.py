import logging
import sys
import functools
from typing import Any, Callable
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

# Logger setup
logger = logging.getLogger("geocluster_mcp")
logger.setLevel(logging.DEBUG)

# File handler for logging to a file
file_handler = logging.FileHandler("geocluster_mcp.log", mode='w')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Stream handler for logging to stderr
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.DEBUG)
stream_format = logging.Formatter('[MCP-LOG] %(levelname)s: %(message)s')
stream_handler.setFormatter(stream_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

mcp = FastMCP("Geocluster MCP")

def log_tool_call(func: Callable) -> Callable:
    """Decorator to log tool calls and results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Calling tool '{func_name}' with args={args} kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Tool '{func_name}' completed. Result: {str(result)[:500]}...") # Truncate long results
            return result
        except Exception as e:
            logger.error(f"Tool '{func_name}' failed with error: {e}", exc_info=True)
            raise e
    return wrapper

# --- Register Tools ---

# # Section A: Hygiene
# mcp.tool()(list_files)
# mcp.tool()(inspect_dataset)
# mcp.tool()(inspect_raster)
# mcp.tool()(check_missing)


# # Section B: Spatial
# mcp.tool()(reproject)
# mcp.tool()(resample)
# mcp.tool()(clip_to_extent)
# mcp.tool()(align_grids)


# # Section C: Transforms
# mcp.tool()(normalize)
# mcp.tool()(standardize)
# mcp.tool()(log_transform)
# mcp.tool()(smooth)


# # Section D: Features
# mcp.tool()(select_bands)
# mcp.tool()(band_math)
# mcp.tool()(compute_gradient)
# mcp.tool()(texture_features)
# mcp.tool()(select_columns)
# mcp.tool()(compute_ratios)
# mcp.tool()(aggregate)


# # Section E: Anomaly Detection
# mcp.tool()(compute_anomaly)
# mcp.tool()(threshold)
# mcp.tool()(rank_by_metric)


# # Section F: Clustering
# mcp.tool()(cluster)
# mcp.tool()(reduce_dimensions)


# # Section G: Visualization
# mcp.tool()(plot_map)
# mcp.tool()(plot_scatter)
# mcp.tool()(plot_histogram)
# mcp.tool()(plot_clusters)


# # Section H: Provenance & Export
# mcp.tool()(summarize_provenance)
# mcp.tool()(export_artifact)

# Section A: Hygiene
mcp.tool()(log_tool_call(list_files))
mcp.tool()(log_tool_call(inspect_dataset))
mcp.tool()(log_tool_call(inspect_raster))
mcp.tool()(log_tool_call(check_missing))


# Section B: Spatial
mcp.tool()(log_tool_call(reproject))
mcp.tool()(log_tool_call(resample))
mcp.tool()(log_tool_call(clip_to_extent))
mcp.tool()(log_tool_call(align_grids))


# Section C: Transforms
mcp.tool()(log_tool_call(normalize))
mcp.tool()(log_tool_call(standardize))
mcp.tool()(log_tool_call(log_transform))
mcp.tool()(log_tool_call(smooth))


# Section D: Features
mcp.tool()(log_tool_call(select_bands))
mcp.tool()(log_tool_call(band_math))
mcp.tool()(log_tool_call(compute_gradient))
mcp.tool()(log_tool_call(texture_features))
mcp.tool()(log_tool_call(select_columns))
mcp.tool()(log_tool_call(compute_ratios))
mcp.tool()(log_tool_call(aggregate))


# Section E: Anomaly Detection
mcp.tool()(log_tool_call(compute_anomaly))
mcp.tool()(log_tool_call(threshold))
mcp.tool()(log_tool_call(rank_by_metric))


# Section F: Clustering
mcp.tool()(log_tool_call(cluster))
mcp.tool()(log_tool_call(reduce_dimensions))


# Section G: Visualization
mcp.tool()(log_tool_call(plot_map))
mcp.tool()(log_tool_call(plot_scatter))
mcp.tool()(log_tool_call(plot_histogram))
mcp.tool()(log_tool_call(plot_clusters))


# Section H: Provenance & Export
mcp.tool()(log_tool_call(summarize_provenance))
mcp.tool()(log_tool_call(export_artifact))

if __name__ == "__main__":
    # mcp.run()

    print("Starting Geocluster MCP on http://0.0.0.0:7654/sse")
    mcp.run(transport="sse", host="0.0.0.0", port=7654)
