import logging
import os

from fastmcp import FastMCP

# Import tools from your modules
from tools.hygiene import list_files, inspect_dataset, check_missing, inspect_raster, inspect_specific_columns, profile_geochem, query_data
from tools.spatial import reproject, resample, clip_to_extent, align_grids
from tools.transforms import normalize, standardize, log_transform, smooth, pivot, melt, merge_datasets, filter_rows, convert_dtype
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
from tools.verification import verify_claims
from tools.cleaning import (
    validate_geology,
    detect_cleaning_issues,
    fix_decimals,
    parse_detection_limits,
    remove_duplicates,
    standardize_terms,
)
from tools.voxel import (
    voxel_init_grid,
    voxel_get_grid,
    voxel_add_point,
    voxel_add_line,
    voxel_add_box,
    voxel_upsert_geometry,
    voxel_set_layer_array,
    voxel_probe_region,
    voxel_list_layers,
    voxel_history,
    voxel_export_bundle,
)

# Section K (training data collection: the five *_training_* tools plus the
# middleware that records every tool call into the active session) is parked.
# Set GEOCLUSTER_TRAINING_ENABLED=1 to register it; default off so the server
# and the specialists never depend on the training package.
TRAINING_ENABLED = os.environ.get("GEOCLUSTER_TRAINING_ENABLED", "0") == "1"

mcp = FastMCP("Geocluster MCP")

_training_log = logging.getLogger("geocluster.training")


def _install_training(mcp_server: FastMCP) -> None:
    """Register Section K tools and the capture middleware (opt-in)."""
    from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

    from tools.training import (
        start_training_session,
        generate_hypotheses,
        evaluate_hypotheses,
        end_training_session,
        get_training_stats,
    )
    from training.session import get_session_manager

    # Tools that drive the training lifecycle itself — skip to avoid recursion
    # and to keep the session's tool_calls log focused on the analysis trace
    # that the hypothesis-generation prompt will see.
    lifecycle_tools = frozenset({
        "start_training_session",
        "generate_hypotheses",
        "evaluate_hypotheses",
        "end_training_session",
        "get_training_stats",
    })

    def _serialize_tool_result(result):
        """Convert a FastMCP ToolResult to a JSON-safe dict for session storage.

        Prefers structured_content (dict) because evaluate_hypotheses expects
        keys like 'clusters', 'anomaly_count', 'pairs' for mechanical grading.
        Falls back to concatenated text content, then str(result).
        """
        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        content = getattr(result, "content", None)
        if isinstance(content, list):
            texts = [getattr(b, "text", None) for b in content if getattr(b, "text", None)]
            if texts:
                return {"text": "\n".join(texts)}
        return {"repr": str(result)}

    class TrainingCaptureMiddleware(Middleware):
        """Appends every non-lifecycle tool invocation to the active training session.

        No-ops when no session is active or when TRAINING_CAPTURE_ENABLED=0.
        Never raises — recording failures are logged and swallowed so analysis
        tool results always reach the specialist unchanged.
        """

        async def on_call_tool(
            self, context: MiddlewareContext, call_next: CallNext
        ):
            result = await call_next(context)

            if os.environ.get("TRAINING_CAPTURE_ENABLED", "1") != "1":
                return result

            tool_name = getattr(context.message, "name", None)
            if not tool_name or tool_name in lifecycle_tools:
                return result

            try:
                manager = get_session_manager()
                session = manager.get_active_session()
                if session is None or session.finalized:
                    return result
                args = getattr(context.message, "arguments", {}) or {}
                session.record_call(tool_name, dict(args), _serialize_tool_result(result))
            except Exception as exc:
                _training_log.warning(
                    "TrainingCaptureMiddleware: failed to record %s: %s",
                    tool_name,
                    exc,
                )

            return result

    mcp_server.add_middleware(TrainingCaptureMiddleware())

    # Section K: Training Data Collection
    mcp_server.tool()(start_training_session)
    mcp_server.tool()(generate_hypotheses)
    mcp_server.tool()(evaluate_hypotheses)
    mcp_server.tool()(end_training_session)
    mcp_server.tool()(get_training_stats)


# --- Register Tools ---

# Section A: Hygiene
mcp.tool()(list_files)
# mcp.tool()(get_dataset_schema)
mcp.tool()(inspect_dataset)
mcp.tool()(inspect_specific_columns)
mcp.tool()(inspect_raster)
mcp.tool()(check_missing)
mcp.tool()(profile_geochem)
mcp.tool()(query_data)


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
mcp.tool()(pivot)
mcp.tool()(melt)
mcp.tool()(merge_datasets)
mcp.tool()(filter_rows)
mcp.tool()(convert_dtype)


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


# Section I: Data Cleaning
mcp.tool()(validate_geology)
mcp.tool()(detect_cleaning_issues)
mcp.tool()(fix_decimals)
mcp.tool()(parse_detection_limits)
mcp.tool()(remove_duplicates)
mcp.tool()(standardize_terms)


# Section J: Verification
mcp.tool()(verify_claims)


# Section K: Training Data Collection (opt-in, see TRAINING_ENABLED above)
if TRAINING_ENABLED:
    _install_training(mcp)


# Section L: Voxel store + viewer bundle (voxel specialist only — names are the ACL)
mcp.tool()(voxel_init_grid)
mcp.tool()(voxel_get_grid)
mcp.tool()(voxel_add_point)
mcp.tool()(voxel_add_line)
mcp.tool()(voxel_add_box)
mcp.tool()(voxel_upsert_geometry)
mcp.tool()(voxel_set_layer_array)
mcp.tool()(voxel_probe_region)
mcp.tool()(voxel_list_layers)
mcp.tool()(voxel_history)
mcp.tool()(voxel_export_bundle)

if __name__ == "__main__":
    # mcp.run()

    print("Starting Geocluster MCP on http://0.0.0.0:7654/sse")
    mcp.run(transport="sse", host="0.0.0.0", port=7654)
