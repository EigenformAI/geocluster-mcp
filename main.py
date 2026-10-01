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

# Shared server (MCP_TRANSPORT=http, one process for every conversation): the per-call run folder and worker threads
# live in tools/runctx.py. Registered first, so it is the outermost middleware (FastMCP applies them in reverse): run_dir
# is popped before any other middleware or FastMCP's argument validation sees it.
from tools import dataframe_cache, runctx  # noqa: E402
from tools.runctx import RunDirMiddleware, threaded  # noqa: E402

mcp.add_middleware(RunDirMiddleware())


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    """Readiness for supervisors. Never probe /mcp itself: each request without a session id opens a session that is
    never closed."""
    from starlette.responses import JSONResponse

    return JSONResponse({"ok": True, "mode": "shared" if runctx.SHARED else "single", "in_flight": runctx.in_flight()})


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
mcp.tool()(threaded(list_files))
# mcp.tool()(get_dataset_schema)
mcp.tool()(threaded(inspect_dataset))
mcp.tool()(threaded(inspect_specific_columns))
mcp.tool()(threaded(inspect_raster))
mcp.tool()(threaded(check_missing))
mcp.tool()(threaded(profile_geochem))
mcp.tool()(threaded(query_data))


# Section B: Spatial
mcp.tool()(threaded(reproject))
mcp.tool()(threaded(resample))
mcp.tool()(threaded(clip_to_extent))
mcp.tool()(threaded(align_grids))


# Section C: Transforms
mcp.tool()(threaded(normalize))
mcp.tool()(threaded(standardize))
mcp.tool()(threaded(log_transform))
mcp.tool()(threaded(smooth))
mcp.tool()(threaded(pivot))
mcp.tool()(threaded(melt))
mcp.tool()(threaded(merge_datasets))
mcp.tool()(threaded(filter_rows))
mcp.tool()(threaded(convert_dtype))


# Section D: Features
mcp.tool()(threaded(select_bands))
mcp.tool()(threaded(band_math))
mcp.tool()(threaded(compute_gradient))
mcp.tool()(threaded(texture_features))
mcp.tool()(threaded(select_columns))
mcp.tool()(threaded(compute_ratios))
mcp.tool()(threaded(aggregate))


# Section E: Anomaly Detection
mcp.tool()(threaded(compute_anomaly))
mcp.tool()(threaded(threshold))
mcp.tool()(threaded(rank_by_metric))


# Section F: Clustering
mcp.tool()(threaded(cluster))
mcp.tool()(threaded(reduce_dimensions))


# Section G: Visualization
mcp.tool()(threaded(plot_map))
mcp.tool()(threaded(plot_scatter))
mcp.tool()(threaded(plot_histogram))
mcp.tool()(threaded(plot_clusters))


# Section H: Provenance & Export
mcp.tool()(threaded(summarize_provenance))
mcp.tool()(threaded(export_artifact))


# Section I: Data Cleaning
mcp.tool()(threaded(validate_geology))
mcp.tool()(threaded(detect_cleaning_issues))
mcp.tool()(threaded(fix_decimals))
mcp.tool()(threaded(parse_detection_limits))
mcp.tool()(threaded(remove_duplicates))
mcp.tool()(threaded(standardize_terms))


# Section J: Verification
mcp.tool()(threaded(verify_claims))


# Section K: Training Data Collection (opt-in, see TRAINING_ENABLED above)
if TRAINING_ENABLED:
    _install_training(mcp)


# Section L: Voxel store + viewer bundle (voxel specialist only — names are the ACL)
mcp.tool()(threaded(voxel_init_grid))
mcp.tool()(threaded(voxel_get_grid))
mcp.tool()(threaded(voxel_add_point))
mcp.tool()(threaded(voxel_add_line))
mcp.tool()(threaded(voxel_add_box))
mcp.tool()(threaded(voxel_upsert_geometry))
mcp.tool()(threaded(voxel_set_layer_array))
mcp.tool()(threaded(voxel_probe_region))
mcp.tool()(threaded(voxel_list_layers))
mcp.tool()(threaded(voxel_history))
mcp.tool()(threaded(voxel_export_bundle))

def configure(env=os.environ) -> dict:
    """
    Transport, host and port from the environment. Exits with a message on a combination that is unsafe for the
    shared server (MCP_TRANSPORT=http, one process serving every conversation).

    SSE (the default) is unchanged: api 2's per-conversation servers (GEOCLUSTER_RUN_DIR, MCP_HOST=127.0.0.1) and the
    host CLI (0.0.0.0:7654).
    """
    raw = env.get("MCP_TRANSPORT", "sse")
    transport = raw.strip().lower()
    if transport not in ("sse", "http"):
        raise SystemExit(f"MCP_TRANSPORT must be 'sse' or 'http', got {raw!r}")
    port = int(env.get("MCP_PORT", "7654"))
    if transport == "sse":
        return {"transport": "sse", "host": env.get("MCP_HOST", "0.0.0.0"), "port": port}

    host = env.get("MCP_HOST", "127.0.0.1")
    problems = []
    if host not in ("127.0.0.1", "localhost", "::1"):
        problems.append(f"MCP_HOST must be a loopback address, got {host!r} (no auth, and each call names its folder)")
    if env.get("GEOCLUSTER_RUN_DIR", "").strip():
        problems.append("unset GEOCLUSTER_RUN_DIR (every call brings its own run_dir)")
    if env.get("GEOCLUSTER_TRAINING_ENABLED", "0") == "1":
        problems.append("unset GEOCLUSTER_TRAINING_ENABLED (one training session would record every conversation)")
    for name in ("MCP_TOOL_THREADS", "MCP_TOOL_THREADS_PER_RUN"):
        if not env.get(name, "1").isdigit() or int(env.get(name, "1")) < 1:
            problems.append(f"{name} must be a positive integer, got {env.get(name)!r}")
    if problems:
        raise SystemExit("geocluster-mcp with MCP_TRANSPORT=http: " + "; ".join(problems))
    return {"transport": "http", "host": host, "port": port}


if __name__ == "__main__":
    settings = configure()
    host, port = settings["host"], settings["port"]
    if settings["transport"] == "http":
        # One server for every conversation (OpenCode 2.0 has no SSE client). Options passed explicitly, so FASTMCP_*
        # env vars or a .env in the working directory can't change them. Stateful: OpenCode re-initializes after the
        # 404 that follows a restart.
        runctx.SHARED = True
        dataframe_cache.SKIP_OVER_HALF = True  # one oversized frame must not flush every conversation's cache
        runctx.warm_up()
        print(f"Starting Geocluster MCP on http://{host}:{port}/mcp (Streamable HTTP, shared by every conversation)")
        mcp.run(transport="http", host=host, port=port, path="/mcp", stateless_http=False, json_response=False)
    else:
        # api 2 starts one server per conversation on a loopback port (MCP_HOST/MCP_PORT, with GEOCLUSTER_RUN_DIR);
        # the host CLI's server keeps 0.0.0.0:7654.
        print(f"Starting Geocluster MCP on http://{host}:{port}/sse")
        mcp.run(transport="sse", host=host, port=port)
