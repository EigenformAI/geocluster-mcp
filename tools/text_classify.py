"""Text classification tools (Section M): the IDE "Classify with AI" pipeline
exposed as MCP tools, so the Cline chat can drive it instead of the CSV
viewer's top-of-screen pickers.

Thin wrappers around direct/classify_text_to_voxel.py -- the exact functions
the CSV viewer's button shells into (same LLM proxy call, same batching over
distinct text values, same voxel stamping and viewer export), so a layer built
from the chat is identical to one built from the button.

Only the Cline "classify" mode sees these (mcpToolFilter ``classify_text_*``).
Every name starts with ``classify_text_`` and avoids substrings other agents
glob on (inspect, column, dataset, log, stat, map, check, ...) and the
``voxel_`` prefix: the name is the ACL.

The pipeline is synchronous and can run for minutes (one LLM call per batch of
distinct text values, per category), so each call runs in a worker thread to
keep the SSE server responsive for other clients.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

VIEWER_LINE = "Open 3D Visualization in the dashboard (Layers mode) to explore: {ids}"
_TEXT_FIELD_HINT = "pick one of the classifiable columns; numeric values such as grades belong in Convert to Voxel"


def _error(tool: str, exc: Exception, next_step: str) -> dict[str, Any]:
    return {"success": False, "tool": tool, "error": str(exc), "next_step": next_step}


def _text_field_problem(dataset_path: str, text_field: str) -> str | None:
    """Why ``text_field`` can't be classified (same rule as the AI column picker), or None if it can."""
    from direct.classify_text_to_voxel import _candidate_columns
    from tools.config import read_tabular, resolve_path

    df = read_tabular(resolve_path(dataset_path), nrows=200)
    if text_field not in df.columns:
        return f"text_field {text_field!r} not found in {dataset_path}"
    allowed = _candidate_columns(df)
    if text_field not in allowed:
        return (
            f"{text_field!r} is not a free-text column (ID, file-name, date, coordinate/depth and numeric columns "
            f"are skipped); classifiable columns: {', '.join(map(str, allowed)) or 'none'}"
        )
    return None


async def classify_text_pick_field(dataset_path: str, goal: str, model: str | None = None) -> dict[str, Any]:
    """Classify with AI, step 1: ask the AI which ONE text field of a CSV to read to accomplish `goal`.

    dataset_path: CSV path (absolute or relative to the workspace).
    goal: what to look for, e.g. "minerals mentioned in the description".
    model: optional OpenRouter model id; omit to use the pipeline default (openai/gpt-4o-mini).
    """
    from direct.classify_text_to_voxel import pick_text_column

    try:
        result = await asyncio.to_thread(pick_text_column, dataset_path, goal, model)
    except Exception as exc:  # noqa: BLE001
        return _error("classify_text_pick_field", exc, "check dataset_path points at a CSV inside the workspace")
    return {
        "success": True,
        "tool": "classify_text_pick_field",
        "text_field": result["column"],
        "candidates": result.get("candidates", []),
    }


async def classify_text_discover(
    dataset_path: str, text_field: str, goal: str, model: str | None = None
) -> dict[str, Any]:
    """Classify with AI, step 2: sample distinct values of `text_field` and let the AI list what is mentioned that fits `goal`.

    Returns candidate categories plus an estimate of the AI calls needed to build one layer per category.
    """
    from direct.classify_text_to_voxel import DEFAULT_BATCH_SIZE, discover_categories

    tool = "classify_text_discover"
    try:
        problem = await asyncio.to_thread(_text_field_problem, dataset_path, text_field)
        if problem:
            return _error(tool, ValueError(problem), _TEXT_FIELD_HINT)
        result = await asyncio.to_thread(discover_categories, dataset_path, text_field, goal, model=model)
    except Exception as exc:  # noqa: BLE001
        return _error(tool, exc, "use the text_field returned by classify_text_pick_field")

    categories = result["categories"]
    distinct = int(result.get("distinct_text_values") or 0)
    # Same estimate as the CSV viewer button: classification only hits distinct values, in batches.
    calls_per_category = max(1, math.ceil(distinct / DEFAULT_BATCH_SIZE))
    return {
        "success": True,
        "tool": tool,
        "categories": categories,
        "sample_size": result.get("sample_size"),
        "distinct_text_values": distinct,
        "rows_before_dedup": result.get("rows_before_dedup"),
        "estimated_ai_calls_per_category": calls_per_category,
        "estimated_ai_calls_total": calls_per_category * len(categories),
    }


async def classify_text_grid_options(dataset_path: str) -> dict[str, Any]:
    """Classify with AI, before building: the project's current 3D grid (cell size, existing layers) and the cell size each resolution would give.

    The first layer of a project fixes the grid for every later layer, so a resolution only matters when grid_exists is false
    (or when the user explicitly wants to rebuild, which deletes current_layers).
    """
    from direct.csv_to_voxel import grid_info

    tool = "classify_text_grid_options"
    try:
        info = await asyncio.to_thread(grid_info, dataset_path)
    except Exception as exc:  # noqa: BLE001
        return _error(tool, exc, "check dataset_path points at a CSV with coordinate columns inside the workspace")

    current = info.get("current") or {}
    current_grid = current.get("grid") or {}
    return {
        "success": True,
        "tool": tool,
        "grid_exists": info["exists"],
        "current_cell_size_m": [round(float(c), 1) for c in current_grid.get("cell_size", [])] or None,
        "current_layers": current.get("layers", []),
        "resolutions": {
            name: {"cell_size_m": preset["cell_size_m"], "shape": preset["shape"]}
            for name, preset in info["presets"].items()
        },
    }


async def classify_text_build_layers(
    dataset_path: str,
    text_field: str,
    goal: str,
    categories: list[str],
    layer_prefix: str,
    model: str | None = None,
    resolution: str | None = None,
    rebuild_grid: bool = False,
) -> dict[str, Any]:
    """Classify with AI, step 3: build one yes/no 3D layer per confirmed category ("does this row mention X?") and publish the viewer bundle.

    categories: the (possibly user-trimmed) list from classify_text_discover.
    layer_prefix: layer names become <prefix>_<category>, sanitised the same way as the CSV viewer button.
    resolution: "standard" | "detailed" | "finest"; only used when the project has no grid yet (see classify_text_grid_options).
    rebuild_grid: true only after the user confirmed replacing the grid; DELETES every existing 3D layer first.
    """
    from direct.classify_text_to_voxel import _slug, classify_multi_label_to_voxel

    tool = "classify_text_build_layers"
    cats = list(dict.fromkeys(str(c).strip() for c in categories if str(c).strip()))
    if not cats:
        return _error(tool, ValueError("no categories given"), "pass the confirmed list from classify_text_discover")

    try:
        problem = await asyncio.to_thread(_text_field_problem, dataset_path, text_field)
        if problem:
            return _error(tool, ValueError(problem), _TEXT_FIELD_HINT)
        result = await asyncio.to_thread(
            classify_multi_label_to_voxel, dataset_path, text_field, goal, cats, layer_prefix,
            model=model, resolution=resolution, rebuild_grid=rebuild_grid,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(tool, exc, "check the dataset has coordinate columns and no other voxel build is running")

    prefix = _slug(layer_prefix, 30)
    per_layer: list[dict[str, Any]] = []
    grid_cell_size = None
    for cat in cats:
        slug = _slug(cat, 30)
        outcome = result.get("layers", {}).get(cat, {})
        stats = outcome.get("classification") or {}
        ok = bool(outcome.get("success"))
        if ok and grid_cell_size is None and (outcome.get("grid") or {}).get("cell_size"):
            grid_cell_size = [round(float(c), 1) for c in outcome["grid"]["cell_size"]]
        per_layer.append({
            "category": cat,
            "layer": f"{prefix}_{slug}" if prefix else slug,
            "success": ok,
            "rows_in": stats.get("rows_in"),
            "rows_classified": stats.get("rows_classified"),
            "n_batch_failures": stats.get("n_batch_failures"),
            "error": None if ok else (outcome.get("error") or outcome.get("step") or "unknown error"),
        })

    built = [entry["layer"] for entry in per_layer if entry["success"]]
    failed = [entry["layer"] for entry in per_layer if not entry["success"]]
    return {
        "success": bool(built),
        "tool": tool,
        "text_field": text_field,
        "grid_cell_size_m": grid_cell_size,
        "layers": per_layer,
        "layers_built": built,
        "layers_failed": failed,
        "viewer": VIEWER_LINE.format(ids=", ".join(built)) if built else None,
    }
