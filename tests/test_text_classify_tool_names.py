"""Tool names are ACLs (see test_voxel_tool_names.py). The classify_text_* tools
are meant only for the Cline "classify" mode (mcpToolFilter ``classify_text_*``):
prefixed, never ``voxel_``-prefixed, and free of every substring the other
agents' filters glob on, so they can't leak into their prompts/permissions."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import tools.text_classify as text_classify

# substrings from the geology / dataops / transform / analytics / geoviz / voxel filters
FORBIDDEN = (
    "inspect", "dataset", "raster", "missing", "select", "column", "query", "validate_geo",
    "cleaning_issues", "verify_claims", "normalize", "standardize", "log", "smooth", "gradient",
    "ratio", "band", "texture", "aggregate", "pivot", "melt", "merge", "filter", "convert",
    "fix_decimal", "detection_limit", "remove_duplicate", "anomaly", "threshold", "rank",
    "cluster", "reduce", "stat", "plot", "map", "scatter", "histogram", "chart", "viz",
    "list_files", "check", "summarize_provenance",
)

EXPECTED = {
    "classify_text_pick_field", "classify_text_discover", "classify_text_grid_options", "classify_text_build_layers",
}


def _public_tools():
    return {
        name for name, obj in inspect.getmembers(text_classify, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == text_classify.__name__
    }


def test_tool_set_is_exactly_the_documented_three():
    assert _public_tools() == EXPECTED


def test_names_are_prefixed_and_collision_free():
    for name in _public_tools():
        assert name.startswith("classify_text_"), name
        assert not name.startswith("voxel_"), name
        for sub in FORBIDDEN:
            assert sub not in name, f"{name} contains {sub!r} — would match another agent's filter"


def test_every_tool_has_a_docstring_first_line():
    for name in _public_tools():
        doc = getattr(text_classify, name).__doc__
        assert doc and doc.strip().splitlines()[0], name


def test_registered_in_main():
    src = (Path(text_classify.__file__).resolve().parents[1] / "main.py").read_text()
    tree = ast.parse(src)
    registered = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
        and getattr(node.func.func, "attr", "") == "tool" and node.args and isinstance(node.args[0], ast.Name)
    }
    assert EXPECTED <= registered


def test_pipeline_errors_are_returned_not_raised(monkeypatch):
    import direct.classify_text_to_voxel as pipeline

    def boom(*_args, **_kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(pipeline, "pick_text_column", boom)
    out = asyncio.run(text_classify.classify_text_pick_field("data.csv", "minerals mentioned"))
    assert out["success"] is False
    assert out["tool"] == "classify_text_pick_field"
    assert "nope" in out["error"]


def test_build_layers_reports_button_layer_names(monkeypatch, workspace):
    import pandas as pd

    import direct.classify_text_to_voxel as pipeline

    pd.DataFrame({
        "EASTING": [500000.0, 500010.0],
        "NORTHING": [6600000.0, 6600010.0],
        "DESCRIPTION": ["granite with pyrite", "gneiss, chip sample"],
    }).to_csv(workspace / "data.csv", index=False)

    def fake(dataset_path, text_col, goal, categories, layer_prefix, **_kwargs):
        return {
            "success": False,
            "categories": categories,
            "layers": {
                "Pyrite": {"success": True, "classification": {"rows_in": 10, "rows_classified": 9, "n_batch_failures": 0}},
                "Chalco pyrite": {"success": False, "step": "voxel_upsert_geometry", "error": "bad"},
            },
        }

    monkeypatch.setattr(pipeline, "classify_multi_label_to_voxel", fake)
    out = asyncio.run(text_classify.classify_text_build_layers(
        "data.csv", "DESCRIPTION", "minerals", ["Pyrite", "Chalco pyrite"], "Minerals Found"
    ))
    assert [entry["layer"] for entry in out["layers"]] == ["minerals_found_pyrite", "minerals_found_chalco_pyrite"]
    assert out["layers_built"] == ["minerals_found_pyrite"]
    assert out["layers_failed"] == ["minerals_found_chalco_pyrite"]
    assert out["success"] is True
    assert "minerals_found_pyrite" in out["viewer"]


def test_build_layers_rejects_empty_category_list():
    out = asyncio.run(text_classify.classify_text_build_layers("data.csv", "DESCRIPTION", "minerals", ["", "  "], "x"))
    assert out["success"] is False
    assert "no categories" in out["error"]
