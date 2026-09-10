"""Tool names are ACLs: specialists glob on them (cline-fork mcpToolFilter).
Every voxel tool must start with ``voxel_`` and avoid substrings other
specialists match, otherwise it silently leaks into their prompts/permissions."""

from __future__ import annotations

import inspect

import tools.voxel as voxel_tools

# substrings from the dataops / transform / analytics / geoviz / geology filters
FORBIDDEN = (
    "inspect", "dataset", "raster", "missing", "select", "column", "query", "validate_geo",
    "cleaning_issues", "verify_claims", "normalize", "standardize", "log", "smooth", "gradient",
    "ratio", "band", "texture", "aggregate", "pivot", "melt", "merge", "filter", "convert",
    "fix_decimal", "detection_limit", "remove_duplicate", "anomaly", "threshold", "rank",
    "cluster", "reduce", "stat", "plot", "map", "scatter", "histogram", "chart", "viz",
    "list_files", "check",
)

EXPECTED = {
    "voxel_init_grid", "voxel_get_grid", "voxel_add_point", "voxel_add_line", "voxel_add_box",
    "voxel_upsert_geometry", "voxel_set_layer_array", "voxel_probe_region", "voxel_list_layers",
    "voxel_history", "voxel_export_bundle",
}


def _public_tools():
    return {
        name for name, obj in inspect.getmembers(voxel_tools, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == voxel_tools.__name__
    }


def test_tool_set_is_exactly_the_documented_eleven():
    assert _public_tools() == EXPECTED


def test_names_are_prefixed_and_collision_free():
    for name in _public_tools():
        assert name.startswith("voxel_"), name
        for sub in FORBIDDEN:
            assert sub not in name, f"{name} contains {sub!r} — would match another specialist's filter"


def test_every_tool_has_a_docstring_first_line():
    for name in _public_tools():
        doc = getattr(voxel_tools, name).__doc__
        assert doc and doc.strip().splitlines()[0], name


def test_registered_in_main():
    import ast
    from pathlib import Path

    src = (Path(voxel_tools.__file__).resolve().parents[1] / "main.py").read_text()
    tree = ast.parse(src)
    registered = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
        and getattr(node.func.func, "attr", "") == "tool" and node.args and isinstance(node.args[0], ast.Name)
    }
    assert EXPECTED <= registered
