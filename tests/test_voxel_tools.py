"""MCP voxel_* tool tests against a seeded workspace (see conftest.workspace)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import tools.voxel as T


def test_get_grid_before_init_is_a_clear_error(workspace):
    r = T.voxel_get_grid()
    assert r["success"] is False and "voxel_init_grid" in r["next_step"] and r["tool"] == "voxel_get_grid"


def test_init_grid_from_dataset_bounds(workspace):
    r = T.voxel_init_grid("drillholes_small.csv")
    assert r["success"], r
    g = r["grid"]
    assert g["shape"] == [64, 64, 16] and g["units"] == "meters"
    assert g["origin"][:2] == [500100.0, 6000100.0] and g["maximum"][:2] == [500800.0, 6000800.0]
    assert g["origin"][2] == 0.0 and g["maximum"][2] == 100.0
    assert r["dataset"]["columns"] == {"x": "EASTING", "y": "NORTHING", "kind": "projected", "depth": "depth_from_m/depth_to_m"}
    assert r["depth_degenerate"] is False
    assert (workspace / "voxel_store" / "index.json").is_file()
    again = T.voxel_get_grid()
    assert again["success"] and again["grid"]["shape"] == [64, 64, 16] and again["layers"] == []


def test_init_grid_cell_sizes_and_overwrite_guard(workspace):
    r = T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=25.0)
    assert r["grid"]["shape"] == [7, 7, 4] and r["grid"]["cell_size"] == [100.0, 100.0, 25.0]
    T.voxel_add_point("l", 500150.0, 6000150.0, 5.0, 1.0)
    blocked = T.voxel_init_grid("drillholes_small.csv")
    assert blocked["success"] is False and "overwrite=True" in blocked["next_step"]
    ok = T.voxel_init_grid("drillholes_small.csv", overwrite=True)
    assert ok["success"] and T.voxel_list_layers()["layers"] == []


def test_init_grid_flat_dataset_is_one_slab(workspace):
    r = T.voxel_init_grid("flat_points.csv")
    assert r["success"] and r["depth_degenerate"] is True and r["grid"]["shape"][2] == 1


def test_init_grid_rejects_missing_coords_and_path_escape(workspace, tmp_path):
    (workspace / "nocoords.csv").write_text("filename,Cu\na,1\n")
    r = T.voxel_init_grid("nocoords.csv")
    assert r["success"] is False and "EASTING/NORTHING" in r["error"]
    outside = tmp_path / "outside.csv"
    outside.write_text("filename,EASTING,NORTHING\na,1,2\n")
    r2 = T.voxel_init_grid(str(outside))
    assert r2["success"] is False and "outside workspace" in r2["error"]


def test_upsert_geometry_categorical_and_export(workspace):
    T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=10.0)
    r = T.voxel_upsert_geometry("clusters_k3", "feature_geometry_points.csv", dtype="categorical", combination_rule="replace")
    assert r["success"], r
    assert r["records_applied"] == 40 and r["records_rejected_source_mismatch"] == 0
    assert r["coordinate_source_counts"] == {"artifact": 40}
    ops = T.voxel_history(layer="clusters_k3")
    assert ops["count"] == 40 and ops["summary"]["source_files"] == ["drillholes_small.csv"]

    # array route
    shape = tuple(T.voxel_get_grid()["grid"]["shape"])
    arr = np.zeros(shape)
    arr[0, 0, 0] = 12.5
    np.save(workspace / "field.npy", arr)
    s = T.voxel_set_layer_array("cu_field", "field.npy", source_file="drillholes_small.csv")
    assert s["success"] and s["nonempty_voxels"] == 1 and s["array_sha256"]

    e = T.voxel_export_bundle(finding="cluster 1 is deep", hypothesis="h1")
    assert e["success"], e
    assert "Open 3D Visualization" in e["viewer"]
    viz = workspace / "viz"
    manifest = json.loads((viz / "manifest.json").read_text())
    kinds = [a["kind"] for a in manifest["artifacts"]]
    assert kinds == ["samples", "grid"]  # clustering run present -> samples too
    grid = next(a for a in manifest["artifacts"] if a["kind"] == "grid")
    ids = [l["id"] for l in grid["layers"]]
    assert "clusters_k3" in ids and "cu_field" in ids and "sample_density" in ids and "grade_cu" in ids
    ncells = int(np.prod(grid["grid"]["shape"]))
    for layer in grid["layers"]:
        path = viz / layer["path"]
        assert path.stat().st_size == layer["bytes"] == ncells * (1 if layer["dtype"] == "uint8" else 4)
        assert os.path.getmtime(viz / "manifest.json") >= os.path.getmtime(path)
    cl = next(l for l in grid["layers"] if l["id"] == "clusters_k3")
    assert cl["kind"] == "categorical" and [c["value"] for c in cl["categories"]] == [0, 1, 2]
    assert cl["provenance"]["finding"] == "cluster 1 is deep" and cl["provenance"]["coordinate_source_counts"] == {"artifact": 40}
    assert cl["provenance"]["operations"]["count"] == 40
    # the store's clusters layer replaced the publisher-binned one, on the store grid
    assert grid["grid"]["shape"] == list(shape)
    assert any("replaces the publisher-binned" in w for w in manifest["provenance"]["warnings"])
    # known answer: majority label per hole in the exported u8 equals the assignments
    codes = np.frombuffer((viz / cl["path"]).read_bytes(), dtype="u1").reshape(shape)
    assigns = pd.read_csv(workspace / "cluster_assignments_k3.csv")
    holes = pd.read_csv(workspace / "drillholes_small.csv")
    from voxel.spatial import SpatialVoxelStore
    store = SpatialVoxelStore(workspace / "voxel_store")
    for hole, grp in holes.groupby("drillhole"):
        expected = int(assigns[assigns.filename.isin(grp.filename)].cluster_label.iloc[0])
        labels = [int(codes[store.coord_to_index(r.EASTING, r.NORTHING, r.depth_m)]) for r in grp.itertuples()]
        assert max(set(labels), key=labels.count) == expected, hole


def test_upsert_rejects_fabricated_coordinates(workspace):
    T.voxel_init_grid("drillholes_small.csv")
    df = pd.read_csv(workspace / "feature_geometry_points.csv").head(3)
    df.loc[df.index[0], "x"] += 40.0  # 40 m off any real row
    df.to_csv(workspace / "fake.csv", index=False)
    r = T.voxel_upsert_geometry("fake", "fake.csv")
    assert r["records_applied"] == 2 and r["records_rejected_source_mismatch"] == 1
    assert list(r["source_mismatches"]) == [df.record_id.iloc[0]]
    warn = T.voxel_upsert_geometry("fake", "fake.csv", source_check="warn")
    assert warn["records_applied"] == 3 and any("source mismatch" in w for w in warn["warnings"])
    off = T.voxel_upsert_geometry("fake", "fake.csv", source_check="off")
    assert off["records_applied"] == 3 and off["source_mismatches"] == {}


def test_upsert_string_labels_get_label_map(workspace):
    T.voxel_init_grid("drillholes_small.csv")
    df = pd.read_csv(workspace / "feature_geometry_points.csv").head(6)
    df["value"] = ["shale", "basalt", "shale", "granite", "basalt", "shale"]
    df.to_csv(workspace / "lith.csv", index=False)
    r = T.voxel_upsert_geometry("lith", "lith.csv", dtype="categorical")
    assert r["success"] and r["records_applied"] == 6
    assert set(r["label_map"].values()) == {"shale", "basalt", "granite"}
    e = T.voxel_export_bundle(layers=["lith"], include_samples=False)
    manifest = json.loads((workspace / "viz" / "manifest.json").read_text())
    assert [a["kind"] for a in manifest["artifacts"]] == ["grid"]
    cats = manifest["artifacts"][0]["layers"][0]["categories"]
    assert {c["label"] for c in cats} == {"shale", "basalt", "granite"}


def test_export_without_layers_or_samples_is_an_error(workspace):
    T.voxel_init_grid("flat_points.csv")
    r = T.voxel_export_bundle(include_samples=False)
    assert r["success"] is False and "nothing to publish" in r["error"]


def test_probe_and_list(workspace):
    T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=10.0)
    T.voxel_add_point("p", 500150.0, 6000150.0, 5.0, 9.0, radius_m=0)
    probe = T.voxel_probe_region(500150.0, 6000150.0, 5.0, 1.0, layers=["p"])
    assert probe["success"] and probe["layers"]["p"]["value_max"] == 9.0
    listing = T.voxel_list_layers()
    assert listing["layers"][0]["name"] == "p" and listing["layers"][0]["nonempty_voxels"] == 1
