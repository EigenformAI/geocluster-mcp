"""viz_bundle: normalisation, publisher-vs-store binning parity, publish() with and without a store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import viz_bundle
from voxel.spatial import SpatialVoxelStore
from voxel.store import GridSpec


def test_normalize_log_contract():
    arr = np.array([0.0, 1.0, 10.0, 100.0, -5.0, np.nan])
    out, lo, hi = viz_bundle.normalize_log(arr)
    assert out.dtype == np.float32 and (lo, hi) == (1.0, 100.0)
    assert out[0] == 0.0 and out[4] == 0.0 and out[5] == 0.0  # empty / negative / nan -> sentinel
    assert out[1] == pytest.approx(0.05) and out[3] == pytest.approx(1.0) and out[2] == pytest.approx(0.525)
    flat, lo2, hi2 = viz_bundle.normalize_log(np.array([3.0, 3.0]))
    assert flat.tolist() == [1.0, 1.0] and lo2 == hi2 == 3.0


def _samples(workspace: Path) -> dict:
    return viz_bundle.build_samples(workspace / "cluster_assignments_k3.csv", workspace / "drillholes_small.csv")


def test_build_samples_shape(workspace):
    s = _samples(workspace)
    assert s["n_samples"] == 40 and s["k"] == 3 and s["elements"] == ["Cu", "Au_ppm"]
    assert s["coordinate_space"]["crs"] == "projected_meters" and s["hole"][0] == "DH001"
    assert len(s["x"]) == len(s["cluster"]) == 40 and s["lithology"] is not None


def test_publisher_bins_match_store_binning(workspace):
    """Parity: build_grid_layers density == a point-per-sample add_geometry_batch on the same grid."""
    s = _samples(workspace)
    grid = GridSpec(origin=(500100.0, 6000100.0, 0.0), maximum=(500800.0, 6000800.0, 100.0), shape=(7, 7, 10), crs="projected_meters")
    _, layers = viz_bundle.build_grid_layers(s, grid.shape, grid=grid)
    density_blob = layers["sample_density"][1]
    density = np.frombuffer(density_blob, dtype="<f4").reshape(grid.shape)

    store = SpatialVoxelStore(workspace / "voxel_store", grid)
    ox, oy = s["coordinate_space"]["origin"]
    records = [
        {"record_id": str(i), "geometry_kind": "point", "x": s["x"][i] + ox, "y": s["y"][i] + oy,
         "depth_m": (s["depth_from"][i] + s["depth_to"][i]) / 2, "radius_m": 0, "value": 1.0,
         "combination_rule": "add", "coordinate_source": "artifact"}
        for i in range(s["n_samples"])
    ]
    r = store.add_geometry_batch("density", records, combination_rule="add")
    assert r["records_applied"] == 40
    counts = store.get_layer_values("density")
    store_density, _, _ = viz_bundle.normalize_log(counts)
    np.testing.assert_array_equal(density, store_density)
    assert int((counts > 0).sum()) == int((density > 0).sum()) == 40  # one sample per cell here

    # legacy path (grid from bounds) at the same shape is the same binning
    spec_legacy, legacy = viz_bundle.build_grid_layers(s, grid.shape)
    assert spec_legacy["shape"] == [7, 7, 10]
    np.testing.assert_array_equal(np.frombuffer(legacy["sample_density"][1], dtype="<f4"), density.ravel())


def test_publish_without_store_is_legacy_bundle(workspace):
    result = viz_bundle.publish(workspace, (4, 4, 2))
    manifest = json.loads((workspace / "viz" / "manifest.json").read_text())
    assert [a["kind"] for a in manifest["artifacts"]] == ["samples", "grid"]
    assert manifest["provenance"]["grid_source"] == "samples_bounds" and manifest["provenance"]["voxel_store"] is None
    assert "clusters_k3" in result["layer_ids"] and result["n_samples"] == 40


def test_publish_with_store_uses_store_grid(workspace):
    grid = GridSpec(origin=(500000.0, 6000000.0, 0.0), maximum=(501000.0, 6001000.0, 100.0), shape=(5, 5, 2), crs="projected_meters")
    store = SpatialVoxelStore(workspace / "voxel_store", grid)
    store.add_point_feature("anom", 500500.0, 6000500.0, 50.0, 2.0, radius_m=0)
    result = viz_bundle.publish(workspace, (64, 64, 16), finding="f")
    manifest = json.loads((workspace / "viz" / "manifest.json").read_text())
    g = next(a for a in manifest["artifacts"] if a["kind"] == "grid")
    assert g["grid"]["shape"] == [5, 5, 2] and manifest["provenance"]["grid_source"] == "voxel_store"
    assert "anom" in result["layer_ids"] and "sample_density" in result["layer_ids"]
    anom = next(l for l in g["layers"] if l["id"] == "anom")
    assert anom["provenance"]["finding"] == "f" and anom["provenance"]["operations"]["count"] == 1


def test_publish_errors_are_value_errors(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    with pytest.raises(ValueError, match="cluster_assignments"):
        viz_bundle.publish(ws, (4, 4, 2))
