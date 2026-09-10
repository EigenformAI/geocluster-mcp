"""SpatialVoxelStore known-answer tests on a projected-metre grid
(port of voxel-features-mcp test_spatial_geometry_batch.py + test_set_layer_array.py)."""

from __future__ import annotations

import numpy as np
import pytest

from voxel.spatial import SpatialVoxelStore
from voxel.store import CATEGORICAL_EMPTY, GridSpec

# 100 m x 100 m x 20 m cells
GRID = GridSpec(origin=(500000.0, 6000000.0, 0.0), maximum=(501000.0, 6001000.0, 100.0), shape=(10, 10, 5), crs="projected_meters")
# single depth slab: 100 m cells, nz = 1 (isolates horizontal geometry)
SLAB = GridSpec(origin=(500000.0, 6000000.0, 0.0), maximum=(501000.0, 6001000.0, 100.0), shape=(10, 10, 1), crs="projected_meters")


def _store(tmp_path, grid=GRID, name="store") -> SpatialVoxelStore:
    return SpatialVoxelStore(tmp_path / name, grid)


def test_sub_cell_radius_claims_its_containing_voxel(tmp_path):
    store = _store(tmp_path)
    r = store.add_point_feature("pts", 500510.0, 6000510.0, 50.0, 1.0, radius_m=1.0)
    assert r["success"] and r["affected_voxels"] == 1
    vals = store.get_layer_values("pts")
    assert vals[5, 5, 2] == 1.0 and vals.sum() == 1.0


def test_metre_radius_touches_expected_cells(tmp_path):
    """150 m radius at a cell centre on 100 m cells: the 3x3 neighbourhood (corners are 141 m away)."""
    store = _store(tmp_path, SLAB)
    r = store.add_point_feature("r150", 500550.0, 6000550.0, 50.0, 2.0, radius_m=150.0)
    assert r["affected_voxels"] == 9
    vals = store.get_layer_values("r150")
    assert vals[4:7, 4:7, 0].sum() == 18.0 and vals.sum() == 18.0
    # 99 m: only the 4-neighbours (100 m away) are excluded -> own cell only
    r2 = store.add_point_feature("r99", 500550.0, 6000550.0, 50.0, 2.0, radius_m=99.0)
    assert r2["affected_voxels"] == 1


def test_point_outside_bounds_fails_and_bad_source_rejected(tmp_path):
    store = _store(tmp_path)
    assert store.add_point_feature("p", 400000.0, 6000500.0, 5.0, 1.0)["success"] is False
    bad = store.add_point_feature("p", 500500.0, 6000500.0, 5.0, 1.0, coordinate_source="creative_fallback")
    assert bad["success"] is False and "artifact" in bad["error"]
    assert "p" not in store.layer_names


def test_line_stamps_cells_along_segment(tmp_path):
    store = _store(tmp_path, SLAB)
    # along x through cell centres of row 3, width 0 -> exactly the crossed cells 1..8
    r = store.add_line_feature("fault", (500150.0, 6000350.0, 50.0), (500850.0, 6000350.0, 50.0), 1.0, width_m=0.0)
    assert r["success"] and r["affected_voxels"] == 8
    vals = store.get_layer_values("fault")
    assert vals[1:9, 3, 0].sum() == 8.0 and vals.sum() == 8.0


def test_box_fills_clipped_slice(tmp_path):
    store = _store(tmp_path)
    r = store.add_box_feature("box", 499500.0, 5999900.0, -10.0, 500210.0, 6000310.0, 45.0, value=0.75)
    assert r["success"]
    expected = np.zeros(GRID.shape)
    expected[0:3, 0:4, 0:3] = 0.75
    np.testing.assert_array_equal(store.get_layer_values("box"), expected)
    assert r["affected_voxels"] == 36
    assert store.add_box_feature("nope", 0, 0, 0, 1, 1, 1, value=1.0)["success"] is False


def test_combination_rules_treat_empty_as_absent(tmp_path):
    store = _store(tmp_path, SLAB)
    store.add_point_feature("m", 500550.0, 6000550.0, 50.0, 4.0, radius_m=0, combination_rule="mean")
    store.add_point_feature("m", 500550.0, 6000550.0, 50.0, 2.0, radius_m=0, combination_rule="mean")
    assert store.get_layer_values("m")[5, 5, 0] == 3.0  # (4+2)/2, not (0+4)/2 first
    store.add_point_feature("a", 500550.0, 6000550.0, 50.0, 4.0, radius_m=0, combination_rule="add")
    store.add_point_feature("a", 500550.0, 6000550.0, 50.0, 2.0, radius_m=0, combination_rule="add")
    assert store.get_layer_values("a")[5, 5, 0] == 6.0


def _rec(i, **kw):
    base = {"record_id": str(i), "geometry_kind": "point", "value": 1.0, "coordinate_source": "artifact"}
    base.update(kw)
    return base


def test_batch_aliases_projected_and_geographic(tmp_path):
    store = SpatialVoxelStore(tmp_path / "s", GRID, meta={"coordinate_conversion": {"kind": "approx_meters_from_latlon", "lat0": -30.0}})
    from voxel.columns import latlon_to_local_meters

    # a lon/lat that converts inside the grid
    lon = 500500.0 / (111320.0 * np.cos(np.radians(-30.0)))
    lat = 6000500.0 / 111320.0
    xm, ym = latlon_to_local_meters([lon], [lat], -30.0)
    assert GRID.in_bounds(float(xm[0]), float(ym[0]), 50.0)
    records = [
        _rec("a", x=500150.0, y=6000150.0, depth_m=10.0),
        _rec("b", easting=500350.0, northing=6000350.0, depth_m=30.0),
        _rec("c", Easting=500550.0, Northing=6000550.0, depth_m=50.0, radius_m=0),
        _rec("d", longitude=lon, latitude=lat, depth_m=50.0),
    ]
    r = store.add_geometry_batch("aliased", records)
    assert r["success"] and r["records_applied"] == 4 and r["records_skipped"] == 0
    assert r["geometry_kind_counts"] == {"point": 4} and r["coordinate_source_counts"] == {"artifact": 4}
    ops = store.ops.read(layer="aliased")
    assert len(ops) == 4 and all("None" not in op["coordinates"] for op in ops)


def test_batch_rejects_non_artifact_and_missing_coords(tmp_path):
    store = _store(tmp_path)
    records = [_rec("ok", x=500150.0, y=6000150.0), _rec("web", x=500150.0, y=6000150.0, coordinate_source="web"), _rec("nocoord")]
    r = store.add_geometry_batch("g", records)
    assert r["records_applied"] == 1 and r["records_skipped"] == 2
    assert any("artifact" in w for w in r["warnings"]) and any("no coordinates" in w for w in r["warnings"])


def test_batch_replace_is_idempotent_and_accumulate_adds(tmp_path):
    store = _store(tmp_path, SLAB)
    records = [_rec("a", x=500150.0, y=6000150.0, radius_m=0, value=1.0, combination_rule="add"),
               _rec("b", x=500150.0, y=6000150.0, radius_m=0, value=2.0, combination_rule="add")]
    first = store.add_geometry_batch("L", records, combination_rule="add")
    v1 = store.get_layer_values("L").copy()
    second = store.add_geometry_batch("L", records, combination_rule="add")
    np.testing.assert_array_equal(v1, store.get_layer_values("L"))
    assert float(v1[1, 1, 0]) == 3.0
    ops = store.ops.read(layer="L")
    assert len(ops) == 2 and {o["record_id"] for o in ops} == {"a", "b"}
    assert all(o["group_id"] == second["group_id"] for o in ops)
    third = store.add_geometry_batch("L", records, mode="accumulate_layer", combination_rule="add")
    assert float(store.get_layer_values("L")[1, 1, 0]) == 6.0 and third["records_applied"] == 2
    assert len(store.ops.read(layer="L")) == 4


def test_batch_bounds_policy_skip_clip_fail(tmp_path):
    store = _store(tmp_path)
    box = {"record_id": "oob", "geometry_kind": "box", "x_min": 499000.0, "y_min": 6000100.0, "depth_min_m": 0.0,
           "x_max": 500110.0, "y_max": 6000210.0, "depth_max_m": 10.0, "value": 1.0, "coordinate_source": "artifact"}
    assert store.add_geometry_batch("skip", [box], bounds_policy="skip")["records_applied"] == 0
    clipped = store.add_geometry_batch("clip", [box], bounds_policy="clip")
    assert clipped["records_applied"] == 1 and clipped["affected_voxels"] > 0
    failed = store.add_geometry_batch("fail", [box], bounds_policy="fail")
    assert failed["success"] is False and "outside grid bounds" in failed["error"]


def test_non_numeric_value_becomes_presence(tmp_path):
    store = _store(tmp_path)
    r = store.add_geometry_batch("pres", [_rec("s", x=500150.0, y=6000150.0, value="Suite A"), _rec("n", x=500250.0, y=6000250.0, value=None)])
    assert r["records_applied"] == 2 and r["value_min"] == 1.0 == r["value_max"]
    assert any("presence" in w for w in r["warnings"])


def test_categorical_batch_keeps_label_zero(tmp_path):
    store = _store(tmp_path, SLAB)
    r = store.add_geometry_batch("k", [_rec("z", x=500150.0, y=6000150.0, radius_m=0, value=0)], dtype="categorical")
    assert r["records_applied"] == 1 and r["nonempty_voxels"] == 1
    vals = store.get_layer_values("k")
    assert vals[1, 1, 0] == 0.0 and vals[0, 0, 0] == CATEGORICAL_EMPTY


def test_set_layer_array_verbatim_replace_and_shape_check(tmp_path):
    store = _store(tmp_path)
    arr = np.zeros(GRID.shape)
    arr[2, 3, 1] = 7.25
    r = store.set_layer_array("field", arr, source_file="x.csv")
    assert r["success"] and r["nonempty_voxels"] == 1 and r["value_max"] == 7.25
    np.testing.assert_array_equal(store.get_layer_values("field"), arr)
    ops = store.ops.read(layer="field")
    assert len(ops) == 1 and ops[0]["op"] == "array" and ops[0]["source_file"] == "x.csv"
    arr2 = np.ones(GRID.shape)
    store.set_layer_array("field", arr2)
    np.testing.assert_array_equal(store.get_layer_values("field"), arr2)
    assert len(store.ops.read(layer="field")) == 1  # replaced, not appended
    with pytest.raises(ValueError):
        store.set_layer_array("bad", np.zeros((2, 2, 2)))


def test_probe_region_reports_values(tmp_path):
    store = _store(tmp_path, SLAB)
    store.add_point_feature("p", 500550.0, 6000550.0, 50.0, 3.0, radius_m=0)
    out = store.probe_region(500550.0, 6000550.0, 50.0, 120.0)
    assert out["voxels_in_region"] == 5 and out["layers"]["p"]["nonempty_voxels"] == 1
    assert out["layers"]["p"]["value_max"] == 3.0 and out["layers"]["p"]["voxels"][0]["index"] == [5, 5, 0]


def test_degree_grid_radius_uses_local_metres(tmp_path):
    grid = GridSpec(origin=(66.5, 49.5, 0.0), maximum=(71.5, 52.5, 80.0), shape=(200, 200, 8), crs="EPSG:4326")
    store = _store(tmp_path, grid)
    centre = store.coord_to_index(68.046, 51.997, 0.0)
    m = store.sphere_mask(68.046, 51.997, 0.0, 100.0)
    assert m[centre] and m.sum() == 1  # 100 m << 1.7 km cells: own cell only
    m2 = store.sphere_mask(68.046, 51.997, 0.0, 3000.0)
    assert m2.sum() > 1
