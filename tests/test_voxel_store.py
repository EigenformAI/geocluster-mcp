"""GridSpec + VoxelStore known-answer tests (port of voxel-features-mcp test_store.py)."""

from __future__ import annotations

import numpy as np
import pytest

from voxel.store import CATEGORICAL_EMPTY, GridSpec, VoxelStore

GRID = GridSpec(origin=(500000.0, 6000000.0, 0.0), maximum=(501000.0, 6001000.0, 100.0), shape=(10, 10, 5), crs="projected_meters")


def test_grid_geometry():
    assert GRID.units == "meters"
    assert GRID.cell_size == (100.0, 100.0, 20.0)
    assert GRID.n_voxels == 500
    assert GridSpec(origin=(0, 0, 0), maximum=(1, 1, 1), shape=(1, 1, 1), crs="EPSG:4326").units == "degrees"


def test_grid_validation():
    with pytest.raises(ValueError):
        GridSpec(origin=(0, 0, 0), maximum=(0, 1, 1), shape=(1, 1, 1))
    with pytest.raises(ValueError):
        GridSpec(origin=(0, 0, 0), maximum=(1, 1, 1), shape=(0, 1, 1))


def test_coord_to_index_is_floor_and_clamped():
    assert GRID.coord_to_index(500000.0, 6000000.0, 0.0) == (0, 0, 0)
    assert GRID.coord_to_index(500099.9, 6000199.9, 19.9) == (0, 1, 0)
    assert GRID.coord_to_index(500100.0, 6000200.0, 20.0) == (1, 2, 1)
    assert GRID.coord_to_index(501000.0, 6001000.0, 100.0) == (9, 9, 4)  # max corner clamps into last cell
    assert GRID.coord_to_index(-1e9, 1e12, 1e6) == (0, 9, 4)
    assert GRID.index_to_coord(0, 0, 0) == (500050.0, 6000050.0, 10.0)
    assert GRID.in_bounds(500500.0, 6000500.0, 50.0) and not GRID.in_bounds(499999.0, 6000500.0, 50.0)


def test_grid_roundtrip_dict():
    assert GridSpec.from_dict(GRID.to_dict()) == GRID


def test_store_create_add_get_reload_remove(tmp_path):
    store = VoxelStore(tmp_path / "store", GRID, meta={"dataset": {"rows": 3}})
    vals = np.zeros(GRID.shape)
    vals[1, 2, 3] = 4.5
    layer = store.add_layer("cu", vals, "float", metadata={"note": "x"}, hypothesis="h")
    assert layer.content_hash
    with pytest.raises(ValueError):
        store.add_layer("cu", vals, "float")  # duplicate

    reopened = VoxelStore(tmp_path / "store")
    assert reopened.grid == GRID
    assert reopened.meta["dataset"] == {"rows": 3}
    assert reopened.layer_names == ["cu"]
    np.testing.assert_array_equal(reopened.get_layer_values("cu"), vals)
    summary = reopened.layer_summary("cu")
    assert summary["nonempty_voxels"] == 1 and summary["value_max"] == 4.5 and summary["hypothesis"] == "h"
    assert summary["metadata"] == {"note": "x"}

    reopened.remove_layer("cu")
    assert reopened.layer_names == [] and not (tmp_path / "store" / "layers" / "cu.npy").exists()
    with pytest.raises(KeyError):
        reopened.get_layer_values("cu")


def test_store_requires_grid_when_new(tmp_path):
    with pytest.raises(ValueError):
        VoxelStore(tmp_path / "new")


def test_layer_name_and_dtype_validation(tmp_path):
    store = VoxelStore(tmp_path / "store", GRID)
    with pytest.raises(ValueError):
        store.add_layer("bad name/with slash", np.zeros(GRID.shape), "float")
    with pytest.raises(ValueError):
        store.add_layer("ok", np.zeros(GRID.shape), "text")
    with pytest.raises(ValueError):
        store.add_layer("ok", np.zeros((2, 2, 2)), "float")
    with pytest.raises(ValueError):
        store.add_layer("cat", np.full(GRID.shape, 0.5), "categorical")  # non-integer codes


def test_categorical_label_zero_is_not_empty(tmp_path):
    store = VoxelStore(tmp_path / "store", GRID)
    vals = store.new_layer_values("categorical")
    assert float(vals[0, 0, 0]) == CATEGORICAL_EMPTY
    vals[0, 0, 0] = 0  # class 0
    store.add_layer("k", vals, "categorical")
    assert store.layer_summary("k")["nonempty_voxels"] == 1
