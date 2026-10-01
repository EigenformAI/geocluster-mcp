"""Tool outputs when calls run in parallel threads (the shared MCP server): no torn files, no lost voxel layers, no
writes outside the run folder or the workspace (X-1, X-7). Each case failed in the 2026-10-01 audit."""

from __future__ import annotations

import contextvars
import json
import os
import threading

import pandas as pd
import pytest

import tools.config as cfg
import tools.voxel as T
import viz_bundle


def in_threads(fns):
    """Run callables at once, each in its own thread with a copy of the current context (as the server does)."""
    results, errors = [None] * len(fns), []
    barrier = threading.Barrier(len(fns))

    def go(i, fn, ctx):
        barrier.wait()
        try:
            results[i] = ctx.run(fn)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i, fn, contextvars.copy_context())) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


@pytest.fixture
def run_dir(workspace):
    token = cfg._CALL_RUN_DIR.set(cfg.validate_call_run_dir("runs/ses_A"))
    yield workspace / "runs" / "ses_A"
    cfg._CALL_RUN_DIR.reset(token)


@pytest.fixture
def data(workspace):
    (workspace / "data").mkdir()
    (workspace / "data" / "a.csv").write_text("x\n1\n")
    return "data/a.csv"


# --- writes --------------------------------------------------------------------------------------------------------


def test_a_reader_never_sees_a_half_written_output(data, run_dir):
    old, new = pd.DataFrame({"x": range(1000)}), pd.DataFrame({"x": range(300_000), "y": 1.5})
    path = cfg.save_csv(old, data, "same")
    seen, done = [], threading.Event()

    def write():
        try:
            for _ in range(3):
                cfg.save_csv(new, data, "same")
        finally:
            done.set()

    def read():
        while not done.is_set():
            try:
                seen.append(len(pd.read_csv(path)))
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                seen.append(repr(exc))

    _, errors = in_threads([write, read])
    assert errors == [] and seen
    assert set(seen) <= {len(old), len(new)}, f"partial reads: {sorted(set(map(str, seen)))[:5]}"
    assert not [p for p in os.listdir(os.path.dirname(path)) if p.startswith(".")], "temp files cleaned"


def test_a_suffix_cannot_add_folders(data, run_dir):
    out = cfg.save_csv(pd.DataFrame({"x": [1]}), data, "/../../../data/pwn")
    assert os.path.dirname(out) == str(run_dir.resolve() / "results")
    assert "/" not in os.path.basename(out)
    assert not (run_dir.parent.parent / "data" / "pwn.csv").exists()


def test_a_symlinked_results_folder_is_refused(data, run_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "results").symlink_to(outside)
    with pytest.raises(ValueError, match="outside this conversation's run folder"):
        cfg.save_csv(pd.DataFrame({"x": [1]}), data, "fixed")
    assert list(outside.iterdir()) == []


def test_inputs_are_workspace_paths(data, workspace, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the server's cwd is not the workspace
    assert cfg.read_tabular(data)["x"].tolist() == [1]
    (tmp_path / "secret.csv").write_text("k\nv\n")
    with pytest.raises(ValueError, match="outside workspace"):
        cfg.read_tabular(str(tmp_path / "secret.csv"))


def test_provenance_log_is_replaced_atomically(run_dir):
    from tools.provenance import summarize_provenance

    out = summarize_provenance()
    assert json.loads((run_dir / "results" / "provenance_log.json").read_text()) == out["inventory"]
    assert [p for p in os.listdir(run_dir / "results") if p.startswith(".")] == []


# --- voxel store ---------------------------------------------------------------------------------------------------


def test_parallel_stamps_keep_every_layer(workspace, run_dir):
    assert T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=10.0)["success"]
    layers = [f"layer_{i}" for i in range(8)]
    results, errors = in_threads([
        lambda name=name: T.voxel_add_point(name, 500150.0, 6000150.0, 5.0, 1.0) for name in layers
    ])
    assert errors == [] and all(r["success"] for r in results), results
    listed = {layer["name"] for layer in T.voxel_list_layers()["layers"]}
    assert listed == set(layers)
    leftovers = [p for p in os.listdir(run_dir / "voxel_store" / "layers") if not p.endswith(".npy") or p.startswith(".")]
    assert leftovers == []


def test_parallel_exports_mirror_cleanly(workspace, run_dir):
    assert T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=10.0)["success"]
    assert T.voxel_add_point("p", 500150.0, 6000150.0, 5.0, 1.0)["success"]
    results, errors = in_threads([lambda: T.voxel_export_bundle(include_samples=False) for _ in range(4)])
    assert errors == [] and all(r["success"] and r.get("legacy_viz_dir") == "viz" for r in results), results
    legacy = json.loads((workspace / "viz" / "manifest.json").read_text())
    assert legacy == json.loads((run_dir / "viz" / "manifest.json").read_text())
    assert [p.name for p in workspace.iterdir() if p.name.startswith(".")] == [], "no temp or lock files left"


def test_concurrent_mirrors_of_different_runs_leave_one_whole_bundle(workspace):
    sources = []
    for tag in "ABCD":
        src = workspace / "runs" / tag / "viz"
        (src / "layers").mkdir(parents=True)
        (src / "layers" / "l.f32").write_bytes(tag.encode() * 64)
        (src / "manifest.json").write_text(json.dumps({"tag": tag}))
        sources.append(src)
    dest = workspace / "viz"
    _, errors = in_threads([lambda s=s: viz_bundle.mirror_bundle(s, dest) for s in sources])
    assert errors == []
    tag = json.loads((dest / "manifest.json").read_text())["tag"]
    assert (dest / "layers" / "l.f32").read_bytes() == tag.encode() * 64, "manifest and blob from the same bundle"
    assert [p.name for p in workspace.iterdir() if p.name.startswith(".")] == [], "no temp or lock files left"


def test_reinitialising_the_grid_fails_a_stamp_opened_on_the_old_one(workspace, run_dir):
    from voxel.spatial import SpatialVoxelStore

    assert T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=100.0, cell_size_z_m=10.0)["success"]
    stale = SpatialVoxelStore(T._store_dir())  # a call that opened the store before the re-init below
    assert T.voxel_init_grid("drillholes_small.csv", cell_size_xy_m=50.0, cell_size_z_m=10.0, overwrite=True)["success"]
    result = stale.add_point_feature("late", 500150.0, 6000150.0, 5.0, 1.0)
    assert not result["success"] and "re-initialised" in result["error"]
    assert T.voxel_list_layers()["layers"] == []


# --- review fixes (2026-10-01) ---------------------------------------------------------------------------------------


def test_a_long_output_name_still_fits(data, run_dir, workspace):
    stem = "d" * 240
    (workspace / "data" / f"{stem}.csv").write_text("x\n1\n")
    out = cfg.save_csv(pd.DataFrame({"x": [1]}), f"data/{stem}.csv", "log")  # final name 248 bytes; temp must fit too
    assert os.path.basename(out) == f"{stem}_log.csv"


def _envi(path, value):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="ENVI", width=16, height=16, count=1, dtype="float32", crs="EPSG:32750",
                       transform=from_origin(500000, 6000000, 10, 10)) as dst:
        dst.write(np.full((16, 16), value, dtype="float32"), 1)


def test_rasters_with_sidecars_keep_them_under_the_final_name(workspace, run_dir):
    import rasterio

    from tools.features import compute_gradient

    _envi(workspace / "scene.dat", 1.0)
    out = compute_gradient("scene.dat")["output_path"]
    results = run_dir / "results"
    assert sorted(os.listdir(results)) == ["scene_gradient.dat", "scene_gradient.hdr"]
    with rasterio.open(out) as src:
        assert src.driver == "ENVI" and src.crs.to_epsg() == 32750


def test_a_replaced_raster_drops_the_old_statistics_and_overviews(workspace, run_dir):
    from tools.features import compute_gradient

    _envi(workspace / "scene.dat", 1.0)
    out = compute_gradient("scene.dat")["output_path"]
    for stale in (".aux.xml", ".ovr"):
        with open(out + stale, "w") as f:
            f.write("stale")
    compute_gradient("scene.dat")
    assert sorted(os.listdir(run_dir / "results")) == ["scene_gradient.dat", "scene_gradient.hdr"]


def test_dataset_discovery_skips_writes_in_progress(workspace):
    real = "filename,EASTING,NORTHING\na.jpg,1,2\n"
    (workspace / "runs" / "r" / "results").mkdir(parents=True)
    (workspace / "runs" / "r" / "results" / "real.csv").write_text(real)
    (workspace / "runs" / "r" / "results" / ".real_fixed.0123456789ab.tmp.csv").write_text(real * 100)
    (workspace / "runs" / "r" / "results" / ".tmp-0123456789ab").mkdir()
    (workspace / "runs" / "r" / "results" / ".tmp-0123456789ab" / "big.csv").write_text(real * 200)
    found = viz_bundle.discover_dataset(workspace / "runs" / "r", None)  # as voxel_export_bundle scopes it
    assert found is not None and found.name == "real.csv"


def test_export_all_skips_temps_and_leaves_no_partial_zip(workspace, run_dir):
    import zipfile

    from tools.provenance import export_artifact

    results = run_dir / "results"
    results.mkdir(parents=True)
    (results / "a.csv").write_text("x\n1\n")
    (results / ".a_fixed.0123456789ab.tmp").write_text("half")
    out = export_artifact()
    with zipfile.ZipFile(out["output_path"]) as z:
        assert z.namelist() == ["a.csv"]
    assert [p for p in os.listdir(results) if p.startswith(".")] == [".a_fixed.0123456789ab.tmp"]
