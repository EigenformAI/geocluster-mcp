"""Tool outputs when calls run in parallel threads (the shared MCP server): no torn files, no
writes outside the run folder or the workspace (X-1, X-7). Each case failed in the 2026-10-01 audit."""

from __future__ import annotations

import contextvars
import json
import os
import threading

import pandas as pd
import pytest

import tools.config as cfg


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


def test_a_repeated_output_never_replaces_the_first(data, run_dir):
    first, again = pd.DataFrame({"x": range(1000)}), pd.DataFrame({"x": range(3), "y": 1.5})
    path = cfg.save_csv(first, data, "same")
    names = [os.path.basename(cfg.save_csv(again, data, "same")) for _ in range(3)]
    assert names == ["a_same-2.csv", "a_same-3.csv", "a_same-4.csv"]
    assert len(pd.read_csv(path)) == len(first), "the first output is untouched"
    assert not [p for p in os.listdir(os.path.dirname(path)) if p.startswith(".")], "temp files cleaned"


def test_parallel_calls_racing_for_one_name_never_overwrite_each_other(data, run_dir):
    frames = [pd.DataFrame({"x": [i] * 50_000}) for i in range(4)]
    results, errors = in_threads([lambda f=f: cfg.save_csv(f, data, "race") for f in frames])
    assert all(isinstance(e, FileExistsError) for e in errors), errors  # the loser of a name is refused
    written = [(i, p) for i, p in enumerate(results) if p]
    assert written and len({p for _, p in written}) == len(written)
    for i, p in written:
        assert set(pd.read_csv(p)["x"]) == {i}, "each file holds what its own call wrote"


def test_a_name_taken_during_the_write_is_refused(data, run_dir):
    target = cfg._output_path(data, "late", ".csv")
    with pytest.raises(FileExistsError, match="nothing was overwritten"):
        with cfg.atomic_output(target, keep_ext=False) as tmp:
            pd.DataFrame({"x": [1]}).to_csv(tmp, index=False)
            with open(target, "w") as f:  # another call finished first
                f.write("theirs")
    assert open(target).read() == "theirs"
    assert not [p for p in os.listdir(os.path.dirname(target)) if p.startswith(".")], "temp removed"


def test_an_existing_file_under_the_name_is_kept(data, run_dir):
    results = run_dir / "results"
    results.mkdir(parents=True)
    (results / "a_log.csv").write_text("someone's own file")
    out = cfg.save_csv(pd.DataFrame({"x": [1]}), data, "log")
    assert os.path.basename(out) == "a_log-2.csv"
    assert (results / "a_log.csv").read_text() == "someone's own file"


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

    summarize_provenance()
    out = summarize_provenance()  # an inventory rebuilt each call: the one file that is replaced, not numbered
    assert json.loads((run_dir / "results" / "provenance_log.json").read_text()) == out["inventory"]
    assert [p for p in os.listdir(run_dir / "results") if p.startswith(".")] == []


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


def test_a_second_raster_leaves_the_first_and_its_sidecars_alone(workspace, run_dir):
    from tools.features import compute_gradient

    _envi(workspace / "scene.dat", 1.0)
    out = compute_gradient("scene.dat")["output_path"]
    with open(out + ".aux.xml", "w") as f:
        f.write("first's statistics")
    again = compute_gradient("scene.dat")["output_path"]
    assert os.path.basename(again) == "scene_gradient-2.dat"
    assert sorted(os.listdir(run_dir / "results")) == [
        "scene_gradient-2.dat", "scene_gradient-2.hdr", "scene_gradient.dat", "scene_gradient.dat.aux.xml",
        "scene_gradient.hdr",
    ]


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
