"""Per-conversation output folder (design invariant X-7).

The runtime gateway starts one MCP server per concurrent conversation with
GEOCLUSTER_RUN_DIR=runs/<id>; every file the tools write must land under that folder.
Without it (shared server, host CLI) outputs go next to their input, as before.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tools.config as cfg
import tools.provenance as P

RUN = "runs/1790000000000"


@pytest.fixture
def run_dir(workspace, monkeypatch):
    monkeypatch.setenv("GEOCLUSTER_RUN_DIR", RUN)
    monkeypatch.setattr(cfg, "RUN_DIR", cfg._load_run_dir())
    return workspace / RUN


def test_unset_keeps_outputs_next_to_input(workspace, monkeypatch):
    monkeypatch.delenv("GEOCLUSTER_RUN_DIR", raising=False)
    assert cfg._load_run_dir() is None
    monkeypatch.setattr(cfg, "RUN_DIR", None)
    (workspace / "data").mkdir()
    (workspace / "data" / "a.csv").write_text("x\n1\n")
    assert cfg.get_output_dir("data/a.csv") == str((workspace / "data" / "results").resolve())
    assert cfg.default_results_dir() == str((workspace / "results").resolve())


def test_run_dir_routes_outside_inputs_to_run_results(run_dir, workspace):
    assert run_dir.is_dir()
    (workspace / "data").mkdir()
    (workspace / "data" / "a.csv").write_text("x\n1\n")
    out = cfg.save_csv(pd.DataFrame({"x": [1]}), "data/a.csv", "fixed")
    assert out == str(run_dir.resolve() / "results" / "a_fixed.csv")
    assert not (workspace / "data" / "results").exists(), "nothing written next to shared input"


def test_run_dir_chains_stay_in_one_folder(run_dir):
    results = run_dir / "results"
    results.mkdir(parents=True)
    (results / "a_fixed.csv").write_text("x\n1\n")
    assert cfg.get_output_dir(f"{RUN}/results/a_fixed.csv") == str(results.resolve()), "no results/results nesting"
    (run_dir / "scratch").mkdir()
    (run_dir / "scratch" / "b.csv").write_text("x\n1\n")
    assert cfg.get_output_dir(f"{RUN}/scratch/b.csv") == str((run_dir / "scratch" / "results").resolve())


@pytest.mark.parametrize("bad", ["/abs/runs/1", "../runs/1", "runs", "other/1", "runs/../../x", "runs/../other"])
def test_invalid_run_dir_stops_the_server(workspace, monkeypatch, bad):
    monkeypatch.setenv("GEOCLUSTER_RUN_DIR", bad)
    with pytest.raises(ValueError):
        cfg._load_run_dir()


def test_run_dir_symlink_escape_rejected(workspace, monkeypatch, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (workspace / "runs").mkdir()
    (workspace / "runs" / "evil").symlink_to(outside)
    monkeypatch.setenv("GEOCLUSTER_RUN_DIR", "runs/evil")
    with pytest.raises(ValueError, match="outside workspace"):
        cfg._load_run_dir()


def test_provenance_default_is_run_results_not_server_cwd(run_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the old fallback resolved "results" against the server's cwd
    out = P.summarize_provenance()
    results = run_dir / "results"
    assert out["status"] == "success" and out["results_dir"] == str(results.resolve())
    assert (results / "provenance_log.json").is_file()
    assert not (tmp_path / "results").exists()
