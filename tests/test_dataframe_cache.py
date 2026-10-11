"""The DataFrame cache under concurrent use (the shared MCP server runs tools in threads for every conversation)."""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import pytest

import tools.dataframe_cache as C


@pytest.fixture(autouse=True)
def empty_cache(workspace):
    C.drop()
    yield
    C.drop()


def write(path, rows=100, label="a"):
    pd.DataFrame({"x": range(rows), "s": [f"{label}{i}" for i in range(rows)]}).to_csv(path, index=False)
    return str(path)


def replace(path, rows, label):
    """Rewrite atomically, as the tools' save_csv now does (a new inode, so a new cache version)."""
    tmp = f"{path}.tmp.csv"
    write(tmp, rows, label)
    os.replace(tmp, path)


def test_concurrent_misses_on_one_file_share_one_read(workspace, monkeypatch):
    path = write(workspace / "big.csv")
    reads = []
    real = C.read_tabular

    def slow(p, **kw):
        reads.append(p)
        time.sleep(0.3)
        return real(p, **kw)

    monkeypatch.setattr(C, "read_tabular", slow)
    frames = [None] * 6

    def load(i):
        frames[i] = C.load(path)

    threads = [threading.Thread(target=load, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reads) == 1, "one read for six concurrent loads"
    assert all(f is not None and f.equals(frames[0]) for f in frames)


def test_concurrent_loads_of_many_files_under_a_tiny_budget(workspace, monkeypatch):
    monkeypatch.setattr(C, "MAX_CACHE_MB", 1)
    paths = [write(workspace / f"f{i}.csv", rows=3000, label=str(i)) for i in range(6)]
    errors = []

    def load(p):
        try:
            for _ in range(5):
                C.load(p)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=load, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert C.stats()["total_mb"] <= 1


def test_callers_cannot_change_the_cached_frame(workspace):
    path = write(workspace / "a.csv", rows=5)
    hits = C.stats()["hits"]
    df = C.load(path)
    df["x"] = 0
    df.loc[0, "s"] = "changed"
    df.rename(columns={"s": "renamed"}, inplace=True)
    df.drop(index=[1], inplace=True)
    again = C.load(path)
    assert list(again.columns) == ["x", "s"]
    assert again["x"].tolist() == [0, 1, 2, 3, 4] and again.loc[0, "s"] == "a0" and len(again) == 5
    assert C.stats()["hits"] == hits + 1


def test_a_frame_over_half_the_budget_is_returned_but_not_cached(workspace, monkeypatch):
    monkeypatch.setattr(C, "MAX_CACHE_MB", 1)
    monkeypatch.setattr(C, "SKIP_OVER_HALF", True)  # the shared server
    small = write(workspace / "small.csv", rows=10)
    big = write(workspace / "big.csv", rows=40000)
    C.load(small)
    df = C.load(big)
    assert len(df) == 40000
    stats = C.stats()
    assert stats["uncached"] == 1 and stats["entries"] == 1, "the small frame was not evicted for the big one"
    assert C.get(small) is not None and C.get(big) is None


def test_a_deleted_file_drops_its_entry(workspace):
    path = write(workspace / "gone.csv")
    C.load(path)
    os.remove(path)
    with pytest.raises(FileNotFoundError):
        C.load(path)
    assert C.stats()["entries"] == 0


def test_a_replaced_file_is_reloaded(workspace):
    path = write(workspace / "v.csv", rows=3, label="old")
    assert C.load(path)["s"].tolist() == ["old0", "old1", "old2"]
    replace(path, rows=2, label="new")
    assert C.load(path)["s"].tolist() == ["new0", "new1"]


def test_get_returns_none_until_loaded_and_a_copy_after(workspace):
    path = write(workspace / "g.csv", rows=3)
    assert C.get(path) is None
    C.load(path)
    got = C.get(path)
    got["x"] = -1
    assert C.get(path)["x"].tolist() == [0, 1, 2]


def test_size_estimate_counts_string_columns(workspace):
    df = pd.DataFrame({"n": range(5000), "s": ["a fairly long string value " * 3] * 5000})
    deep = int(df.memory_usage(deep=True).sum())
    estimate = C._estimate_bytes(df)
    assert 0.7 * deep <= estimate <= 1.3 * deep


def test_a_single_conversation_server_caches_up_to_the_full_budget(workspace, monkeypatch):
    monkeypatch.setattr(C, "MAX_CACHE_MB", 1)
    assert C.SKIP_OVER_HALF is False  # api 2's per-conversation servers keep the old rule
    path = write(workspace / "mid.csv", rows=12000)
    df = C.load(path)
    assert 0.5 * 1024 * 1024 < C._estimate_bytes(df) <= 1024 * 1024, "the frame is between half and the full budget"
    assert C.get(path) is not None and C.stats()["entries"] == 1


def test_arrow_strings_are_not_counted_twice():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"s": pd.array([f"value {i}" for i in range(20000)], dtype="string[pyarrow]")})
    deep = int(df.memory_usage(deep=True).sum())
    assert C._estimate_bytes(df) <= 1.1 * deep
