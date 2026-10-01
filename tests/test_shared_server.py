"""One MCP server for every conversation (MCP_TRANSPORT=http): per-call run folders, worker threads, startup checks.

The SSE server (api 2's per-conversation servers, the host CLI) must behave exactly as before; the shared server must
route every call to its own runs/<id>/, refuse calls it can't route (X-6, X-7), and run tools in threads that each see
their own run folder. Plan: docs/OPENCODE_MIGRATION_PLAN.md §5.2 in the IDE repo.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

import main
import tools.config as cfg
from tools import runctx


def run(coro):
    return asyncio.run(coro)


def text(result) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content)


async def call(server, name, args, meta=None):
    async with Client(server) as client:
        return await client.call_tool(name, args, raise_on_error=False, meta=meta)


@pytest.fixture
def data(workspace):
    (workspace / "data").mkdir()
    (workspace / "data" / "a.csv").write_text("x,y\n1,10\n2,20\n3,30\n")
    return "data/a.csv"


@pytest.fixture
def shared(monkeypatch):
    monkeypatch.setattr(runctx, "SHARED", True)
    monkeypatch.setattr(cfg, "RUN_DIR", None)


# --- the public API doesn't change ------------------------------------------------------------------------------


def test_threaded_registration_keeps_every_tool_schema():
    async def listing(server):
        async with Client(server) as client:
            return {t.name: t.model_dump() for t in await client.list_tools()}

    plain = FastMCP("plain")
    for tool in run(main.mcp.get_tools()).values():
        plain.tool()(getattr(tool.fn, "__wrapped__", tool.fn))
    wrapped, unwrapped = run(listing(main.mcp)), run(listing(plain))
    assert len(wrapped) == 56
    assert wrapped == unwrapped


# --- SSE: byte-compatible with the pre-OpenCode server -----------------------------------------------------------


def test_sse_mode_rejects_run_dir_like_any_unknown_argument(data, monkeypatch):
    monkeypatch.setattr(runctx, "SHARED", False)
    res = run(call(main.mcp, "normalize", {"path": data, "columns": ["x"], "run_dir": "runs/ses_A"}))
    assert res.is_error and "Unexpected keyword argument" in text(res)


def test_sse_mode_keeps_geocluster_run_dir(data, workspace, monkeypatch):
    monkeypatch.setattr(runctx, "SHARED", False)
    monkeypatch.setenv("GEOCLUSTER_RUN_DIR", "runs/1790000000000")
    monkeypatch.setattr(cfg, "RUN_DIR", cfg._load_run_dir())
    res = run(call(main.mcp, "normalize", {"path": data, "columns": ["x"]}))
    assert not res.is_error, text(res)
    assert (workspace / "runs/1790000000000/results/a_norm.csv").is_file()


# --- shared mode: routing and refusals ----------------------------------------------------------------------------


def test_each_call_writes_into_its_own_run_folder(data, workspace, shared):
    async def both():
        async with Client(main.mcp) as client:
            return await asyncio.gather(
                client.call_tool("normalize", {"path": data, "columns": ["x"], "run_dir": "runs/ses_A"}, raise_on_error=False),
                client.call_tool("normalize", {"path": data, "columns": ["y"], "run_dir": "runs/ses_B"}, raise_on_error=False),
            )

    a, b = run(both())
    assert not a.is_error and not b.is_error, (text(a), text(b))
    assert (workspace / "runs/ses_A/results/a_norm.csv").read_text().splitlines()[0] == "x,y,x_norm"
    assert (workspace / "runs/ses_B/results/a_norm.csv").read_text().splitlines()[0] == "x,y,y_norm"
    assert not (workspace / "data/results").exists(), "nothing next to the shared input"


def test_missing_run_dir_is_an_error_not_a_shared_write(data, workspace, shared):
    res = run(call(main.mcp, "normalize", {"path": data, "columns": ["x"]}))
    assert res.is_error
    msg = text(res)
    assert "geocluster normalize" in msg and "without run_dir" in msg and "do not retry" in msg  # X-6
    assert not (workspace / "data/results").exists() and not (workspace / "results").exists()


@pytest.mark.parametrize("bad", [
    "runs/ses_B/results", "runs/ses_A/../ses_B", "runs/a b", "runs/x\ny", " runs/x", "runs", "runs/", "../runs/x",
    "/abs/runs/x", "runs/..", "other/ses_A", 5,
])
def test_run_dir_must_be_exactly_one_run_folder(data, workspace, shared, bad):
    res = run(call(main.mcp, "normalize", {"path": data, "columns": ["x"], "run_dir": bad}))
    assert res.is_error and "run_dir must be 'runs/<conversation id>'" in text(res), text(res)
    assert not list((workspace / "runs").rglob("*.csv")) if (workspace / "runs").exists() else True


def test_run_dir_symlinked_outside_the_workspace_is_refused(data, workspace, shared, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (workspace / "runs").mkdir()
    (workspace / "runs" / "ses_evil").symlink_to(tmp_path / "elsewhere")
    res = run(call(main.mcp, "normalize", {"path": data, "columns": ["x"], "run_dir": "runs/ses_evil"}))
    assert res.is_error and "outside workspace" in text(res)
    assert not list((tmp_path / "elsewhere").iterdir())


def test_acting_session_from_meta_is_logged(data, shared):
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    runctx.log.addHandler(handler)
    try:
        res = run(call(main.mcp, "check_missing", {"path": data, "run_dir": "runs/ses_root"},
                       meta={"ai.opencode/sessionID": "ses_child"}))
    finally:
        runctx.log.removeHandler(handler)
    assert not res.is_error, text(res)
    assert any("call-start tool=check_missing run_dir=runs/ses_root session='ses_child'" in line for line in lines), lines
    assert any(line.startswith("call-end tool=check_missing") for line in lines)


# --- shared mode: threads ----------------------------------------------------------------------------------------


def _probe_server(fn):
    server = FastMCP("probe")
    server.add_middleware(runctx.RunDirMiddleware())
    server.tool()(runctx.threaded(fn))
    return server


def test_tool_threads_see_their_own_run_folder(workspace, shared):
    def where(delay: float) -> dict:
        time.sleep(delay)
        return {"root": cfg.output_root(), "thread": threading.current_thread().name}

    server = _probe_server(where)

    async def four():
        async with Client(server) as client:
            return await asyncio.gather(*[
                client.call_tool("where", {"delay": 0.3, "run_dir": f"runs/ses_{i}"}) for i in range(4)
            ])

    t0 = time.monotonic()
    results = run(four())
    assert time.monotonic() - t0 < 1.0, "the four calls ran in parallel"
    for i, res in enumerate(results):
        assert res.structured_content["root"] == str((workspace / f"runs/ses_{i}").resolve())
        assert res.structured_content["thread"] != threading.main_thread().name
    assert runctx.in_flight() == 0


def test_one_conversation_cannot_take_every_worker(workspace, shared, monkeypatch):
    monkeypatch.setattr(runctx, "TOOL_THREADS", 4)
    monkeypatch.setattr(runctx, "TOOL_THREADS_PER_RUN", 2)
    release = threading.Event()

    def slow(tag: str) -> str:
        if tag.startswith("A"):
            release.wait(5)
        return tag

    server = _probe_server(slow)

    async def scenario():
        async with Client(server) as client:
            a_calls = [asyncio.create_task(client.call_tool("slow", {"tag": f"A{i}", "run_dir": "runs/ses_A"}))
                       for i in range(3)]
            await asyncio.sleep(0.2)
            busy = runctx.in_flight()  # A holds only its 2 per-run slots; its third call waits outside the pool
            b = await asyncio.wait_for(client.call_tool("slow", {"tag": "B", "run_dir": "runs/ses_B"}), 2)
            release.set()
            await asyncio.gather(*a_calls)
            return busy, b

    busy, b = run(scenario())
    assert busy == 2
    assert b.structured_content["result"] == "B", "conversation B ran while A's calls were blocked"


def test_inline_outside_shared_mode(workspace, monkeypatch):
    monkeypatch.setattr(runctx, "SHARED", False)

    def where() -> str:
        return threading.current_thread().name

    res = run(call(_probe_server(where), "where", {}))
    assert res.structured_content["result"] == threading.main_thread().name


def test_warm_up_builds_sklearns_threadpool_controller():
    runctx.warm_up()
    import sklearn.utils.parallel as p

    assert getattr(p, "_threadpool_controller", None) is not None


def test_plot_tools_close_their_figures_even_on_errors(data, shared):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.visualization import plot_scatter

    out = plot_scatter(data, "x", "y", color_col="missing")  # returns an error after creating its figure
    assert "not found" in str(out)
    assert plt.get_fignums() == []


# --- startup -------------------------------------------------------------------------------------------------------


def test_configure_keeps_sse_defaults():
    assert main.configure({}) == {"transport": "sse", "host": "0.0.0.0", "port": 7654}
    assert main.configure({"MCP_HOST": "127.0.0.1", "MCP_PORT": "7701", "GEOCLUSTER_RUN_DIR": "runs/1"}) == {
        "transport": "sse", "host": "127.0.0.1", "port": 7701}


def test_configure_http_defaults_to_loopback():
    assert main.configure({"MCP_TRANSPORT": " HTTP "}) == {"transport": "http", "host": "127.0.0.1", "port": 7654}


@pytest.mark.parametrize("env, needle", [
    ({"MCP_TRANSPORT": "streamable-http"}, "must be 'sse' or 'http'"),
    ({"MCP_TRANSPORT": "http", "MCP_HOST": "0.0.0.0"}, "loopback"),
    ({"MCP_TRANSPORT": "http", "GEOCLUSTER_RUN_DIR": "runs/1"}, "GEOCLUSTER_RUN_DIR"),
    ({"MCP_TRANSPORT": "http", "GEOCLUSTER_TRAINING_ENABLED": "1"}, "GEOCLUSTER_TRAINING_ENABLED"),
    ({"MCP_TRANSPORT": "http", "MCP_TOOL_THREADS": "many"}, "MCP_TOOL_THREADS"),
    ({"MCP_TRANSPORT": "http", "MCP_TOOL_THREADS_PER_RUN": "0"}, "MCP_TOOL_THREADS_PER_RUN"),
])
def test_configure_refuses_unsafe_shared_server(env, needle):
    with pytest.raises(SystemExit, match=needle):
        main.configure(env)


def test_healthz_reports_mode_without_touching_mcp(shared):
    import httpx

    async def get():
        app = main.mcp.http_app(path="/mcp")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            return await client.get("/healthz")

    res = run(get())
    assert res.status_code == 200 and res.json() == {"ok": True, "mode": "shared", "in_flight": 0}


def test_main_registers_middleware_first():
    assert isinstance(main.mcp.middleware[0], runctx.RunDirMiddleware)
    assert Path(main.__file__).read_text().count("mcp.tool()(threaded(") == 56


def test_per_run_limit_holds_while_a_woken_call_takes_over(workspace, shared, monkeypatch):
    # anyio counts a woken waiter as a borrower only when it resumes; the run's limiter must not be dropped meanwhile
    monkeypatch.setattr(runctx, "TOOL_THREADS_PER_RUN", 1)
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def hold(seconds: float) -> str:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(seconds)
        with lock:
            state["now"] -= 1
        return "ok"

    server = _probe_server(hold)

    async def scenario():
        async with Client(server) as client:
            first = asyncio.create_task(client.call_tool("hold", {"seconds": 0.4, "run_dir": "runs/ses_A"}))
            await asyncio.sleep(0.1)
            second = asyncio.create_task(client.call_tool("hold", {"seconds": 0.4, "run_dir": "runs/ses_A"}))
            await first
            third = asyncio.create_task(client.call_tool("hold", {"seconds": 0.1, "run_dir": "runs/ses_A"}))
            await asyncio.gather(second, third)

    run(scenario())
    assert state["peak"] == 1
