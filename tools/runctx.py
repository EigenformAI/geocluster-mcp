"""One MCP server for every conversation (MCP_TRANSPORT=http; docs/OPENCODE_MIGRATION_PLAN.md §5.2 in the IDE repo).

SSE servers (api 2's per-conversation servers, the host CLI) keep their behaviour: SHARED stays False, tools run
inline on the event loop as FastMCP 2.14.5 does by itself, and a `run_dir` argument is rejected by FastMCP's argument
validation like any unknown keyword. main.py sets SHARED for the shared server, which then gets:

- RunDirMiddleware: pops the `run_dir` argument the OpenCode run-dir plugin adds to every geocluster_* call, checks it
  (exactly runs/<id>, inside WORKSPACE_ROOT: X-1, X-7) and sets the per-call ContextVar that
  tools.config.output_root()/get_output_dir() read. A missing or invalid value is a tool error (X-6), never a silent
  write into the shared workspace.
- threaded(): runs a sync tool in a worker thread (anyio.to_thread copies the ContextVar), so one conversation's slow
  tool doesn't stall the others. Bounded by a global limiter and a per-run limiter. A cancelled call returns at once
  and its thread runs to completion in the background (abandon_on_cancel=True); outputs are written atomically, so an
  abandoned or killed thread never leaves a partial file. Shielding the call instead kept a SIGTERM'd server alive
  until the tool finished (over 60 s for a UMAP, measured 2026-10-01). An abandoned thread no longer holds its limiter
  slot; in_flight() still counts it.
- warm_up(): builds sklearn's threadpool controller before any tool thread exists (see its docstring).

pyplot_locked() applies in both modes: pyplot keeps one process-global "current figure".
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import sys
import threading
import time
import weakref

import anyio
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from . import config as _cfg

log = logging.getLogger("geocluster.runctx")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(name)s %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False



def _int_env(name: str, default: int) -> int:
    # main.configure() rejects a bad value with a message; importing must not fail first
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


SHARED = False  # set by main.py for MCP_TRANSPORT=http
TOOL_THREADS = _int_env("MCP_TOOL_THREADS", 8)
TOOL_THREADS_PER_RUN = _int_env("MCP_TOOL_THREADS_PER_RUN", 4)

# OpenCode 2.0 sends the id of the session that made the call: a specialist's own (child) session, not the root
# conversation. Logged for tracing only; the folder always comes from run_dir.
OPENCODE_SESSION_META = "ai.opencode/sessionID"

_running = 0
_running_lock = threading.Lock()


def in_flight() -> int:
    """Tool threads running right now (shown by /healthz; a drain before restart waits for 0)."""
    return _running


def _acting_session(context: MiddlewareContext):
    # FastMCP rebuilds context.message without _meta; the request's _meta is on the request context.
    try:
        meta = context.fastmcp_context.request_context.meta
        return (meta.model_extra or {}).get(OPENCODE_SESSION_META) if meta is not None else None
    except Exception:  # noqa: BLE001 - tracing only
        return None


class RunDirMiddleware(Middleware):
    """Per-call run folder from the injected `run_dir` argument (X-7 on the shared server)."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        if not SHARED:
            return await call_next(context)
        name = getattr(context.message, "name", "?")
        args = context.message.arguments
        raw = args.pop("run_dir", None) if isinstance(args, dict) else None
        if raw is None:
            # The plugin sets run_dir on every call, so retrying can't help (X-6: tool, cause, next step).
            raise ToolError(
                f"geocluster {name}: this call arrived without run_dir, so its outputs have no conversation folder. "
                "Next step: do not retry; tell the user the analysis server did not receive this conversation's "
                "run folder."
            )
        try:
            run_dir = _cfg.validate_call_run_dir(raw)
        except ValueError as exc:
            raise ToolError(
                f"geocluster {name}: {exc}. Next step: do not retry; tell the user this conversation's run folder "
                "is invalid."
            ) from exc
        session = _acting_session(context)
        token = _cfg._CALL_RUN_DIR.set(run_dir)
        t0 = time.monotonic()
        log.info("call-start tool=%s run_dir=%s session=%r", name, raw, session)
        try:
            return await call_next(context)
        finally:
            _cfg._CALL_RUN_DIR.reset(token)
            log.info("call-end tool=%s run_dir=%s session=%r dur_ms=%d", name, raw, session,
                     (time.monotonic() - t0) * 1000)


class _Limiters:
    """
    Thread limiters of the running event loop (anyio limiters bind to one loop; tests start several).

    per_run holds weak references: a run's limiter lives exactly as long as some call of that run holds it or waits
    for it. Dropping it by reading borrowed/waiting counts was wrong: a woken waiter is not counted until it resumes,
    so the limiter was dropped under it and the run's next call got a fresh set of slots.
    """

    loop = None
    total: anyio.CapacityLimiter | None = None
    per_run: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _limiters(run_dir):
    loop = asyncio.get_running_loop()
    if _Limiters.loop is not loop:
        _Limiters.loop, _Limiters.total = loop, anyio.CapacityLimiter(TOOL_THREADS)
        _Limiters.per_run = weakref.WeakValueDictionary()
    if run_dir is None:
        return _Limiters.total, None
    per_run = _Limiters.per_run.get(run_dir)
    if per_run is None:
        per_run = _Limiters.per_run[run_dir] = anyio.CapacityLimiter(TOOL_THREADS_PER_RUN)
    return _Limiters.total, per_run


def threaded(fn):
    """Register a sync tool through this: in shared mode it runs in a worker thread, otherwise inline as before."""
    if inspect.iscoroutinefunction(fn):
        return fn

    def counted(*args, **kwargs):
        global _running
        with _running_lock:
            _running += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _running_lock:
                _running -= 1

    @functools.wraps(fn)
    async def _tool(*args, **kwargs):
        if not SHARED:
            return fn(*args, **kwargs)
        run_dir = _cfg._CALL_RUN_DIR.get()
        total, per_run = _limiters(run_dir)
        call = functools.partial(counted, *args, **kwargs)
        if per_run is None:
            return await anyio.to_thread.run_sync(call, limiter=total, abandon_on_cancel=True)
        async with per_run:  # one conversation can't take every worker (e.g. parallel stamps on one store lock)
            return await anyio.to_thread.run_sync(call, limiter=total, abandon_on_cancel=True)

    return _tool


def warm_up() -> None:
    """
    Build sklearn's threadpool controller in the main thread, before tool threads exist.

    The first KMeans.fit builds it by walking the loaded libraries with dl_iterate_phdr; its Python callback needs the
    GIL while glibc's loader lock is held. A first extension import in another thread (umap, numba) holds the GIL
    inside dlopen and waits for that loader lock: the whole process deadlocks, event loop included (8 of 8 cold pairs
    in the 2026-10-01 audit). sklearn caches the controller, so building it once removes the trigger. About 2 s.
    """
    t0 = time.monotonic()
    try:
        from sklearn.utils.parallel import _get_threadpool_controller

        _get_threadpool_controller()
    except ImportError:  # private helper moved: a tiny fit builds the same cached controller
        from sklearn.cluster import KMeans

        KMeans(n_clusters=1, n_init=1).fit([[0.0], [1.0]])
    log.info("warm-up: sklearn threadpool controller ready in %d ms", (time.monotonic() - t0) * 1000)


PYPLOT_LOCK = threading.Lock()


def pyplot_locked(fn):
    """Serialise pyplot use (its current-figure state is process-global) and leave no figure open, even on errors."""

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with PYPLOT_LOCK:
            try:
                return fn(*args, **kwargs)
            finally:
                plt = sys.modules.get("matplotlib.pyplot")
                if plt is not None:
                    plt.close("all")

    return _locked
