"""
Module-level DataFrame cache with version-based invalidation and memory budget.

Shared across all tool calls within the MCP server process. On the shared server
(MCP_TRANSPORT=http) that means every conversation, from worker threads at once:
- the lock only guards dict and integer work; reading a file and sizing a frame
  happen outside it (sizing a string-heavy frame with memory_usage(deep=True)
  took seconds and held the GIL);
- concurrent misses on one file version share a single read;
- callers get a shallow copy, so pandas' public mutation APIs (df[c] = ...,
  inplace=True, .loc assignment, rename, columns =) can't change the cached
  frame. Writing through .values/.array/._mgr of a returned frame still can:
  a tool that mutates takes .copy() first, as the cleaning tools do;
- on the shared server (SKIP_OVER_HALF, set by main.py), a frame larger than half
  the budget is returned without being cached instead of evicting every other
  conversation's frames. A per-conversation server caches up to the full budget,
  as before.

Entries are keyed by resolved path and invalidated when the file's
(mtime_ns, size, inode) changes; least-recently-used entries are evicted when the
estimated total exceeds MAX_CACHE_MB.

NOTE: Heavy imports (pandas) are deferred to function bodies for fast startup.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd

from .config import resolve_path, read_tabular

logger = logging.getLogger("geocluster-mcp.cache")

MAX_CACHE_MB = int(os.environ.get("MCP_CACHE_MAX_MB", "512"))
SKIP_OVER_HALF = False  # main.py sets it for the shared server (MCP_TRANSPORT=http)
SAMPLE_VALUES = 1000  # values sampled per string/object column to estimate its memory

# resolved_path -> (df, version, last_access_time, estimated_bytes); version = (mtime_ns, size, inode)
_cache: dict[str, tuple] = {}
_total_bytes = 0
# (resolved_path, version) -> Future of the read in progress
_inflight: dict[tuple, Future] = {}
_lock = threading.Lock()

_stats = {"hits": 0, "misses": 0, "evictions": 0, "uncached": 0}


def _version(resolved: str) -> tuple:
    st = os.stat(resolved)
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _estimate_bytes(df) -> int:
    """Memory of ``df``: exact for fixed-width columns, sampled for object/string ones (O(sample), not O(rows))."""
    import pandas as pd

    total = int(df.memory_usage(index=True, deep=False).sum())
    rows = len(df)
    if rows == 0:
        return total
    step = max(1, rows // SAMPLE_VALUES)
    for i, dtype in enumerate(df.dtypes):
        # Python objects are counted as pointers by deep=False; Arrow-backed strings are already counted in full
        if pd.api.types.is_object_dtype(dtype) or (isinstance(dtype, pd.StringDtype) and dtype.storage == "python"):
            sample = df.iloc[::step, i].iloc[:SAMPLE_VALUES]
            total += int(sum(sys.getsizeof(v) for v in sample) / max(1, len(sample)) * rows)
    return total


def _max_bytes() -> int:
    return MAX_CACHE_MB * 1024 * 1024


def _forget_locked(resolved: str) -> None:
    global _total_bytes
    entry = _cache.pop(resolved, None)
    if entry is not None:
        _total_bytes -= entry[3]


def _evict_locked() -> list:
    """Evict least-recently-accessed entries until under budget. Returns (path, bytes) of each eviction."""
    evicted = []
    while _cache and _total_bytes > _max_bytes():
        lru_path = min(_cache, key=lambda p: _cache[p][2])
        nbytes = _cache[lru_path][3]
        _forget_locked(lru_path)
        _stats["evictions"] += 1
        evicted.append((lru_path, nbytes))
    return evicted


def load(path: str, force_reload: bool = False) -> pd.DataFrame:
    """Load DataFrame, cache by resolved path. Auto-invalidates on file change."""
    global _total_bytes
    resolved = resolve_path(path)
    try:
        version = _version(resolved)
    except FileNotFoundError:
        with _lock:
            _forget_locked(resolved)
        raise
    key = (resolved, version)

    frame = flight = None
    with _lock:
        hit = _cache.get(resolved)
        if not force_reload and hit is not None and hit[1] == version:
            _cache[resolved] = (hit[0], version, time.monotonic(), hit[3])
            _stats["hits"] += 1
            frame = hit[0]
        else:
            flight = None if force_reload else _inflight.get(key)
            if flight is None:
                mine = Future()
                if not force_reload:
                    _inflight[key] = mine
    if frame is not None:  # the copy walks every column: made outside the lock
        logger.debug(f"Cache HIT: {resolved}")
        return frame.copy(deep=False)

    if flight is not None:  # another call is reading this exact version: share its frame
        return flight.result().copy(deep=False)

    try:
        df = read_tabular(resolved)
        nbytes = _estimate_bytes(df)
    except BaseException as exc:
        with _lock:
            if _inflight.get(key) is mine:
                del _inflight[key]
        mine.set_exception(exc)
        raise

    with _lock:
        if _inflight.get(key) is mine:
            del _inflight[key]
        _stats["misses"] += 1
        cached = nbytes <= (_max_bytes() // 2 if SKIP_OVER_HALF else _max_bytes())
        if cached:
            _forget_locked(resolved)
            _cache[resolved] = (df, version, time.monotonic(), nbytes)  # stamped at insert, not at the start of the read
            _total_bytes += nbytes
            evicted = _evict_locked()
        else:
            _stats["uncached"] += 1
            evicted = []
        total = _total_bytes
    mine.set_result(df)

    mb = nbytes / (1024 * 1024)
    if cached:
        logger.info(f"Cache LOAD: {resolved} ({len(df)} rows, ~{mb:.1f}MB, total cached: ~{total / (1024 * 1024):.1f}MB)")
    else:
        logger.info(f"Cache SKIP: {resolved} ({len(df)} rows, ~{mb:.1f}MB, too large for the {MAX_CACHE_MB}MB budget)")
    for lru_path, lru_bytes in evicted:
        logger.info(f"Cache EVICT: {lru_path} (~{lru_bytes / (1024 * 1024):.1f}MB)")
    return df.copy(deep=False)


def get(path: str) -> Optional[pd.DataFrame]:
    """Get cached DataFrame if available and fresh, else None."""
    resolved = resolve_path(path)
    with _lock:
        if resolved not in _cache:
            return None
    try:
        version = _version(resolved)
    except FileNotFoundError:
        with _lock:
            _forget_locked(resolved)
        raise
    with _lock:
        hit = _cache.get(resolved)
        if hit is None:
            return None
        fresh = hit[1] == version
        if fresh:
            _cache[resolved] = (hit[0], version, time.monotonic(), hit[3])
            _stats["hits"] += 1
        else:
            _forget_locked(resolved)
            _stats["misses"] += 1
    if fresh:
        return hit[0].copy(deep=False)
    logger.info(f"Cache INVALIDATE (file changed): {resolved}")
    return None


def drop(path: str = None):
    """Drop one or all cached DataFrames."""
    global _total_bytes
    with _lock:
        if path:
            _forget_locked(resolve_path(path))
        else:
            _cache.clear()
            _total_bytes = 0


def stats() -> dict:
    """Return cache statistics for diagnostics."""
    with _lock:
        return {
            **_stats,
            "entries": len(_cache),
            "total_mb": round(_total_bytes / (1024 * 1024), 1),
            "max_mb": MAX_CACHE_MB,
        }
