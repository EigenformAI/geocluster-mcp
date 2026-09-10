"""Append-only JSONL provenance log for voxel operations.

Replaces the source repo's SQLite ``spatial_operations`` table: no native
extension, human-readable, and it travels with the store as a plain file.
One JSON object per line; ``reset_layer`` rewrites the file without a layer's
rows (``replace_layer`` semantics). Newest-first reads.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path)
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _LOCKS[key] = lk
        return lk


class OperationsLog:
    FILENAME = "operations.jsonl"

    def __init__(self, path: Path | str):
        self.path = Path(path)

    # --- write -----------------------------------------------------------------
    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        return self.append_many([entry])[0]

    def append_many(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stamped = []
        ts = datetime.now(timezone.utc).isoformat()
        for e in entries:
            row = {"ts": ts, **e}
            stamped.append(row)
        if not stamped:
            return stamped
        with _lock_for(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                for row in stamped:
                    f.write(json.dumps(row, default=str) + "\n")
        return stamped

    def reset_layer(self, layer: str) -> int:
        """Drop every row for ``layer``; returns the number removed."""
        with _lock_for(self.path):
            rows = self._read_all()
            keep = [r for r in rows if r.get("layer") != layer]
            removed = len(rows) - len(keep)
            if removed:
                tmp = self.path.with_suffix(f".jsonl.{os.getpid()}.tmp")
                with open(tmp, "w") as f:
                    for r in keep:
                        f.write(json.dumps(r, default=str) + "\n")
                os.replace(tmp, self.path)
            return removed

    # --- read ------------------------------------------------------------------
    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn trailing line never hides earlier provenance
        return rows

    def read(self, layer: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Rows newest-first, optionally filtered by layer and capped."""
        with _lock_for(self.path):
            rows = self._read_all()
        if layer is not None:
            rows = [r for r in rows if r.get("layer") == layer]
        rows.reverse()
        return rows[:limit] if limit else rows

    def count(self, layer: str | None = None) -> int:
        return len(self.read(layer=layer))

    def summary(self, layer: str) -> dict[str, Any]:
        """Compact provenance for a layer — what the manifest carries."""
        rows = self.read(layer=layer)
        kinds: dict[str, int] = {}
        sources: dict[str, int] = {}
        files: set[str] = set()
        for r in rows:
            kinds[r.get("op", "?")] = kinds.get(r.get("op", "?"), 0) + 1
            cs = str(r.get("coordinate_source") or "unknown")
            sources[cs] = sources.get(cs, 0) + 1
            if r.get("source_file"):
                files.add(str(r["source_file"]))
        return {
            "operations": len(rows),
            "kinds": dict(sorted(kinds.items())),
            "coordinate_source_counts": dict(sorted(sources.items())),
            "source_files": sorted(files),
            "first_ts": rows[-1]["ts"] if rows else None,
            "last_ts": rows[0]["ts"] if rows else None,
        }
