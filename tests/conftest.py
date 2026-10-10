"""Shared pytest setup: put the MCP server root on sys.path and provide a
workspace fixture that monkeypatches tools.config.WORKSPACE_ROOT (it is read
at import time from MCP_WORKSPACE_ROOT)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """A throwaway workspace; tools resolve paths inside it."""
    import tools.config as cfg

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", str(ws))
    return ws
