"""Shared file logger for direct/ scripts.

These run as one-off subprocesses (spawned by the IDE button, or by hand),
not inside the long-lived MCP server -- so without this, nothing they do is
recorded anywhere. Added after a voxel_store wipe that couldn't be traced
because there was no log to look at.
"""

from __future__ import annotations

import logging
import os

_LOG_PATH = os.environ.get("DIRECT_VOXEL_LOG", "/var/log/theia/direct-voxel.log")
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"direct.{name}")
    if name not in _configured:
        _configured.add(name)
        logger.setLevel(logging.INFO)
        try:
            handler: logging.Handler = logging.FileHandler(_LOG_PATH)
        except OSError:
            handler = logging.StreamHandler()  # log dir not writable here -- stderr beats nothing
        handler.setFormatter(logging.Formatter("%(asctime)s pid=%(process)d %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
