"""Per-project voxel store for GeoCluster.

A regular 3-D grid (projected metres by default) with named feature layers
persisted as ``layers/<name>.npy`` plus an ``index.json`` and an append-only
``operations.jsonl`` provenance log. Ported from combo-geology-nsl
``voxel_features`` (commit fd3091f) — see VOXEL_TRANSCRIPTION_PLAN.md at the
platform repo root for the deviations.

Module-level imports are numpy only: this package is used by the MCP server
venv (Python >= 3.12) and by the system-python publisher CLI.
"""

from .store import CATEGORICAL_EMPTY, FeatureLayer, GridSpec, VoxelStore
from .spatial import SpatialVoxelStore

__all__ = ["CATEGORICAL_EMPTY", "FeatureLayer", "GridSpec", "VoxelStore", "SpatialVoxelStore"]
