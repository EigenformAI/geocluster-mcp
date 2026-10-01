# NOTE: All heavy imports (pandas, rasterio, matplotlib) are deferred to
# function bodies for fast MCP server startup (<2s). Do NOT add top-level
# imports of heavy C-extension libraries here. Use inline imports instead.
# After the first call, Python's sys.modules cache makes repeated imports free.

import os
import re
import shutil
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

WORKSPACE_ROOT = os.environ.get("MCP_WORKSPACE_ROOT", os.getcwd())


def resolve_path(path: str) -> str:
    """
    Resolve a path to an absolute path within WORKSPACE_ROOT.
    Raises ValueError if the resolved path escapes the workspace.
    """
    if os.path.isabs(path):
        resolved = os.path.realpath(path)
    else:
        resolved = os.path.realpath(os.path.join(WORKSPACE_ROOT, path))

    workspace_real = os.path.realpath(WORKSPACE_ROOT)
    if not resolved.startswith(workspace_real + os.sep) and resolved != workspace_real:
        raise ValueError(
            f"Path '{path}' resolves to '{resolved}' which is outside workspace '{workspace_real}'"
        )
    return resolved


def read_tabular(path: str, **kwargs):
    """Read CSV, Excel, or LAS file into DataFrame. Relative paths are workspace paths (X-1)."""
    import pandas as pd

    path = resolve_path(path)  # was read as given: relative to the server's cwd, and unchecked
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, **kwargs)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(path, **kwargs)
    elif ext == ".las":
        import lasio

        las = lasio.read(path)
        df = las.df().reset_index()
        if "nrows" in kwargs:
            df = df.head(kwargs["nrows"])
        if "usecols" in kwargs:
            df = df[kwargs["usecols"]]
        return df
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .csv, .xlsx, or .las")


def validate_run_dir(raw: str, source: str = "GEOCLUSTER_RUN_DIR") -> str:
    """
    Check a conversation run folder and return its absolute path (design invariant X-7).

    It must be a relative folder under runs/ in the workspace, and it must stay inside
    WORKSPACE_ROOT after symlinks are resolved (X-1). The folder is created if missing.
    """
    rel = os.path.normpath(raw.strip())
    if os.path.isabs(rel) or rel.split(os.sep)[0] != "runs" or rel == "runs":
        raise ValueError(f"{source} must be a folder under runs/ in the workspace, got '{raw}'")
    run_dir = resolve_path(rel)  # X-1: stays inside WORKSPACE_ROOT (symlinks included)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# One folder, named like an OpenCode session id (ses_...) or a gateway id (13 digits). Checked on the raw value, so
# nothing is normalised into another conversation's folder or a subfolder of one.
_CALL_RUN_DIR_PATTERN = re.compile(r"runs/[A-Za-z0-9_-]{1,128}")


def validate_call_run_dir(raw) -> str:
    """The run folder of one tool call on the shared server (the `run_dir` argument): exactly runs/<id>."""
    if not isinstance(raw, str) or not _CALL_RUN_DIR_PATTERN.fullmatch(raw):
        raise ValueError(f"run_dir must be 'runs/<conversation id>' (letters, digits, '_' or '-'), got {raw!r}")
    return validate_run_dir(raw, source="run_dir")


def _load_run_dir():
    """
    Server-wide run folder from GEOCLUSTER_RUN_DIR (api 2: one server per conversation).

    The runtime gateway starts one MCP server per concurrent conversation with
    GEOCLUSTER_RUN_DIR=runs/<conversation id>, so conversations sharing a workspace never
    write over each other's outputs. Unset (the shared server, host CLI, tests): None and
    outputs go next to their input as before. An invalid value stops the server at startup
    instead of silently writing to the shared workspace (X-6).
    """
    raw = os.environ.get("GEOCLUSTER_RUN_DIR", "").strip()
    if not raw:
        return None
    return validate_run_dir(raw)


RUN_DIR = _load_run_dir()

# The run folder of the conversation that made THIS tool call, on the shared server (MCP_TRANSPORT=http). Set per call
# by tools.runctx.RunDirMiddleware from the `run_dir` argument the OpenCode run-dir plugin injects; worker threads get
# a copy of it (anyio.to_thread copies the context). Unset everywhere else, where RUN_DIR applies.
_CALL_RUN_DIR: ContextVar = ContextVar("geocluster_call_run_dir", default=None)


def current_run_dir():
    """This call's run folder, else the server-wide GEOCLUSTER_RUN_DIR, else None."""
    return _CALL_RUN_DIR.get() or RUN_DIR


def output_root() -> str:
    """Folder that owns this call's outputs: the conversation's run folder, else the workspace."""
    return current_run_dir() or os.path.realpath(WORKSPACE_ROOT)


def _contained(output_dir: str) -> str:
    """
    Return ``output_dir`` if it really lies in the run folder (or, without one, the workspace).

    Catches a results/ folder that is a symlink to somewhere else: the path looks right, the files would not be
    (X-1, X-7).
    """
    run_dir = current_run_dir()
    root = os.path.realpath(run_dir or WORKSPACE_ROOT)
    real = os.path.realpath(output_dir)
    if real != root and not real.startswith(root + os.sep):
        where = "this conversation's run folder" if run_dir else "the workspace"
        raise ValueError(
            f"output folder '{output_dir}' resolves to '{real}', outside {where} '{root}'. "
            "Next step: remove the link, or pass an input file that lies inside the workspace."
        )
    return output_dir


def default_results_dir() -> str:
    """results/ under output_root(). Used when a tool has no input file to put results next to."""
    output_dir = os.path.join(output_root(), "results")
    os.makedirs(output_dir, exist_ok=True)
    return _contained(output_dir)


def get_output_dir(input_path: str) -> str:
    """
    Get the output directory for results derived from ``input_path``.

    Without a run folder: a 'results' folder next to the input file (unchanged).
    With a run folder (the call's run_dir, or GEOCLUSTER_RUN_DIR): inputs outside it write to
    <run>/results/; inputs already inside it (a previous step's output) write next to themselves,
    so a chain of steps stays in one folder instead of nesting results/results/.

    Args:
        input_path: Path to the input file being processed

    Returns:
        Absolute path to the results directory
    """
    resolved_path = resolve_path(input_path)
    input_dir = os.path.dirname(os.path.abspath(resolved_path))
    run_dir = current_run_dir()
    if run_dir and not (input_dir == run_dir or input_dir.startswith(run_dir + os.sep)):
        return default_results_dir()
    if run_dir and os.path.basename(input_dir) == "results":
        output_dir = input_dir
    else:
        output_dir = os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    return _contained(output_dir)


# Path separators and control characters: a suffix built from a column name or a method argument must not add a folder.
_UNSAFE_IN_NAME = re.compile(r"[\x00-\x1f/\\]")


def _output_path(input_path: str, suffix: str, ext: str = None) -> str:
    """<output dir for input_path>/<input stem>_<suffix><ext, default the input's>."""
    resolved_path = resolve_path(input_path)
    output_dir = get_output_dir(resolved_path)
    name, orig_ext = os.path.splitext(os.path.basename(resolved_path))
    safe_suffix = _UNSAFE_IN_NAME.sub("_", str(suffix))
    return os.path.join(output_dir, f"{name}_{safe_suffix}{ext if ext else orig_ext}")


@contextmanager
def atomic_output(output_path: str, keep_ext: bool = True):
    """
    Yield a hidden temp path next to ``output_path``; on success it replaces ``output_path``.

    A parallel call or the DataFrame cache reading the file sees the old version or the new one, never a torn one,
    and a server stopped mid-write leaves no partial file under the final name. Keep the extension for writers that
    pick the format from it (savefig); without it, scanners looking for *.csv never meet a half-written temp.
    The stem is shortened so the temp name stays within NAME_MAX whenever the final name does.
    """
    folder, name = os.path.split(output_path)
    stem, ext = os.path.splitext(name)
    tmp = os.path.join(folder, f".{stem[:100]}.{uuid.uuid4().hex[:12]}.tmp{ext if keep_ext else ''}")
    try:
        yield tmp
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# Files GDAL attaches to a raster by name. A replaced raster must not inherit the old one's overviews or statistics.
_GDAL_AUXILIARY = (".aux.xml", ".ovr", ".msk")


@contextmanager
def atomic_raster(output_path: str):
    """
    Like atomic_output, for a raster that may come with sidecar files (ENVI/EHdr .hdr, .prj, .aux.xml).

    The raster is written under its final name into a hidden folder next to it, so GDAL names the sidecars right.
    Then stale auxiliary files of the old raster are removed and every new file is moved in, the main file last.
    """
    folder, name = os.path.split(output_path)
    staging = os.path.join(folder, f".tmp-{uuid.uuid4().hex[:12]}")
    os.mkdir(staging)
    try:
        yield os.path.join(staging, name)
        for suffix in _GDAL_AUXILIARY:
            try:
                os.unlink(output_path + suffix)
            except FileNotFoundError:
                pass
        for produced in sorted(os.listdir(staging), key=lambda f: f == name):
            os.replace(os.path.join(staging, produced), os.path.join(folder, produced))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def save_csv(df, input_path: str, suffix: str) -> str:
    """Save a DataFrame to results folder next to input file."""
    output_path = _output_path(input_path, suffix, ".csv")
    with atomic_output(output_path, keep_ext=False) as tmp:
        df.to_csv(tmp, index=False)
    return output_path


def save_raster(data, meta, input_path: str, suffix: str) -> str:
    """Save raster data to results folder next to input file."""
    import rasterio

    output_path = _output_path(input_path, suffix)
    with atomic_raster(output_path) as tmp:
        with rasterio.open(tmp, "w", **meta) as dest:
            if data.ndim == 3:
                for i in range(data.shape[0]):
                    dest.write(data[i], i + 1)
            else:
                dest.write(data, 1)

    return output_path


def save_plot(input_path: str, suffix: str) -> str:
    """Save matplotlib plot to results folder next to input file."""
    import matplotlib.pyplot as plt

    output_path = _output_path(input_path, suffix, ".png")
    with atomic_output(output_path) as tmp:
        plt.savefig(tmp, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def save_file(input_path: str, suffix: str, ext: str = None) -> str:
    """Get output path for any file type."""
    return _output_path(input_path, suffix, ext)
