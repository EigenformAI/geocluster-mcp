# NOTE: All heavy imports (pandas, rasterio, matplotlib) are deferred to
# function bodies for fast MCP server startup (<2s). Do NOT add top-level
# imports of heavy C-extension libraries here. Use inline imports instead.
# After the first call, Python's sys.modules cache makes repeated imports free.

import os

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
    """Read CSV, Excel, or LAS file into DataFrame."""
    import pandas as pd

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


def _load_run_dir():
    """
    This server's conversation output folder (design invariant X-7).

    The runtime gateway starts one MCP server per concurrent conversation with
    GEOCLUSTER_RUN_DIR=runs/<conversation id>, so conversations sharing a workspace never
    write over each other's outputs. Unset (the shared server, host CLI, tests): None and
    outputs go next to their input as before. An invalid value stops the server at startup
    instead of silently writing to the shared workspace (X-6).
    """
    raw = os.environ.get("GEOCLUSTER_RUN_DIR", "").strip()
    if not raw:
        return None
    rel = os.path.normpath(raw)
    if os.path.isabs(rel) or rel.split(os.sep)[0] != "runs" or rel == "runs":
        raise ValueError(f"GEOCLUSTER_RUN_DIR must be a folder under runs/ in the workspace, got '{raw}'")
    run_dir = resolve_path(rel)  # X-1: stays inside WORKSPACE_ROOT (symlinks included)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


RUN_DIR = _load_run_dir()


def output_root() -> str:
    """Folder that owns this server's outputs: the conversation's run folder, else the workspace."""
    return RUN_DIR or os.path.realpath(WORKSPACE_ROOT)


def default_results_dir() -> str:
    """results/ under output_root(). Used when a tool has no input file to put results next to."""
    output_dir = os.path.join(output_root(), "results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_output_dir(input_path: str) -> str:
    """
    Get the output directory for results derived from ``input_path``.

    Without a run folder: a 'results' folder next to the input file (unchanged).
    With a run folder (GEOCLUSTER_RUN_DIR): inputs outside it write to <run>/results/;
    inputs already inside it (a previous step's output) write next to themselves, so a
    chain of steps stays in one folder instead of nesting results/results/.

    Args:
        input_path: Path to the input file being processed

    Returns:
        Absolute path to the results directory
    """
    resolved_path = resolve_path(input_path)
    input_dir = os.path.dirname(os.path.abspath(resolved_path))
    if RUN_DIR and not (input_dir == RUN_DIR or input_dir.startswith(RUN_DIR + os.sep)):
        return default_results_dir()
    if RUN_DIR and os.path.basename(input_dir) == "results":
        output_dir = input_dir
    else:
        output_dir = os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_csv(df, input_path: str, suffix: str) -> str:
    """Save a DataFrame to results folder next to input file."""
    resolved_path = resolve_path(input_path)
    output_dir = get_output_dir(resolved_path)
    name = os.path.splitext(os.path.basename(resolved_path))[0]
    output_path = os.path.join(output_dir, f"{name}_{suffix}.csv")
    df.to_csv(output_path, index=False)
    return output_path


def save_raster(data, meta, input_path: str, suffix: str) -> str:
    """Save raster data to results folder next to input file."""
    import rasterio

    resolved_path = resolve_path(input_path)
    output_dir = get_output_dir(resolved_path)
    filename = os.path.basename(resolved_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_{suffix}{ext}")

    with rasterio.open(output_path, "w", **meta) as dest:
        if data.ndim == 3:
            for i in range(data.shape[0]):
                dest.write(data[i], i + 1)
        else:
            dest.write(data, 1)

    return output_path


def save_plot(input_path: str, suffix: str) -> str:
    """Save matplotlib plot to results folder next to input file."""
    import matplotlib.pyplot as plt

    resolved_path = resolve_path(input_path)
    output_dir = get_output_dir(resolved_path)
    filename = os.path.basename(resolved_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_{suffix}.png")

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def save_file(input_path: str, suffix: str, ext: str = None) -> str:
    """Get output path for any file type."""
    resolved_path = resolve_path(input_path)
    output_dir = get_output_dir(resolved_path)
    filename = os.path.basename(resolved_path)
    name, orig_ext = os.path.splitext(filename)
    final_ext = ext if ext else orig_ext
    return os.path.join(output_dir, f"{name}_{suffix}{final_ext}")
