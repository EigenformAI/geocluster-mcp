import os


def get_output_dir(input_path: str) -> str:
    """
    Get the output directory based on the input file's location.
    Creates a 'results' folder next to the input file.
    
    Args:
        input_path: Path to the input file being processed
        
    Returns:
        Absolute path to the results directory
    """
    input_dir = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_csv(df, input_path: str, suffix: str) -> str:
    """Save a DataFrame to results folder next to input file."""
    output_dir = get_output_dir(input_path)
    filename = os.path.basename(input_path).replace(".csv", f"_{suffix}.csv")
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    return output_path


def save_raster(data, meta, input_path: str, suffix: str) -> str:
    """Save raster data to results folder next to input file."""
    import rasterio
    
    output_dir = get_output_dir(input_path)
    filename = os.path.basename(input_path)
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
    
    output_dir = get_output_dir(input_path)
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_{suffix}.png")
    
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def save_file(input_path: str, suffix: str, ext: str = None) -> str:
    """Get output path for any file type."""
    output_dir = get_output_dir(input_path)
    filename = os.path.basename(input_path)
    name, orig_ext = os.path.splitext(filename)
    final_ext = ext if ext else orig_ext
    return os.path.join(output_dir, f"{name}_{suffix}{final_ext}")