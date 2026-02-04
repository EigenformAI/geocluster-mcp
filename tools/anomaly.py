import pandas as pd
import numpy as np
import os


def _save_result(df, path, suffix):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(path).replace(".csv", f"_{suffix}.csv")
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    return output_path


def compute_anomaly(path: str, columns: list[str], method: str = "zscore"):
    """
    Compute anomaly scores to highlight interesting targets.
    Methods:
    - 'zscore': Standard deviation from mean (Good for normal data).
    - 'mad': Median Absolute Deviation (Better for data with crazy outliers).
    - 'ratio': Value divided by background (mean).
    """
    try:
        df = pd.read_csv(path)
        df_new = df.copy()

        for col in columns:
            if col not in df.columns:
                return f"Error: Column {col} not found."

            data = df[col]

            if method == "zscore":
                # (Value - Mean) / StdDev
                mean = data.mean()
                std = data.std()
                df_new[f"{col}_anom_z"] = (data - mean) / std

            elif method == "mad":
                # Robust Z-Score using Median
                median = data.median()
                mad = (data - median).abs().median()
                # 0.6745 constant makes it comparable to Z-score
                df_new[f"{col}_anom_mad"] = 0.6745 * (data - median) / mad

            elif method == "ratio":
                # Value / Mean (Times Background)
                bg = data.mean()
                df_new[f"{col}_anom_ratio"] = data / bg

            else:
                return "Error: Method not supported. Use 'zscore', 'mad', or 'ratio'."

        output_path = _save_result(df_new, path, f"anom_{method}")
        return {"status": "success", "method": method, "output_path": output_path}
    except Exception as e:
        return f"Error: {str(e)}"


def threshold(path: str, column: str, value: float, mode: str = "above"):
    """
    Filter data to keep only significant targets.
    mode: 'above' (Keep > value) or 'below' (Keep < value).
    """
    try:
        df = pd.read_csv(path)

        if column not in df.columns:
            return f"Error: Column {column} not found."

        if mode == "above":
            df_filtered = df[df[column] > value]
        elif mode == "below":
            df_filtered = df[df[column] < value]
        else:
            return "Error: Mode must be 'above' or 'below'."

        output_path = _save_result(df_filtered, path, f"thresh_{column}")

        return {
            "status": "success",
            "original_count": len(df),
            "filtered_count": len(df_filtered),
            "output_path": output_path,
        }
    except Exception as e:
        return f"Error: {str(e)}"


def rank_by_metric(path: str, metric: str, top_n: int = 10):
    """
    Sort the dataset to find the top N highest priority targets.
    """
    try:
        df = pd.read_csv(path)

        if metric not in df.columns:
            return f"Error: Column {metric} not found."

        # Sort descending (Highest first)
        df_sorted = df.sort_values(by=metric, ascending=False).head(top_n)

        output_path = _save_result(df_sorted, path, f"top_{top_n}_{metric}")

        return {
            "status": "success",
            "top_n": top_n,
            "top_targets": df_sorted.to_dict(orient="records"),
            "output_path": output_path,
        }
    except Exception as e:
        return f"Error: {str(e)}"
