import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import save_csv, read_tabular


def log_transform(path: str, columns: list[str], base: str = "e"):
    """Log transform columns (log1p). base: 'e' or '10'."""
    try:
        df = read_tabular(path)
        df_new = df.copy()

        for col in columns:
            if col not in df.columns:
                return f"Error: Column {col} not found."

            if (df[col] < 0).any():
                return f"Error: Column {col} contains negative values. Log transform impossible."

            if base == "10":
                df_new[f"{col}_log10"] = np.log1p(df[col]) / np.log(10)
            else:
                df_new[f"{col}_log"] = np.log1p(df[col])

        output_path = save_csv(df_new, path, "log")

        return {"output_path": output_path}
    except Exception as e:
        return f"Error: {str(e)}"


def standardize(path: str, columns: list[str]):
    """Z-score standardize columns (mean=0, std=1)."""
    try:
        df = read_tabular(path)
        scaler = StandardScaler()

        data = df[columns].fillna(df[columns].mean())
        transformed = scaler.fit_transform(data)

        df_new = df.copy()
        new_cols = []
        for i, col in enumerate(columns):
            new_col_name = f"{col}_z"
            df_new[new_col_name] = transformed[:, i]
            new_cols.append(new_col_name)

        output_path = save_csv(df_new, path, "std")

        return {"output_path": output_path}
    except Exception as e:
        return f"Error: {str(e)}"


def normalize(path: str, columns: list[str], method: str = "minmax"):
    """Normalize columns to [0,1] range."""
    try:
        df = read_tabular(path)

        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            return "Error: Only 'minmax' method supported currently."

        data = df[columns].fillna(df[columns].mean())
        transformed = scaler.fit_transform(data)

        df_new = df.copy()
        for i, col in enumerate(columns):
            df_new[f"{col}_norm"] = transformed[:, i]

        output_path = save_csv(df_new, path, "norm")
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        return f"Error: {str(e)}"


def smooth(path: str, columns: list[str], window: int = 3, method: str = "mean"):
    """Rolling window smooth. method: 'mean' or 'median'."""
    try:
        df = read_tabular(path)
        df_new = df.copy()

        for col in columns:
            if method == "mean":
                df_new[f"{col}_smooth"] = (
                    df[col].rolling(window=window, center=True).mean()
                )
            elif method == "median":
                df_new[f"{col}_smooth"] = (
                    df[col].rolling(window=window, center=True).median()
                )

        output_path = save_csv(df_new, path, f"smooth_{window}")
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        return f"Error: {str(e)}"
