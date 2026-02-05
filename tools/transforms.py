import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import save_csv


def log_transform(path: str, columns: list[str], base: str = "e"):
    """
    Apply Log transformation (log1p to handle zeros).
    Critical for geochemical elements like Gold (Au) which follow log-normal distribution.
    """
    try:
        df = pd.read_csv(path)
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

        return {
            "status": "success",
            "transformed_columns": columns,
            "output_path": output_path,
            "preview": df_new[[c for c in df_new.columns if "log" in c]]
            .head(3)
            .to_dict(orient="records"),
        }
    except Exception as e:
        return f"Error: {str(e)}"


def standardize(path: str, columns: list[str]):
    """
    Z-Score Standardization (Mean=0, Std=1).
    Best for algorithms that assume Gaussian distribution (e.g., K-Means, PCA).
    """
    try:
        df = pd.read_csv(path)
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

        return {
            "status": "success",
            "output_path": output_path,
            "stats": df_new[new_cols].describe().to_dict(),
        }
    except Exception as e:
        return f"Error: {str(e)}"


def normalize(path: str, columns: list[str], method: str = "minmax"):
    """
    Scale data to a fixed range [0, 1].
    Useful for Neural Networks or visualization.
    """
    try:
        df = pd.read_csv(path)

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
    """
    Apply a rolling window smoothing to remove noise from drillhole/line data.
    """
    try:
        df = pd.read_csv(path)
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
