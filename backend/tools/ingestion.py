"""Data ingestion tools — load datasets and detect schema."""

import os
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from CSV, Excel, or JSON based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV, Excel, or JSON.")

    return df


def detect_schema(df: pd.DataFrame) -> dict:
    """Detect and return schema information for a DataFrame."""
    columns = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
        columns.append(col_info)

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": columns,
    }


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return high-level dataset information."""
    info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    # Detect date columns and report range
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.8:  # >80% parsed as dates
                    info["date_column"] = col
                    info["date_range"] = {
                        "min": str(parsed.min()),
                        "max": str(parsed.max()),
                    }
                    break
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            info["date_column"] = col
            info["date_range"] = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
            }
            break

    return info
