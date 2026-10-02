"""Data Profiler and Semantic Context Generator for AutoAnalytics.

Constructs rich, privacy-preserving DataProfile structures from local datasets
without exposing raw tabular rows to the LLM agent. Profiles schemas,
distributions, cardinality, semantic types, and column relationships.
"""

from typing import Any
import numpy as np
import pandas as pd

from backend.state import ColumnProfile, DataProfile, AutoAnalyticsState
from backend.tools.ingestion import load_dataset


def profile_dataframe(df: pd.DataFrame, dataset_name: str = "dataset", dataset_path: str = "") -> DataProfile:
    """Generate a comprehensive, privacy-safe DataProfile from a Pandas DataFrame."""
    row_count = len(df)
    col_count = len(df.columns)
    
    columns_profile: dict[str, ColumnProfile] = {}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    id_cols: list[str] = []
    
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        null_cnt = int(series.isnull().sum())
        null_pct = round(float((null_cnt / row_count) * 100), 2) if row_count > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        is_unique = (unique_cnt == row_count)
        
        # Determine semantic type
        col_lower = str(col).lower()
        if is_unique and ("id" in col_lower or "code" in col_lower or "key" in col_lower):
            semantic_type = "identifier"
            id_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series) or "date" in col_lower or "time" in col_lower:
            # Check if parseable as datetime
            semantic_type = "datetime"
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Check if low-cardinality discrete numeric (e.g. status code 0/1)
            if unique_cnt <= 5 and not any(k in col_lower for k in ["amount", "revenue", "price", "cost", "quantity", "discount"]):
                semantic_type = "categorical"
                categorical_cols.append(col)
            else:
                semantic_type = "numeric"
                numeric_cols.append(col)
        else:
            semantic_type = "categorical"
            categorical_cols.append(col)
            
        stats: dict[str, float] = {}
        sample_summary: list[str] = []
        
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                q25 = float(non_null.quantile(0.25))
                q75 = float(non_null.quantile(0.75))
                stats = {
                    "min": round(float(non_null.min()), 2),
                    "max": round(float(non_null.max()), 2),
                    "mean": round(float(non_null.mean()), 2),
                    "median": round(float(non_null.median()), 2),
                    "std": round(float(non_null.std()), 2) if len(non_null) > 1 else 0.0,
                    "q25": round(q25, 2),
                    "q75": round(q75, 2),
                    "iqr": round(q75 - q25, 2),
                }
        elif semantic_type == "categorical":
            # Safe categorical summary: top-k category labels (without individual user associations)
            top_counts = series.value_counts(dropna=True).head(5)
            sample_summary = [f"{k} ({v})" for k, v in top_counts.items()]
            
        columns_profile[col] = {
            "name": str(col),
            "dtype": dtype_str,
            "semantic_type": semantic_type,
            "null_count": null_cnt,
            "null_percentage": null_pct,
            "unique_count": unique_cnt,
            "is_unique": is_unique,
            "sample_summary": sample_summary,
            "stats": stats,
        }

    # Discover lightweight inter-column relationships & potential correlations
    relationships = []
    if len(numeric_cols) >= 2:
        try:
            corr_mat = df[numeric_cols].corr()
            for i, c1 in enumerate(numeric_cols):
                for j, c2 in enumerate(numeric_cols):
                    if i < j:
                        c_val = corr_mat.loc[c1, c2]
                        if not np.isnan(c_val) and abs(c_val) >= 0.4:
                            relationships.append({
                                "type": "correlation",
                                "source": c1,
                                "target": c2,
                                "strength": round(float(c_val), 3),
                                "description": f"Strong correlation ({c_val:.2f}) between {c1} and {c2}",
                            })
        except Exception:
            pass

    # Detect primary entities / date ranges
    date_range: dict[str, str] = {}
    for dt_col in datetime_cols:
        try:
            dt_series = pd.to_datetime(df[dt_col], errors="coerce").dropna()
            if len(dt_series) > 0:
                date_range[dt_col] = {
                    "min": str(dt_series.min()),
                    "max": str(dt_series.max()),
                    "span_days": int((dt_series.max() - dt_series.min()).days),
                }
        except Exception:
            pass

    mem_usage = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3)
    
    # Synthesize semantic dataset description
    desc = (
        f"Dataset '{dataset_name}' with {row_count} records and {col_count} attributes. "
        f"Key identifiers: {id_cols or 'None'}; "
        f"Metrics: {numeric_cols}; "
        f"Segments: {categorical_cols}."
    )

    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns_profile,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "id_columns": id_cols,
        "relationships": relationships,
        "semantic_description": desc,
        "date_range": date_range,
        "memory_usage_mb": mem_usage,
    }


def profile_dataset_file(file_path: str) -> DataProfile:
    """Load and profile a dataset file from disk."""
    df = load_dataset(file_path)
    import os
    dataset_name = os.path.basename(file_path)
    return profile_dataframe(df, dataset_name=dataset_name, dataset_path=file_path)


def data_profiling_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node that profiles the local dataset and populates data_context."""
    dataset_path = state.get("dataset_name") or "data/sample_ecommerce.csv"
    try:
        profile = profile_dataset_file(dataset_path)
    except Exception as e:
        # Fallback profile
        profile = {
            "dataset_name": dataset_path,
            "dataset_path": dataset_path,
            "row_count": 0,
            "column_count": 0,
            "columns": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "id_columns": [],
            "relationships": [],
            "semantic_description": f"Error profiling dataset: {str(e)}",
            "date_range": {},
            "memory_usage_mb": 0.0,
        }

    trace_entry = {
        "node": "data_profiling",
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "numeric_columns": profile["numeric_columns"],
    }

    return {
        "data_context": profile,
        "schema_info": {
            "num_rows": profile["row_count"],
            "num_columns": profile["column_count"],
            "columns": list(profile["columns"].keys()),
            "dtypes": {k: v["dtype"] for k, v in profile["columns"].items()},
        },
        "trace": (state.get("trace") or []) + [trace_entry],
    }
