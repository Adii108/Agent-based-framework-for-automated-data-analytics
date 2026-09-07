"""Preprocessing tools — encoding, scaling, and feature preparation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categoricals(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns df and encoding mappings."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Skip columns that look like IDs or dates
    skip_patterns = ["_id", "date", "order_id", "customer_id", "product_id"]
    columns = [
        c for c in columns
        if not any(p in c.lower() for p in skip_patterns)
    ]

    mappings = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        mappings[col] = {
            "classes": le.classes_.tolist(),
            "mapping": {label: int(idx) for idx, label in enumerate(le.classes_)},
        }

    return df, mappings


def scale_numericals(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """StandardScaler on numeric columns. Returns df and scaler info."""
    df = df.copy()
    if columns is None:
        # Scale only continuous numeric columns, skip IDs and encoded categoricals
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        skip_patterns = ["_id", "order_id", "customer_id", "product_id"]
        columns = [
            c for c in numeric_cols
            if not any(p in c.lower() for p in skip_patterns)
        ]

    if not columns:
        return df, {}

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    info = {}
    for i, col in enumerate(columns):
        info[col] = {
            "mean": round(float(scaler.mean_[i]), 4),
            "std": round(float(scaler.scale_[i]), 4),
        }

    return df, info


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run full preprocessing: encode categoricals, scale numericals.

    Note: Scaling is deferred — we encode categoricals now and store
    the scaler info, but actual scaling is applied only when preparing
    features for ML models (Part 3). This keeps the data human-readable
    for EDA and SQL queries.
    """
    report = {}

    df, encoding_mappings = encode_categoricals(df)
    report["categorical_encoding"] = encoding_mappings
    report["columns_encoded"] = list(encoding_mappings.keys())

    # Compute scaler info for reference but don't apply yet.
    # Scaling will be applied in prepare_features() for ML models.
    _, scaler_info = scale_numericals(df.copy())
    report["scaler_info"] = scaler_info
    report["note"] = "Scaling computed but not applied. Applied during ML feature preparation."

    return df, report
