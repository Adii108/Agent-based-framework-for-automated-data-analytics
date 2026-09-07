"""Data cleaning tools — handle missing values, duplicates, and outliers."""

import pandas as pd
import numpy as np


def detect_missing_values(df: pd.DataFrame) -> dict:
    """Detect missing values per column."""
    missing = df.isnull().sum()
    total = len(df)
    result = {}
    for col in df.columns:
        count = int(missing[col])
        if count > 0:
            result[col] = {
                "missing_count": count,
                "missing_percentage": round(count / total * 100, 2),
            }
    return result


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill missing values: median for numeric, mode for categorical."""
    df = df.copy()
    report = {}

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "filled": missing_count,
                "method": "median",
                "fill_value": float(fill_value),
            }
        else:
            fill_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "filled": missing_count,
                "method": "mode",
                "fill_value": str(fill_value),
            }

    return df, report


def detect_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows."""
    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns cleaned df and count removed."""
    count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    return df, count


def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Detect outliers using IQR method on numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if outlier_count > 0:
            result[col] = {
                "outlier_count": outlier_count,
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
            }
    return result


def handle_outliers(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Cap outliers at IQR bounds."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    report = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        capped_lower = int((df[col] < lower).sum())
        capped_upper = int((df[col] > upper).sum())

        if capped_lower > 0 or capped_upper > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            report[col] = {
                "capped_lower": capped_lower,
                "capped_upper": capped_upper,
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
            }

    return df, report


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline. Returns cleaned df and report."""
    report = {
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
    }

    # Missing values
    missing_before = detect_missing_values(df)
    report["missing_values_detected"] = missing_before
    report["total_missing"] = sum(v["missing_count"] for v in missing_before.values())

    df, missing_report = handle_missing_values(df)
    report["missing_values_handled"] = missing_report

    # Duplicates
    dup_count = detect_duplicates(df)
    report["duplicates_detected"] = dup_count
    df, removed = remove_duplicates(df)
    report["duplicates_removed"] = removed

    # Outliers
    outliers = detect_outliers(df)
    report["outliers_detected"] = outliers
    df, outlier_report = handle_outliers(df)
    report["outliers_handled"] = outlier_report

    report["final_rows"] = len(df)
    report["final_columns"] = len(df.columns)

    return df, report
