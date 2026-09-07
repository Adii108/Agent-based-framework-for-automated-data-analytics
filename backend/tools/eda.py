"""EDA tools — statistics, correlations, and business metrics."""

import pandas as pd
import numpy as np


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}

    stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col]
        stats[col] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "q75": round(float(series.quantile(0.75)), 2),
        }
    return stats


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}

    corr = numeric_df.corr()
    # Convert to nested dict with rounded values
    result = {}
    for col in corr.columns:
        result[col] = {
            row: round(float(corr.loc[row, col]), 3)
            for row in corr.index
        }
    return result


def get_column_summaries(df: pd.DataFrame) -> dict:
    """Value counts for categoricals, distribution info for numerics."""
    summaries = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summaries[col] = {
                "type": "numeric",
                "non_null": int(df[col].notna().sum()),
                "unique": int(df[col].nunique()),
            }
        else:
            value_counts = df[col].value_counts().head(10)
            summaries[col] = {
                "type": "categorical",
                "unique": int(df[col].nunique()),
                "top_values": {
                    str(k): int(v) for k, v in value_counts.items()
                },
            }
    return summaries


def calculate_business_metrics(df: pd.DataFrame) -> dict:
    """E-commerce business metrics computed from the dataset."""
    metrics = {}

    # Total revenue
    if "revenue" in df.columns:
        metrics["total_revenue"] = round(float(df["revenue"].sum()), 2)
        metrics["average_order_value"] = round(float(df["revenue"].mean()), 2)

    # Order count
    if "order_id" in df.columns:
        metrics["total_orders"] = int(df["order_id"].nunique())

    # Customer count
    if "customer_id" in df.columns:
        metrics["total_customers"] = int(df["customer_id"].nunique())

    # Revenue by category
    if "product_category" in df.columns and "revenue" in df.columns:
        rev_by_cat = (
            df.groupby("product_category")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        metrics["revenue_by_category"] = {
            str(k): round(float(v), 2) for k, v in rev_by_cat.items()
        }

    # Revenue by region
    if "region" in df.columns and "revenue" in df.columns:
        rev_by_region = (
            df.groupby("region")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        metrics["revenue_by_region"] = {
            str(k): round(float(v), 2) for k, v in rev_by_region.items()
        }

    # Monthly revenue trend
    if "order_date" in df.columns and "revenue" in df.columns:
        df_copy = df.copy()
        df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], errors="coerce")
        df_copy["month"] = df_copy["order_date"].dt.to_period("M")
        monthly = (
            df_copy.groupby("month")["revenue"]
            .sum()
            .sort_index()
        )
        metrics["monthly_revenue"] = {
            str(k): round(float(v), 2) for k, v in monthly.items()
        }

    # Top customers by revenue
    if "customer_id" in df.columns and "revenue" in df.columns:
        top_cust = (
            df.groupby("customer_id")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        metrics["top_customers"] = {
            str(k): round(float(v), 2) for k, v in top_cust.items()
        }

    # Orders by payment method
    if "payment_method" in df.columns:
        pay_counts = df["payment_method"].value_counts()
        metrics["orders_by_payment_method"] = {
            str(k): int(v) for k, v in pay_counts.items()
        }

    return metrics


def run_eda(df: pd.DataFrame) -> dict:
    """Run the full EDA pipeline. Returns combined results."""
    return {
        "statistics": calculate_statistics(df),
        "correlations": calculate_correlations(df),
        "column_summaries": get_column_summaries(df),
        "business_metrics": calculate_business_metrics(df),
    }
