"""Part 1 test — Run the full data pipeline and verify each stage."""

import os
import sys
import json

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from backend.tools.ingestion import load_dataset, detect_schema, get_dataset_info
from backend.tools.cleaning import clean_data
from backend.tools.preprocessing import preprocess_data
from backend.tools.eda import run_eda
from backend.graph import get_pipeline


DATA_PATH = os.path.join("data", "sample_ecommerce.csv")


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_individual_tools():
    """Test each tool module independently."""
    separator("1. INGESTION")

    df = load_dataset(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")

    schema = detect_schema(df)
    print(f"  Schema detected: {schema['num_rows']} rows, {schema['num_columns']} cols")

    info = get_dataset_info(df)
    print(f"  Memory: {info['memory_usage_mb']} MB")
    if "date_column" in info:
        print(f"  Date column: {info['date_column']}, range: {info['date_range']}")

    assert len(df) > 0, "DataFrame should not be empty"
    assert schema["num_rows"] == len(df), "Schema row count mismatch"

    # ── Cleaning ──
    separator("2. CLEANING")

    cleaned_df, cleaning_report = clean_data(df)
    print(f"  Initial rows: {cleaning_report['initial_rows']}")
    print(f"  Missing values detected: {cleaning_report['total_missing']}")
    print(f"  Duplicates detected: {cleaning_report['duplicates_detected']}")
    print(f"  Duplicates removed: {cleaning_report['duplicates_removed']}")
    print(f"  Outliers detected: {list(cleaning_report['outliers_detected'].keys())}")
    print(f"  Outliers handled: {list(cleaning_report['outliers_handled'].keys())}")
    print(f"  Final rows: {cleaning_report['final_rows']}")

    assert cleaning_report["total_missing"] > 0, "Should detect missing values"
    assert cleaning_report["duplicates_removed"] > 0, "Should detect duplicates"
    assert cleaned_df.isnull().sum().sum() == 0, "No missing values after cleaning"

    # ── Preprocessing ──
    separator("3. PREPROCESSING")

    processed_df, preprocess_report = preprocess_data(cleaned_df)
    print(f"  Columns encoded: {preprocess_report['columns_encoded']}")
    print(f"  Scaler info available for: {list(preprocess_report['scaler_info'].keys())}")
    print(f"  Note: {preprocess_report['note']}")

    assert len(preprocess_report["columns_encoded"]) > 0, "Should encode some columns"

    # ── EDA ──
    separator("4. EDA")

    eda_results = run_eda(cleaned_df)
    stats = eda_results["statistics"]
    corr = eda_results["correlations"]
    biz = eda_results["business_metrics"]

    print(f"  Statistics computed for: {list(stats.keys())}")
    print(f"  Correlations computed for: {list(corr.keys())}")

    if "total_revenue" in biz:
        print(f"  Total Revenue: Rs.{biz['total_revenue']:,.2f}")
    if "average_order_value" in biz:
        print(f"  Avg Order Value: Rs.{biz['average_order_value']:,.2f}")
    if "total_orders" in biz:
        print(f"  Total Orders: {biz['total_orders']}")
    if "total_customers" in biz:
        print(f"  Total Customers: {biz['total_customers']}")
    if "revenue_by_category" in biz:
        print(f"  Revenue by Category:")
        for cat, rev in biz["revenue_by_category"].items():
            print(f"    {cat}: Rs.{rev:,.2f}")
    if "revenue_by_region" in biz:
        print(f"  Revenue by Region:")
        for reg, rev in biz["revenue_by_region"].items():
            print(f"    {reg}: Rs.{rev:,.2f}")
    if "monthly_revenue" in biz:
        print(f"  Monthly Revenue Trend ({len(biz['monthly_revenue'])} months)")

    assert len(stats) > 0, "Should compute statistics"
    assert len(biz) > 0, "Should compute business metrics"

    return True


def test_langgraph_pipeline():
    """Test the full LangGraph pipeline end-to-end."""
    separator("5. LANGGRAPH PIPELINE (end-to-end)")

    pipeline = get_pipeline()
    result = pipeline.invoke({
        "dataset_name": DATA_PATH,
        "errors": [],
    })

    errors = result.get("errors", [])
    if errors:
        print(f"  [WARN] Pipeline errors: {errors}")
    else:
        print(f"  [PASS] Pipeline completed with no errors")

    # Verify all stages produced output
    checks = {
        "schema_info": "Schema detection",
        "raw_data": "Data ingestion",
        "cleaning_report": "Cleaning",
        "cleaned_data": "Cleaned data",
        "preprocessing_report": "Preprocessing",
        "processed_data": "Processed data",
        "eda_results": "EDA",
    }

    all_passed = True
    for key, label in checks.items():
        if result.get(key):
            print(f"  [PASS] {label}: present")
        else:
            print(f"  [FAIL] {label}: MISSING")
            all_passed = False

    # Summary stats from pipeline results
    schema = result.get("schema_info", {})
    cleaning = result.get("cleaning_report", {})
    eda = result.get("eda_results", {})

    print(f"\n  Pipeline Summary:")
    print(f"    Rows ingested: {schema.get('num_rows', 'N/A')}")
    print(f"    Rows after cleaning: {cleaning.get('final_rows', 'N/A')}")
    print(f"    Missing values handled: {cleaning.get('total_missing', 'N/A')}")
    print(f"    Duplicates removed: {cleaning.get('duplicates_removed', 'N/A')}")

    biz = eda.get("business_metrics", {})
    if biz:
        print(f"    Total Revenue: Rs.{biz.get('total_revenue', 0):,.2f}")
        print(f"    Total Customers: {biz.get('total_customers', 'N/A')}")

    assert all_passed, "Some pipeline stages missing"
    assert not errors, f"Pipeline had errors: {errors}"

    return True


def main():
    print("=" * 60)
    print("  AutoAnalytics — Part 1 Pipeline Test")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: Sample data not found at {DATA_PATH}")
        print("  Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    try:
        test_individual_tools()
        test_langgraph_pipeline()

        separator("RESULT: ALL TESTS PASSED")
        print("  Part 1 — Data Pipeline Foundation is fully functional.")
        print()

    except AssertionError as e:
        separator("RESULT: TEST FAILED")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        separator("RESULT: ERROR")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
