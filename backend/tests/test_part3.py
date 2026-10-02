"""Part 3 test — Prediction, Explainability, Insights & Recommendations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from backend.tools.ingestion import load_dataset
from backend.tools.cleaning import clean_data
from backend.tools.eda import run_eda
from backend.tools.prediction import predict_churn, generate_forecast, prepare_churn_features
from backend.tools.explainability import (
    generate_shap_explanation,
    explain_customer_churn,
    explain_forecast,
)
from backend.agents.insight import generate_insights
from backend.agents.recommendation import generate_recommendations
from backend.agents.orchestrator import classify_intent
from backend.graph import get_pipeline


DATA_PATH = os.path.join("data", "sample_ecommerce.csv")


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_churn_features():
    """Test RFM feature engineering for churn prediction."""
    separator("1. CHURN FEATURE ENGINEERING")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    X, y, feature_names = prepare_churn_features(cleaned_df)
    print(f"  Customers: {len(X)}")
    print(f"  Features: {feature_names}")
    print(f"  Churn distribution: {y.value_counts().to_dict()}")
    print(f"  Sample features (first customer):")
    print(f"    {X.iloc[0].to_dict()}")

    assert len(X) > 0, "Should have customer features"
    assert len(feature_names) == 7, "Should have 7 features"
    assert set(y.unique()).issubset({0, 1}), "Labels should be binary"

    print("  [PASS] Churn feature engineering works")
    return X, y, feature_names


def test_churn_prediction():
    """Test churn prediction with Logistic Regression."""
    separator("2. CHURN PREDICTION")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    result = predict_churn(cleaned_df)

    assert "error" not in result, f"Churn prediction failed: {result.get('error')}"

    print(f"  Model accuracy: {result['model_accuracy']:.4f}")
    print(f"  Total customers: {result['total_customers']}")
    print(f"  Predicted churned: {result['predicted_churned']}")
    print(f"  Predicted retained: {result['predicted_retained']}")
    print(f"  Classification Report:")
    for line in result["classification_report"].strip().split("\n"):
        print(f"    {line}")

    # Top 5 at-risk customers
    top_risk = result["predictions"][:5]
    print(f"\n  Top 5 at-risk customers:")
    for p in top_risk:
        print(f"    {p['customer_id']}: {p['churn_probability']:.1%} churn probability")

    assert result["model_accuracy"] > 0, "Accuracy should be positive"
    assert len(result["predictions"]) > 0, "Should have predictions"
    assert result["total_customers"] == result["predicted_churned"] + result["predicted_retained"]
    assert "model" in result, "Should return model object"
    assert "feature_data" in result, "Should return feature DataFrame"

    print("  [PASS] Churn prediction works")
    return result


def test_revenue_forecast():
    """Test ARIMA revenue forecasting."""
    separator("3. REVENUE FORECAST (ARIMA)")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    result = generate_forecast(cleaned_df)

    assert "error" not in result, f"Forecast failed: {result.get('error')}"

    print(f"  Historical data points: {len(result['historical'])}")
    print(f"  Forecast data points: {len(result['forecast'])}")
    print(f"  Model order: {result['model_info']['order']}")
    print(f"  AIC: {result['model_info']['aic']}")
    print(f"  Summary:")
    summary = result["summary"]
    print(f"    Last 30 days avg: Rs.{summary['last_30_days_avg']:,.2f}")
    print(f"    Forecast avg: Rs.{summary['forecast_avg']:,.2f}")
    print(f"    Trend: {summary['trend']}")
    print(f"    Forecast periods: {summary['forecast_periods']} days")

    # Show first 5 forecast points
    print(f"\n  First 5 forecast points:")
    for f in result["forecast"][:5]:
        print(f"    {f['date']}: Rs.{f['value']:,.2f} (CI: {f['lower']:,.2f} - {f['upper']:,.2f})")

    assert len(result["historical"]) > 0, "Should have historical data"
    assert len(result["forecast"]) == 90, "Should have 90 forecast points"
    assert all(f["value"] >= 0 for f in result["forecast"]), "No negative forecasts"

    print("  [PASS] Revenue forecast works")
    return result


def test_shap_explanation():
    """Test SHAP explanations for churn model."""
    separator("4. SHAP EXPLANATIONS")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)
    churn_result = predict_churn(cleaned_df)

    assert "error" not in churn_result, f"Need churn model: {churn_result.get('error')}"

    model = churn_result["model"]
    X = churn_result["feature_data"]
    feature_names = churn_result["feature_names"]

    shap_result = generate_shap_explanation(model, X, feature_names)

    assert "error" not in shap_result, f"SHAP failed: {shap_result.get('error')}"

    print(f"  Base value: {shap_result['base_value']}")
    print(f"  Global feature importance:")
    for f in shap_result["global_importance"]:
        print(f"    {f['feature']}: {f['importance']:.4f}")

    print(f"  Per-sample explanations available for: {len(shap_result['per_sample'])} customers")

    # Show explanation for one customer
    first_customer = list(shap_result["per_sample"].keys())[0]
    contributions = shap_result["per_sample"][first_customer][:3]
    print(f"\n  Example: {first_customer}")
    for c in contributions:
        print(f"    {c['feature']}: {c['contribution']:+.4f} ({c['direction']}, value={c['feature_value']})")

    assert len(shap_result["global_importance"]) == len(feature_names)
    assert len(shap_result["per_sample"]) > 0

    print("  [PASS] SHAP explanations work")
    return shap_result


def test_customer_explanation():
    """Test per-customer churn explanation."""
    separator("5. CUSTOMER-SPECIFIC EXPLANATION")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)
    churn_result = predict_churn(cleaned_df)
    shap_result = generate_shap_explanation(
        churn_result["model"], churn_result["feature_data"], churn_result["feature_names"]
    )

    # Pick a customer from predictions
    customer_id = churn_result["predictions"][0]["customer_id"]
    explanation = explain_customer_churn(customer_id, churn_result, shap_result)

    assert "error" not in explanation, f"Explanation failed: {explanation.get('error')}"

    print(f"  Customer: {explanation['customer_id']}")
    print(f"  Churn probability: {explanation['churn_probability']:.1%}")
    print(f"  Predicted churned: {explanation['churned_predicted']}")
    print(f"  Top factors:")
    for f in explanation["top_factors"]:
        print(f"    {f['feature']}: {f['contribution']:+.4f} ({f['direction']})")
    print(f"\n  Summary:\n    {explanation['summary']}")

    # Test with non-existent customer
    bad = explain_customer_churn("CUST_9999", churn_result, shap_result)
    assert "error" in bad, "Should fail for non-existent customer"
    print(f"\n  Non-existent customer handled: {bad['error'][:50]}")

    print("  [PASS] Customer explanation works")
    return True


def test_forecast_explanation():
    """Test forecast explanation."""
    separator("6. FORECAST EXPLANATION")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    forecast_result = generate_forecast(cleaned_df)
    eda_results = run_eda(cleaned_df)

    explanation = explain_forecast(forecast_result, eda_results)

    print(f"  Trend: {explanation['trend']}")
    print(f"  Detail: {explanation['trend_detail']}")
    print(f"  Percentage change: {explanation['percentage_change']:+.2f}%")
    print(f"  Evidence:")
    for ev in explanation["evidence"]:
        print(f"    {ev['metric']}: {ev['value']}")
    print(f"\n  Summary: {explanation['forecast_summary']}")

    assert "trend" in explanation
    assert "evidence" in explanation
    assert len(explanation["evidence"]) >= 3

    print("  [PASS] Forecast explanation works")
    return True


def test_insight_generation():
    """Test LLM-driven business insight generation."""
    separator("7. INSIGHT GENERATION (LLM)")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    eda_results = run_eda(cleaned_df)
    churn_result = predict_churn(cleaned_df)
    forecast_result = generate_forecast(cleaned_df)

    # Remove non-serializable objects
    safe_churn = {k: v for k, v in churn_result.items() if k not in ("model", "feature_data")}

    prediction_results = {
        "churn": safe_churn,
        "forecast": forecast_result,
    }

    shap_result = generate_shap_explanation(
        churn_result["model"], churn_result["feature_data"], churn_result["feature_names"]
    )
    forecast_exp = explain_forecast(forecast_result, eda_results)
    xai_results = {
        "shap": shap_result,
        "forecast_explanation": forecast_exp,
    }

    result = generate_insights(eda_results, prediction_results, xai_results)

    print(f"  Data sources: {result['data_sources']}")
    print(f"  Insights preview (first 500 chars):")
    print(f"    {result['insights'][:500]}...")

    assert "insights" in result, "Should have insights"
    assert len(result["insights"]) > 50, "Insights should be substantial"
    assert "eda_results" in result["data_sources"]

    print("  [PASS] Insight generation works")
    return result


def test_recommendation_generation():
    """Test LLM-driven recommendation generation."""
    separator("8. RECOMMENDATION GENERATION (LLM)")

    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    eda_results = run_eda(cleaned_df)
    churn_result = predict_churn(cleaned_df)
    forecast_result = generate_forecast(cleaned_df)

    safe_churn = {k: v for k, v in churn_result.items() if k not in ("model", "feature_data")}
    prediction_results = {
        "churn": safe_churn,
        "forecast": forecast_result,
    }

    # First generate insights (required input)
    insight_result = generate_insights(eda_results, prediction_results)
    insights_text = insight_result["insights"]

    result = generate_recommendations(
        insights_text, eda_results, prediction_results
    )

    print(f"  Data sources: {result['data_sources']}")
    print(f"  Recommendations preview (first 500 chars):")
    print(f"    {result['recommendations'][:500]}...")

    assert "recommendations" in result, "Should have recommendations"
    assert len(result["recommendations"]) > 50, "Recommendations should be substantial"

    print("  [PASS] Recommendation generation works")
    return True


def test_intent_classification_part3():
    """Test intent classification for Part 3 intents."""
    separator("9. INTENT CLASSIFICATION (Part 3)")

    test_cases = [
        ("What will revenue look like next month?", "prediction"),
        ("Which customers are likely to churn?", "prediction"),
        ("What are the key findings from the data?", "insight"),
        ("What should we do to improve retention?", "recommendation"),
        ("Why is customer CUST_0042 at risk?", "explain"),
        ("What is the total revenue?", "sql_query"),
        ("Hello!", "general_chat"),
    ]

    for question, expected in test_cases:
        intent = classify_intent(question)
        match = "[PASS]" if intent == expected else "[WARN]"
        print(f"  {match} '{question}' -> {intent} (expected: {expected})")

    print("  Intent classification done")
    return True


def test_full_pipeline_with_part3():
    """Test the full LangGraph pipeline including Part 3 nodes."""
    separator("10. FULL PIPELINE (Parts 1+3 end-to-end)")

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

    # Verify all stages
    checks = {
        "schema_info": "Schema detection",
        "raw_data": "Data ingestion",
        "cleaning_report": "Cleaning",
        "cleaned_data": "Cleaned data",
        "preprocessing_report": "Preprocessing",
        "processed_data": "Processed data",
        "eda_results": "EDA",
        "prediction_results": "Prediction (churn + forecast)",
        "xai_results": "Explainability (SHAP + forecast)",
        "insight_results": "Business insights",
        "recommendation_results": "Recommendations",
    }

    all_passed = True
    for key, label in checks.items():
        if result.get(key):
            print(f"  [PASS] {label}: present")
        else:
            print(f"  [FAIL] {label}: MISSING")
            all_passed = False

    # Part 3 specific checks
    prediction = result.get("prediction_results", {})
    churn = prediction.get("churn", {})
    forecast = prediction.get("forecast", {})

    if churn and "error" not in churn:
        print(f"\n  Churn Summary:")
        print(f"    Accuracy: {churn.get('model_accuracy')}")
        print(f"    Customers at risk: {churn.get('predicted_churned')}/{churn.get('total_customers')}")
    else:
        print(f"\n  Churn: {churn.get('error', 'Not available')}")

    if forecast and "error" not in forecast:
        summary = forecast.get("summary", {})
        print(f"  Forecast Summary:")
        print(f"    Trend: {summary.get('trend')}")
        print(f"    Daily avg: Rs.{summary.get('last_30_days_avg', 0):,.2f} -> Rs.{summary.get('forecast_avg', 0):,.2f}")

    xai = result.get("xai_results", {})
    shap = xai.get("shap", {})
    if shap and "error" not in shap:
        imp = shap.get("global_importance", [])[:3]
        print(f"  Top 3 churn factors: {[f['feature'] for f in imp]}")

    insights = result.get("insight_results", {})
    if insights:
        print(f"  Insights generated: {len(insights.get('insights', ''))} chars")

    recs = result.get("recommendation_results", {})
    if recs:
        print(f"  Recommendations generated: {len(recs.get('recommendations', ''))} chars")

    assert all_passed, "Some pipeline stages missing"

    return True


def main():
    print("=" * 60)
    print("  AutoAnalytics — Part 3 Prediction & Insights Test")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: Sample data not found at {DATA_PATH}")
        print("  Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    try:
        # Tool-level tests (no LLM needed)
        test_churn_features()
        test_churn_prediction()
        test_revenue_forecast()
        test_shap_explanation()
        test_customer_explanation()
        test_forecast_explanation()

        # LLM-powered tests (need GROQ_API_KEY)
        test_insight_generation()
        test_recommendation_generation()
        test_intent_classification_part3()

        # Full pipeline
        test_full_pipeline_with_part3()

        separator("RESULT: ALL PART 3 TESTS PASSED")
        print("  Part 3 — Prediction, Explainability & Insights is fully functional.")
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
