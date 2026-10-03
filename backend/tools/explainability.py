"""Explainability tools — SHAP-based model explanations."""

import numpy as np
import pandas as pd


def generate_shap_explanation(model, X: pd.DataFrame, feature_names: list[str]) -> dict:
    """Generate SHAP values for a trained model.

    Uses a lightweight approach: for linear models (Logistic Regression),
    compute feature contributions directly from coefficients * feature values.
    Falls back to SHAP's LinearExplainer for proper Shapley values.

    Returns:
        {
            "global_importance": [{"feature": ..., "importance": ...}],
            "per_sample": [{"index": ..., "contributions": [...]}],
            "base_value": float
        }
    """
    X_array = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        mean_X = np.mean(X_array, axis=0)
        # Exact Shapley values for linear model: w_i * (x_i - E[x_i])
        shap_values = (X_array - mean_X) * coefs
        intercept = float(model.intercept_[0]) if hasattr(model.intercept_, "__len__") else float(model.intercept_)
        base_value = float(intercept + np.dot(mean_X, coefs))
    else:
        try:
            import shap
            explainer = shap.LinearExplainer(model, X_array)
            shap_values = explainer.shap_values(X_array)
            base_value = float(explainer.expected_value)
        except Exception:
            return {"error": "Cannot generate explanations for this model type"}


    # Global feature importance (mean absolute SHAP value)
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = []
    for i, name in enumerate(feature_names):
        global_importance.append({
            "feature": name,
            "importance": round(float(mean_abs[i]), 4),
        })
    global_importance.sort(key=lambda x: x["importance"], reverse=True)

    # Per-sample contributions (store all for later lookup)
    per_sample = {}
    sample_indices = X.index if isinstance(X, pd.DataFrame) else range(len(X_array))
    for i, idx in enumerate(sample_indices):
        contributions = []
        for j, name in enumerate(feature_names):
            val = float(shap_values[i][j])
            contributions.append({
                "feature": name,
                "contribution": round(val, 4),
                "direction": "increases churn" if val > 0 else "decreases churn",
                "feature_value": round(float(X_array[i][j]), 2),
            })
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        per_sample[str(idx)] = contributions

    return {
        "global_importance": global_importance,
        "per_sample": per_sample,
        "base_value": round(base_value, 4),
    }


def explain_customer_churn(
    customer_id: str,
    churn_results: dict,
    shap_results: dict,
) -> dict:
    """Extract a human-readable churn explanation for a specific customer.

    Returns:
        {
            "customer_id": str,
            "churn_probability": float,
            "top_factors": [{"feature": ..., "contribution": ..., "direction": ..., "feature_value": ...}],
            "summary": str
        }
    """
    # Find customer prediction
    prediction = None
    for p in churn_results.get("predictions", []):
        if p["customer_id"] == customer_id:
            prediction = p
            break

    if not prediction:
        return {"error": f"Customer {customer_id} not found in predictions"}

    # Find SHAP contributions
    contributions = shap_results.get("per_sample", {}).get(customer_id, [])
    if not contributions:
        return {"error": f"No SHAP explanation available for customer {customer_id}"}

    # Top 5 factors
    top_factors = contributions[:5]

    # Build summary text
    factor_lines = []
    for f in top_factors:
        factor_lines.append(
            f"- {f['feature']}: {f['feature_value']} ({f['direction']}, contribution: {f['contribution']:+.4f})"
        )

    summary = (
        f"Customer {customer_id} has a churn probability of {prediction['churn_probability']:.1%}.\n\n"
        f"Main contributing factors:\n" + "\n".join(factor_lines)
    )

    return {
        "customer_id": customer_id,
        "churn_probability": prediction["churn_probability"],
        "churned_predicted": prediction["churned_predicted"],
        "top_factors": top_factors,
        "summary": summary,
    }


def explain_forecast(forecast_result: dict, eda_results: dict | None = None) -> dict:
    """Generate a structured explanation of forecast results using actual data.

    This is NOT LLM-generated — it uses the computed forecast and EDA data.

    Returns:
        {
            "trend": str,
            "trend_detail": str,
            "evidence": [...],
            "forecast_summary": str
        }
    """
    summary = forecast_result.get("summary", {})
    trend = summary.get("trend", "unknown")
    last_30_avg = summary.get("last_30_days_avg", 0)
    forecast_avg = summary.get("forecast_avg", 0)
    periods = summary.get("forecast_periods", 90)

    # Calculate percentage change
    if last_30_avg > 0:
        pct_change = ((forecast_avg - last_30_avg) / last_30_avg) * 100
    else:
        pct_change = 0

    evidence = []
    evidence.append({
        "metric": "Recent daily average (last 30 days)",
        "value": f"Rs.{last_30_avg:,.2f}",
    })
    evidence.append({
        "metric": f"Forecast daily average (next {periods} days)",
        "value": f"Rs.{forecast_avg:,.2f}",
    })
    evidence.append({
        "metric": "Projected change",
        "value": f"{pct_change:+.1f}%",
    })

    # Add EDA context if available
    if eda_results:
        biz = eda_results.get("business_metrics", {})
        if "monthly_revenue" in biz:
            months = biz["monthly_revenue"]
            if len(months) >= 2:
                values = list(months.values())
                recent_trend = "growing" if values[-1] > values[-2] else "declining"
                evidence.append({
                    "metric": "Recent monthly trend",
                    "value": recent_trend,
                })

    trend_detail = (
        f"Based on ARIMA analysis, daily revenue is projected to "
        f"{'increase' if trend == 'increasing' else 'decrease' if trend == 'decreasing' else 'remain stable'} "
        f"by approximately {abs(pct_change):.1f}% over the next {periods} days."
    )

    forecast_summary = (
        f"The {periods}-day revenue forecast projects a {trend} trend. "
        f"Average daily revenue is expected to move from Rs.{last_30_avg:,.2f} "
        f"to Rs.{forecast_avg:,.2f} ({pct_change:+.1f}%)."
    )

    return {
        "trend": trend,
        "trend_detail": trend_detail,
        "evidence": evidence,
        "forecast_summary": forecast_summary,
        "percentage_change": round(pct_change, 2),
    }
