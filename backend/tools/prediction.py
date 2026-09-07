"""Prediction tools — revenue forecasting (ARIMA) and customer churn (Logistic Regression)."""

import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def prepare_churn_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Engineer customer-level RFM features from transaction data.

    Features computed per customer:
        - recency: days since last purchase (from max date in dataset)
        - frequency: number of orders
        - monetary: total revenue
        - avg_order_value: average revenue per order
        - avg_quantity: average items per order
        - avg_discount: average discount used
        - distinct_categories: number of product categories purchased
        - distinct_regions: number of regions ordered from

    Churn label: customer with no purchase in last 60 days = churned (1).

    Returns:
        (feature_df, labels, feature_names)
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    max_date = df["order_date"].max()

    # Aggregate per customer
    customer_agg = df.groupby("customer_id").agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        avg_quantity=("quantity", "mean"),
        avg_discount=("discount", "mean"),
        distinct_categories=("product_category", "nunique"),
    ).reset_index()

    # Recency: days since last purchase
    customer_agg["recency"] = (max_date - customer_agg["last_purchase"]).dt.days

    # Churn label: no purchase in last 60 days
    customer_agg["churned"] = (customer_agg["recency"] > 60).astype(int)

    feature_names = [
        "recency", "frequency", "monetary", "avg_order_value",
        "avg_quantity", "avg_discount", "distinct_categories",
    ]

    X = customer_agg[feature_names].copy()
    y = customer_agg["churned"]

    # Store customer_id for later reference
    X.index = customer_agg["customer_id"].values

    return X, y, feature_names


def predict_churn(df: pd.DataFrame) -> dict:
    """Train a churn prediction model and return per-customer predictions.

    Returns:
        {
            "predictions": [{"customer_id": ..., "churn_probability": ..., "churned_predicted": ...}],
            "model_accuracy": float,
            "classification_report": str,
            "feature_names": [...],
            "model": LogisticRegression instance,
            "feature_data": DataFrame (for SHAP)
        }
    """
    X, y, feature_names = prepare_churn_features(df)

    if len(X) < 20:
        return {"error": "Not enough customers for churn prediction (need at least 20)"}

    # Check if we have both classes
    if y.nunique() < 2:
        return {"error": "Cannot train churn model: all customers are in the same class"}

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(y_test, y_pred, zero_division=0)

    # Predict on all customers
    probabilities = model.predict_proba(X)
    churn_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

    predictions = []
    for i, cust_id in enumerate(X.index):
        predictions.append({
            "customer_id": str(cust_id),
            "churn_probability": round(float(churn_probs[i]), 4),
            "churned_predicted": bool(churn_probs[i] >= 0.5),
        })

    # Sort by churn probability descending
    predictions.sort(key=lambda x: x["churn_probability"], reverse=True)

    return {
        "predictions": predictions,
        "model_accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "feature_names": feature_names,
        "total_customers": len(predictions),
        "predicted_churned": sum(1 for p in predictions if p["churned_predicted"]),
        "predicted_retained": sum(1 for p in predictions if not p["churned_predicted"]),
        "model": model,
        "feature_data": X,
    }


def generate_forecast(
    df: pd.DataFrame,
    date_col: str = "order_date",
    value_col: str = "revenue",
    periods: int = 90,
) -> dict:
    """Generate a revenue forecast using ARIMA.

    Aggregates daily revenue, fits an ARIMA model, and forecasts future periods.

    Returns:
        {
            "historical": [{"date": ..., "value": ...}],
            "forecast": [{"date": ..., "value": ..., "lower": ..., "upper": ...}],
            "model_info": {"order": ..., "aic": ...},
            "summary": {"last_30_days_avg": ..., "forecast_avg": ..., "trend": ...}
        }
    """
    from statsmodels.tsa.arima.model import ARIMA

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Aggregate daily revenue
    daily = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
    daily.columns = ["date", "value"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").set_index("date")

    # Fill missing dates with 0
    full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "date"

    if len(daily) < 30:
        return {"error": "Not enough data points for forecasting (need at least 30 days)"}

    # Fit ARIMA — use a simple order, suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(daily["value"], order=(5, 1, 2))
            fitted = model.fit()
        except Exception:
            # Fallback to simpler model
            try:
                model = ARIMA(daily["value"], order=(1, 1, 1))
                fitted = model.fit()
            except Exception as e:
                return {"error": f"ARIMA fitting failed: {str(e)}"}

    # Forecast
    forecast_result = fitted.get_forecast(steps=periods)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()

    # Historical data (last 90 days for context)
    historical = []
    for date, value in daily.tail(90).iterrows():
        historical.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(float(value["value"]), 2),
        })

    # Forecast data
    forecast = []
    for i, (date, value) in enumerate(forecast_mean.items()):
        lower = float(forecast_ci.iloc[i, 0]) if len(forecast_ci.columns) >= 2 else float(value) * 0.8
        upper = float(forecast_ci.iloc[i, 1]) if len(forecast_ci.columns) >= 2 else float(value) * 1.2
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(max(0, float(value)), 2),  # No negative revenue
            "lower": round(max(0, lower), 2),
            "upper": round(max(0, upper), 2),
        })

    # Summary statistics
    last_30 = daily.tail(30)["value"].mean()
    forecast_avg = np.mean([f["value"] for f in forecast])
    trend = "increasing" if forecast_avg > last_30 else "decreasing" if forecast_avg < last_30 else "stable"

    return {
        "historical": historical,
        "forecast": forecast,
        "model_info": {
            "order": str(fitted.specification["order"]),
            "aic": round(float(fitted.aic), 2),
        },
        "summary": {
            "last_30_days_avg": round(float(last_30), 2),
            "forecast_avg": round(float(forecast_avg), 2),
            "trend": trend,
            "forecast_periods": periods,
        },
    }
