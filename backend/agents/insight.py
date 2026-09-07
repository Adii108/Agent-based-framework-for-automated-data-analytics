"""Insight agent — generates business insights from actual computed data."""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import GROQ_API_KEY, GROQ_MODEL


INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a business data analyst. Generate actionable insights from the data provided.

Rules:
- ONLY reference metrics and numbers that appear in the data below. Never invent numbers.
- Focus on findings that a business owner would care about.
- For each insight, provide the evidence (the actual numbers).
- Be specific and concise. No generic advice.
- Structure your response as a numbered list of insights.
- If prediction/XAI data is provided, incorporate those findings too.""",
    ),
    (
        "human",
        """Here is the analysis data:

EDA Results:
{eda_summary}

Prediction Results:
{prediction_summary}

Explainability Results:
{xai_summary}

Generate business insights based on this data:""",
    ),
])


def _format_eda_summary(eda_results: dict) -> str:
    """Format EDA results into a concise text block for the prompt."""
    if not eda_results:
        return "No EDA data available."

    parts = []
    biz = eda_results.get("business_metrics", {})

    if "total_revenue" in biz:
        parts.append(f"Total Revenue: Rs.{biz['total_revenue']:,.2f}")
    if "average_order_value" in biz:
        parts.append(f"Average Order Value: Rs.{biz['average_order_value']:,.2f}")
    if "total_orders" in biz:
        parts.append(f"Total Orders: {biz['total_orders']}")
    if "total_customers" in biz:
        parts.append(f"Total Customers: {biz['total_customers']}")
    if "revenue_by_category" in biz:
        parts.append("Revenue by Category:")
        for cat, rev in biz["revenue_by_category"].items():
            parts.append(f"  {cat}: Rs.{rev:,.2f}")
    if "revenue_by_region" in biz:
        parts.append("Revenue by Region:")
        for reg, rev in biz["revenue_by_region"].items():
            parts.append(f"  {reg}: Rs.{rev:,.2f}")
    if "monthly_revenue" in biz:
        parts.append("Monthly Revenue Trend:")
        for month, rev in biz["monthly_revenue"].items():
            parts.append(f"  {month}: Rs.{rev:,.2f}")

    return "\n".join(parts) if parts else "No business metrics available."


def _format_prediction_summary(prediction_results: dict) -> str:
    """Format prediction results for the prompt."""
    if not prediction_results:
        return "No prediction data available."

    parts = []

    # Churn
    churn = prediction_results.get("churn", {})
    if churn and "error" not in churn:
        parts.append(f"Churn Model Accuracy: {churn.get('model_accuracy', 'N/A')}")
        parts.append(f"Total Customers: {churn.get('total_customers', 'N/A')}")
        parts.append(f"Predicted to Churn: {churn.get('predicted_churned', 'N/A')}")
        parts.append(f"Predicted to Stay: {churn.get('predicted_retained', 'N/A')}")
        # Top 5 at-risk customers
        top_risk = churn.get("predictions", [])[:5]
        if top_risk:
            parts.append("Top at-risk customers:")
            for p in top_risk:
                parts.append(f"  {p['customer_id']}: {p['churn_probability']:.1%} churn probability")

    # Forecast
    forecast = prediction_results.get("forecast", {})
    if forecast and "error" not in forecast:
        summary = forecast.get("summary", {})
        parts.append(f"\nForecast Trend: {summary.get('trend', 'N/A')}")
        parts.append(f"Recent Daily Avg: Rs.{summary.get('last_30_days_avg', 0):,.2f}")
        parts.append(f"Forecast Daily Avg: Rs.{summary.get('forecast_avg', 0):,.2f}")

    return "\n".join(parts) if parts else "No prediction data available."


def _format_xai_summary(xai_results: dict) -> str:
    """Format XAI results for the prompt."""
    if not xai_results:
        return "No explainability data available."

    parts = []

    # Global feature importance
    shap = xai_results.get("shap", {})
    if shap:
        global_imp = shap.get("global_importance", [])
        if global_imp:
            parts.append("Churn Model - Top Feature Importances (SHAP):")
            for f in global_imp[:5]:
                parts.append(f"  {f['feature']}: {f['importance']:.4f}")

    # Forecast explanation
    forecast_exp = xai_results.get("forecast_explanation", {})
    if forecast_exp:
        parts.append(f"\nForecast Explanation:")
        parts.append(forecast_exp.get("trend_detail", ""))
        for ev in forecast_exp.get("evidence", []):
            parts.append(f"  {ev['metric']}: {ev['value']}")

    return "\n".join(parts) if parts else "No explainability data available."


def generate_insights(
    eda_results: dict,
    prediction_results: dict | None = None,
    xai_results: dict | None = None,
) -> dict:
    """Generate business insights from actual computed data.

    Args:
        eda_results: EDA results from run_eda().
        prediction_results: Dict with 'churn' and/or 'forecast' keys.
        xai_results: Dict with 'shap' and/or 'forecast_explanation' keys.

    Returns:
        {"insights": str, "data_sources": [...]}
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1500,
    )

    chain = INSIGHT_PROMPT | llm | StrOutputParser()

    eda_text = _format_eda_summary(eda_results)
    pred_text = _format_prediction_summary(prediction_results or {})
    xai_text = _format_xai_summary(xai_results or {})

    insights = chain.invoke({
        "eda_summary": eda_text,
        "prediction_summary": pred_text,
        "xai_summary": xai_text,
    })

    # Track which data sources were used
    sources = ["eda_results"]
    if prediction_results:
        sources.append("prediction_results")
    if xai_results:
        sources.append("xai_results")

    return {
        "insights": insights,
        "data_sources": sources,
    }
