"""Recommendation agent — generates evidence-backed business recommendations."""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import GROQ_API_KEY, GROQ_MODEL


RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strategic business advisor for an e-commerce company. Generate practical, actionable recommendations.

Rules:
- Base every recommendation on the evidence provided. Never invent metrics.
- Each recommendation must include: the action, why (evidence), expected impact, and timeframe.
- Focus on realistic, implementable actions for a small-to-medium e-commerce business.
- Prioritize recommendations by expected impact.
- Keep to 3-5 recommendations maximum.
- Be specific — avoid generic advice like "improve marketing".""",
    ),
    (
        "human",
        """Business Insights:
{insights}

Prediction & Explainability Data:
{prediction_data}

EDA Summary:
{eda_summary}

Generate prioritized recommendations:""",
    ),
])


def generate_recommendations(
    insights: str,
    eda_results: dict | None = None,
    prediction_results: dict | None = None,
    xai_results: dict | None = None,
) -> dict:
    """Generate evidence-backed business recommendations.

    Args:
        insights: Text output from the insight agent.
        eda_results: EDA metrics.
        prediction_results: Churn/forecast results.
        xai_results: SHAP/forecast explanations.

    Returns:
        {"recommendations": str, "data_sources": [...]}
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1500,
    )

    # Build prediction data summary
    pred_parts = []
    if prediction_results:
        churn = prediction_results.get("churn", {})
        if churn and "error" not in churn:
            pred_parts.append(f"Churn: {churn.get('predicted_churned', 0)} customers at risk out of {churn.get('total_customers', 0)}")
            top_risk = churn.get("predictions", [])[:3]
            for p in top_risk:
                pred_parts.append(f"  {p['customer_id']}: {p['churn_probability']:.1%}")

        forecast = prediction_results.get("forecast", {})
        if forecast and "error" not in forecast:
            summary = forecast.get("summary", {})
            pred_parts.append(f"Revenue forecast: {summary.get('trend', 'N/A')} trend")
            pred_parts.append(f"  Recent avg: Rs.{summary.get('last_30_days_avg', 0):,.2f}/day")
            pred_parts.append(f"  Forecast avg: Rs.{summary.get('forecast_avg', 0):,.2f}/day")

    if xai_results:
        shap = xai_results.get("shap", {})
        if shap:
            imp = shap.get("global_importance", [])[:3]
            if imp:
                pred_parts.append("Top churn factors: " + ", ".join(f['feature'] for f in imp))

    # EDA summary
    eda_parts = []
    if eda_results:
        biz = eda_results.get("business_metrics", {})
        if "revenue_by_category" in biz:
            top_cat = list(biz["revenue_by_category"].items())
            if top_cat:
                eda_parts.append(f"Top category: {top_cat[0][0]} (Rs.{top_cat[0][1]:,.2f})")
                if len(top_cat) > 1:
                    eda_parts.append(f"Lowest category: {top_cat[-1][0]} (Rs.{top_cat[-1][1]:,.2f})")
        if "revenue_by_region" in biz:
            top_reg = list(biz["revenue_by_region"].items())
            if top_reg:
                eda_parts.append(f"Top region: {top_reg[0][0]} (Rs.{top_reg[0][1]:,.2f})")

    chain = RECOMMENDATION_PROMPT | llm | StrOutputParser()

    recommendations = chain.invoke({
        "insights": insights,
        "prediction_data": "\n".join(pred_parts) if pred_parts else "No prediction data available.",
        "eda_summary": "\n".join(eda_parts) if eda_parts else "No EDA data available.",
    })

    sources = ["insights"]
    if prediction_results:
        sources.append("prediction_results")
    if xai_results:
        sources.append("xai_results")
    if eda_results:
        sources.append("eda_results")

    return {
        "recommendations": recommendations,
        "data_sources": sources,
    }
