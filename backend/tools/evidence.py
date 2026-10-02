"""Evidence Extraction Layer for AutoAnalytics.

Transforms raw execution outcomes and intermediate analytical artifacts
into structured, verifiable EvidenceItems establishing full analytical traceability:
Claim -> Evidence -> Analysis Step -> Executed Code -> Data Source.
"""

from typing import Any
import uuid
from backend.state import EvidenceItem, ExecutionRecord, AutoAnalyticsState, TaskSpecification


def extract_evidence_from_execution(
    history: list[ExecutionRecord],
    task_spec: TaskSpecification | None = None,
) -> list[EvidenceItem]:
    """Convert execution records and outputs into structured, traceable evidence items."""
    evidence_items: list[EvidenceItem] = []

    for record in history:
        if not record.get("success"):
            continue

        step_id = record.get("step_id", 1)
        code_ref = record.get("code", "")[:120].strip().replace("\n", " ")
        ret_val = record.get("return_value")
        metrics = record.get("extracted_metrics", {})

        # 1. Process structured return_values
        if isinstance(ret_val, dict):
            # Forecasting evidence
            if "forecast_avg" in ret_val or "trend" in ret_val:
                trend = ret_val.get("trend", "stable")
                fc_avg = ret_val.get("forecast_avg", 0.0)
                hist_avg = ret_val.get("historical_avg", 0.0)
                evidence_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "claim": f"Projected revenue trend is {trend} with forecasted daily average of Rs.{fc_avg:,.2f} compared to historical average of Rs.{hist_avg:,.2f}.",
                    "metric_name": "forecasted_daily_revenue_avg",
                    "metric_value": fc_avg,
                    "step_id": step_id,
                    "code_reference": code_ref,
                    "artifact_id": "forecast_series",
                    "statistical_support": {
                        "test_name": "ARIMA / Autoregressive Trend Projection",
                        "p_value": None,
                        "statistic_value": None,
                        "confidence_interval": None,
                        "sample_size": ret_val.get("points_count", 90),
                        "effect_size": None,
                    },
                    "confidence_score": 0.90,
                })

            # Classification / Churn evidence
            if "accuracy" in ret_val or "predicted_churned" in ret_val:
                acc = ret_val.get("accuracy", 0.0)
                churned = ret_val.get("predicted_churned", 0)
                total = ret_val.get("total_customers", 0)
                churn_pct = round((churned / total * 100), 1) if total > 0 else 0.0
                evidence_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "claim": f"Churn predictive model identified {churned} of {total} customers ({churn_pct}%) at high churn risk with validation accuracy of {acc:.1%}.",
                    "metric_name": "predicted_churn_rate",
                    "metric_value": churn_pct,
                    "step_id": step_id,
                    "code_reference": code_ref,
                    "artifact_id": "churn_model_predictions",
                    "statistical_support": {
                        "test_name": "Logistic Regression Classification",
                        "p_value": None,
                        "statistic_value": acc,
                        "confidence_interval": None,
                        "sample_size": total,
                        "effect_size": None,
                    },
                    "confidence_score": round(float(acc), 2) if acc else 0.85,
                })

            # Explainability evidence
            if "global_importance" in ret_val:
                top_factors = ret_val.get("global_importance", [])[:3]
                factor_names = [f.get("feature", "unknown") for f in top_factors]
                evidence_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "claim": f"SHAP attribution identifies primary drivers of customer behavior as: {', '.join(factor_names)}.",
                    "metric_name": "shap_feature_importance",
                    "metric_value": factor_names,
                    "step_id": step_id,
                    "code_reference": code_ref,
                    "artifact_id": "shap_global_importance",
                    "statistical_support": {
                        "test_name": "SHAP LinearExplainer Attribution",
                        "p_value": None,
                        "statistic_value": None,
                        "confidence_interval": None,
                        "sample_size": None,
                        "effect_size": None,
                    },
                    "confidence_score": 0.92,
                })

            # Breakdown / Aggregation evidence
            if "breakdown" in ret_val:
                breakdown = ret_val.get("breakdown", [])
                if breakdown and isinstance(breakdown, list):
                    top_seg = breakdown[0]
                    dim_key = [k for k in top_seg.keys() if k not in ("sum", "mean", "count")][0] if top_seg else "segment"
                    evidence_items.append({
                        "id": str(uuid.uuid4())[:8],
                        "claim": f"Leading segment for {dim_key} is '{top_seg.get(dim_key)}' generating sum of {top_seg.get('sum', 0):,} with average {top_seg.get('mean', 0):,}.",
                        "metric_name": f"{dim_key}_top_segment_contribution",
                        "metric_value": top_seg.get("sum"),
                        "step_id": step_id,
                        "code_reference": code_ref,
                        "artifact_id": "segment_aggregation_table",
                        "statistical_support": {
                            "test_name": "Segment Aggregation",
                            "p_value": None,
                            "statistic_value": None,
                            "confidence_interval": None,
                            "sample_size": len(breakdown),
                            "effect_size": None,
                        },
                        "confidence_score": 0.95,
                    })

        # 2. Process scalar metrics in locals
        for m_name, m_val in metrics.items():
            if isinstance(m_val, (int, float)) and not any(e["metric_name"] == m_name for e in evidence_items):
                evidence_items.append({
                    "id": str(uuid.uuid4())[:8],
                    "claim": f"Computed metric '{m_name}' evaluated to {m_val}.",
                    "metric_name": m_name,
                    "metric_value": m_val,
                    "step_id": step_id,
                    "code_reference": code_ref,
                    "artifact_id": None,
                    "statistical_support": None,
                    "confidence_score": 0.90,
                })

    return evidence_items


def evidence_extraction_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node extracting structured evidence items from execution history."""
    history = state.get("execution_history") or []
    task_spec = state.get("task_spec")

    evidence = extract_evidence_from_execution(history, task_spec)

    trace_entry = {
        "node": "evidence_extraction",
        "evidence_items_count": len(evidence),
        "claims_extracted": [e["claim"][:60] for e in evidence],
    }

    return {
        "evidence": evidence,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
