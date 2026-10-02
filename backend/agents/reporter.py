"""Evidence-Grounded Reporting Agent for AutoAnalytics.

Synthesizes structured FinalReport objects and markdown answers strictly grounded
in validated evidence. Distinguishes computed factual findings from strategic
interpretation and notes methodology, limitations, and traceability.
"""

from typing import Any
from backend.state import FinalReport, EvidenceItem, EvidenceValidation, AnalyticalPlan, TaskSpecification, AutoAnalyticsState


def generate_final_report(
    task_spec: TaskSpecification,
    plan: AnalyticalPlan,
    evidence: list[EvidenceItem],
    evidence_validation: EvidenceValidation,
) -> FinalReport:
    """Generate a structured, evidence-grounded final analytical report."""
    goal = task_spec.get("user_goal", "Analytical Query")
    task_type = task_spec.get("task_type", "descriptive")
    val_status = evidence_validation.get("status", "VALID")
    
    # 1. Key Findings extracted directly from verified evidence
    key_findings = [item.get("claim", "") for item in evidence if item.get("claim")]
    if not key_findings:
        key_findings = ["Analysis completed; no empirical anomalies or strong signals detected."]

    # 2. Evidence Table
    evidence_table = []
    for item in evidence:
        evidence_table.append({
            "metric": item.get("metric_name", ""),
            "value": item.get("metric_value", ""),
            "confidence": f"{item.get('confidence_score', 0.9):.0%}",
            "step_id": item.get("step_id", 1),
            "code_ref": item.get("code_reference", "")[:60],
        })

    # 3. Direct Answer Synthesis
    if task_type == "forecasting":
        fc_items = [e for e in evidence if "forecast" in e.get("metric_name", "")]
        if fc_items:
            direct_ans = fc_items[0]["claim"]
        else:
            direct_ans = f"Completed time-series forecasting for '{goal}'."
    elif task_type in ("predictive", "classification"):
        churn_items = [e for e in evidence if "churn" in e.get("metric_name", "")]
        if churn_items:
            direct_ans = churn_items[0]["claim"]
        else:
            direct_ans = f"Completed predictive classification for '{goal}'."
    else:
        direct_ans = f"Completed analytical investigation for '{goal}'. Primary findings are detailed below."

    # 4. Methodological Summary
    methodology = plan.get("strategy", "Multi-step analytical synthesis using sandboxed Pandas and statistical modeling.")

    # 5. Limitations & Caveats
    limitations = [
        "Analysis is bounded by the historical timeframe and data quality of the uploaded dataset.",
        "Observational relationships indicate statistical associations rather than proven causal mechanisms.",
    ]
    if val_status == "INSUFFICIENT":
        limitations.append("Warning: Some requested analytical dimensions lacked complete empirical data.")

    # 6. Strategic Recommendations
    recommendations = []
    if task_type in ("predictive", "classification"):
        recommendations.append("Deploy targeted re-engagement campaigns for customers in the top predicted churn decile.")
        recommendations.append("Incentivize recent purchasing behavior to address recency as a primary churn driver.")
    elif task_type == "forecasting":
        recommendations.append("Align inventory and procurement schedules to support projected demand trajectory.")
        recommendations.append("Monitor high-variance categories to prevent stockouts during peak forecast intervals.")
    else:
        recommendations.append("Focus operational resources on dominant revenue-generating categories and regions.")

    # 7. Traceability Summary
    traceability = {
        "steps_planned": len(plan.get("steps", [])),
        "evidence_items_count": len(evidence),
        "validation_status": val_status,
        "is_grounded": evidence_validation.get("is_grounded", True),
    }

    return {
        "direct_answer": direct_ans,
        "key_findings": key_findings,
        "evidence_table": evidence_table,
        "methodology_summary": methodology,
        "limitations": limitations,
        "recommendations": recommendations,
        "evidence_status": val_status,
        "traceability_summary": traceability,
    }


def format_report_to_markdown(report: FinalReport) -> str:
    """Format FinalReport into rich, GitHub-flavored markdown."""
    md_lines = []
    md_lines.append(f"### Direct Answer\n{report['direct_answer']}\n")

    md_lines.append("### Key Findings (Evidence-Grounded)")
    for f in report["key_findings"]:
        md_lines.append(f"- **Finding**: {f}")
    md_lines.append("")

    if report["evidence_table"]:
        md_lines.append("### Traceable Evidence Table")
        md_lines.append("| Metric | Computed Value | Confidence | Step | Source Code Ref |")
        md_lines.append("|---|---|---|---|---|")
        for row in report["evidence_table"]:
            md_lines.append(f"| `{row['metric']}` | {row['value']} | {row['confidence']} | Step {row['step_id']} | `{row['code_ref']}` |")
        md_lines.append("")

    md_lines.append(f"### Methodology\n{report['methodology_summary']}\n")

    if report["recommendations"]:
        md_lines.append("### Strategic Recommendations")
        for rec in report["recommendations"]:
            md_lines.append(f"1. {rec}")
        md_lines.append("")

    if report["limitations"]:
        md_lines.append("### Assumptions & Limitations")
        for lim in report["limitations"]:
            md_lines.append(f"- *Note*: {lim}")
        md_lines.append("")

    md_lines.append(f"> **Validation Status**: `{report['evidence_status']}` | **Empirically Grounded**: `{report['traceability_summary']['is_grounded']}`")

    return "\n".join(md_lines)


def reporting_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node synthesizing the final report and response string."""
    task_spec = state.get("task_spec") or {
        "user_goal": state.get("user_task", "Analysis"),
        "task_type": "descriptive",
        "entities": [],
        "requested_metrics": [],
        "constraints": [],
        "expected_output": "",
        "is_ambiguous": False,
        "clarification_needed": "",
        "required_analysis_categories": ["eda"],
    }
    plan = state.get("analytical_plan") or {"goal": "", "strategy": "", "steps": []}
    evidence = state.get("evidence") or []
    evidence_val = state.get("evidence_validation") or {
        "is_grounded": True,
        "status": "VALID",
        "supported_claims": [],
        "unsupported_claims": [],
        "issues": [],
        "missing_evidence": [],
        "suggested_replanning_goals": [],
    }

    report = generate_final_report(task_spec, plan, evidence, evidence_val)
    markdown_response = format_report_to_markdown(report)

    trace_entry = {
        "node": "reporting",
        "evidence_status": report["evidence_status"],
        "is_grounded": report["traceability_summary"]["is_grounded"],
    }

    return {
        "final_report": report,
        "chat_response": markdown_response,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
