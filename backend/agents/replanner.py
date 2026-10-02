"""Adaptive Re-Planning Agent for AutoAnalytics.

Implements differentiated recovery strategies:
- CODE_ERROR -> Code repair
- ANALYSIS_ERROR / DATA_ERROR -> Methodological re-planning
- INSUFFICIENT_EVIDENCE -> Supplementary step augmentation
- TASK_MISMATCH -> Task re-interpretation
- Iteration bounding & loop prevention
"""

from typing import Any
from backend.state import AnalyticalPlan, ExecutionInspection, AutoAnalyticsState, TaskSpecification, DataProfile


MAX_DEFAULT_ITERATIONS = 3


def should_continue_adaptive_loop(state: AutoAnalyticsState) -> str:
    """LangGraph conditional edge router based on inspection outcome and budget."""
    iter_count = state.get("iteration_count", 1)
    max_iters = state.get("max_iterations", MAX_DEFAULT_ITERATIONS)

    # 1. Budget check
    if iter_count >= max_iters:
        return "route_to_reporting"  # Budget exhausted, synthesize best available evidence

    # 2. Check Plan Validation Status (if coming from plan_validator)
    plan_val = state.get("plan_validation")
    if plan_val and not plan_val.get("is_valid", True):
        return "route_to_replanner"

    # 3. Check Execution Inspection
    inspection = state.get("inspection_result")
    if not inspection:
        return "route_to_evidence_extraction"

    outcome = inspection.get("outcome", "SUCCESS")
    action = inspection.get("suggested_action", "PROCEED")

    if outcome == "SUCCESS" or action == "PROCEED":
        return "route_to_evidence_extraction"
    elif outcome == "CODE_ERROR":
        return "route_to_code_repair"
    elif outcome in ("ANALYSIS_ERROR", "DATA_ERROR", "STATISTICAL_WARNING"):
        return "route_to_replanner"
    elif outcome == "INSUFFICIENT_EVIDENCE":
        return "route_to_supplement_evidence"
    elif outcome == "TASK_MISMATCH":
        return "route_to_task_understanding"
    else:
        return "route_to_reporting"


def adapt_plan(
    current_plan: AnalyticalPlan,
    inspection: ExecutionInspection,
    data_context: DataProfile,
    task_spec: TaskSpecification | None = None,
) -> AnalyticalPlan:
    """Mutate and adapt analytical plan based on specific failure diagnostics."""
    outcome = inspection.get("outcome", "ANALYSIS_ERROR")
    details = inspection.get("details", "")
    target_step_id = inspection.get("target_step_id", 1)

    steps = list(current_plan.get("steps", []))
    new_steps = []

    if outcome == "DATA_ERROR":
        # Adjust requested data to use safe available columns
        available_numeric = data_context.get("numeric_columns", ["revenue", "quantity"])
        for s in steps:
            s_copy = dict(s)
            if s.get("id") == target_step_id:
                s_copy["required_data"] = available_numeric[:2]
                s_copy["objective"] = f"Adapted step {target_step_id} using verified numeric columns {available_numeric[:2]}"
            new_steps.append(s_copy)

    elif outcome == "INSUFFICIENT_EVIDENCE":
        # Add supplementary statistical or segmentation step
        new_steps = list(steps)
        supp_id = len(steps) + 1
        new_steps.append({
            "id": supp_id,
            "objective": "Compute granular cross-sectional distributions to supplement evidence",
            "operation": "aggregate",
            "required_data": data_context.get("categorical_columns", ["product_category"])[:1] + data_context.get("numeric_columns", ["revenue"])[:1],
            "expected_output": "Segment-level breakdown table",
            "depends_on": [target_step_id],
            "status": "pending",
        })

    else:
        # Generic methodological simplification
        for s in steps:
            s_copy = dict(s)
            if s.get("id") == target_step_id:
                s_copy["operation"] = "aggregate"
                s_copy["objective"] = f"Fallback simplified aggregation for step {target_step_id}"
            new_steps.append(s_copy)

    return {
        "goal": current_plan.get("goal", "Adapted Goal"),
        "strategy": f"Adapted Strategy in response to {outcome}: {details[:60]}",
        "steps": new_steps or steps,
        "assumptions": current_plan.get("assumptions", []) + [f"Adapted after {outcome}"],
        "validation_criteria": current_plan.get("validation_criteria", []),
    }


def adaptive_replanning_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node executing analytical plan adaptation."""
    curr_plan = state.get("analytical_plan") or {"goal": "", "strategy": "", "steps": []}
    inspection = state.get("inspection_result") or {
        "outcome": "ANALYSIS_ERROR",
        "details": "Replanning requested",
        "suggested_action": "REPLAN",
        "target_step_id": 1,
    }
    data_ctx = state.get("data_context") or {}
    task_spec = state.get("task_spec")
    
    adapted = adapt_plan(curr_plan, inspection, data_ctx, task_spec)
    new_iter = state.get("iteration_count", 0) + 1

    trace_entry = {
        "node": "adaptive_replanner",
        "iteration": new_iter,
        "trigger_outcome": inspection["outcome"],
        "new_step_count": len(adapted["steps"]),
    }

    return {
        "analytical_plan": adapted,
        "iteration_count": new_iter,
        "replanning_reason": f"Adapted on iteration {new_iter} due to {inspection['outcome']}",
        "trace": (state.get("trace") or []) + [trace_entry],
    }
