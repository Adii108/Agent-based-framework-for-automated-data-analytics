"""Human-In-The-Loop (HITL) Intervention Framework for AutoAnalytics.

Enables selective, conditional human intervention without compromising
autonomous execution for standard workflows. Triggers when:
- Unresolved ambiguity is detected during Task Understanding
- Repeated analytical failures exceed iteration limits
- Critical data transformations exceed safety bounds
"""

from typing import Any
from backend.state import AutoAnalyticsState, TaskSpecification, PlanValidation, ExecutionInspection


def evaluate_human_intervention_need(state: AutoAnalyticsState) -> dict[str, Any]:
    """Determine whether human intervention is required based on state signals."""
    task_spec = state.get("task_spec")
    inspection = state.get("inspection_result")
    plan_val = state.get("plan_validation")
    iter_count = state.get("iteration_count", 0)
    max_iters = state.get("max_iterations", 3)

    # 1. Unresolved critical ambiguity in user request
    if task_spec and task_spec.get("is_ambiguous") and task_spec.get("task_type") != "general_chat":
        return {
            "needed": True,
            "reason": f"Ambiguity Detected: {task_spec.get('clarification_needed', 'Task goals are underspecified.')}",
            "intervention_type": "CLARIFICATION",
        }

    # 2. Repeated failure loop exhaustion
    if iter_count >= max_iters and inspection and inspection.get("outcome") != "SUCCESS":
        return {
            "needed": True,
            "reason": f"Max Iterations Reached ({iter_count}/{max_iters}): Agent encountered persistent failure ({inspection.get('outcome')}: {inspection.get('details')}).",
            "intervention_type": "FAILURE_RESOLUTION",
        }

    # 3. Invalid Plan without viable automatic correction
    if plan_val and not plan_val.get("is_valid") and len(plan_val.get("reasons", [])) > 3:
        return {
            "needed": True,
            "reason": f"Complex Plan Invalidation: {'; '.join(plan_val.get('reasons', [])[:2])}",
            "intervention_type": "PLAN_APPROVAL",
        }

    return {
        "needed": False,
        "reason": None,
        "intervention_type": None,
    }


def human_intervention_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node assessing human intervention triggers."""
    assessment = evaluate_human_intervention_need(state)

    trace_entry = {
        "node": "human_intervention",
        "needed": assessment["needed"],
        "reason": assessment["reason"],
    }

    return {
        "human_intervention_required": assessment["needed"],
        "trace": (state.get("trace") or []) + [trace_entry],
    }


def apply_human_feedback(state: AutoAnalyticsState, feedback: str) -> dict[str, Any]:
    """Incorporate user feedback directly into the analytical plan or task specification."""
    task_spec = dict(state.get("task_spec") or {})
    task_spec["clarification_needed"] = ""
    task_spec["is_ambiguous"] = False
    task_spec["constraints"] = list(task_spec.get("constraints", [])) + [f"User feedback: {feedback}"]

    return {
        "task_spec": task_spec,
        "human_feedback": feedback,
        "human_intervention_required": False,
        "iteration_count": 0,  # Reset retry budget on human direction
    }
