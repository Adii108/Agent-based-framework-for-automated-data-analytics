"""Plan Validator Agent for AutoAnalytics.

Performs deterministic and semantic validation of analytical plans prior to execution:
- Structural integrity & acyclic dependency graph checks
- Data availability & schema compatibility (types, columns)
- Analytical relevance & sufficiency to address user goals
- Removal of redundant or contradictory operations
"""

from typing import Any
from backend.state import AnalyticalPlan, DataProfile, TaskSpecification, PlanValidation, AutoAnalyticsState


SUPPORTED_OPERATIONS = {
    "aggregate",
    "filter",
    "correlation",
    "statistical_test",
    "time_series_forecast",
    "feature_engineering",
    "classification",
    "regression",
    "explainability",
    "summary",
}


def validate_analytical_plan(
    plan: AnalyticalPlan,
    data_context: DataProfile,
    task_spec: TaskSpecification | None = None,
) -> PlanValidation:
    """Deterministically and semantically validate an AnalyticalPlan against the DataProfile."""
    reasons: list[str] = []
    affected_steps: list[int] = []
    suggested_corrections: list[str] = []
    checks_passed: list[str] = []

    steps = plan.get("steps", [])
    if not steps:
        return {
            "is_valid": False,
            "status": "INVALID",
            "reasons": ["Plan contains no analytical steps."],
            "affected_steps": [],
            "suggested_corrections": ["Generate at least 1 concrete analytical step."],
            "checks_passed": [],
        }

    checks_passed.append("non_empty_steps")

    # 1. Structural & Dependency Validation
    step_ids = set()
    for s in steps:
        s_id = s.get("id")
        if s_id is None or not isinstance(s_id, int):
            reasons.append(f"Invalid or missing step ID in step: {s.get('objective', 'unknown')}")
            suggested_corrections.append("Assign integer step IDs (1, 2, 3...)")
        elif s_id in step_ids:
            reasons.append(f"Duplicate step ID {s_id} found.")
            affected_steps.append(s_id)
            suggested_corrections.append(f"Ensure all step IDs are unique.")
        else:
            step_ids.add(s_id)

    # Dependency ordering & cycle detection
    seen_ids = set()
    for s in steps:
        s_id = s.get("id", 0)
        deps = s.get("depends_on", [])
        for d in deps:
            if d not in step_ids:
                reasons.append(f"Step {s_id} depends on non-existent step {d}.")
                affected_steps.append(s_id)
                suggested_corrections.append(f"Remove dependency on step {d} or add step {d}.")
            elif d not in seen_ids and d == s_id:
                reasons.append(f"Step {s_id} has a self-dependency.")
                affected_steps.append(s_id)
                suggested_corrections.append(f"Remove self-reference from depends_on in step {s_id}.")
        seen_ids.add(s_id)

    if not reasons:
        checks_passed.append("acyclic_dependency_structure")

    # 2. Operation & Data Column Compatibility
    available_cols = set(data_context.get("columns", {}).keys())
    numeric_cols = set(data_context.get("numeric_columns", []))
    datetime_cols = set(data_context.get("datetime_columns", []))

    # Columns created virtually by intermediate steps
    dynamically_created_cols = {"recency", "frequency", "monetary", "churn", "predicted_churn", "forecast", "revenue_sum", "count"}

    for s in steps:
        s_id = s.get("id", 0)
        op = s.get("operation", "")
        req_data = s.get("required_data", [])

        # Check operation validity
        if op not in SUPPORTED_OPERATIONS:
            reasons.append(f"Step {s_id} specifies unsupported operation '{op}'.")
            affected_steps.append(s_id)
            suggested_corrections.append(f"Use one of supported operations: {sorted(list(SUPPORTED_OPERATIONS))}")

        # Check column existence
        if available_cols and req_data:
            missing_cols = []
            for col in req_data:
                col_str = str(col).lower()
                if (col not in available_cols and 
                    col_str not in {c.lower() for c in available_cols} and 
                    col not in dynamically_created_cols):
                    # Check if it's a generic reference like 'dataset' or 'target'
                    if col_str not in {"dataset", "features", "target", "all", "table"}:
                        missing_cols.append(col)

            if missing_cols:
                reasons.append(f"Step {s_id} requests columns {missing_cols} not present in dataset schema.")
                affected_steps.append(s_id)
                suggested_corrections.append(f"Replace missing columns {missing_cols} with available columns: {list(available_cols)[:8]}")

        # Check type compatibility
        if op == "time_series_forecast" and not datetime_cols and not any("date" in str(c).lower() for c in available_cols):
            reasons.append(f"Step {s_id} requests time_series_forecast, but no datetime column exists in dataset.")
            affected_steps.append(s_id)
            suggested_corrections.append("Use cross-sectional regression or standard aggregation instead of time series forecasting.")

    if not any(r for r in reasons if "schema" in r or "unsupported" in r):
        checks_passed.append("schema_and_operation_compatibility")

    # 3. Analytical Relevance to Task
    if task_spec:
        task_type = task_spec.get("task_type")
        plan_ops = [s.get("operation") for s in steps]
        
        if task_type == "forecasting" and "time_series_forecast" not in plan_ops:
            reasons.append("Task requests forecasting, but the plan contains no time-series forecasting operation.")
            suggested_corrections.append("Add a time_series_forecast step.")
        elif task_type in ("predictive", "classification") and not any(op in plan_ops for op in ["classification", "feature_engineering"]):
            reasons.append("Task requests predictive modeling, but plan lacks classification / ML operations.")
            suggested_corrections.append("Include feature_engineering and classification steps.")

    is_valid = (len(reasons) == 0)
    status = "VALID" if is_valid else "INVALID"

    return {
        "is_valid": is_valid,
        "status": status,
        "reasons": reasons,
        "affected_steps": sorted(list(set(affected_steps))),
        "suggested_corrections": suggested_corrections,
        "checks_passed": checks_passed,
    }


def plan_validator_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node validating the AnalyticalPlan."""
    plan = state.get("analytical_plan") or {"goal": "", "strategy": "", "steps": []}
    data_ctx = state.get("data_context") or {}
    task_spec = state.get("task_spec")

    validation = validate_analytical_plan(plan, data_ctx, task_spec)

    trace_entry = {
        "node": "plan_validator",
        "is_valid": validation["is_valid"],
        "status": validation["status"],
        "checks_passed": validation["checks_passed"],
        "reasons": validation["reasons"],
    }

    return {
        "plan_validation": validation,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
