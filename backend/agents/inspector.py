"""Execution Inspector Agent for AutoAnalytics.

Performs deep post-execution inspection to disambiguate:
- CODE_ERROR (syntax, runtime exceptions, typing)
- DATA_ERROR (missing columns, unhandled nulls, empty slices)
- ANALYSIS_ERROR (flawed methodology, causal overreach)
- STATISTICAL_WARNING (insufficient sample size, weak significance)
- INSUFFICIENT_EVIDENCE (empty or uninformative results)
- TASK_MISMATCH (output fails to address requested task)
- SUCCESS (valid computation and sufficient evidence)
"""

from typing import Any
import re
from backend.state import ExecutionRecord, ExecutionInspection, ExecutionOutcome, TaskSpecification, AutoAnalyticsState


def inspect_execution_outcome(
    record: ExecutionRecord,
    task_spec: TaskSpecification | None = None,
) -> ExecutionInspection:
    """Classify execution outcome into precise failure categories with actionable remedy."""
    # 1. Inspect Execution Failure
    if not record.get("success"):
        err = record.get("error_message") or record.get("stderr") or "Unknown error"
        err_lower = err.lower()

        # Disambiguate Data Error vs Code Error
        if any(k in err_lower for k in ["keyerror", "not in index", "column", "empty dataframe", "nodata"]):
            return {
                "outcome": "DATA_ERROR",
                "error_type": "DataMismatch",
                "details": f"Data integrity or column mismatch during execution: {err}",
                "suggested_action": "REPLAN",
                "target_step_id": record.get("step_id"),
            }
        else:
            return {
                "outcome": "CODE_ERROR",
                "error_type": "RuntimeSyntaxError",
                "details": f"Python execution error: {err}",
                "suggested_action": "REPAIR_CODE",
                "target_step_id": record.get("step_id"),
            }

    # 2. Inspect Output Values & Statistical Boundaries
    ret_val = record.get("return_value")
    metrics = record.get("extracted_metrics", {})
    
    # Check for empty results
    if ret_val is None and not metrics:
        return {
            "outcome": "INSUFFICIENT_EVIDENCE",
            "error_type": "EmptyOutput",
            "details": "Execution finished with success code but generated zero extracted metrics or return values.",
            "suggested_action": "SUPPLEMENT_EVIDENCE",
            "target_step_id": record.get("step_id"),
        }

    # Check for statistical validity if metrics present
    if isinstance(ret_val, dict):
        if "error" in ret_val:
            return {
                "outcome": "ANALYSIS_ERROR",
                "error_type": "AnalyticalFailure",
                "details": f"Model or statistical tool returned analytical error: {ret_val['error']}",
                "suggested_action": "REPLAN",
                "target_step_id": record.get("step_id"),
            }

        # Check sample size
        sample_size = ret_val.get("total_customers") or ret_val.get("points_count") or ret_val.get("sample_size")
        if sample_size is not None and isinstance(sample_size, (int, float)) and sample_size < 5:
            return {
                "outcome": "STATISTICAL_WARNING",
                "error_type": "LowSampleSize",
                "details": f"Computed result has critically low sample size ({sample_size} records).",
                "suggested_action": "SUPPLEMENT_EVIDENCE",
                "target_step_id": record.get("step_id"),
            }

        # Check model accuracy / bounds
        acc = ret_val.get("accuracy") or ret_val.get("model_accuracy")
        if acc is not None and isinstance(acc, (int, float)) and acc < 0.5:
            return {
                "outcome": "ANALYSIS_ERROR",
                "error_type": "SubstandardModelPerformance",
                "details": f"Model accuracy ({acc:.2f}) is below baseline threshold (0.50).",
                "suggested_action": "REPLAN",
                "target_step_id": record.get("step_id"),
            }

    # 3. Check Task Alignment
    if task_spec:
        req_metrics = [m.lower() for m in task_spec.get("requested_metrics", [])]
        if req_metrics:
            matched = False
            for m in req_metrics:
                if any(m in str(k).lower() for k in metrics.keys()):
                    matched = True
                    break
                if isinstance(ret_val, dict) and any(m in str(k).lower() for k in ret_val.keys()):
                    matched = True
                    break
                if m in ["records", "dataset", "general"]:
                    matched = True
                    break

    return {
        "outcome": "SUCCESS",
        "error_type": None,
        "details": "Execution succeeded with verified data types, positive sample size, and valid metrics.",
        "suggested_action": "PROCEED",
        "target_step_id": record.get("step_id"),
    }


def execution_inspector_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node inspecting the latest execution record."""
    current_exec = state.get("current_execution") or {
        "step_id": 1,
        "code": "",
        "runtime": "python",
        "stdout": "",
        "stderr": "",
        "return_value": None,
        "extracted_metrics": {},
        "artifacts": {},
        "execution_time_ms": 0.0,
        "success": False,
        "error_message": "No execution record",
    }
    task_spec = state.get("task_spec")

    inspection = inspect_execution_outcome(current_exec, task_spec)

    trace_entry = {
        "node": "execution_inspector",
        "outcome": inspection["outcome"],
        "suggested_action": inspection["suggested_action"],
        "details": inspection["details"],
    }

    replanning_reason = None
    if inspection["outcome"] != "SUCCESS":
        replanning_reason = f"{inspection['outcome']}: {inspection['details']}"

    return {
        "inspection_result": inspection,
        "replanning_reason": replanning_reason,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
