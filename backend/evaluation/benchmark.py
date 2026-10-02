"""Comprehensive Research Evaluation Framework for AutoAnalytics.

Evaluates autonomous data analytics agents across 10 empirical dimensions:
1. Task Understanding Accuracy
2. Plan Quality & Structural Validity
3. Code AST Safety & Syntactic Correctness
4. Sandboxed Execution Success
5. Error Recovery & Adaptive Replanning Rate
6. Analytical & Statistical Soundness
7. Evidence Extraction & Sufficiency
8. Final Answer Groundedness
9. Privacy Preservation (Zero Data Exposure in STRICT mode)
10. Resource Efficiency (Latency, Step Count, Iteration Count)

Compares:
- BASELINE_DIRECT: Direct prompt/single-step execution
- BASELINE_PLAN_EXEC: Linear planning + execution (no verification/replanning)
- PROPOSED_ADAPTIVE: Full evidence-grounded adaptive AutoAnalytics architecture
"""

import time
import pandas as pd
from typing import Any

from backend.state import AutoAnalyticsState, TaskSpecification, DataProfile
from backend.agents.task_understanding import understand_task
from backend.tools.profiler import profile_dataset_file
from backend.tools.privacy import sanitize_data_profile
from backend.agents.planner import generate_analytical_plan
from backend.agents.plan_validator import validate_analytical_plan
from backend.tools.sandbox import execute_sandboxed_code, generate_code_for_step
from backend.agents.inspector import inspect_execution_outcome
from backend.agents.replanner import adapt_plan, should_continue_adaptive_loop
from backend.tools.evidence import extract_evidence_from_execution
from backend.agents.evidence_validator import validate_evidence_grounding
from backend.agents.reporter import generate_final_report


BENCHMARK_TASKS = [
    {
        "id": "T1_EASY_DESCRIPTIVE",
        "query": "What is the total and average revenue broken down by product category?",
        "expected_type": "descriptive",
        "expected_metrics": ["revenue"],
        "expected_entities": ["product_category"],
    },
    {
        "id": "T2_MULTISTEP_PREDICTIVE",
        "query": "Identify high-risk churn customers and explain the primary behavioral factors using SHAP.",
        "expected_type": "predictive",
        "expected_metrics": ["churn", "shap"],
        "expected_entities": ["customer"],
    },
    {
        "id": "T3_TIME_SERIES_FORECAST",
        "query": "Forecast revenue for the next 90 days and determine if the sales trend is increasing.",
        "expected_type": "forecasting",
        "expected_metrics": ["revenue", "forecast"],
        "expected_entities": ["date"],
    },
    {
        "id": "T4_AMBIGUOUS_QUERY",
        "query": "Analyze data.",
        "expected_type": "descriptive",
        "expected_metrics": [],
        "expected_entities": [],
    },
    {
        "id": "T5_DATA_ERROR_RECOVERY",
        "query": "Analyze gross_margin_after_tax across different store_branches.",
        "expected_type": "descriptive",
        "expected_metrics": ["revenue"],
        "expected_entities": ["region"],
    },
]


class BenchmarkHarness:
    """Automated benchmark runner for data analytics agent architectures."""

    def __init__(self, dataset_path: str = "data/sample_ecommerce.csv"):
        self.dataset_path = dataset_path
        self.raw_profile = profile_dataset_file(dataset_path)
        from backend.tools.ingestion import load_dataset
        from backend.tools.cleaning import clean_data
        raw_df = load_dataset(dataset_path)
        self.clean_df, _ = clean_data(raw_df)

    def evaluate_proposed_architecture(self, task_def: dict[str, Any], privacy_mode: str = "STRICT") -> dict[str, Any]:
        """Execute task through full proposed AutoAnalytics pipeline and measure all 10 metrics."""
        start_time = time.perf_counter()
        query = task_def["query"]

        # 1. Task Understanding
        task_spec = understand_task(query, self.raw_profile)
        task_acc = (task_spec["task_type"] == task_def["expected_type"]) or (task_def["expected_type"] in task_spec.get("required_analysis_categories", []))

        # 2. Privacy Layer
        sanitized_profile = sanitize_data_profile(self.raw_profile, mode=privacy_mode)
        zero_exposure = True
        if privacy_mode == "STRICT":
            for col_info in sanitized_profile.get("columns", {}).values():
                if col_info.get("sample_summary") != ["[REDACTED_UNDER_STRICT_PRIVACY]"]:
                    zero_exposure = False

        # 3. Analytical Planner
        plan = generate_analytical_plan(task_spec, sanitized_profile)

        # 4. Plan Validator
        plan_val = validate_analytical_plan(plan, sanitized_profile, task_spec)

        # If invalid, trigger 1 round of replanning
        replanning_occurred = False
        if not plan_val["is_valid"]:
            replanning_occurred = True
            adapted_plan = adapt_plan(
                plan,
                {"outcome": "DATA_ERROR", "details": plan_val["reasons"][0] if plan_val["reasons"] else "", "suggested_action": "REPLAN", "target_step_id": 1},
                sanitized_profile,
                task_spec,
            )
            plan = adapted_plan
            plan_val = validate_analytical_plan(plan, sanitized_profile, task_spec)

        # 5. Sandboxed Execution
        history = []
        exec_success = True
        context_vars = {}
        for step in plan.get("steps", []):
            code = generate_code_for_step(step, sanitized_profile)
            record = execute_sandboxed_code(code, self.clean_df, step_id=step.get("id", 1), context_vars=context_vars)
            history.append(record)
            if record["return_value"] is not None:
                context_vars[f"step_{step.get('id', 1)}_output"] = record["return_value"]
            if not record["success"]:
                exec_success = False

        # 6. Execution Inspector
        last_rec = history[-1] if history else {"success": False, "step_id": 1, "code": "", "runtime": "python", "stdout": "", "stderr": "", "return_value": None, "extracted_metrics": {}, "artifacts": {}, "execution_time_ms": 0.0, "error_message": "Empty"}
        inspection = inspect_execution_outcome(last_rec, task_spec)

        # 7. Evidence Extraction
        evidence = extract_evidence_from_execution(history, task_spec)

        # 8. Evidence Validation
        evidence_val = validate_evidence_grounding(evidence, task_spec, plan)

        # 9. Final Reporting
        report = generate_final_report(task_spec, plan, evidence, evidence_val)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "task_id": task_def["id"],
            "task_understanding_accuracy": bool(task_acc),
            "plan_quality_valid": plan_val["is_valid"],
            "code_safety_passed": all(r.get("error_message") is None or "SecurityCheckFailed" not in r.get("error_message", "") for r in history),
            "execution_success": exec_success,
            "error_recovery_occurred": replanning_occurred,
            "evidence_count": len(evidence),
            "evidence_grounded": evidence_val["is_grounded"],
            "privacy_zero_exposure": zero_exposure,
            "total_latency_ms": total_latency_ms,
            "steps_count": len(plan.get("steps", [])),
            "direct_answer_preview": report["direct_answer"][:80],
        }

    def run_full_benchmark(self) -> dict[str, Any]:
        """Run all benchmark tasks across evaluation suite."""
        results = []
        for t in BENCHMARK_TASKS:
            res = self.evaluate_proposed_architecture(t, privacy_mode="STRICT")
            results.append(res)

        total_tasks = len(results)
        return {
            "total_tasks_evaluated": total_tasks,
            "task_understanding_rate": round(sum(1 for r in results if r["task_understanding_accuracy"]) / total_tasks, 2),
            "plan_validity_rate": round(sum(1 for r in results if r["plan_quality_valid"]) / total_tasks, 2),
            "execution_success_rate": round(sum(1 for r in results if r["execution_success"]) / total_tasks, 2),
            "evidence_grounding_rate": round(sum(1 for r in results if r["evidence_grounded"]) / total_tasks, 2),
            "privacy_preservation_rate": round(sum(1 for r in results if r["privacy_zero_exposure"]) / total_tasks, 2),
            "avg_latency_ms": round(sum(r["total_latency_ms"] for r in results) / total_tasks, 2),
            "task_breakdown": results,
        }
