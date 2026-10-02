"""Comprehensive Automated Test Suite for AutoAnalytics Research Architecture.

Tests all 14 research stages and end-to-end LangGraph execution:
- Task understanding & ambiguity detection
- Data profiling & relationship discovery
- Privacy tiers (STRICT, STANDARD, FULL)
- Analytical planning & Plan validation
- Sandboxed execution & AST security screening
- Execution inspection & tripartite failure disambiguation
- Adaptive replanning & loop bounding
- Evidence extraction & Evidence validation
- Analytical memory persistence & retrieval
- Evidence-grounded final reporting
- Human-in-the-loop intervention
- Full end-to-end LangGraph pipeline execution
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.state import AutoAnalyticsState
from backend.agents.task_understanding import understand_task, _rule_based_task_understanding
from backend.tools.profiler import profile_dataframe, profile_dataset_file
from backend.tools.privacy import sanitize_data_profile, PRIVACY_POLICY_DEFINITIONS, mask_pii_in_text
from backend.agents.planner import generate_analytical_plan
from backend.agents.plan_validator import validate_analytical_plan
from backend.tools.sandbox import execute_sandboxed_code, validate_code_safety, generate_code_for_step
from backend.agents.inspector import inspect_execution_outcome
from backend.agents.replanner import adapt_plan, should_continue_adaptive_loop
from backend.tools.evidence import extract_evidence_from_execution
from backend.agents.evidence_validator import validate_evidence_grounding
from backend.memory.analytical_memory import AnalyticalMemory
from backend.agents.reporter import generate_final_report, format_report_to_markdown
from backend.agents.hitl import evaluate_human_intervention_need, apply_human_feedback
from backend.graph import get_research_pipeline


DATA_PATH = os.path.join("data", "sample_ecommerce.csv")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5"],
        "customer_id": ["C1", "C2", "C1", "C3", "C2"],
        "order_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "revenue": [100.0, 250.0, 150.0, 500.0, 300.0],
        "quantity": [1, 2, 1, 4, 3],
        "product_category": ["Electronics", "Clothing", "Electronics", "Books", "Clothing"],
    })


def test_task_understanding():
    """Verify natural language task decomposition into structured TaskSpecification."""
    spec_fc = _rule_based_task_understanding("Forecast sales for next 90 days")
    assert spec_fc["task_type"] == "forecasting"
    assert "revenue" in spec_fc["requested_metrics"] or "sales" in spec_fc["requested_metrics"]

    spec_churn = _rule_based_task_understanding("Which customers are at risk of churning?")
    assert spec_churn["task_type"] == "predictive"
    assert "customer" in spec_churn["entities"]

    spec_ambig = _rule_based_task_understanding("Analyze")
    assert spec_ambig["is_ambiguous"] is True


def test_data_profiling(sample_df):
    """Verify privacy-safe data profiling without exposing raw rows."""
    profile = profile_dataframe(sample_df, dataset_name="test_dataset")
    assert profile["row_count"] == 5
    assert profile["column_count"] == 6
    assert "revenue" in profile["numeric_columns"]
    assert "product_category" in profile["categorical_columns"]
    assert "revenue" in profile["columns"]
    assert profile["columns"]["revenue"]["stats"]["mean"] == 260.0
    assert profile["columns"]["revenue"]["stats"]["min"] == 100.0


def test_privacy_modes(sample_df):
    """Verify STRICT mode redacts sample rows while STANDARD permits safe aggregates."""
    profile = profile_dataframe(sample_df)
    
    strict_profile = sanitize_data_profile(profile, mode="STRICT")
    assert strict_profile["privacy_mode"] == "STRICT"
    for col in strict_profile["columns"].values():
        assert col["sample_summary"] == ["[REDACTED_UNDER_STRICT_PRIVACY]"]

    standard_profile = sanitize_data_profile(profile, mode="STANDARD")
    assert standard_profile["privacy_mode"] == "STANDARD"
    cat_summary = standard_profile["columns"]["product_category"]["sample_summary"]
    assert len(cat_summary) > 0

    # PII masking
    masked = mask_pii_in_text("Contact user at test.user@example.com or 555-123-4567")
    assert "[REDACTED_EMAIL]" in masked
    assert "[REDACTED_PHONE]" in masked


def test_analytical_planner_and_validator(sample_df):
    """Verify structured plan generation and validation."""
    profile = profile_dataframe(sample_df)
    task_spec = _rule_based_task_understanding("Forecast revenue for 90 days")
    
    plan = generate_analytical_plan(task_spec, profile)
    assert len(plan["steps"]) > 0
    assert plan["goal"] != ""
    assert all("id" in s and "operation" in s for s in plan["steps"])

    # Validate valid plan
    validation = validate_analytical_plan(plan, profile, task_spec)
    assert validation["is_valid"] is True
    assert validation["status"] == "VALID"

    # Test invalid plan with non-existent column
    bad_plan = {
        "goal": "Bad Plan",
        "strategy": "Flawed",
        "steps": [{
            "id": 1,
            "objective": "Use invalid column",
            "operation": "aggregate",
            "required_data": ["non_existent_column_xyz"],
            "expected_output": "Fail",
            "depends_on": [],
        }],
    }
    bad_val = validate_analytical_plan(bad_plan, profile, task_spec)
    assert bad_val["is_valid"] is False
    assert bad_val["status"] == "INVALID"


def test_sandboxed_executor_and_security(sample_df):
    """Verify safe sandboxed Python execution and AST security scanner."""
    # Valid execution
    code = "result = {'total_rev': float(df['revenue'].sum()), 'avg_rev': float(df['revenue'].mean())}"
    rec = execute_sandboxed_code(code, sample_df)
    assert rec["success"] is True
    assert rec["return_value"]["total_rev"] == 1300.0

    # AST Security Blocking
    unsafe_code = "import os\nos.system('echo hacked')"
    safe, violations = validate_code_safety(unsafe_code)
    assert safe is False
    assert len(violations) > 0

    rec_unsafe = execute_sandboxed_code(unsafe_code, sample_df)
    assert rec_unsafe["success"] is False
    assert "SecurityCheckFailed" in rec_unsafe["error_message"]


def test_execution_inspector():
    """Verify tripartite classification of CODE_ERROR, DATA_ERROR, and SUCCESS."""
    code_err_rec = {
        "step_id": 1, "code": "", "runtime": "python", "stdout": "", "stderr": "",
        "return_value": None, "extracted_metrics": {}, "artifacts": {},
        "execution_time_ms": 1.0, "success": False, "error_message": "IndexError: list index out of range",
    }
    insp_code = inspect_execution_outcome(code_err_rec)
    assert insp_code["outcome"] == "CODE_ERROR"
    assert insp_code["suggested_action"] == "REPAIR_CODE"

    data_err_rec = {
        "step_id": 1, "code": "", "runtime": "python", "stdout": "", "stderr": "",
        "return_value": None, "extracted_metrics": {}, "artifacts": {},
        "execution_time_ms": 1.0, "success": False, "error_message": "KeyError: 'revenue_target' not in index",
    }
    insp_data = inspect_execution_outcome(data_err_rec)
    assert insp_data["outcome"] == "DATA_ERROR"
    assert insp_data["suggested_action"] == "REPLAN"

    succ_rec = {
        "step_id": 1, "code": "", "runtime": "python", "stdout": "", "stderr": "",
        "return_value": {"total_revenue": 1300.0, "total_customers": 50},
        "extracted_metrics": {"total_revenue": 1300.0}, "artifacts": {},
        "execution_time_ms": 5.0, "success": True, "error_message": None,
    }
    insp_succ = inspect_execution_outcome(succ_rec)
    assert insp_succ["outcome"] == "SUCCESS"
    assert insp_succ["suggested_action"] == "PROCEED"


def test_adaptive_replanning(sample_df):
    """Verify adaptive replanner mutates plan and loop router controls retry budget."""
    profile = profile_dataframe(sample_df)
    plan = {
        "goal": "Initial",
        "strategy": "Initial",
        "steps": [{"id": 1, "objective": "Fail step", "operation": "aggregate", "required_data": ["missing"], "depends_on": []}],
    }
    inspection = {
        "outcome": "DATA_ERROR",
        "details": "Column missing",
        "suggested_action": "REPLAN",
        "target_step_id": 1,
    }
    adapted = adapt_plan(plan, inspection, profile)
    assert len(adapted["steps"]) > 0
    assert "Adapted" in adapted["strategy"]

    # Budget exhaustion check
    state_budget_exhausted = {"iteration_count": 3, "max_iterations": 3}
    assert should_continue_adaptive_loop(state_budget_exhausted) == "route_to_reporting"


def test_evidence_extraction_and_validation():
    """Verify evidence extraction and empirical grounding validator."""
    history = [
        {
            "step_id": 1,
            "code": "forecast_res = generate_forecast(df)",
            "runtime": "python",
            "stdout": "",
            "stderr": "",
            "return_value": {
                "forecast_avg": 120000.0,
                "historical_avg": 100000.0,
                "trend": "increasing",
                "points_count": 90,
            },
            "extracted_metrics": {"forecast_avg": 120000.0},
            "artifacts": {},
            "execution_time_ms": 10.0,
            "success": True,
            "error_message": None,
        }
    ]
    evidence = extract_evidence_from_execution(history)
    assert len(evidence) > 0
    assert evidence[0]["metric_name"] == "forecasted_daily_revenue_avg"
    assert evidence[0]["confidence_score"] >= 0.8

    # Grounding check
    val = validate_evidence_grounding(evidence)
    assert val["is_grounded"] is True
    assert val["status"] == "VALID"

    # Causal overreach check
    overreach_evidence = [
        {
            "id": "e_bad",
            "claim": "Discounts cause customer churn directly",
            "metric_name": "churn",
            "metric_value": 0.5,
            "step_id": 1,
            "code_reference": "corr",
            "artifact_id": None,
            "statistical_support": None,
            "confidence_score": 0.9,
        }
    ]
    val_overreach = validate_evidence_grounding(overreach_evidence)
    assert val_overreach["is_grounded"] is False
    assert val_overreach["status"] in ("INSUFFICIENT", "OVERREACH")


def test_analytical_memory(tmp_path):
    """Verify persistent memory recording and heuristic retrieval."""
    mem_file = str(tmp_path / "test_memory.json")
    mem = AnalyticalMemory(memory_file_path=mem_file)
    profile = {"columns": {"order_date": {}, "revenue": {}}}
    
    ctx = mem.retrieve_context({"task_type": "forecasting"}, profile)
    assert len(ctx["relevant_heuristics"]) > 0

    mem.record_experience(
        task_type="forecasting",
        profile=profile,
        plan_summary="Daily aggregation with Holt-Winters",
        successful_steps=["Verified daily monotonic index."],
        failures=[],
        recovery_strategies=[],
        insights=["Seasonality observed on weekends."],
    )
    assert len(mem.experiences) >= 3


def test_evidence_grounded_reporting():
    """Verify structured final report format and markdown synthesis."""
    task_spec = {"user_goal": "Forecast revenue", "task_type": "forecasting"}
    plan = {"goal": "Forecast", "strategy": "ARIMA projection", "steps": []}
    evidence = [{
        "id": "ev1",
        "claim": "Revenue projected to rise 15%",
        "metric_name": "forecast_avg",
        "metric_value": 115000.0,
        "step_id": 1,
        "code_reference": "forecast()",
        "confidence_score": 0.9,
    }]
    val = {"is_grounded": True, "status": "VALID", "supported_claims": ["Revenue projected to rise 15%"], "unsupported_claims": [], "issues": []}

    report = generate_final_report(task_spec, plan, evidence, val)
    assert report["direct_answer"] == "Revenue projected to rise 15%"
    assert len(report["key_findings"]) == 1
    assert len(report["evidence_table"]) == 1
    assert len(report["limitations"]) > 0

    md = format_report_to_markdown(report)
    assert "### Direct Answer" in md
    assert "### Traceable Evidence Table" in md
    assert "### Methodology" in md


def test_human_in_the_loop():
    """Verify HITL triggers on unresolved ambiguity and handles human direction."""
    ambig_state = {
        "task_spec": {"user_goal": "x", "task_type": "predictive", "is_ambiguous": True, "clarification_needed": "Specify target"},
        "iteration_count": 0,
    }
    need = evaluate_human_intervention_need(ambig_state)
    assert need["needed"] is True
    assert need["intervention_type"] == "CLARIFICATION"

    updated = apply_human_feedback(ambig_state, "Predict 90-day churn")
    assert updated["task_spec"]["is_ambiguous"] is False
    assert updated["human_intervention_required"] is False


def test_end_to_end_research_pipeline():
    """Verify complete multi-stage execution through compiled LangGraph research pipeline."""
    pipeline = get_research_pipeline()
    result = pipeline.invoke(
        {
            "user_task": "What is the revenue breakdown across product categories?",
            "dataset_name": DATA_PATH,
            "privacy_mode": "STRICT",
            "iteration_count": 0,
            "max_iterations": 3,
        },
        config={"recursion_limit": 50},
    )

    assert "final_report" in result
    assert "chat_response" in result
    assert "evidence" in result
    assert "trace" in result
    assert len(result["trace"]) >= 5
    assert result["final_report"]["evidence_status"] in ("VALID", "INSUFFICIENT")
    assert "Direct Answer" in result["chat_response"]
