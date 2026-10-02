"""Analytical Planner Agent for AutoAnalytics.

Generates explicit, structured, multi-step analytical plans (DAGs)
prior to code synthesis. Formulates verifiable objectives, required data fields,
expected analytical outputs, and dependency graphs.
"""

import json
import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.state import AnalyticalPlan, TaskSpecification, DataProfile, AutoAnalyticsState


PLANNER_SYSTEM_PROMPT = """You are an expert Autonomous Data Analytics Planner.
Your goal is to formulate a structured, multi-step analytical plan (DAG) to solve the user's analytical task.
Do NOT generate executable Python code here. Formulate pure analytical logic and operations.

Available operations include:
- 'filter': Subset rows based on conditions or date ranges
- 'aggregate': Group by dimensions and compute sum/mean/median/count
- 'correlation': Compute correlation matrix or relationship coefficients
- 'statistical_test': T-test, Chi-squared, ANOVA, or distribution checks
- 'time_series_forecast': Fit ARIMA, trend, or seasonality model
- 'feature_engineering': Construct RFM or predictive feature representations
- 'classification': Train and evaluate classification model (e.g. churn)
- 'regression': Train and evaluate regression model
- 'explainability': SHAP feature importance or factor contribution
- 'summary': Synthesize high-level business findings and recommendations

Return ONLY a valid JSON object strictly matching this schema:
{
  "goal": "Clear statement of what the analytical workflow will establish",
  "strategy": "High-level analytical methodology",
  "steps": [
    {
      "id": 1,
      "objective": "Specific goal for this step",
      "operation": "aggregate" | "filter" | "correlation" | "statistical_test" | "time_series_forecast" | "feature_engineering" | "classification" | "regression" | "explainability" | "summary",
      "required_data": ["col1", "col2"],
      "expected_output": "description of computed metric, table, or model output",
      "depends_on": []
    }
  ],
  "assumptions": ["assumption1", "assumption2"],
  "validation_criteria": ["criterion1", "criterion2"]
}
"""


def _generate_template_plan(task_spec: TaskSpecification, profile: DataProfile) -> AnalyticalPlan:
    """Deterministic fallback planner based on task type and available columns."""
    task_type = task_spec.get("task_type", "descriptive")
    goal = task_spec.get("user_goal", "Perform data analysis")
    cols = list(profile.get("columns", {}).keys())
    num_cols = profile.get("numeric_columns", [])
    cat_cols = profile.get("categorical_columns", [])
    dt_cols = profile.get("datetime_columns", [])
    
    steps = []
    
    if task_type == "forecasting":
        date_col = dt_cols[0] if dt_cols else ("order_date" if "order_date" in cols else (cols[0] if cols else "date"))
        val_col = "revenue" if "revenue" in cols else (num_cols[0] if num_cols else "value")
        steps = [
            {
                "id": 1,
                "objective": f"Aggregate historical time series data by date on {val_col}",
                "operation": "aggregate",
                "required_data": [date_col, val_col],
                "expected_output": f"Daily aggregated time series of {val_col}",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": 2,
                "objective": f"Fit time-series forecasting model on {val_col} and generate 90-day projection",
                "operation": "time_series_forecast",
                "required_data": [date_col, val_col],
                "expected_output": "90-day point forecast with upper and lower confidence intervals",
                "depends_on": [1],
                "status": "pending",
            },
            {
                "id": 3,
                "objective": "Compute trend statistics and forecast percentage changes",
                "operation": "summary",
                "required_data": [val_col],
                "expected_output": "Projected growth rate and baseline comparison",
                "depends_on": [2],
                "status": "pending",
            },
        ]
        strategy = "Daily time-series aggregation followed by autoregressive forecasting and trend synthesis."
        assumptions = ["Historical trends provide predictive value for future periods.", "No severe structural regime shifts."]
        criteria = ["Confidence intervals are well-defined and positive.", "Forecast covers the required horizon."]

    elif task_type in ("predictive", "classification"):
        id_col = profile.get("id_columns", ["customer_id"])[0] if profile.get("id_columns") else "customer_id"
        steps = [
            {
                "id": 1,
                "objective": "Extract customer behavioral and RFM features",
                "operation": "feature_engineering",
                "required_data": cols[:6],
                "expected_output": "Customer feature matrix with recency, frequency, monetary metrics",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": 2,
                "objective": "Train classification model to predict customer churn probability",
                "operation": "classification",
                "required_data": ["recency", "frequency", "monetary"],
                "expected_output": "Model accuracy, ROC-AUC, and customer risk probabilities",
                "depends_on": [1],
                "status": "pending",
            },
            {
                "id": 3,
                "objective": "Compute SHAP feature importance to explain churn drivers",
                "operation": "explainability",
                "required_data": ["recency", "frequency", "monetary"],
                "expected_output": "Global and customer-specific feature importance rankings",
                "depends_on": [2],
                "status": "pending",
            },
        ]
        strategy = "Feature engineering + supervised classification + SHAP feature attribution."
        assumptions = ["Customer behavior in recency and frequency correlates with churn risk."]
        criteria = ["Model achieves valid classification metrics.", "Risk scores are normalized between 0 and 1."]

    elif task_type == "comparative":
        dim_col = cat_cols[0] if cat_cols else (cols[1] if len(cols) > 1 else "category")
        val_col = num_cols[0] if num_cols else "revenue"
        steps = [
            {
                "id": 1,
                "objective": f"Group records by {dim_col} and calculate summary metrics for {val_col}",
                "operation": "aggregate",
                "required_data": [dim_col, val_col],
                "expected_output": f"Summary table showing total and mean {val_col} across {dim_col}",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": 2,
                "objective": f"Compare variance and distributions across {dim_col} segments",
                "operation": "statistical_test",
                "required_data": [dim_col, val_col],
                "expected_output": "Comparative percentage contribution and segment ranking",
                "depends_on": [1],
                "status": "pending",
            },
        ]
        strategy = f"Cross-segment aggregation and statistical ranking across {dim_col}."
        assumptions = [f"{dim_col} segments contain sufficient sample sizes."]
        criteria = ["All major segments are represented in the breakdown."]

    else:
        # Default descriptive / exploratory plan
        steps = [
            {
                "id": 1,
                "objective": "Compute descriptive statistics and distributional summaries",
                "operation": "aggregate",
                "required_data": num_cols[:4] or cols[:4],
                "expected_output": "Descriptive statistics table (mean, median, IQR, quantiles)",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": 2,
                "objective": "Identify primary trends, dominant segments, and anomalous values",
                "operation": "summary",
                "required_data": cols[:6],
                "expected_output": "Grounded analytical summary addressing the user query",
                "depends_on": [1],
                "status": "pending",
            },
        ]
        strategy = "Descriptive aggregation and distributional synthesis."
        assumptions = ["Cleaned data accurately represents the operational domain."]
        criteria = ["Calculated metrics directly address the user query."]

    return {
        "goal": goal,
        "strategy": strategy,
        "steps": steps,
        "assumptions": assumptions,
        "validation_criteria": criteria,
    }


def generate_analytical_plan(
    task_spec: TaskSpecification,
    data_context: DataProfile,
    memory_context: dict[str, Any] | None = None,
) -> AnalyticalPlan:
    """Generate a multi-step analytical plan matching the task and data context."""
    if GROQ_API_KEY:
        try:
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=800,
            )
            cols_summary = {
                k: {"type": v.get("semantic_type"), "stats": v.get("stats")}
                for k, v in list(data_context.get("columns", {}).items())[:15]
            }
            prompt_payload = {
                "task_spec": task_spec,
                "available_columns": cols_summary,
                "relationships": data_context.get("relationships", []),
                "memory_guidance": memory_context.get("relevant_heuristics", []) if memory_context else [],
            }
            
            response = llm.invoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=f"Generate Analytical Plan for:\n{json.dumps(prompt_payload, indent=2)}"),
            ])
            
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
                
            parsed = json.loads(content)
            steps = []
            for s in parsed.get("steps", []):
                steps.append({
                    "id": int(s.get("id", len(steps) + 1)),
                    "objective": s.get("objective", ""),
                    "operation": s.get("operation", "aggregate"),
                    "required_data": s.get("required_data", []),
                    "expected_output": s.get("expected_output", ""),
                    "depends_on": [int(d) for d in s.get("depends_on", [])],
                    "status": "pending",
                })
                
            if steps:
                return {
                    "goal": parsed.get("goal", task_spec.get("user_goal", "")),
                    "strategy": parsed.get("strategy", "Multi-step analytical synthesis"),
                    "steps": steps,
                    "assumptions": parsed.get("assumptions", []),
                    "validation_criteria": parsed.get("validation_criteria", []),
                }
        except Exception:
            pass

    return _generate_template_plan(task_spec, data_context)


def analytical_planner_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node generating the structured AnalyticalPlan."""
    task_spec = state.get("task_spec") or {
        "user_goal": state.get("user_task", ""),
        "task_type": "descriptive",
        "entities": [],
        "requested_metrics": [],
        "constraints": [],
        "expected_output": "",
        "is_ambiguous": False,
        "clarification_needed": "",
        "required_analysis_categories": ["eda"],
    }
    data_ctx = state.get("data_context") or {}
    mem_ctx = state.get("memory_context")
    
    plan = generate_analytical_plan(task_spec, data_ctx, mem_ctx)
    
    trace_entry = {
        "node": "analytical_planner",
        "step_count": len(plan["steps"]),
        "strategy": plan["strategy"],
    }
    
    return {
        "analytical_plan": plan,
        "analytical_goal": plan["goal"],
        "trace": (state.get("trace") or []) + [trace_entry],
    }
