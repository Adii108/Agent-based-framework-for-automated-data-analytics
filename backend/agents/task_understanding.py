"""Task Understanding Agent for AutoAnalytics.

Parses natural-language user queries into structured TaskSpecifications
without immediately generating executable code. Detects intent, entities,
metrics, constraints, expected output formats, and ambiguities.
"""

import json
import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.state import TaskSpecification, TaskType, AutoAnalyticsState


TASK_UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Data Analytics Task Understanding Agent.
Your responsibility is to analyze the user's natural language request and extract a formal, structured specification.
DO NOT generate executable Python or SQL code. Focus exclusively on task decomposition.

Return ONLY a valid JSON object with the following schema:
{
  "user_goal": "Concise summary of what the user wants to achieve",
  "analytical_question": "Precise mathematical/analytical question to answer",
  "task_type": "descriptive" | "diagnostic" | "comparative" | "statistical" | "predictive" | "forecasting" | "classification" | "clustering" | "anomaly_detection" | "recommendation" | "mixed" | "general_chat",
  "entities": ["entity1", "entity2"],
  "requested_metrics": ["metric1", "metric2"],
  "constraints": ["constraint1", "constraint2"],
  "expected_output": "description of the target analytical artifact or response",
  "is_ambiguous": false | true,
  "clarification_needed": "Explanation if ambiguous, otherwise empty string",
  "required_analysis_categories": ["eda", "ml", "time_series", "statistical_test", "sql_query", "explainability"]
}
"""


def _rule_based_task_understanding(user_query: str) -> TaskSpecification:
    """Deterministic fallback parser when LLM is unavailable or offline."""
    query_lower = user_query.lower()
    
    # Task type heuristics
    task_type: TaskType = "descriptive"
    required_categories = ["eda"]
    
    if any(k in query_lower for k in ["forecast", "predict revenue", "future", "next month", "trend", "arima", "sales projection"]):
        task_type = "forecasting"
        required_categories = ["time_series", "eda"]
    elif any(k in query_lower for k in ["churn", "retention", "leave", "attrition", "risk", "predict customer"]):
        task_type = "predictive"
        required_categories = ["ml", "explainability"]
    elif any(k in query_lower for k in ["why", "cause", "driver", "shap", "factor", "reason", "explain"]):
        task_type = "diagnostic"
        required_categories = ["explainability", "statistical_test"]
    elif any(k in query_lower for k in ["compare", "difference", "vs", "versus", "higher than", "by region", "by category"]):
        task_type = "comparative"
        required_categories = ["eda", "sql_query"]
    elif any(k in query_lower for k in ["recommend", "action", "strategy", "improve", "optimize"]):
        task_type = "recommendation"
        required_categories = ["eda", "ml"]
    elif any(k in query_lower for k in ["correlat", "p-value", "significant", "hypothesis", "distribution", "variance"]):
        task_type = "statistical"
        required_categories = ["statistical_test", "eda"]
    elif any(k in query_lower for k in ["hello", "hi", "who are you", "what can you do", "help"]):
        task_type = "general_chat"
        required_categories = []
    
    # Entity extraction
    entities = []
    for candidate in ["customer", "order", "product", "region", "payment_method", "revenue", "category", "date"]:
        if candidate in query_lower:
            entities.append(candidate)
            
    # Metric extraction
    metrics = []
    for m in ["revenue", "sales", "churn", "count", "average_order_value", "quantity", "discount", "profit", "cost"]:
        if m in query_lower or m.replace("_", " ") in query_lower:
            metrics.append(m)
            
    # Constraints
    constraints = []
    year_match = re.search(r'\b(20\d\d)\b', user_query)
    if year_match:
        constraints.append(f"year = {year_match.group(1)}")
    top_match = re.search(r'\btop\s*(\d+)\b', query_lower)
    if top_match:
        constraints.append(f"limit = {top_match.group(1)}")
        
    is_ambiguous = len(user_query.strip().split()) < 2 and task_type != "general_chat"
    clarification = "Query is too brief to identify target metrics or entities." if is_ambiguous else ""

    return {
        "user_goal": user_query.strip(),
        "analytical_question": f"Analyze: {user_query.strip()}",
        "task_type": task_type,
        "entities": entities or ["dataset"],
        "requested_metrics": metrics or ["records"],
        "constraints": constraints,
        "expected_output": f"Analytical response for {task_type} question.",
        "is_ambiguous": is_ambiguous,
        "clarification_needed": clarification,
        "required_analysis_categories": required_categories,
    }


def understand_task(user_query: str, data_context: dict[str, Any] | None = None) -> TaskSpecification:
    """Analyze the user's natural language request and return a structured TaskSpecification."""
    if not user_query or not user_query.strip():
        return {
            "user_goal": "",
            "analytical_question": "",
            "task_type": "general_chat",
            "entities": [],
            "requested_metrics": [],
            "constraints": [],
            "expected_output": "Prompt for input",
            "is_ambiguous": True,
            "clarification_needed": "No task provided.",
            "required_analysis_categories": [],
        }

    # Attempt LLM-based structured extraction
    if GROQ_API_KEY:
        try:
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=600,
            )
            context_hint = ""
            if data_context:
                cols = list(data_context.get("columns", {}).keys()) if "columns" in data_context else []
                if cols:
                    context_hint = f"\nAvailable dataset columns: {cols[:20]}"

            prompt_text = f"User Request: {user_query}{context_hint}"
            response = llm.invoke([
                SystemMessage(content=TASK_UNDERSTANDING_SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ])

            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            
            parsed = json.loads(content)
            return {
                "user_goal": parsed.get("user_goal", user_query),
                "analytical_question": parsed.get("analytical_question", user_query),
                "task_type": parsed.get("task_type", "descriptive"),
                "entities": parsed.get("entities", []),
                "requested_metrics": parsed.get("requested_metrics", []),
                "constraints": parsed.get("constraints", []),
                "expected_output": parsed.get("expected_output", "Analytical response"),
                "is_ambiguous": bool(parsed.get("is_ambiguous", False)),
                "clarification_needed": parsed.get("clarification_needed", ""),
                "required_analysis_categories": parsed.get("required_analysis_categories", ["eda"]),
            }
        except Exception:
            # Fall back safely to rule-based parser on any LLM or network exception
            pass

    return _rule_based_task_understanding(user_query)


def task_understanding_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node wrapper for task understanding."""
    user_task = state.get("user_task") or state.get("user_query") or ""
    data_ctx = state.get("data_context")
    task_spec = understand_task(user_task, data_ctx)
    
    trace_entry = {
        "node": "task_understanding",
        "task_type": task_spec["task_type"],
        "is_ambiguous": task_spec["is_ambiguous"],
        "required_analysis_categories": task_spec["required_analysis_categories"],
    }
    
    return {
        "task_spec": task_spec,
        "task_type": task_spec["task_type"],
        "task_constraints": task_spec.get("constraints", []),
        "analytical_goal": task_spec.get("user_goal", user_task),
        "trace": (state.get("trace") or []) + [trace_entry],
    }
