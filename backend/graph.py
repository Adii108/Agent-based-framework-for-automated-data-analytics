"""LangGraph workflows for the analytics pipeline.

Contains two compiled graphs:
1. Data pipeline: ingest -> clean -> preprocess -> eda -> predict -> explain -> insights -> recommendations
2. Chat/query graph: classify -> route -> (sql | prediction | insight | recommendation | explain | chat) -> respond
"""

import re
import pandas as pd
from langgraph.graph import StateGraph, START, END

from backend.state import PipelineState
from backend.tools.ingestion import load_dataset, detect_schema, get_dataset_info
from backend.tools.cleaning import clean_data
from backend.tools.preprocessing import preprocess_data
from backend.tools.eda import run_eda
from backend.tools.sql import run_sql_query
from backend.tools.prediction import predict_churn, generate_forecast
from backend.tools.explainability import (
    generate_shap_explanation,
    explain_customer_churn,
    explain_forecast,
)
from backend.database.database import get_schema
from backend.agents.nl2sql import generate_sql, correct_sql
from backend.agents.orchestrator import (
    classify_intent,
    format_sql_response,
    format_prediction_response,
    format_explain_response,
    general_chat,
)
from backend.agents.insight import generate_insights
from backend.agents.recommendation import generate_recommendations


# ── Data Pipeline Nodes (Part 1) ────────────────────────────────


def ingest_node(state: PipelineState) -> dict:
    """Load the dataset and detect its schema."""
    try:
        file_path = state.get("dataset_name", "")
        df = load_dataset(file_path)
        schema = detect_schema(df)
        info = get_dataset_info(df)
        schema.update(info)

        return {
            "raw_data": df.to_dict(orient="list"),
            "schema_info": schema,
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Ingestion error: {str(e)}"]}


def clean_node(state: PipelineState) -> dict:
    """Clean the raw data."""
    try:
        raw = state.get("raw_data")
        if not raw:
            return {"errors": state.get("errors", []) + ["No raw data to clean"]}

        df = pd.DataFrame(raw)
        cleaned_df, report = clean_data(df)

        return {
            "cleaned_data": cleaned_df.to_dict(orient="list"),
            "cleaning_report": report,
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Cleaning error: {str(e)}"]}


def preprocess_node(state: PipelineState) -> dict:
    """Preprocess the cleaned data."""
    try:
        cleaned = state.get("cleaned_data")
        if not cleaned:
            return {"errors": state.get("errors", []) + ["No cleaned data to preprocess"]}

        df = pd.DataFrame(cleaned)
        processed_df, report = preprocess_data(df)

        return {
            "processed_data": processed_df.to_dict(orient="list"),
            "preprocessing_report": report,
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Preprocessing error: {str(e)}"]}


def eda_node(state: PipelineState) -> dict:
    """Run EDA on the cleaned data (not preprocessed, to keep values readable)."""
    try:
        cleaned = state.get("cleaned_data")
        if not cleaned:
            return {"errors": state.get("errors", []) + ["No cleaned data for EDA"]}

        df = pd.DataFrame(cleaned)
        results = run_eda(df)

        return {"eda_results": results}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"EDA error: {str(e)}"]}


# ── Prediction & Explainability Nodes (Part 3) ──────────────────


def prediction_node(state: PipelineState) -> dict:
    """Run churn prediction and revenue forecasting on cleaned data."""
    try:
        cleaned = state.get("cleaned_data")
        if not cleaned:
            return {"errors": state.get("errors", []) + ["No cleaned data for prediction"]}

        df = pd.DataFrame(cleaned)
        results = {}

        # Churn prediction
        try:
            churn_result = predict_churn(df)
            results["churn"] = churn_result
        except Exception as e:
            results["churn"] = {"error": f"Churn prediction failed: {str(e)}"}

        # Revenue forecast
        try:
            forecast_result = generate_forecast(df)
            results["forecast"] = forecast_result
        except Exception as e:
            results["forecast"] = {"error": f"Forecast failed: {str(e)}"}

        return {"prediction_results": results}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Prediction error: {str(e)}"]}


def explainability_node(state: PipelineState) -> dict:
    """Generate SHAP explanations and forecast explanations."""
    try:
        prediction_results = state.get("prediction_results", {})
        eda_results = state.get("eda_results", {})
        xai_results = {}

        # SHAP for churn model
        churn = prediction_results.get("churn", {})
        if churn and "error" not in churn and "model" in churn:
            try:
                model = churn["model"]
                X = churn["feature_data"]
                feature_names = churn["feature_names"]
                shap_result = generate_shap_explanation(model, X, feature_names)
                xai_results["shap"] = shap_result
            except Exception as e:
                xai_results["shap"] = {"error": f"SHAP failed: {str(e)}"}
        else:
            xai_results["shap"] = {"error": "No churn model available for SHAP"}

        # Forecast explanation
        forecast = prediction_results.get("forecast", {})
        if forecast and "error" not in forecast:
            try:
                forecast_exp = explain_forecast(forecast, eda_results)
                xai_results["forecast_explanation"] = forecast_exp
            except Exception as e:
                xai_results["forecast_explanation"] = {"error": f"Forecast explanation failed: {str(e)}"}
        else:
            xai_results["forecast_explanation"] = {"error": "No forecast data available"}

        return {"xai_results": xai_results}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Explainability error: {str(e)}"]}


def insight_node(state: PipelineState) -> dict:
    """Generate LLM-driven business insights from computed data."""
    try:
        eda_results = state.get("eda_results", {})
        prediction_results = state.get("prediction_results", {})
        xai_results = state.get("xai_results", {})

        # Remove non-serializable objects before passing to LLM
        safe_prediction = _make_serializable(prediction_results)

        result = generate_insights(eda_results, safe_prediction, xai_results)
        return {"insight_results": result}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Insight error: {str(e)}"]}


def recommendation_node(state: PipelineState) -> dict:
    """Generate LLM-driven business recommendations."""
    try:
        insight_results = state.get("insight_results", {})
        eda_results = state.get("eda_results", {})
        prediction_results = state.get("prediction_results", {})
        xai_results = state.get("xai_results", {})

        insights_text = insight_results.get("insights", "")
        safe_prediction = _make_serializable(prediction_results)

        result = generate_recommendations(
            insights_text, eda_results, safe_prediction, xai_results
        )
        return {"recommendation_results": result}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Recommendation error: {str(e)}"]}


def _make_serializable(prediction_results: dict) -> dict:
    """Remove non-serializable objects (model, DataFrame) from prediction results."""
    if not prediction_results:
        return {}

    safe = {}
    for key, value in prediction_results.items():
        if isinstance(value, dict):
            safe[key] = {
                k: v for k, v in value.items()
                if k not in ("model", "feature_data")
            }
        else:
            safe[key] = value
    return safe


# ── Chat/Query Nodes (Part 2) ───────────────────────────────────


def classify_node(state: PipelineState) -> dict:
    """Classify the user's question intent."""
    question = state.get("user_query", "")
    history = state.get("conversation_history", [])

    intent = classify_intent(question, history)
    return {"intent": intent}


def nl2sql_node(state: PipelineState) -> dict:
    """Convert the user's question to SQL."""
    question = state.get("user_query", "")
    db_path = state.get("db_path", "")
    history = state.get("conversation_history", [])

    schema = get_schema(db_path)
    sql = generate_sql(question, schema, history)

    return {
        "sql_query": sql,
        "db_schema": schema,
        "sql_retry_count": 0,
    }


def execute_sql_node(state: PipelineState) -> dict:
    """Execute the generated SQL query."""
    db_path = state.get("db_path", "")
    sql = state.get("sql_query", "")

    result = run_sql_query(db_path, sql)
    return {"sql_result": result}


def check_sql_result(state: PipelineState) -> str:
    """Conditional edge: route based on SQL success/failure."""
    result = state.get("sql_result", {})
    retry_count = state.get("sql_retry_count", 0)

    if result.get("success"):
        return "format_response"
    elif retry_count < 2:
        return "correct_sql"
    else:
        return "sql_error_response"


def correct_sql_node(state: PipelineState) -> dict:
    """Ask the LLM to fix a failed SQL query."""
    question = state.get("user_query", "")
    sql = state.get("sql_query", "")
    error = state.get("sql_result", {}).get("error", "Unknown error")
    db_path = state.get("db_path", "")

    schema = state.get("db_schema") or get_schema(db_path)
    corrected = correct_sql(sql, error, schema, question)

    return {
        "sql_query": corrected,
        "sql_retry_count": state.get("sql_retry_count", 0) + 1,
    }


def format_response_node(state: PipelineState) -> dict:
    """Format a successful SQL result into a natural language response."""
    question = state.get("user_query", "")
    sql = state.get("sql_query", "")
    result = state.get("sql_result", {})

    response = format_sql_response(question, sql, result)

    # Update conversation history
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
    }


def sql_error_response_node(state: PipelineState) -> dict:
    """Handle SQL that failed after all retries."""
    error = state.get("sql_result", {}).get("error", "Unknown error")
    question = state.get("user_query", "")

    response = (
        f"I wasn't able to query the data for your question. "
        f"The SQL query produced an error: {error}. "
        f"Could you try rephrasing your question?"
    )

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
    }


def general_chat_node(state: PipelineState) -> dict:
    """Handle general/non-data questions."""
    question = state.get("user_query", "")
    response = general_chat(question)

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
    }


# ── Part 3 Chat Nodes ───────────────────────────────────────────


def prediction_chat_node(state: PipelineState) -> dict:
    """Handle prediction-related questions in chat."""
    question = state.get("user_query", "")
    prediction_results = state.get("prediction_results", {})

    # Build a text summary of prediction data
    parts = []

    churn = prediction_results.get("churn", {})
    if churn and "error" not in churn:
        parts.append(f"Churn Model Accuracy: {churn.get('model_accuracy', 'N/A')}")
        parts.append(f"Total Customers: {churn.get('total_customers', 'N/A')}")
        parts.append(f"Predicted to Churn: {churn.get('predicted_churned', 'N/A')}")
        parts.append(f"Predicted to Stay: {churn.get('predicted_retained', 'N/A')}")
        top_risk = churn.get("predictions", [])[:10]
        if top_risk:
            parts.append("\nTop at-risk customers:")
            for p in top_risk:
                parts.append(f"  {p['customer_id']}: {p['churn_probability']:.1%} churn probability")
    elif churn:
        parts.append(f"Churn prediction: {churn.get('error', 'Not available')}")

    forecast = prediction_results.get("forecast", {})
    if forecast and "error" not in forecast:
        summary = forecast.get("summary", {})
        parts.append(f"\nRevenue Forecast Trend: {summary.get('trend', 'N/A')}")
        parts.append(f"Recent Daily Avg: Rs.{summary.get('last_30_days_avg', 0):,.2f}")
        parts.append(f"Forecast Daily Avg: Rs.{summary.get('forecast_avg', 0):,.2f}")
        parts.append(f"Forecast Period: {summary.get('forecast_periods', 90)} days")
    elif forecast:
        parts.append(f"Revenue forecast: {forecast.get('error', 'Not available')}")

    pred_text = "\n".join(parts) if parts else "No prediction data available. Please upload and analyze a dataset first."

    response = format_prediction_response(question, pred_text)

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
    }


def insight_chat_node(state: PipelineState) -> dict:
    """Handle insight-related questions in chat."""
    question = state.get("user_query", "")
    insight_results = state.get("insight_results", {})

    if insight_results and "insights" in insight_results:
        response = insight_results["insights"]
    else:
        # Generate insights on-the-fly if not cached
        eda_results = state.get("eda_results", {})
        prediction_results = state.get("prediction_results", {})
        xai_results = state.get("xai_results", {})

        if eda_results:
            safe_prediction = _make_serializable(prediction_results)
            result = generate_insights(eda_results, safe_prediction, xai_results)
            response = result.get("insights", "Unable to generate insights at this time.")
        else:
            response = "No analysis data available. Please upload and analyze a dataset first."

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
        "insight_results": {"insights": response, "data_sources": ["eda_results"]},
    }


def recommendation_chat_node(state: PipelineState) -> dict:
    """Handle recommendation-related questions in chat."""
    question = state.get("user_query", "")
    recommendation_results = state.get("recommendation_results", {})

    if recommendation_results and "recommendations" in recommendation_results:
        response = recommendation_results["recommendations"]
    else:
        # Generate recommendations on-the-fly
        insight_results = state.get("insight_results", {})
        eda_results = state.get("eda_results", {})
        prediction_results = state.get("prediction_results", {})
        xai_results = state.get("xai_results", {})

        insights_text = insight_results.get("insights", "")
        if not insights_text and eda_results:
            safe_prediction = _make_serializable(prediction_results)
            insight_result = generate_insights(eda_results, safe_prediction, xai_results)
            insights_text = insight_result.get("insights", "")

        if insights_text:
            safe_prediction = _make_serializable(prediction_results)
            result = generate_recommendations(
                insights_text, eda_results, safe_prediction, xai_results
            )
            response = result.get("recommendations", "Unable to generate recommendations at this time.")
        else:
            response = "No analysis data available. Please upload and analyze a dataset first."

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
        "recommendation_results": {"recommendations": response, "data_sources": ["insights"]},
    }


def explain_chat_node(state: PipelineState) -> dict:
    """Handle explain/SHAP-related questions in chat."""
    question = state.get("user_query", "")
    xai_results = state.get("xai_results", {})
    prediction_results = state.get("prediction_results", {})

    # Try to extract customer ID from the question
    customer_id = _extract_customer_id(question)

    if customer_id and xai_results.get("shap") and prediction_results.get("churn"):
        churn_results = prediction_results["churn"]
        shap_results = xai_results["shap"]

        explanation = explain_customer_churn(customer_id, churn_results, shap_results)

        if "error" in explanation:
            explanation_text = explanation["error"]
        else:
            explanation_text = explanation.get("summary", "")
    else:
        # General explanation — show global feature importance
        shap = xai_results.get("shap", {})
        if shap and "error" not in shap:
            global_imp = shap.get("global_importance", [])
            parts = ["Global Churn Feature Importance (SHAP):"]
            for f in global_imp[:7]:
                parts.append(f"  {f['feature']}: importance = {f['importance']:.4f}")
            explanation_text = "\n".join(parts)
        else:
            explanation_text = "No explainability data available. Please upload and analyze a dataset first."

    response = format_explain_response(question, explanation_text)

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return {
        "chat_response": response,
        "conversation_history": history,
    }


def _extract_customer_id(question: str) -> str | None:
    """Try to extract a customer ID from a question like 'Why is CUST_0042 at risk?'"""
    match = re.search(r"CUST_\d+", question, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None


def route_intent(state: PipelineState) -> str:
    """Route to the right subgraph based on classified intent."""
    intent = state.get("intent", "general_chat")

    routing = {
        "sql_query": "nl2sql",
        "prediction": "prediction_chat",
        "insight": "insight_chat",
        "recommendation": "recommendation_chat",
        "explain": "explain_chat",
    }
    return routing.get(intent, "general_chat")


# ── Graph Builders ───────────────────────────────────────────────


def build_pipeline_graph() -> StateGraph:
    """Build the Part 1+3 data pipeline graph.

    Flow: ingest -> clean -> preprocess -> eda -> predict -> explain -> insights -> recommendations
    """
    graph = StateGraph(PipelineState)

    # Part 1 nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("clean", clean_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("eda", eda_node)

    # Part 3 nodes
    graph.add_node("predict", prediction_node)
    graph.add_node("explain", explainability_node)
    graph.add_node("insights", insight_node)
    graph.add_node("recommendations", recommendation_node)

    # Edges: Part 1 pipeline
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "clean")
    graph.add_edge("clean", "preprocess")
    graph.add_edge("preprocess", "eda")

    # Part 3 pipeline (continues after EDA)
    graph.add_edge("eda", "predict")
    graph.add_edge("predict", "explain")
    graph.add_edge("explain", "insights")
    graph.add_edge("insights", "recommendations")
    graph.add_edge("recommendations", END)

    return graph


def build_chat_graph() -> StateGraph:
    """Build the Part 2+3 chat/query graph with self-healing SQL and Part 3 routing.

    Flow:
        classify -> route:
            sql_query  -> nl2sql -> execute_sql -> check:
                success -> format_response -> END
                error (retries < 2) -> correct_sql -> execute_sql
                error (retries >= 2) -> sql_error_response -> END
            prediction -> prediction_chat -> END
            insight -> insight_chat -> END
            recommendation -> recommendation_chat -> END
            explain -> explain_chat -> END
            general_chat -> general_chat_node -> END
    """
    graph = StateGraph(PipelineState)

    # Part 2 nodes
    graph.add_node("classify", classify_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("correct_sql", correct_sql_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("sql_error_response", sql_error_response_node)
    graph.add_node("general_chat", general_chat_node)

    # Part 3 nodes
    graph.add_node("prediction_chat", prediction_chat_node)
    graph.add_node("insight_chat", insight_chat_node)
    graph.add_node("recommendation_chat", recommendation_chat_node)
    graph.add_node("explain_chat", explain_chat_node)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_intent, {
        "nl2sql": "nl2sql",
        "general_chat": "general_chat",
        "prediction_chat": "prediction_chat",
        "insight_chat": "insight_chat",
        "recommendation_chat": "recommendation_chat",
        "explain_chat": "explain_chat",
    })

    # SQL subgraph
    graph.add_edge("nl2sql", "execute_sql")
    graph.add_conditional_edges("execute_sql", check_sql_result, {
        "format_response": "format_response",
        "correct_sql": "correct_sql",
        "sql_error_response": "sql_error_response",
    })
    graph.add_edge("correct_sql", "execute_sql")
    graph.add_edge("format_response", END)
    graph.add_edge("sql_error_response", END)
    graph.add_edge("general_chat", END)

    # Part 3 terminal edges
    graph.add_edge("prediction_chat", END)
    graph.add_edge("insight_chat", END)
    graph.add_edge("recommendation_chat", END)
    graph.add_edge("explain_chat", END)

    return graph


def get_pipeline():
    """Compile and return the data pipeline graph."""
    graph = build_pipeline_graph()
    return graph.compile()


def get_chat_graph():
    """Compile and return the chat/query graph."""
    graph = build_chat_graph()
    return graph.compile()


# ── Research-Oriented Adaptive Analytics Pipeline ─────────────────

from backend.agents.task_understanding import task_understanding_node
from backend.tools.profiler import data_profiling_node
from backend.tools.privacy import privacy_layer_node
from backend.agents.planner import analytical_planner_node
from backend.agents.plan_validator import plan_validator_node
from backend.tools.sandbox import analytical_executor_node
from backend.agents.inspector import execution_inspector_node
from backend.agents.replanner import adaptive_replanning_node, should_continue_adaptive_loop
from backend.tools.evidence import evidence_extraction_node
from backend.agents.evidence_validator import evidence_validator_node
from backend.memory.analytical_memory import analytical_memory_node
from backend.agents.reporter import reporting_node


def route_plan_validation(state: PipelineState) -> str:
    """Conditional edge from plan validation."""
    val = state.get("plan_validation")
    iter_count = state.get("iteration_count", 0)
    max_iters = state.get("max_iterations", 2)

    if val and not val.get("is_valid", True):
        if iter_count >= max_iters:
            return "analytical_executor"
        return "adaptive_replanner"
    return "analytical_executor"


def route_execution_inspection(state: PipelineState) -> str:
    """Conditional edge from execution inspection."""
    route = should_continue_adaptive_loop(state)
    if route == "route_to_evidence_extraction":
        return "evidence_extraction"
    elif route == "route_to_reporting":
        return "reporting"
    elif route == "route_to_task_understanding":
        return "task_understanding"
    else:
        return "adaptive_replanner"


def route_evidence_validation(state: PipelineState) -> str:
    """Conditional edge from evidence validation."""
    val = state.get("evidence_validation")
    iter_count = state.get("iteration_count", 0)
    max_iters = state.get("max_iterations", 2)

    if iter_count >= max_iters or (val and val.get("is_grounded", True)):
        return "reporting"
    return "adaptive_replanner"


def build_research_graph() -> StateGraph:
    """Build the master research-grade adaptive analytics graph."""
    graph = StateGraph(PipelineState)

    # Add all research nodes
    graph.add_node("task_understanding", task_understanding_node)
    graph.add_node("data_profiling", data_profiling_node)
    graph.add_node("privacy_layer", privacy_layer_node)
    graph.add_node("analytical_planner", analytical_planner_node)
    graph.add_node("plan_validator", plan_validator_node)
    graph.add_node("analytical_executor", analytical_executor_node)
    graph.add_node("execution_inspector", execution_inspector_node)
    graph.add_node("adaptive_replanner", adaptive_replanning_node)
    graph.add_node("evidence_extraction", evidence_extraction_node)
    graph.add_node("evidence_validator", evidence_validator_node)
    graph.add_node("reporting", reporting_node)
    graph.add_node("analytical_memory", analytical_memory_node)

    # Linear entry flow
    graph.add_edge(START, "task_understanding")
    graph.add_edge("task_understanding", "data_profiling")
    graph.add_edge("data_profiling", "privacy_layer")
    graph.add_edge("privacy_layer", "analytical_planner")
    graph.add_edge("analytical_planner", "plan_validator")

    # Plan validation branching
    graph.add_conditional_edges("plan_validator", route_plan_validation, {
        "analytical_executor": "analytical_executor",
        "adaptive_replanner": "adaptive_replanner",
    })

    # Execution and Inspection
    graph.add_edge("analytical_executor", "execution_inspector")
    graph.add_conditional_edges("execution_inspector", route_execution_inspection, {
        "evidence_extraction": "evidence_extraction",
        "adaptive_replanner": "adaptive_replanner",
        "reporting": "reporting",
        "task_understanding": "task_understanding",
    })

    # Replanning feedback loop
    graph.add_edge("adaptive_replanner", "plan_validator")

    # Evidence flow
    graph.add_edge("evidence_extraction", "evidence_validator")
    graph.add_conditional_edges("evidence_validator", route_evidence_validation, {
        "reporting": "reporting",
        "adaptive_replanner": "adaptive_replanner",
    })

    # Reporting and Memory terminal sequence
    graph.add_edge("reporting", "analytical_memory")
    graph.add_edge("analytical_memory", END)

    return graph


def get_research_pipeline():
    """Compile and return the master adaptive research analytics pipeline."""
    graph = build_research_graph()
    return graph.compile()
