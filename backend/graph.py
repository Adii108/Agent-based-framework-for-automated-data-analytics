"""LangGraph workflows for the analytics pipeline.

Contains two compiled graphs:
1. Data pipeline: ingest -> clean -> preprocess -> eda
2. Chat/query graph: classify -> route -> (sql | chat) -> respond
"""

import pandas as pd
from langgraph.graph import StateGraph, START, END

from backend.state import PipelineState
from backend.tools.ingestion import load_dataset, detect_schema, get_dataset_info
from backend.tools.cleaning import clean_data
from backend.tools.preprocessing import preprocess_data
from backend.tools.eda import run_eda
from backend.tools.sql import run_sql_query
from backend.database.database import get_schema
from backend.agents.nl2sql import generate_sql, correct_sql
from backend.agents.orchestrator import classify_intent, format_sql_response, general_chat


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


def route_intent(state: PipelineState) -> str:
    """Route to the right subgraph based on classified intent."""
    intent = state.get("intent", "general_chat")
    if intent == "sql_query":
        return "nl2sql"
    return "general_chat"


# ── Graph Builders ───────────────────────────────────────────────


def build_pipeline_graph() -> StateGraph:
    """Build the Part 1 data pipeline graph: ingest -> clean -> preprocess -> eda."""
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("clean", clean_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("eda", eda_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "clean")
    graph.add_edge("clean", "preprocess")
    graph.add_edge("preprocess", "eda")
    graph.add_edge("eda", END)

    return graph


def build_chat_graph() -> StateGraph:
    """Build the Part 2 chat/query graph with self-healing SQL.

    Flow:
        classify -> route:
            sql_query  -> nl2sql -> execute_sql -> check:
                success -> format_response -> END
                error (retries < 2) -> correct_sql -> execute_sql
                error (retries >= 2) -> sql_error_response -> END
            general_chat -> general_chat_node -> END
    """
    graph = StateGraph(PipelineState)

    # Nodes
    graph.add_node("classify", classify_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("correct_sql", correct_sql_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("sql_error_response", sql_error_response_node)
    graph.add_node("general_chat", general_chat_node)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_intent, {
        "nl2sql": "nl2sql",
        "general_chat": "general_chat",
    })
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

    return graph


def get_pipeline():
    """Compile and return the data pipeline graph."""
    graph = build_pipeline_graph()
    return graph.compile()


def get_chat_graph():
    """Compile and return the chat/query graph."""
    graph = build_chat_graph()
    return graph.compile()
