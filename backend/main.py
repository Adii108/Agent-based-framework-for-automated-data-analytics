"""FastAPI application -- entry point for the AutoAnalytics backend."""

import os
import shutil
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import DATABASE_PATH
from backend.graph import get_pipeline, get_chat_graph
from backend.database.database import load_dataframe_to_table, get_schema, execute_sql


# ── Request/Response Models ───────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class QueryRequest(BaseModel):
    sql: str


# ── Application state (in-memory, single-user for now) ────────────

_app_state: dict = {
    "pipeline_results": None,
    "dataset_path": None,
    "db_path": DATABASE_PATH,
    "conversation_history": [],
}

UPLOAD_DIR = os.path.join("data", "uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    yield
    # Cleanup uploads on shutdown
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


# ── FastAPI app ───────────────────────────────────────────────────

app = FastAPI(
    title="AutoAnalytics",
    description="Agent Based Framework for Data Analytics -- E-commerce analytics pipeline",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "version": "0.2.0"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV/Excel/JSON file, run the data pipeline, and load into SQLite."""
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".csv", ".xlsx", ".xls", ".json"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Use CSV, Excel, or JSON.",
        )

    # Save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run the LangGraph data pipeline
    try:
        pipeline = get_pipeline()
        result = pipeline.invoke({
            "dataset_name": file_path,
            "errors": [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Load cleaned data into SQLite for querying
    db_path = _app_state["db_path"]
    cleaned = result.get("cleaned_data")
    if cleaned:
        try:
            df = pd.DataFrame(cleaned)
            load_info = load_dataframe_to_table(df, "transactions", db_path)
        except Exception as e:
            result.setdefault("errors", []).append(f"SQLite load error: {str(e)}")
            load_info = {"status": "failed", "error": str(e)}
    else:
        load_info = {"status": "skipped", "reason": "No cleaned data available"}

    # Store results and reset conversation
    _app_state["pipeline_results"] = result
    _app_state["dataset_path"] = file_path
    _app_state["conversation_history"] = []

    # Build response summary
    errors = result.get("errors", [])
    schema = result.get("schema_info", {})
    cleaning = result.get("cleaning_report", {})
    eda = result.get("eda_results", {})

    return {
        "status": "success" if not errors else "completed_with_errors",
        "dataset": file.filename,
        "schema": {
            "num_rows": schema.get("num_rows"),
            "num_columns": schema.get("num_columns"),
            "columns": [c["name"] for c in schema.get("columns", [])],
        },
        "cleaning_summary": {
            "initial_rows": cleaning.get("initial_rows"),
            "final_rows": cleaning.get("final_rows"),
            "missing_values_handled": cleaning.get("total_missing", 0),
            "duplicates_removed": cleaning.get("duplicates_removed", 0),
        },
        "database": load_info,
        "eda_available": bool(eda),
        "errors": errors,
    }


@app.get("/dataset")
async def get_dataset_info():
    """Return current dataset info and schema."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    schema = results.get("schema_info", {})
    return {
        "dataset_path": _app_state.get("dataset_path"),
        "schema": schema,
    }


@app.post("/analyze")
async def analyze():
    """Return full analysis results (EDA, cleaning report, preprocessing)."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    return {
        "schema": results.get("schema_info", {}),
        "cleaning_report": results.get("cleaning_report", {}),
        "preprocessing_report": results.get("preprocessing_report", {}),
        "eda_results": results.get("eda_results", {}),
        "errors": results.get("errors", []),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Conversational analytics endpoint.

    Classifies intent, routes to SQL or general chat, returns structured response.
    """
    if not _app_state.get("pipeline_results"):
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Upload a file first via /upload.",
        )

    db_path = _app_state["db_path"]
    history = _app_state.get("conversation_history", [])

    try:
        chat_graph = get_chat_graph()
        result = chat_graph.invoke({
            "user_query": request.message,
            "db_path": db_path,
            "conversation_history": history,
            "errors": [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    # Update shared conversation history
    _app_state["conversation_history"] = result.get("conversation_history", history)

    # Build structured response
    response = {
        "answer": result.get("chat_response", ""),
        "intent": result.get("intent", ""),
    }

    # Include SQL info when relevant
    if result.get("intent") == "sql_query":
        response["sql"] = result.get("sql_query", "")
        sql_result = result.get("sql_result", {})
        if sql_result.get("success"):
            response["data"] = sql_result.get("data", [])
            response["columns"] = sql_result.get("columns", [])
            response["row_count"] = sql_result.get("row_count", 0)
        else:
            response["sql_error"] = sql_result.get("error", "")

    return response


@app.post("/query")
async def direct_query(request: QueryRequest):
    """Execute a raw SQL query directly (for testing)."""
    db_path = _app_state["db_path"]

    if not os.path.exists(db_path):
        raise HTTPException(
            status_code=404,
            detail="Database not found. Upload a dataset first.",
        )

    result = execute_sql(db_path, request.sql)
    return result


@app.get("/results")
async def get_results():
    """Return latest pipeline results (everything)."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    # Don't return raw/cleaned/processed data dicts (they're huge)
    return {
        "schema_info": results.get("schema_info", {}),
        "cleaning_report": results.get("cleaning_report", {}),
        "preprocessing_report": results.get("preprocessing_report", {}),
        "eda_results": results.get("eda_results", {}),
        "errors": results.get("errors", []),
    }


@app.get("/schema")
async def get_db_schema():
    """Return the SQLite database schema (for debugging/testing)."""
    db_path = _app_state["db_path"]

    if not os.path.exists(db_path):
        raise HTTPException(
            status_code=404,
            detail="Database not found. Upload a dataset first.",
        )

    schema = get_schema(db_path)
    return schema
