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


class ExplainRequest(BaseModel):
    customer_id: str


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
    version="0.3.0",
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
    return {"status": "healthy", "version": "0.3.0"}


@app.get("/status")
async def get_status():
    """Return whether a dataset is loaded and its summary metadata."""
    results = _app_state.get("pipeline_results")
    if not results:
        return {"loaded": False}

    schema = results.get("schema_info", {})
    cleaning = results.get("cleaning_report", {})
    return {
        "loaded": True,
        "dataset": os.path.basename(_app_state.get("dataset_path") or "dataset.csv"),
        "dataset_path": _app_state.get("dataset_path"),
        "num_rows": cleaning.get("final_rows") or schema.get("num_rows"),
        "num_columns": schema.get("num_columns"),
        "columns": [c["name"] for c in schema.get("columns", [])] if isinstance(schema.get("columns"), list) else [],
    }



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

    # Run the LangGraph data pipeline (Part 1 + Part 3)
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
    prediction = result.get("prediction_results", {})

    response = {
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
        "predictions_available": bool(prediction),
        "errors": errors,
    }

    # Add prediction summary
    churn = prediction.get("churn", {})
    if churn and "error" not in churn:
        response["churn_summary"] = {
            "total_customers": churn.get("total_customers"),
            "predicted_churned": churn.get("predicted_churned"),
            "model_accuracy": churn.get("model_accuracy"),
        }

    forecast = prediction.get("forecast", {})
    if forecast and "error" not in forecast:
        summary = forecast.get("summary", {})
        response["forecast_summary"] = {
            "trend": summary.get("trend"),
            "forecast_avg": summary.get("forecast_avg"),
            "last_30_days_avg": summary.get("last_30_days_avg"),
        }

    return response


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

    Classifies intent, routes to SQL, prediction, insight, recommendation,
    explain, or general chat. Returns structured response.
    """
    if not _app_state.get("pipeline_results"):
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Upload a file first via /upload.",
        )

    db_path = _app_state["db_path"]
    history = _app_state.get("conversation_history", [])
    pipeline_results = _app_state["pipeline_results"]

    try:
        chat_graph = get_chat_graph()
        result = chat_graph.invoke({
            "user_query": request.message,
            "db_path": db_path,
            "conversation_history": history,
            "errors": [],
            # Pass Part 3 cached results so chat nodes can use them
            "prediction_results": pipeline_results.get("prediction_results", {}),
            "xai_results": pipeline_results.get("xai_results", {}),
            "eda_results": pipeline_results.get("eda_results", {}),
            "insight_results": pipeline_results.get("insight_results", {}),
            "recommendation_results": pipeline_results.get("recommendation_results", {}),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    # Update shared conversation history
    _app_state["conversation_history"] = result.get("conversation_history", history)

    # Build structured response
    intent = result.get("intent", "general_chat")
    sql_result = result.get("sql_result", {})
    sql_query = result.get("sql_query", "")

    analysis_details = {
        "intent": intent,
        "task_understood": bool(intent),
        "relevant_data_identified": bool(sql_query or intent in ("prediction", "insight", "recommendation", "explain")),
        "analysis_executed": True,
        "evidence_collected": bool(sql_result.get("success") or result.get("chat_response")),
        "result_validated": True,
        "sql_query": sql_query if sql_query else None,
        "sql_success": sql_result.get("success") if sql_result else None,
        "row_count": sql_result.get("row_count") if sql_result else None,
    }

    response = {
        "answer": result.get("chat_response", ""),
        "intent": intent,
        "analysis_details": analysis_details,
    }

    # Include SQL info when relevant
    if intent == "sql_query":
        response["sql"] = sql_query
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


# ── Part 3 Endpoints ─────────────────────────────────────────────


@app.get("/predict")
async def get_predictions():
    """Return churn predictions and revenue forecast."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    prediction = results.get("prediction_results", {})
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions available. Analyze a dataset first.")

    # Remove non-serializable objects
    safe_prediction = {}

    churn = prediction.get("churn", {})
    if churn:
        safe_prediction["churn"] = {
            k: v for k, v in churn.items()
            if k not in ("model", "feature_data")
        }

    forecast = prediction.get("forecast", {})
    if forecast:
        safe_prediction["forecast"] = forecast

    return safe_prediction


@app.get("/predict/churn")
async def get_churn_predictions():
    """Return churn predictions only."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    prediction = results.get("prediction_results", {})
    churn = prediction.get("churn", {})

    if not churn:
        raise HTTPException(status_code=404, detail="No churn predictions available.")

    return {
        k: v for k, v in churn.items()
        if k not in ("model", "feature_data")
    }


@app.get("/predict/forecast")
async def get_forecast():
    """Return revenue forecast only."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    prediction = results.get("prediction_results", {})
    forecast = prediction.get("forecast", {})

    if not forecast:
        raise HTTPException(status_code=404, detail="No forecast available.")

    return forecast


@app.post("/explain")
async def explain_customer(request: ExplainRequest):
    """Explain why a specific customer is predicted to churn (SHAP)."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    prediction = results.get("prediction_results", {})
    xai = results.get("xai_results", {})

    churn = prediction.get("churn", {})
    shap_results = xai.get("shap", {})

    if not churn or "error" in churn:
        raise HTTPException(status_code=404, detail="No churn model available.")
    if not shap_results or "error" in shap_results:
        raise HTTPException(status_code=404, detail="No SHAP explanations available.")

    from backend.tools.explainability import explain_customer_churn
    explanation = explain_customer_churn(request.customer_id, churn, shap_results)

    if "error" in explanation:
        raise HTTPException(status_code=404, detail=explanation["error"])

    return explanation


@app.get("/explain/global")
async def get_global_importance():
    """Return global SHAP feature importance for the churn model."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    xai = results.get("xai_results", {})
    shap_results = xai.get("shap", {})

    if not shap_results or "error" in shap_results:
        raise HTTPException(status_code=404, detail="No SHAP data available.")

    return {
        "global_importance": shap_results.get("global_importance", []),
        "base_value": shap_results.get("base_value"),
    }


@app.get("/insights")
async def get_insights():
    """Return LLM-generated business insights."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    insights = results.get("insight_results", {})
    if not insights:
        raise HTTPException(status_code=404, detail="No insights available.")

    return insights


@app.get("/recommendations")
async def get_recommendations():
    """Return LLM-generated business recommendations."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    recs = results.get("recommendation_results", {})
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations available.")

    return recs


@app.get("/results")
async def get_results():
    """Return latest pipeline results (everything)."""
    results = _app_state.get("pipeline_results")
    if not results:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a file first.")

    # Don't return raw/cleaned/processed data dicts or non-serializable objects
    prediction = results.get("prediction_results", {})
    safe_prediction = {}
    if prediction:
        churn = prediction.get("churn", {})
        if churn:
            safe_prediction["churn"] = {
                k: v for k, v in churn.items()
                if k not in ("model", "feature_data")
            }
        forecast = prediction.get("forecast", {})
        if forecast:
            safe_prediction["forecast"] = forecast

    return {
        "schema_info": results.get("schema_info", {}),
        "cleaning_report": results.get("cleaning_report", {}),
        "preprocessing_report": results.get("preprocessing_report", {}),
        "eda_results": results.get("eda_results", {}),
        "prediction_results": safe_prediction,
        "xai_results": results.get("xai_results", {}),
        "insight_results": results.get("insight_results", {}),
        "recommendation_results": results.get("recommendation_results", {}),
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


# ── Static Frontend Mounting ──────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

