# AutoAnalytics — Project Status & Next Workflow

> **Generated**: September 12, 2026  
> **Repository**: [Agent-based-framework-for-automated-data-analytics](https://github.com/Adii108/Agent-based-framework-for-automated-data-analytics)

---

## What Has Been Built

### Part 1 — Data Pipeline Foundation ✅

The complete data ingestion-to-EDA pipeline is implemented and tested.

| Component | File | What It Does |
|---|---|---|
| **Ingestion** | [`ingestion.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/ingestion.py) | Loads CSV / Excel / JSON, detects schema (column types, nulls, uniques, sample values), reports dataset info (date ranges, memory) |
| **Cleaning** | [`cleaning.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/cleaning.py) | Fills missing values (median for numeric, mode for categorical), removes duplicates, detects & caps outliers via IQR |
| **Preprocessing** | [`preprocessing.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/preprocessing.py) | Label-encodes categoricals (skipping IDs/dates), computes StandardScaler params (deferred — not applied until ML) |
| **EDA** | [`eda.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/eda.py) | Descriptive stats, correlation matrix, column summaries, and e-commerce business metrics (total revenue, AOV, revenue by category/region/month, top customers, payment method distribution) |
| **Database** | [`database.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/database/database.py) | SQLite wrapper — `load_dataframe_to_table`, `get_schema`, `execute_sql` |
| **Sample Data** | [`generate_sample_data.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/scripts/generate_sample_data.py) | Generates a 1000-row synthetic e-commerce CSV with intentional dirty data (~185 missing values, 30 duplicates, 15 revenue outliers) |

**LangGraph Pipeline Flow:**
```
START → ingest → clean → preprocess → eda → (continues to Part 3) → END
```

---

### Part 2 — Natural Language Querying ✅

Users can ask questions in plain English. The system classifies intent, generates SQL, executes it, and returns a human-readable answer — with self-healing retry logic for broken queries.

| Component | File | What It Does |
|---|---|---|
| **Orchestrator Agent** | [`orchestrator.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/orchestrator.py) | Classifies user intent into 6 categories (`sql_query`, `prediction`, `insight`, `recommendation`, `explain`, `general_chat`), formats responses via Groq LLM |
| **NL-to-SQL Agent** | [`nl2sql.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/nl2sql.py) | Converts natural language → SQLite SQL, includes conversation context, auto-strips code fences. Self-healing: on SQL error, retries up to 2× with error feedback |
| **SQL Tool** | [`sql.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/sql.py) | Read-only SQL execution with safety regex blocking `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE` |

**Chat Graph Flow:**
```
classify → route:
  ├─ sql_query  → nl2sql → execute_sql → check:
  │     ├─ success → format_response → END
  │     ├─ error (retries < 2) → correct_sql → execute_sql (loop)
  │     └─ error (retries ≥ 2) → sql_error_response → END
  ├─ prediction → prediction_chat → END
  ├─ insight → insight_chat → END
  ├─ recommendation → recommendation_chat → END
  ├─ explain → explain_chat → END
  └─ general_chat → general_chat → END
```

---

### Part 3 — Prediction, Explainability & Insights ✅

ML models, SHAP explanations, and LLM-driven business intelligence are all implemented and wired into both the pipeline and chat graphs.

| Component | File | What It Does |
|---|---|---|
| **Churn Prediction** | [`prediction.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/prediction.py) | Engineers RFM features per customer (recency, frequency, monetary, avg_order_value, avg_quantity, avg_discount, distinct_categories). Trains Logistic Regression with 80/20 stratified split. Churn label = no purchase in 60 days |
| **Revenue Forecast** | [`prediction.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/prediction.py#L131-L227) | Aggregates daily revenue, fills missing dates, fits ARIMA(5,1,2) with fallback to (1,1,1). Produces 90-day forecast with confidence intervals |
| **SHAP Explainability** | [`explainability.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/explainability.py) | `LinearExplainer` for per-customer SHAP values, global feature importance, per-customer top-5 factor breakdowns with direction ("increases/decreases churn") |
| **Forecast Explanation** | [`explainability.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/explainability.py#L127-L198) | Data-driven (not LLM) trend analysis: % change projection, evidence metrics, monthly trend context |
| **Insight Agent** | [`insight.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/insight.py) | LLM generates numbered business insights from EDA + prediction + XAI data. Grounded in actual metrics — never invents numbers |
| **Recommendation Agent** | [`recommendation.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/recommendation.py) | LLM generates 3-5 prioritized, evidence-backed action plans with impact estimates and timeframes |

**Extended Pipeline:**
```
ingest → clean → preprocess → eda → predict → explain → insights → recommendations → END
```

---

### API Layer — FastAPI ✅

The backend exposes a full REST API via [`main.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/main.py):

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/upload` | POST | Upload CSV/Excel/JSON → runs full pipeline (Parts 1+3) → loads into SQLite |
| `/dataset` | GET | Current dataset schema |
| `/analyze` | POST | Full analysis results (schema, cleaning, preprocessing, EDA) |
| `/chat` | POST | Conversational analytics (intent classification → routing → response) |
| `/query` | POST | Direct raw SQL execution |
| `/predict` | GET | Churn + forecast results |
| `/predict/churn` | GET | Churn predictions only |
| `/predict/forecast` | GET | Revenue forecast only |
| `/explain` | POST | Per-customer SHAP explanation |
| `/explain/global` | GET | Global feature importance |
| `/insights` | GET | LLM-generated business insights |
| `/recommendations` | GET | LLM-generated recommendations |
| `/results` | GET | Full pipeline results |
| `/schema` | GET | SQLite database schema |

---

### LangGraph State

All state flows through a single [`PipelineState`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/state.py) TypedDict with fields organized by part (Part 1: data pipeline, Part 2: NL querying, Part 3: prediction/XAI/insights).

### Testing

Three test files covering each part:

| Test | File | Coverage |
|---|---|---|
| Part 1 | [`test_part1.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tests/test_part1.py) | Ingestion, cleaning, preprocessing, EDA |
| Part 2 | [`test_part2.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tests/test_part2.py) | NL-to-SQL, self-healing, conversation context |
| Part 3 | [`test_part3.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tests/test_part3.py) | Churn prediction, ARIMA forecast, SHAP, insights |

### Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | LangChain + Groq (`qwen/qwen3.8-27b`) |
| ML | Scikit-learn (Logistic Regression), Statsmodels (ARIMA) |
| Explainability | SHAP (LinearExplainer) |
| Data | Pandas, NumPy |
| Database | SQLite |
| API | FastAPI + Uvicorn |
| Config | python-dotenv (`.env` for `GROQ_API_KEY`) |

---

## Current Project Status

```
┌──────────────────────────────────────────────────────────┐
│  Part 1  Data Pipeline (ingest, clean, preprocess, EDA)  │  ✅ Complete
│  Part 2  NL-to-SQL, self-healing, conversation context   │  ✅ Complete
│  Part 3  ARIMA forecast, churn, SHAP, insights           │  ✅ Complete
│  Part 4  Full chat orchestration + production UI         │  ⬜ Pending
└──────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> Parts 1–3 are fully implemented and tested. The backend is feature-complete for the core analytics pipeline. **Part 4** is the remaining work.

---

## What's Next — Part 4 Workflow

Part 4 focuses on **full chat orchestration polish** and building a **production-ready frontend UI**. Here's the proposed roadmap:

### Phase 1: Frontend Dashboard UI

This is the biggest remaining piece. The README mentions *"production UI via Stitch — coming soon"* and the current frontend is just a basic HTML/JS test page.

**What needs to be built:**

| Feature | Description |
|---|---|
| **File Upload View** | Drag-and-drop CSV/Excel/JSON upload with progress indicator and pipeline status feedback |
| **Dataset Overview** | Schema viewer, cleaning report summary, data quality metrics |
| **Chat Interface** | Conversational analytics panel — type questions, see answers with SQL queries, formatted tables |
| **Dashboard Panels** | EDA visualizations: revenue by category (bar chart), revenue by region (map/chart), monthly trend (line chart), top customers (table) |
| **Predictions View** | Churn risk table (sortable, searchable), revenue forecast chart (historical + projected with confidence band) |
| **Explainability View** | SHAP feature importance bar chart (global), per-customer drill-down with factor breakdown |
| **Insights & Recommendations** | Formatted cards/panels showing LLM-generated insights and action plans |

### Phase 2: Chat Orchestration Polish

| Task | Description |
|---|---|
| **Multi-session support** | Replace the in-memory `_app_state` singleton with session-based state (keyed by `conversation_id`) to support multiple users |
| **Streaming responses** | Add SSE/WebSocket streaming for chat responses so users see real-time LLM output instead of waiting for the full response |
| **Error recovery UX** | Better error messages and retry prompts when the pipeline or LLM fails |
| **Conversation memory persistence** | Persist conversation history to SQLite or a file so it survives server restarts |

### Phase 3: Visualization & Charts

| Task | Description |
|---|---|
| **Revenue trend chart** | Line chart with historical data + ARIMA forecast overlay and confidence intervals |
| **Churn risk distribution** | Histogram or scatter plot of churn probabilities |
| **SHAP waterfall charts** | Per-customer SHAP waterfall/force plots |
| **Category/Region comparison** | Bar/pie charts for revenue distribution |

### Phase 4: Production Hardening

| Task | Description |
|---|---|
| **Authentication** | Add user auth (JWT or API key) |
| **Rate limiting** | Protect Groq API calls with rate limits |
| **Database migration** | Move from SQLite to Supabase/PostgreSQL for multi-user |
| **Deployment** | Dockerize, add CI/CD, deploy to cloud |
| **Part 4 tests** | Integration tests for the full chat flow, end-to-end upload → chat → insight cycle |

---

## Recommended Next Step

> [!TIP]
> The highest-impact next step is **building the frontend dashboard**. The entire backend is ready — all API endpoints are functional. A polished UI will make the project demo-ready and showcase all the capabilities that are currently only accessible via `curl` or API docs.

### Decision Points for You

1. **Frontend framework**: Pure HTML/CSS/JS, or a framework like React/Next.js/Vite?
2. **Charting library**: Chart.js, Plotly.js, or D3?
3. **Design system**: Build from scratch, or use a component library (Shadcn, Material UI)?
4. **Stitch integration**: The README mentions Stitch for UI — do you want to use that, or build manually?
5. **Scope**: Full dashboard, or start with just the chat interface + file upload?
