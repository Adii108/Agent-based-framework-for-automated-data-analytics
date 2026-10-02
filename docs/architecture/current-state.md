# AutoAnalytics — Current State Architecture Audit

**Document Date**: October 2026  
**Repository**: `Agent-based-framework-for-automated-data-analytics`

---

## 1. Executive Summary

AutoAnalytics was initially implemented as a 3-part system:
1. **Data Pipeline Foundation**: Ingestion, cleaning, preprocessing, and exploratory data analysis (EDA).
2. **Natural Language Querying**: LLM-driven intent classification and NL-to-SQL execution with self-healing retries.
3. **Prediction & Insights**: Churn modeling (Logistic Regression), time-series forecasting (ARIMA / trend), SHAP explainability, and LLM-generated business insights/recommendations.

While the existing tools and execution capabilities function well individually, the overall agent flow operates largely as a **fixed linear pipeline** or **direct prompt-to-SQL/LLM execution**.

---

## 2. Component Inventory & Audit

### 2.1 Core Modules & Tools

| Component | Path | Current Capabilities | Limitations / Technical Debt |
|---|---|---|---|
| **Ingestion** | [`backend/tools/ingestion.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/ingestion.py) | Loads CSV, Excel, JSON; detects schema, dtypes, nulls, unique values, memory footprint. | Schema detection is passive; does not construct semantic relationship graphs or privacy profiles. |
| **Cleaning** | [`backend/tools/cleaning.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/cleaning.py) | Imputes missing values, drops duplicates, IQR outlier capping. | Fixed heuristic strategies; does not validate whether transformations preserve data integrity or statistical distributions. |
| **Preprocessing** | [`backend/tools/preprocessing.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/preprocessing.py) | Label-encodes categorical variables; calculates scaling parameters. | Basic label encoding without cardinality checks or leakage prevention. |
| **EDA** | [`backend/tools/eda.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/eda.py) | Descriptive statistics, Pearson correlation matrix, e-commerce business metrics. | Hardcoded domain metrics (revenue, orders, category distribution) rather than dynamic analytical metrics. |
| **Prediction** | [`backend/tools/prediction.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/prediction.py) | RFM feature extraction, Logistic Regression churn prediction, ARIMA/trend revenue forecasting. | Hardcoded churn definition (60-day window); ARIMA sensitive to binary extension mismatches (now wrapped with linear trend fallback). |
| **Explainability** | [`backend/tools/explainability.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/explainability.py) | SHAP `LinearExplainer` for global/local churn importance; deterministic forecast trend metrics. | Explanations are isolated from the iterative reasoning loop. |
| **SQL & Database** | [`backend/database/database.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/database/database.py), [`backend/tools/sql.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/tools/sql.py) | SQLite database loading, read-only SQL execution, schema extraction. | SQL execution is isolated; lacks sandboxed Python runtime for multi-step arbitrary analytical computations. |

### 2.2 Agent & Orchestration Layer

| Agent | Path | Current Implementation | Architectural Gaps |
|---|---|---|---|
| **Orchestrator** | [`backend/agents/orchestrator.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/orchestrator.py) | 6-intent classification (`sql_query`, `prediction`, `insight`, `recommendation`, `explain`, `general_chat`). | Single-turn intent routing. Does not formulate structured multi-step analytical plans. |
| **NL-to-SQL** | [`backend/agents/nl2sql.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/nl2sql.py) | Prompt-based SQL generator with syntax retry (up to 2 iterations). | Only addresses syntax errors in SQL; does not detect semantic mismatches, statistical flaws, or insufficient evidence. |
| **Insights & Recommendations** | [`backend/agents/insight.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/insight.py), [`backend/agents/recommendation.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/agents/recommendation.py) | Formats EDA/ML outputs into markdown bullet points. | LLM can hallucinate if prompted loosely; lacks formal claim-to-evidence validation. |
| **LangGraph Graphs** | [`backend/graph.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/graph.py) | Contains `pipeline_graph` (linear ingest-to-recommendations) and `chat_graph` (intent router). | Rigid graph topologies; lacks adaptive replanning, execution inspection, evidence extraction, and memory loops. |
| **State Representation** | [`backend/state.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/state.py) | Flat `PipelineState` TypedDict storing full raw/cleaned/processed data dicts in memory. | Storing raw datasets directly in LangGraph state causes state bloat and breaks privacy boundaries. |

---

## 3. What Works, What Should Be Retained, Refactored, or Replaced

### 3.1 Retained & Preserved
- **FastAPI backend** ([`backend/main.py`](file:///c:/Users/adity/Desktop/AutoAnalytics/backend/main.py)) and REST endpoints.
- **Core statistical and ML tools** (EDA calculations, RFM feature engineering, Logistic Regression, SHAP computation, time-series forecasting).
- **SQLite integration** and read-only SQL query safety guardrails.
- **LangChain / Groq LLM integration** with configurable model provider abstraction.

### 3.2 Refactored
- **State Architecture**: Replace flat state with a typed, modular, privacy-aware `AutoAnalyticsState` where raw data remains in local storage/sandbox and only metadata/traces/evidence flow through state.
- **Execution Engine**: Upgrade from pure SQL executor to a sandboxed multi-mode analytical executor (Python + Pandas + SQL) with resource constraints, timeouts, and execution capture.
- **Prompt Engineering**: Transition from free-form generation to structured schemas for task understanding, plan generation, code synthesis, inspection, and evidence extraction.

### 3.3 Replaced / Newly Built
- **Task Understanding**: Dissect natural language goals into task types, constraints, entities, metrics, ambiguity, and required analysis.
- **Data Profiler & Privacy Layer**: Generate privacy-preserving schema profiles (STRICT, STANDARD, FULL modes) without exposing raw data values.
- **Analytical Planner & Plan Validator**: Formulate verifiable multi-step execution plans before generating code.
- **Execution Inspector & Adaptive Replanner**: Distinguish `CODE_ERROR` vs. `DATA_ERROR` vs. `ANALYSIS_ERROR` vs. `INSUFFICIENT_EVIDENCE` vs. `TASK_MISMATCH`.
- **Evidence Extractor & Evidence Validator**: Formally link factual claims to computed metrics, code artifacts, and statistical tests.
- **Analytical Memory**: Persist structured trajectory learnings across sessions.
- **Evaluation & Benchmark Framework**: Automated harness for measuring task parsing, planning, execution, error recovery, and evidence fidelity.
