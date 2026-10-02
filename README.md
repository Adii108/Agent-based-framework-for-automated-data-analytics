# AutoAnalytics: Evidence-Grounded, Privacy-Aware, Adaptive Autonomous Data Analytics

An open-source, research-oriented autonomous data analytics agent built with **LangGraph**, **LangChain**, **FastAPI**, **Pandas**, **Scikit-learn**, and **SHAP**.

---

## 1. Research Direction & Paradigm

AutoAnalytics re-architects automated data analysis around a core research distinction:

$$\mathbf{Code\ Failure \neq Analysis\ Failure \neq Evidence\ Failure}$$

- **Code Failure (`CODE_ERROR`)**: Syntax errors, runtime exceptions $\to$ *Local code repair*.
- **Analysis Failure (`ANALYSIS_ERROR`, `DATA_ERROR`)**: Inappropriate statistical tests, invalid column assumptions, correlation treated as causation $\to$ *Analytical re-planning*.
- **Evidence Failure (`INSUFFICIENT_EVIDENCE`)**: Inconclusive results, zero variance, weak empirical support $\to$ *Hypothesis refinement & supplementary step addition*.

---

## 2. System Architecture

```
                 ┌────────────────────────────────┐
                 │       USER NATURAL TASK        │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   TASK UNDERSTANDING  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ DATA PROFILING LAYER  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ PRIVACY CONTROL LAYER │ (STRICT / STANDARD / FULL)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
             ┌───────┤  ANALYTICAL PLANNER   │◄──────────────┐
             │       └───────────┬───────────┘               │
             │                   │                           │
             │                   ▼                           │
             │       ┌───────────────────────┐               │
             │       │    PLAN VALIDATOR     │               │
             │       └───────────┬───────────┘               │
             │              /         \                      │
             │      [INVALID]         [VALID]                │
             │            /             \                    │
             │           /               ▼                   │
             │          /    ┌───────────────────────┐       │
             │         │     │  ANALYTICAL EXECUTOR  │       │
             │         │     │ (Sandboxed Py & SQL)  │       │
             │         │     └───────────┬───────────┘       │
             │         │                 │                   │
             │         │                 ▼                   │
             │         │     ┌───────────────────────┐       │
             │         │     │  EXECUTION INSPECTOR  │       │
             │         │     └───────────┬───────────┘       │
             │         │                 │                   │
             │         │                 ▼                   │
             │         │     ┌───────────────────────┐       │
             │         │     │  EVIDENCE EXTRACTION  │       │
             │         │     └───────────┬───────────┘       │
             │         │                 │                   │
             │         │                 ▼                   │
             │         │     ┌───────────────────────┐       │
             │         │     │  EVIDENCE VALIDATOR   │       │
             │         │     └───────────┬───────────┘       │
             │         │            /         \              │
             │         │   [INSUFFICIENT]   [GROUNDED]       │
             │         │          /             \            │
             └─────────┴─────────┘               ▼           │
             (Adaptive Replanning Loop) ┌─────────────────┐  │
                                        │ FINAL REPORTING │  │
                                        └────────┬────────┘  │
                                                 │           │
                                                 ▼           │
                                        ┌─────────────────┐  │
                                        │ANALYTICAL MEMORY├──┘
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                                END
```

---

## 3. Key Capabilities & Research Modules

| Layer | Component | Implementation | Research Function |
|---|---|---|---|
| **State** | [`state.py`](backend/state.py) | `AutoAnalyticsState` | Typed, extensible state tracking tasks, profiles, plans, AST traces, evidence graphs, and memory context. |
| **Task Understanding** | [`task_understanding.py`](backend/agents/task_understanding.py) | `understand_task()` | Decomposes raw user queries into intent, entities, requested metrics, constraints, and ambiguity flags without immediate code generation. |
| **Data Profiling** | [`profiler.py`](backend/tools/profiler.py) | `profile_dataframe()` | Safe summary profiling (missingness, cardinality, IQR, mean, correlations) without raw record exposure. |
| **Privacy Control** | [`privacy.py`](backend/tools/privacy.py) | `sanitize_data_profile()` | **STRICT Mode** (zero raw data exposure to LLM), **STANDARD Mode** (sanitized categorical levels), **FULL Mode** (opt-in sample debugging). |
| **Planning & Validation** | [`planner.py`](backend/agents/planner.py), [`plan_validator.py`](backend/agents/plan_validator.py) | `generate_analytical_plan()`, `validate_analytical_plan()` | Multi-step DAG planning with deterministic structural, schema, and dependency validation. |
| **Execution Sandbox** | [`sandbox.py`](backend/tools/sandbox.py) | `execute_sandboxed_code()` | Controlled runtime with AST security scanner blocking unauthorized system modules (`os`, `subprocess`, `requests`). |
| **Execution Inspector** | [`inspector.py`](backend/agents/inspector.py) | `inspect_execution_outcome()` | Multi-class outcome classification (`CODE_ERROR`, `DATA_ERROR`, `ANALYSIS_ERROR`, `STATISTICAL_WARNING`, `INSUFFICIENT_EVIDENCE`, `SUCCESS`). |
| **Adaptive Replanner** | [`replanner.py`](backend/agents/replanner.py) | `adapt_plan()`, `should_continue_adaptive_loop()` | Dynamic plan mutation with bounded iteration budgets preventing infinite loops. |
| **Evidence Extraction & Validation** | [`evidence.py`](backend/tools/evidence.py), [`evidence_validator.py`](backend/agents/evidence_validator.py) | `extract_evidence_from_execution()`, `validate_evidence_grounding()` | Extracts traceable `EvidenceItem` tuples ($\text{Claim} \to \text{Metric} \to \text{Step} \to \text{Code}$) and validates against causal overreach. |
| **Analytical Memory** | [`analytical_memory.py`](backend/memory/analytical_memory.py) | `AnalyticalMemory` | Persistent experience store indexing successful patterns, failure trajectories, and domain heuristics. |
| **Evidence-Grounded Reporting** | [`reporter.py`](backend/agents/reporter.py) | `generate_final_report()` | Produces direct answers, key findings, traceable evidence tables, methodologies, assumptions, and limitations. |
| **Evaluation Framework** | [`benchmark.py`](backend/evaluation/benchmark.py) | `BenchmarkHarness` | Automated benchmark evaluating 10 empirical research dimensions across diverse task complexities. |

---

## 4. Running the Tests & Benchmark

### Run the Research Architecture Test Suite:
```bash
pytest backend/tests/test_agentic_research.py -v
```

### Run the Empirical Benchmark Suite:
```bash
python -c "from backend.evaluation.benchmark import BenchmarkHarness; harness = BenchmarkHarness(); res = harness.run_full_benchmark(); print(res)"
```

### Run the FastAPI Server:
```bash
uvicorn backend.main:app --reload --port 8000
```

---

## 5. Repository Structure

```
AutoAnalytics/
├── docs/
│   ├── architecture/
│   │   ├── current-state.md         # Baseline system audit
│   │   └── research-target.md       # Target research specification
│   └── research/
│       └── methodology.md           # Research methodology document
├── data/
│   └── sample_ecommerce.csv         # 1000-row synthetic benchmark dataset
├── backend/
│   ├── config.py                    # Provider configurations
│   ├── state.py                     # Master AutoAnalyticsState definition
│   ├── graph.py                     # LangGraph adaptive research pipeline & pipelines
│   ├── main.py                      # FastAPI REST endpoints
│   ├── agents/
│   │   ├── task_understanding.py    # Task decomposition & ambiguity detection
│   │   ├── planner.py               # Analytical multi-step DAG planner
│   │   ├── plan_validator.py        # Pre-execution plan validation
│   │   ├── inspector.py             # Tripartite failure execution inspector
│   │   ├── replanner.py             # Adaptive replanner & loop router
│   │   ├── evidence_validator.py    # Empirical grounding & causal overreach checks
│   │   ├── reporter.py              # Evidence-grounded report synthesis
│   │   ├── hitl.py                  # Human-in-the-loop intervention hooks
│   │   ├── orchestrator.py          # Legacy chat router
│   │   ├── nl2sql.py                # NL-to-SQL agent
│   │   ├── insight.py               # Insight generator
│   │   └── recommendation.py        # Recommendation generator
│   ├── tools/
│   │   ├── profiler.py              # Privacy-safe data profiler
│   │   ├── privacy.py               # Privacy control & PII sanitization
│   │   ├── sandbox.py               # AST-secured code executor
│   │   ├── evidence.py              # Structured evidence extraction
│   │   ├── ingestion.py             # Dataset ingestion
│   │   ├── cleaning.py              # Data cleaning & imputation
│   │   ├── preprocessing.py         # Encoding & scaling
│   │   ├── eda.py                   # Exploratory data analysis
│   │   ├── prediction.py            # Churn & forecasting
│   │   └── explainability.py        # SHAP explainability
│   ├── memory/
│   │   └── analytical_memory.py     # Persistent experience store
│   ├── evaluation/
│   │   └── benchmark.py             # 10-dimension research evaluation harness
│   └── tests/
│       ├── test_agentic_research.py # Master 12-stage automated test suite
│       ├── test_part1.py            # Data pipeline test
│       ├── test_part2.py            # NL-to-SQL test
│       └── test_part3.py            # Prediction & XAI test
└── README.md
```