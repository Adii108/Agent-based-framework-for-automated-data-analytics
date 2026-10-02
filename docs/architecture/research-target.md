# AutoAnalytics — Research-Target Architecture Specification

**Topic**: Evidence-Grounded, Privacy-Aware, Adaptive Autonomous Data Analytics  
**Paradigm**: Systematic separation of *Code Failure*, *Analysis Failure*, and *Evidence Failure*.

---

## 1. Core Architectural Vision

Conventional LLM data analytics frameworks frequently conflate code execution success with analytical correctness, and direct prompt completion with factual evidence. 

AutoAnalytics re-architects data analytics around an **evidence-grounded, privacy-aware, multi-stage adaptive workflow**:

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

## 2. Research Principles & Distinctions

### Principle 1: Tripartite Failure Disambiguation
- **Code Failure (`CODE_ERROR`)**: Syntax errors, missing imports, shape mismatches, runtime exceptions. Handled via *local code repair*.
- **Analysis Failure (`ANALYSIS_ERROR`, `DATA_ERROR`)**: Inappropriate statistical tests, missing control variables, invalid column assumptions, correlation treated as causation. Handled via *analytical replanning*.
- **Evidence Failure (`INSUFFICIENT_EVIDENCE`)**: Syntactically valid code that produces weak support, non-significant p-values, or inconclusive sample sizes. Handled via *hypothesis refinement and supplementary analysis*.

### Principle 2: Zero-Data-Exposure Privacy Tiering
Raw tabular records stay entirely inside the local execution environment / sandbox. The LLM agent receives:
- **STRICT Mode**: Column names, data types, value shapes, semantic tags, and aggregate execution results only. No row-level values or PII.
- **STANDARD Mode**: Schema + statistical metrics (IQR, quantiles, histograms, anonymized top-k categories).
- **FULL Mode**: Explicit developer/user opt-in for full context debugging.

### Principle 3: Traceable Evidence Graph
Every assertion in the generated final report is linked to an underlying factual claim tuple:
$$\text{Claim} \longrightarrow \text{Evidence Tuple} (\text{Metric}, \text{Value}, \text{Step ID}, \text{Code Artifact}, \text{Statistical Support})$$

### Principle 4: Deterministic Guardrails with LLM Semantic Synthesis
- **Deterministic Python Engines**: Schema parsing, IQR calculation, sandboxed code execution, AST safety scanning, metric validation.
- **Probabilistic LLM Components**: Task intent extraction, hypothesis formulation, step planning, code generation, evidence interpretation.

---

## 3. Detailed Component Roadmap

1. **State Definition (`backend/state.py`)**: `AutoAnalyticsState` with complete typed sub-models (`TaskSpecification`, `DataProfile`, `AnalyticalPlan`, `ExecutionRecord`, `EvidenceItem`, `ValidationResult`, `MemoryItem`).
2. **Task Understanding (`backend/agents/task_understanding.py`)**: Structured breakdown of intent, entity extraction, metric specification, constraints, ambiguity detection.
3. **Data Profiler & Privacy Guard (`backend/tools/profiler.py`, `backend/tools/privacy.py`)**: Multi-level safe profiling with column cardinality, missingness patterns, semantic classifications.
4. **Analytical Planner & Validator (`backend/agents/planner.py`, `backend/agents/plan_validator.py`)**: Dependency-ordered DAG generation and deterministic AST/semantic plan validation.
5. **Sandboxed Executor & Execution Inspector (`backend/tools/sandbox.py`, `backend/agents/inspector.py`)**: Isolated code runtime with resource/timeout constraints, capturing artifacts and categorizing execution outcomes.
6. **Adaptive Replanner (`backend/agents/replanner.py`)**: Dynamic state router with retry budget, backoff, and targeted strategy mutation.
7. **Evidence Extraction & Validation (`backend/tools/evidence.py`, `backend/agents/evidence_validator.py`)**: Automated extraction of metrics/tables/figures, cross-validation against original plan requirements.
8. **Analytical Memory & Experience Store (`backend/memory/analytical_memory.py`)**: Lightweight, persistent store indexing prior plans, successes, failure causes, and data characteristics.
9. **Evidence-Grounded Reporter (`backend/agents/reporter.py`)**: Structured reporting distinguishing computational findings from contextual interpretations.
10. **Human-in-the-Loop Interventions (`backend/agents/hitl.py`)**: Safe pause/resume interrupts for ambiguous queries or critical decision points.
11. **Evaluation Benchmark Suite (`backend/evaluation/benchmark.py`)**: Automated scoring across 10 evaluation dimensions comparing baseline vs. proposed architectures.
