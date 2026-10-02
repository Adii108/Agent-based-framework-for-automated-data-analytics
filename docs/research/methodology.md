# AutoAnalytics — Research Methodology & System Specification

**Topic**: Evidence-Grounded, Privacy-Aware, Adaptive Autonomous Data Analytics  
**Repository**: `Agent-based-framework-for-automated-data-analytics`

---

## 1. Executive Summary & Research Motivation

Standard LLM-as-Data-Analyst architectures suffer from three critical structural deficiencies:
1. **Conflation of Code Failure with Analysis Failure**: A runtime exception triggers code retries, but invalid statistical methodology (e.g. causal overreach, omitted variables, invalid distributions) passes silently if the code executes without error.
2. **Unregulated Tabular Data Exposure**: Naively serializing raw dataframe rows into LLM prompt context leaks private records and breaches enterprise confidentiality.
3. **Hallucinated or Unvalidated Claims**: LLMs frequently synthesize plausible business narratives disconnected from underlying numerical evidence.

AutoAnalytics resolves these deficiencies through an **evidence-grounded, privacy-aware, multi-stage adaptive workflow** that formally decouples:
$$\text{Code Failure} \neq \text{Analysis Failure} \neq \text{Evidence Failure}$$

---

## 2. Formal Architecture Pipeline

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

## 3. Core Research Stages & Components

### 3.1 Structured State Model (`backend/state.py`)
- Master TypedDict `AutoAnalyticsState` maintaining clean typing across all 14 stages.
- Separates local dataset handles from state, preventing state bloat and external token leakage.

### 3.2 Task Understanding (`backend/agents/task_understanding.py`)
- Analyzes goals into formal `TaskSpecification` (task type, entities, requested metrics, constraints, ambiguity flags, required categories).
- Avoids premature code generation.

### 3.3 Data Profiling & Privacy Tiering (`backend/tools/profiler.py`, `backend/tools/privacy.py`)
- Constructs comprehensive `DataProfile` structures (dtypes, missingness, cardinality, numeric stats, column relationships).
- **STRICT Mode**: Guarantees zero raw tabular record exposure to external LLMs. LLMs receive only abstract schemas, types, and aggregate stats.
- **STANDARD Mode**: Safe summary distributions and sanitized top-k categorical levels.
- **FULL Mode**: Opt-in row inspection for debugging.

### 3.4 Analytical Planner & Plan Validator (`backend/agents/planner.py`, `backend/agents/plan_validator.py`)
- Decomposes tasks into verifiable multi-step DAGs (`PlanStep` with operations, required data, expected output, dependencies).
- Plan Validator checks structural integrity, column existence, operation compatibility, and goal alignment *before* execution.

### 3.5 Sandboxed Executor & Execution Inspector (`backend/tools/sandbox.py`, `backend/agents/inspector.py`)
- Sandboxed Python environment with AST security scanning blocking unauthorized system calls (`os`, `subprocess`, `requests`).
- Multi-class Execution Inspector disambiguates:
  - `CODE_ERROR` $\to$ Code repair
  - `DATA_ERROR` $\to$ Re-planning with available columns
  - `ANALYSIS_ERROR` $\to$ Methodological re-planning
  - `INSUFFICIENT_EVIDENCE` $\to$ Supplementary analytical step augmentation
  - `SUCCESS` $\to$ Evidence extraction

### 3.6 Evidence Extraction & Validation (`backend/tools/evidence.py`, `backend/agents/evidence_validator.py`)
- Extracts verifiable `EvidenceItem` tuples creating full provenance:
  $$\text{Claim} \longrightarrow \text{Evidence} \longrightarrow \text{Analysis Step} \longrightarrow \text{Executed Code} \longrightarrow \text{Data Source}$$
- Evidence Validator detects unsupported claims, numerical discrepancies, and causal overreach.

### 3.7 Analytical Memory (`backend/memory/analytical_memory.py`)
- Persistent store indexing past successful plan patterns, failure causes, and domain heuristics to guide future planning without rigid duplication.

### 3.8 Evidence-Grounded Reporter (`backend/agents/reporter.py`)
- Synthesizes direct answers, key findings, traceable evidence tables, methodologies, assumptions, and limitations.

### 3.9 Benchmark Evaluation Suite (`backend/evaluation/benchmark.py`)
- Automated harness measuring performance across 10 empirical dimensions:
  1. Task Understanding Accuracy
  2. Plan Quality & Structural Validity
  3. Code AST Safety & Syntactic Correctness
  4. Sandboxed Execution Success
  5. Error Recovery & Adaptive Replanning Rate
  6. Analytical & Statistical Soundness
  7. Evidence Extraction & Sufficiency
  8. Final Answer Groundedness
  9. Privacy Preservation (Zero Data Exposure)
  10. Resource Efficiency (Latency, Iterations)

---

## 4. Current Milestone Status & Roadmap

```
┌──────────────────────────────────────────────────────────┐
│  Phase 0   Baseline Audit & Target Architecture Docs     │  ✅ Complete
│  Phase 1   Typed Master State Model (AutoAnalyticsState) │  ✅ Complete
│  Phase 2   Task Understanding Agent & Ambiguity Check    │  ✅ Complete
│  Phase 3   Data Profiler & Relationship Discovery        │  ✅ Complete
│  Phase 4   Privacy Layer (STRICT / STANDARD / FULL)      │  ✅ Complete
│  Phase 5   Analytical Planner (Multi-Step DAGs)          │  ✅ Complete
│  Phase 6   Plan Validator (Deterministic & Semantic)     │  ✅ Complete
│  Phase 7   Sandboxed Executor & AST Security Scanner     │  ✅ Complete
│  Phase 8   Execution Inspector (Tripartite Disambiguation│  ✅ Complete
│  Phase 9   Adaptive Replanning & Bounded Feedback Loops  │  ✅ Complete
│  Phase 10  Evidence Extraction & Traceability Mapping    │  ✅ Complete
│  Phase 11  Evidence Validator & Causal Overreach Checks  │  ✅ Complete
│  Phase 12  Analytical Memory & Persistent Experience Store│  ✅ Complete
│  Phase 13  Evidence-Grounded Reporting & Formatter       │  ✅ Complete
│  Phase 14  Human-in-the-Loop Hooks & Feedback Router     │  ✅ Complete
│  Phase 15  Benchmark Evaluation Harness (10 Dimensions)  │  ✅ Complete
│  Phase 16  Automated Test Suite (12 Passing Tests)       │  ✅ Complete
│  Phase 17  Documentation & Research Positioning         │  ✅ Complete
│  Phase 18  FastAPI Integration Polish & Frontend UI      │  ⏳ Next Session
└──────────────────────────────────────────────────────────┘
```
