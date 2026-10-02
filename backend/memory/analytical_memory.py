"""Structured Analytical Memory Store for AutoAnalytics.

Persists successful analytical patterns, failure trajectories, and recovery heuristics
across sessions. Retrieves contextual guidance during planning without exposing
raw historical records or forcing rigid plan duplication.
"""

import json
import os
import hashlib
from typing import Any
from backend.state import MemoryExperience, MemoryContext, AutoAnalyticsState, TaskSpecification, DataProfile


DEFAULT_MEMORY_FILE = os.path.join("data", "analytical_memory.json")


class AnalyticalMemory:
    """Persistent experience store for analytical workflows."""

    def __init__(self, memory_file_path: str = DEFAULT_MEMORY_FILE):
        self.memory_file_path = memory_file_path
        self.experiences: list[MemoryExperience] = []
        self._load()

    def _load(self):
        """Load stored experiences from disk."""
        if os.path.exists(self.memory_file_path):
            try:
                with open(self.memory_file_path, "r", encoding="utf-8") as f:
                    self.experiences = json.load(f)
            except Exception:
                self.experiences = []
        else:
            self.experiences = self._seed_default_heuristics()
            self._save()

    def _save(self):
        """Save experiences to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
            with open(self.memory_file_path, "w", encoding="utf-8") as f:
                json.dump(self.experiences, f, indent=2)
        except Exception:
            pass

    def _seed_default_heuristics(self) -> list[MemoryExperience]:
        """Bootstrap default domain experiences."""
        return [
            {
                "task_type": "forecasting",
                "dataset_signature": "generic_time_series",
                "plan_summary": "Daily aggregation followed by ARIMA or trend projection with 90-day horizon",
                "successful_steps": [
                    "Aggregate daily revenue ensuring zero-fill for missing calendar dates.",
                    "Estimate prediction confidence intervals using standard error bounds.",
                ],
                "failures_encountered": [
                    "Passing unsorted or duplicate timestamps causes non-monotonic index errors.",
                ],
                "recovery_strategies": [
                    "Reindex with pd.date_range(freq='D') before model estimation.",
                ],
                "insights_learned": [
                    "Time series with strong weekly seasonality require daily aggregation.",
                ],
            },
            {
                "task_type": "predictive",
                "dataset_signature": "generic_customer_behavior",
                "plan_summary": "RFM behavioral feature engineering followed by Logistic Regression and SHAP attribution",
                "successful_steps": [
                    "Compute customer-level Recency (days since last purchase), Frequency (order count), and Monetary (total revenue).",
                    "Compute SHAP LinearExplainer values to isolate top feature drivers.",
                ],
                "failures_encountered": [
                    "Directly predicting on un-aggregated transactional rows leads to label leakage.",
                ],
                "recovery_strategies": [
                    "Aggregate strictly at customer_id granularity before training classifier.",
                ],
                "insights_learned": [
                    "Recency is consistently the highest-importance SHAP feature in churn modeling.",
                ],
            },
        ]

    def compute_dataset_signature(self, profile: DataProfile) -> str:
        """Create a deterministic signature of dataset schema."""
        cols = sorted(list(profile.get("columns", {}).keys()))
        col_str = "_".join(cols)
        return hashlib.md5(col_str.encode()).hexdigest()[:12]

    def record_experience(
        self,
        task_type: str,
        profile: DataProfile,
        plan_summary: str,
        successful_steps: list[str],
        failures: list[str],
        recovery_strategies: list[str],
        insights: list[str],
    ) -> None:
        """Persist a completed analytical experience."""
        sig = self.compute_dataset_signature(profile)
        experience: MemoryExperience = {
            "task_type": task_type,
            "dataset_signature": sig,
            "plan_summary": plan_summary,
            "successful_steps": successful_steps,
            "failures_encountered": failures,
            "recovery_strategies": recovery_strategies,
            "insights_learned": insights,
        }
        self.experiences.append(experience)
        self._save()

    def retrieve_context(
        self,
        task_spec: TaskSpecification,
        profile: DataProfile,
        top_k: int = 2,
    ) -> MemoryContext:
        """Retrieve relevant heuristics and past experiences for the current task."""
        target_type = task_spec.get("task_type", "descriptive")
        matches = [e for e in self.experiences if e.get("task_type") == target_type]
        if not matches:
            matches = self.experiences[:top_k]
        else:
            matches = matches[:top_k]

        heuristics = []
        for exp in matches:
            heuristics.extend(exp.get("successful_steps", []))
            heuristics.extend([f"Caution: {f}" for f in exp.get("failures_encountered", [])])

        return {
            "retrieved_experiences": matches,
            "relevant_heuristics": heuristics[:5],
        }


_global_memory = AnalyticalMemory()


def analytical_memory_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node persisting completed analytical run into memory store."""
    task_spec = state.get("task_spec") or {}
    data_ctx = state.get("data_context") or {}
    plan = state.get("analytical_plan") or {}
    evidence = state.get("evidence") or []
    history = state.get("execution_history") or []

    successful_steps = [s.get("objective", "") for s in plan.get("steps", []) if s.get("status") != "failed"]
    failures = [r.get("error_message") for r in history if r.get("error_message")]
    insights = [e.get("claim", "") for e in evidence[:3]]

    _global_memory.record_experience(
        task_type=task_spec.get("task_type", "descriptive"),
        profile=data_ctx,
        plan_summary=plan.get("strategy", ""),
        successful_steps=successful_steps,
        failures=[f for f in failures if f],
        recovery_strategies=[],
        insights=insights,
    )

    trace_entry = {
        "node": "analytical_memory",
        "persisted_task_type": task_spec.get("task_type", "descriptive"),
        "insights_recorded": len(insights),
    }

    return {
        "trace": (state.get("trace") or []) + [trace_entry],
    }
