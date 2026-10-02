"""Privacy Control and Data Sanitization Layer for AutoAnalytics.

Enforces zero-data-exposure and tiered privacy policies:
- STRICT: Raw records, row samples, and PII are strictly barred from LLM prompts.
  LLM receives only abstract schemas, statistical aggregates, and execution summaries.
- STANDARD: Allows sanitized summary distributions and top-k categorical frequencies.
- FULL: Developer opt-in allowing limited sample row inspection.

Note: STRICT mode prevents direct raw data leakage to external LLM providers;
it does not constitute formal epsilon-delta differential privacy.
"""

from typing import Any
import re
from backend.state import PrivacyConfig, PrivacyMode, DataProfile, AutoAnalyticsState


PRIVACY_POLICY_DEFINITIONS: dict[PrivacyMode, PrivacyConfig] = {
    "STRICT": {
        "mode": "STRICT",
        "allow_sample_values": False,
        "allow_raw_errors": False,
        "anonymize_identifiers": True,
        "differential_privacy_applied": False,
        "policy_description": (
            "STRICT PRIVACY: Zero raw tabular row exposure. LLM agent receives only "
            "column names, abstract data types, cardinality counts, and aggregated metrics. "
            "Raw records never leave the local execution sandbox."
        ),
    },
    "STANDARD": {
        "mode": "STANDARD",
        "allow_sample_values": True,
        "allow_raw_errors": True,
        "anonymize_identifiers": True,
        "differential_privacy_applied": False,
        "policy_description": (
            "STANDARD PRIVACY: Raw tabular data remains local. LLM agent receives "
            "statistical summaries, quartile distributions, and sanitized top-k categorical labels."
        ),
    },
    "FULL": {
        "mode": "FULL",
        "allow_sample_values": True,
        "allow_raw_errors": True,
        "anonymize_identifiers": False,
        "differential_privacy_applied": False,
        "policy_description": (
            "FULL ACCESS (Opt-In): Full error messages and representative non-PII sample rows "
            "may be provided to the LLM for deep debugging."
        ),
    },
}


def sanitize_data_profile(profile: DataProfile, mode: PrivacyMode = "STRICT") -> dict[str, Any]:
    """Filter and sanitize a DataProfile according to the active PrivacyMode before LLM ingestion."""
    config = PRIVACY_POLICY_DEFINITIONS.get(mode, PRIVACY_POLICY_DEFINITIONS["STRICT"])
    
    sanitized_cols = {}
    for col_name, col_data in profile.get("columns", {}).items():
        entry = {
            "name": col_data.get("name"),
            "dtype": col_data.get("dtype"),
            "semantic_type": col_data.get("semantic_type"),
            "null_count": col_data.get("null_count"),
            "null_percentage": col_data.get("null_percentage"),
            "unique_count": col_data.get("unique_count"),
        }
        
        # In STRICT mode, omit sample values entirely; include only aggregate statistical boundaries
        if config["allow_sample_values"]:
            entry["sample_summary"] = col_data.get("sample_summary", [])
        else:
            entry["sample_summary"] = ["[REDACTED_UNDER_STRICT_PRIVACY]"]
            
        if "stats" in col_data and col_data["stats"]:
            entry["stats"] = col_data["stats"]
            
        sanitized_cols[col_name] = entry

    return {
        "dataset_name": profile.get("dataset_name"),
        "row_count": profile.get("row_count"),
        "column_count": profile.get("column_count"),
        "columns": sanitized_cols,
        "numeric_columns": profile.get("numeric_columns", []),
        "categorical_columns": profile.get("categorical_columns", []),
        "datetime_columns": profile.get("datetime_columns", []),
        "id_columns": profile.get("id_columns", []),
        "relationships": profile.get("relationships", []),
        "date_range": profile.get("date_range", {}),
        "privacy_mode": mode,
        "privacy_notice": config["policy_description"],
    }


def mask_pii_in_text(text: str) -> str:
    """Mask common potential PII strings (emails, credit cards, phone numbers)."""
    # Email mask
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', text)
    # Phone number mask
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', text)
    # Credit card mask
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[REDACTED_CARD]', text)
    return text


def privacy_layer_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node configuring privacy boundaries and generating sanitized context."""
    mode: PrivacyMode = state.get("privacy_mode") or "STRICT"
    config = PRIVACY_POLICY_DEFINITIONS.get(mode, PRIVACY_POLICY_DEFINITIONS["STRICT"])
    
    raw_profile = state.get("data_context") or {}
    sanitized_profile = sanitize_data_profile(raw_profile, mode=mode)
    
    trace_entry = {
        "node": "privacy_layer",
        "privacy_mode": mode,
        "allow_sample_values": config["allow_sample_values"],
    }
    
    return {
        "privacy_mode": mode,
        "privacy_config": config,
        "data_context": sanitized_profile,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
