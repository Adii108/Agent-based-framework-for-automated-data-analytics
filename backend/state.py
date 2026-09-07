"""LangGraph state definition for the analytics pipeline."""

from typing import Any
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph analytics pipeline.

    Part 1 fields -- Data Pipeline Foundation:
        dataset_name: Name of the uploaded file
        raw_data: Raw DataFrame as dict (df.to_dict())
        schema_info: Column names, dtypes, sample values, row/col counts
        cleaning_report: Missing values handled, duplicates removed, outlier info
        cleaned_data: Cleaned DataFrame as dict
        preprocessing_report: Encoding/scaling details
        processed_data: Preprocessed DataFrame as dict
        eda_results: Statistics, correlations, business metrics
        errors: List of errors encountered during pipeline execution

    Part 2 fields -- Natural Language Querying:
        user_query: The user's natural language question
        db_path: Path to the SQLite database
        db_schema: Database schema for NL-to-SQL
        intent: Classified intent (sql_query, general_chat, etc.)
        sql_query: Generated SQL query
        sql_result: Result from SQL execution
        sql_retry_count: Number of SQL correction retries attempted
        chat_response: Final formatted response to user
        conversation_history: List of {role, content} message dicts

    Fields for Parts 3-4 will be added when those parts are implemented.
    """

    # Part 1 -- Data Pipeline
    dataset_name: str
    raw_data: dict[str, Any]
    schema_info: dict[str, Any]
    cleaning_report: dict[str, Any]
    cleaned_data: dict[str, Any]
    preprocessing_report: dict[str, Any]
    processed_data: dict[str, Any]
    eda_results: dict[str, Any]
    errors: list[str]

    # Part 2 -- Natural Language Querying
    user_query: str
    db_path: str
    db_schema: dict[str, Any]
    intent: str
    sql_query: str
    sql_result: dict[str, Any]
    sql_retry_count: int
    chat_response: str
    conversation_history: list[dict[str, str]]
