"""NL-to-SQL agent — converts natural language questions to SQL using Groq LLM."""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import GROQ_API_KEY, GROQ_MODEL


def _build_schema_text(schema: dict) -> str:
    """Format database schema as a concise text block for the LLM prompt."""
    parts = []
    for table_name, info in schema.items():
        cols = ", ".join(
            f"{c['name']} ({c['type']})" for c in info["columns"]
        )
        parts.append(f"Table: {table_name}\n  Columns: {cols}")

        # Include sample rows so the LLM understands actual data values
        if info.get("sample_rows"):
            sample = info["sample_rows"][0]
            sample_str = ", ".join(f"{k}={v!r}" for k, v in sample.items())
            parts.append(f"  Sample row: {sample_str}")

        parts.append(f"  Row count: {info.get('row_count', '?')}")
        parts.append("")

    return "\n".join(parts)


NL2SQL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a SQL expert. Convert the user's question into a SQLite SQL query.

Rules:
- Use ONLY the tables and columns listed in the schema below. Never invent columns or tables.
- Return ONLY the SQL query, nothing else. No markdown, no explanation, no code fences.
- Use appropriate aggregations (SUM, COUNT, AVG) when the question asks for totals or averages.
- For date filtering, the order_date column is in 'YYYY-MM-DD' format.
- Use LIKE for partial text matching if needed.
- Always include ORDER BY when ranking results.
- LIMIT results to 20 unless the user asks for all.

Database schema:
{schema}""",
    ),
    (
        "human",
        """Conversation context:
{history}

Current question: {question}

SQL query:""",
    ),
])


def _get_llm():
    """Create the Groq LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
        max_tokens=512,
    )


def generate_sql(question: str, schema: dict, conversation_history: list[dict] | None = None) -> str:
    """Generate a SQL query from a natural language question.

    Args:
        question: The user's natural language question.
        schema: Database schema dict from get_schema().
        conversation_history: List of {"role": ..., "content": ...} dicts.

    Returns:
        SQL query string.
    """
    schema_text = _build_schema_text(schema)

    # Format recent conversation history (last 6 messages max)
    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in recent
        )

    llm = _get_llm()
    chain = NL2SQL_PROMPT | llm | StrOutputParser()

    sql = chain.invoke({
        "schema": schema_text,
        "history": history_text or "No previous context.",
        "question": question,
    })

    # Clean up: strip markdown code fences if the LLM adds them despite instructions
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        # Remove first and last lines (```sql and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()

    return sql


def correct_sql(
    original_sql: str,
    error_message: str,
    schema: dict,
    question: str,
) -> str:
    """Ask the LLM to fix a SQL query that produced an error.

    Args:
        original_sql: The SQL that failed.
        error_message: The error from SQLite.
        schema: Database schema for reference.
        question: The original user question.

    Returns:
        Corrected SQL query string.
    """
    schema_text = _build_schema_text(schema)

    correction_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a SQL expert. Fix the SQL query that produced an error.

Rules:
- Use ONLY the tables and columns in the schema. Never invent columns.
- Return ONLY the corrected SQL query. No explanation, no markdown.

Database schema:
{schema}""",
        ),
        (
            "human",
            """Original question: {question}

Failed SQL:
{original_sql}

Error:
{error}

Corrected SQL:""",
        ),
    ])

    llm = _get_llm()
    chain = correction_prompt | llm | StrOutputParser()

    sql = chain.invoke({
        "schema": schema_text,
        "question": question,
        "original_sql": original_sql,
        "error": error_message,
    })

    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()

    return sql
