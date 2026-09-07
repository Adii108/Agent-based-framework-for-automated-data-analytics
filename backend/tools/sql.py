"""SQL execution tool — safe, read-only SQL runner."""

import re
from backend.database.database import execute_sql


# Patterns that indicate a write/destructive operation
_WRITE_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def run_sql_query(db_path: str, sql: str) -> dict:
    """Execute a read-only SQL query against the database.

    Blocks INSERT/UPDATE/DELETE/DROP/ALTER/CREATE for safety.
    Returns the result from execute_sql or an error dict.
    """
    sql = sql.strip().rstrip(";") + ";"

    # Safety check — block write operations
    if _WRITE_PATTERNS.search(sql):
        return {
            "success": False,
            "error": "Write operations are not allowed. Only SELECT queries are permitted.",
        }

    return execute_sql(db_path, sql)
