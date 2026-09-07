"""SQLite database layer — load data, query schema, execute SQL.

Designed so that only this file needs to change when switching to
Supabase/PostgreSQL later.
"""

import sqlite3
import pandas as pd


def init_db(db_path: str) -> sqlite3.Connection:
    """Create or connect to a SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_dataframe_to_table(
    df: pd.DataFrame, table_name: str, db_path: str, if_exists: str = "replace"
) -> dict:
    """Load a DataFrame into a SQLite table.

    Returns summary of what was loaded.
    """
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        row_count = len(df)
        col_count = len(df.columns)
        return {
            "table_name": table_name,
            "rows_loaded": row_count,
            "columns": df.columns.tolist(),
            "status": "success",
        }
    finally:
        conn.close()


def get_table_names(db_path: str) -> list[str]:
    """List all user tables in the database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_schema(db_path: str, table_name: str | None = None) -> dict:
    """Return schema information for one or all tables.

    For each table returns column names, types, and a few sample values.
    This is what gets sent to the LLM for NL-to-SQL generation.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = (
            [table_name]
            if table_name
            else get_table_names(db_path)
        )

        schema = {}
        for tbl in tables:
            cursor = conn.execute(f"PRAGMA table_info('{tbl}')")
            columns = []
            for row in cursor.fetchall():
                col_info = {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                }
                columns.append(col_info)

            # Grab a few sample values per column
            sample_cursor = conn.execute(f"SELECT * FROM '{tbl}' LIMIT 3")
            sample_rows = [dict(r) for r in sample_cursor.fetchall()]

            # Row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM '{tbl}'")
            row_count = count_cursor.fetchone()[0]

            schema[tbl] = {
                "columns": columns,
                "row_count": row_count,
                "sample_rows": sample_rows,
            }

        return schema
    finally:
        conn.close()


def execute_sql(db_path: str, sql: str) -> dict:
    """Execute a SQL query and return results.

    Returns:
        {"success": True, "data": [...], "columns": [...], "row_count": N}
        or
        {"success": False, "error": "error message"}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        data = [dict(row) for row in rows]

        return {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": len(data),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        conn.close()
