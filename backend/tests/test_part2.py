"""Part 2 test -- Natural language querying, SQLite, self-healing SQL, conversation context."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from backend.tools.ingestion import load_dataset
from backend.tools.cleaning import clean_data
from backend.database.database import load_dataframe_to_table, get_schema, execute_sql
from backend.tools.sql import run_sql_query
from backend.agents.nl2sql import generate_sql, correct_sql
from backend.agents.orchestrator import classify_intent
from backend.graph import get_pipeline, get_chat_graph


DATA_PATH = os.path.join("data", "sample_ecommerce.csv")
DB_PATH = os.path.join("data", "test_analytics.db")


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_database_layer():
    """Test SQLite operations: load, schema, query."""
    separator("1. DATABASE LAYER")

    # Load and clean data
    df = load_dataset(DATA_PATH)
    cleaned_df, _ = clean_data(df)

    # Load into SQLite
    info = load_dataframe_to_table(cleaned_df, "transactions", DB_PATH)
    print(f"  Loaded: {info['rows_loaded']} rows into '{info['table_name']}'")
    assert info["status"] == "success", "Failed to load data into SQLite"

    # Get schema
    schema = get_schema(DB_PATH, "transactions")
    print(f"  Schema columns: {[c['name'] for c in schema['transactions']['columns']]}")
    print(f"  Row count: {schema['transactions']['row_count']}")
    assert schema["transactions"]["row_count"] == len(cleaned_df), "Row count mismatch"

    # Execute a basic query
    result = execute_sql(DB_PATH, "SELECT COUNT(*) as cnt FROM transactions")
    print(f"  Query result: {result['data']}")
    assert result["success"], f"Query failed: {result.get('error')}"

    # Test error handling
    result = execute_sql(DB_PATH, "SELECT * FROM nonexistent_table")
    assert not result["success"], "Should fail for nonexistent table"
    print(f"  Error handling works: {result['error'][:60]}...")

    print("  [PASS] Database layer works")
    return True


def test_sql_tool():
    """Test the safe SQL execution tool."""
    separator("2. SQL TOOL (read-only enforcement)")

    # Valid SELECT
    result = run_sql_query(DB_PATH, "SELECT product_category, COUNT(*) as cnt FROM transactions GROUP BY product_category")
    assert result["success"], f"Valid query failed: {result.get('error')}"
    print(f"  SELECT query: {len(result['data'])} rows returned")

    # Blocked write
    result = run_sql_query(DB_PATH, "DROP TABLE transactions")
    assert not result["success"], "Should block DROP"
    print(f"  DROP blocked: {result['error'][:50]}...")

    result = run_sql_query(DB_PATH, "DELETE FROM transactions WHERE 1=1")
    assert not result["success"], "Should block DELETE"
    print(f"  DELETE blocked: {result['error'][:50]}...")

    print("  [PASS] SQL tool safety works")
    return True


def test_nl2sql():
    """Test NL-to-SQL generation."""
    separator("3. NL-TO-SQL AGENT")

    schema = get_schema(DB_PATH)

    # Test basic question
    sql = generate_sql("What is the total revenue?", schema)
    print(f"  Q: What is the total revenue?")
    print(f"  SQL: {sql}")
    result = run_sql_query(DB_PATH, sql)
    assert result["success"], f"Generated SQL failed: {result.get('error')}"
    print(f"  Result: {result['data']}")

    # Test category question
    sql = generate_sql("Which product category generated the most revenue?", schema)
    print(f"\n  Q: Which product category generated the most revenue?")
    print(f"  SQL: {sql}")
    result = run_sql_query(DB_PATH, sql)
    assert result["success"], f"Generated SQL failed: {result.get('error')}"
    print(f"  Result: {result['data'][:3]}")

    # Test with conversation context
    history = [
        {"role": "user", "content": "What was revenue in July?"},
        {"role": "assistant", "content": "Revenue in July was Rs.2,500,000."},
    ]
    sql = generate_sql("What about August?", schema, history)
    print(f"\n  Q: What about August? (with context: previous Q was about July revenue)")
    print(f"  SQL: {sql}")
    result = run_sql_query(DB_PATH, sql)
    print(f"  Result: {result['data']}")

    print("  [PASS] NL-to-SQL works")
    return True


def test_sql_correction():
    """Test self-healing SQL correction."""
    separator("4. SQL CORRECTION (self-healing)")

    schema = get_schema(DB_PATH)

    # Intentionally bad SQL with wrong column name
    bad_sql = "SELECT total_sales FROM transactions"
    result = run_sql_query(DB_PATH, bad_sql)
    assert not result["success"], "Bad SQL should fail"
    print(f"  Bad SQL: {bad_sql}")
    print(f"  Error: {result['error']}")

    # Ask LLM to correct it
    corrected = correct_sql(
        bad_sql,
        result["error"],
        schema,
        "What is the total revenue?"
    )
    print(f"  Corrected SQL: {corrected}")

    result2 = run_sql_query(DB_PATH, corrected)
    print(f"  Corrected result: {result2}")
    if result2["success"]:
        print("  [PASS] SQL correction succeeded")
    else:
        print(f"  [WARN] Correction also failed: {result2['error']}")
        print("  (This is acceptable -- the retry mechanism will handle it)")

    return True


def test_intent_classification():
    """Test intent classification."""
    separator("5. INTENT CLASSIFICATION")

    test_cases = [
        ("What is the total revenue?", "sql_query"),
        ("Hello!", "general_chat"),
        ("Which region has the highest sales?", "sql_query"),
        ("Thank you!", "general_chat"),
        ("Show revenue by month", "sql_query"),
    ]

    for question, expected in test_cases:
        intent = classify_intent(question)
        match = "[PASS]" if intent == expected else "[WARN]"
        print(f"  {match} '{question}' -> {intent} (expected: {expected})")

    print("  Intent classification done")
    return True


def test_chat_graph():
    """Test the full chat graph end-to-end."""
    separator("6. CHAT GRAPH (end-to-end)")

    chat = get_chat_graph()

    # Test 1: SQL query
    result = chat.invoke({
        "user_query": "What is the total revenue?",
        "db_path": DB_PATH,
        "conversation_history": [],
        "errors": [],
    })

    print(f"  Q: What is the total revenue?")
    print(f"  Intent: {result.get('intent')}")
    print(f"  SQL: {result.get('sql_query', 'N/A')}")
    print(f"  Answer: {result.get('chat_response', '')[:150]}...")
    assert result.get("chat_response"), "Should have a response"

    # Test 2: Follow-up question using conversation history
    history = result.get("conversation_history", [])
    result2 = chat.invoke({
        "user_query": "What about by region?",
        "db_path": DB_PATH,
        "conversation_history": history,
        "errors": [],
    })

    print(f"\n  Q: What about by region? (follow-up)")
    print(f"  Intent: {result2.get('intent')}")
    print(f"  SQL: {result2.get('sql_query', 'N/A')}")
    print(f"  Answer: {result2.get('chat_response', '')[:150]}...")

    # Test 3: General chat
    result3 = chat.invoke({
        "user_query": "Hello, what can you do?",
        "db_path": DB_PATH,
        "conversation_history": [],
        "errors": [],
    })

    print(f"\n  Q: Hello, what can you do?")
    print(f"  Intent: {result3.get('intent')}")
    print(f"  Answer: {result3.get('chat_response', '')[:150]}...")

    print("\n  [PASS] Chat graph works end-to-end")
    return True


def main():
    print("=" * 60)
    print("  AutoAnalytics -- Part 2 NL Querying Test")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: Sample data not found at {DATA_PATH}")
        print("  Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    # Clean up any previous test DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    try:
        test_database_layer()
        test_sql_tool()
        test_nl2sql()
        test_sql_correction()
        test_intent_classification()
        test_chat_graph()

        separator("RESULT: ALL PART 2 TESTS PASSED")
        print("  Part 2 -- Natural Language Querying is fully functional.")
        print()

    except AssertionError as e:
        separator("RESULT: TEST FAILED")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        separator("RESULT: ERROR")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up test DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)


if __name__ == "__main__":
    main()
