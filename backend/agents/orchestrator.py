"""Chat orchestrator — classifies intent and routes to the right handler.

Part 2: Supports sql_query and general_chat intents.
Parts 3-4 will add prediction, analytics, and recommendation routing.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import GROQ_API_KEY, GROQ_MODEL


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intent classifier for a business analytics chatbot.
Given a user question about their e-commerce data, classify it into exactly ONE of these categories:

- sql_query: Questions that can be answered by querying data (revenue, sales, counts, rankings, comparisons, trends, filtering)
- general_chat: Greetings, thanks, help requests, or questions not about the data

Return ONLY the category name, nothing else.""",
    ),
    (
        "human",
        """Conversation context:
{history}

User question: {question}

Category:""",
    ),
])


FORMAT_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful business data analyst. The user asked a question and you have the SQL query result.
Provide a clear, concise answer based ONLY on the data provided. Do not invent numbers.
If the data is empty, say you found no results.
Keep responses conversational but data-driven.""",
    ),
    (
        "human",
        """User question: {question}

SQL query used:
{sql}

Query result:
{result}

Provide a helpful answer:""",
    ),
])


CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly business analytics assistant. You help users analyze their e-commerce data.
If the user greets you, greet them back. If they ask what you can do, explain your capabilities:
- Query their dataset using natural language
- Analyze revenue, sales, customers, products, regions
- Show trends and comparisons
Keep responses brief and helpful.""",
    ),
    (
        "human",
        "{question}",
    ),
])


def _get_llm(temperature: float = 0):
    """Create the Groq LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=temperature,
        max_tokens=1024,
    )


def classify_intent(
    question: str, conversation_history: list[dict] | None = None
) -> str:
    """Classify the user's question into an intent category.

    Returns one of: 'sql_query', 'general_chat'
    (Parts 3-4 will add: 'prediction', 'analytics', 'recommendation')
    """
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in recent
        )

    llm = _get_llm()
    chain = CLASSIFY_PROMPT | llm | StrOutputParser()

    intent = chain.invoke({
        "history": history_text or "No previous context.",
        "question": question,
    })

    intent = intent.strip().lower().replace(" ", "_")

    # Validate — default to sql_query for data-related questions
    valid_intents = {"sql_query", "general_chat"}
    if intent not in valid_intents:
        intent = "sql_query"

    return intent


def format_sql_response(
    question: str, sql: str, result: dict
) -> str:
    """Use the LLM to format a SQL result into a natural language answer."""
    # Truncate large results for the prompt
    data = result.get("data", [])
    if len(data) > 20:
        result_text = str(data[:20]) + f"\n... and {len(data) - 20} more rows"
    else:
        result_text = str(data)

    llm = _get_llm(temperature=0.1)
    chain = FORMAT_RESPONSE_PROMPT | llm | StrOutputParser()

    return chain.invoke({
        "question": question,
        "sql": sql,
        "result": result_text,
    })


def general_chat(question: str) -> str:
    """Handle general/non-data questions."""
    llm = _get_llm(temperature=0.3)
    chain = CHAT_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})
