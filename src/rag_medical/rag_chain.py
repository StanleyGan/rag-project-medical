"""
STEP 3: THE RAG CHAIN — Retrieval + Generation
================================================

This is where everything comes together. The RAG pattern is:

    User question
         ↓
    [1] RETRIEVE: embed the question, find top-k similar chunks
         ↓
    [2] AUGMENT: inject retrieved chunks into the LLM prompt as context
         ↓
    [3] GENERATE: LLM answers using ONLY the provided context

KEY CONCEPT — WHY RAG BEATS PLAIN LLM:
- Plain LLM: "What's the 5-year survival rate for pancreatic cancer?"
  → Answers from training data (may be outdated or hallucinated)

- RAG LLM:   Same question + "Here are relevant excerpts from your docs: ..."
  → Answers grounded in YOUR documents, with sources you can verify

KEY CONCEPT — THE PROMPT TEMPLATE:
The prompt is critical. You must tell the LLM to:
  1. Only use the provided context
  2. Say "I don't know" if the context doesn't contain the answer
  3. Cite which source(s) it used
This prevents hallucination — the #1 problem in naive LLM apps.
"""

import os

import chromadb
import tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langfuse import get_client, observe
from langfuse.openai import OpenAI

from rag_medical.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_HISTORY_TOKENS,
    MODEL,
    PERSIST_DIR,
    TOP_K,
)

# Local fallback prompts — used when Langfuse prompt management is not configured
_DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context documents.

Rules:
1. If the context doesn't contain enough information to answer, say "I don't have enough information in my documents to answer that."
2. Always cite which source document(s) you used in your answer.
3. Be precise and factual. Do not add information beyond what's in the context.
4. If the context contains conflicting information, note the discrepancy.
"""

_DEFAULT_USER_PROMPT_TEMPLATE = """Context documents:
---
{{context}}
---

Question: {{question}}

Answer based on the context above:"""

langfuse = get_client()


def get_prompts():
    """
    Fetch prompts from Langfuse if available, otherwise use local defaults.

    Returns (system_text, user_template_text, prompt_object) where prompt_object
    is the Langfuse user prompt object for linking to generations (None if using defaults).

    To use Langfuse prompt management:
    1. Create a text prompt named "rag-system-prompt" in Langfuse
    2. Create a text prompt named "rag-user-prompt" with {{context}} and {{question}} variables
    3. Label both as "production"
    """
    try:
        system_obj = langfuse.get_prompt("rag-system-prompt")
        user_obj = langfuse.get_prompt("rag-user-prompt")
        return system_obj.compile(), user_obj.prompt, user_obj
    except Exception:
        return _DEFAULT_SYSTEM_PROMPT, _DEFAULT_USER_PROMPT_TEMPLATE, None


def get_collection() -> chromadb.Collection:
    """Load the existing vector store from disk."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


@observe(as_type="retriever", name="retrieve_documents")
def retrieve(collection, question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a question.

    Under the hood:
    1. The question is embedded into a vector
    2. ChromaDB finds the k nearest vectors (cosine similarity)
    3. Returns the original text + metadata
    """
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            }
        )
    return retrieved


def format_context(retrieved: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, chunk in enumerate(retrieved, 1):
        parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(parts)


"""
CONVERSATION MEMORY
====================

KEY CONCEPT — WHY MEMORY MATTERS:
Without memory, each question is independent. The user asks "What are the
symptoms?" then follows up with "Which of those appear first?" — but the
LLM has no idea what "those" refers to.

The fix: pass conversation history as messages to the LLM, just like ChatGPT does.

KEY CONCEPT — THE TOKEN BUDGET PROBLEM:
History grows with every exchange. Eventually it won't fit in the context window
(or it gets expensive). The simplest strategy:

  1. Count tokens in the history
  2. When it exceeds a threshold, ask the LLM to summarize older messages
  3. Replace those messages with the summary
  4. Continue with: [system] + [summary] + [recent messages] + [new question]

This is called "summarization memory" — it trades perfect recall for bounded cost.
"""


def _get_text(content) -> str:
    """Extract plain text from Gradio's message content (str or list of parts)."""
    if isinstance(content, list):
        return " ".join(part["text"] for part in content if part.get("text"))
    return content or ""


def count_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Count the total tokens in a list of chat messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(encoding.encode(_get_text(msg["content"]))) + 4  # +4 for message overhead
    return total


@observe(name="summarize_history")
def summarize_history(history: list[dict]) -> str:
    """
    Ask the LLM to compress conversation history into a concise summary.

    This is the key trick: instead of dropping old messages (losing info)
    or keeping everything (blowing the token budget), we *distill* them.
    """
    client = OpenAI()
    conversation_text = "\n".join(f"{msg['role'].upper()}: {_get_text(msg['content'])}" for msg in history)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this conversation concisely. Preserve key facts, "
                    "questions asked, and answers given. Focus on information the "
                    "user would need for follow-up questions."
                ),
            },
            {"role": "user", "content": conversation_text},
        ],
        temperature=0.0,
        max_tokens=300,
        name="summarize_history",
    )
    return response.choices[0].message.content


def prepare_history(history: list[dict]) -> list[dict]:
    """
    Prepare conversation history for the LLM, summarizing if too long.

    Strategy:
    - If history fits within MAX_HISTORY_TOKENS → use it as-is
    - If not → summarize older messages, keep recent ones verbatim

    Returns a list of message dicts ready to insert into the LLM call.
    """
    if not history:
        return []

    token_count = count_tokens(history)
    if token_count <= MAX_HISTORY_TOKENS:
        return list(history)

    # Split: summarize the older half, keep the recent half verbatim
    midpoint = len(history) // 2
    older = history[:midpoint]
    recent = history[midpoint:]

    print(f"  Memory: {token_count} tokens exceeds limit ({MAX_HISTORY_TOKENS}). Summarizing older messages...")
    summary = summarize_history(older)

    # Return summary as a system-ish message + recent messages
    return [
        {"role": "user", "content": f"[Summary of earlier conversation: {summary}]"},
        {"role": "assistant", "content": "Understood, I have the context from our earlier discussion."},
        *recent,
    ]


@observe(name="generate_answer")
def generate_answer(question: str, context: str, history: list[dict] | None = None) -> str:
    """
    Call the LLM with the RAG prompt and conversation history.

    Args:
        question: The user's current question
        context:  Retrieved document chunks formatted as text
        history:  Optional conversation history (list of role/content dicts).
                  Will be summarized if it exceeds MAX_HISTORY_TOKENS.
    """
    client = OpenAI()
    system_prompt, user_template, prompt_obj = get_prompts()

    # Build the messages list
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (summarized if needed)
    if history:
        messages.extend(prepare_history(history))

    # Add the current question with retrieved context
    # Langfuse templates use {{var}} mustache syntax; replace manually
    user_content = user_template.replace("{{context}}", context).replace("{{question}}", question)
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        name="rag_generation",
        langfuse_prompt=prompt_obj,
    )
    return response.choices[0].message.content


@observe(name="rag_pipeline")
def ask(question: str) -> str:
    """Full RAG pipeline: retrieve → augment → generate."""
    print(f"\n  Question: {question}")

    # Step 1: Retrieve
    collection = get_collection()
    retrieved = retrieve(collection, question)
    print(f"  Retrieved {len(retrieved)} chunks:")
    for chunk in retrieved:
        print(f"    - [{chunk['source']}] (distance: {chunk['distance']:.3f})")

    # Step 2: Augment (format context)
    context = format_context(retrieved)

    # Step 3: Generate
    print("  Generating answer...")
    answer = generate_answer(question, context)

    # Flush traces to ensure delivery
    get_client().flush()

    return answer
