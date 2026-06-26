"""
GRADIO CHAT UI FOR RAG
=======================

This wraps the RAG chain in a web-based chat interface.

Gradio's ChatInterface handles:
- Conversation display (message bubbles)
- Text input + submit button
- Chat history state

We just provide a function that takes (message, history) and returns an answer.

RUN:
  uv add gradio
  uv run python app.py
  → Opens at http://localhost:7860
"""

import gradio as gr
from dotenv import load_dotenv

from rag_medical.rag_chain import format_context, generate_answer, get_collection, retrieve

load_dotenv()


# Load the vector store once at startup (not per-request)
collection = get_collection()


def respond(message: str, history: list[dict]) -> str:
    """
    Called by Gradio on each user message.

    Args:
        message: The user's latest question
        history: List of {"role": "user"|"assistant", "content": "..."} dicts
                 (managed by Gradio — we don't use it here, but you could
                  for conversation-aware retrieval)
    """
    # Retrieve relevant chunks
    retrieved = retrieve(collection, message)

    # Format sources for display
    sources = set(chunk["source"] for chunk in retrieved)
    source_line = f"\n\n---\n*Sources: {', '.join(sources)}*"

    # Generate answer with conversation history for follow-up awareness
    context = format_context(retrieved)
    answer = generate_answer(message, context, history=history)

    return answer + source_line


# --- Build the UI ---
demo = gr.ChatInterface(
    fn=respond,
    title="Pancreatic Cancer Q&A Bot",
    description="Ask questions about pancreatic cancer. Answers are grounded in the loaded documents.",
    examples=[
        "What are the early symptoms of pancreatic cancer?",
        "What is the 5-year survival rate?",
        "What is the Whipple procedure?",
        "What genetic mutations increase risk?",
    ],
)

if __name__ == "__main__":
    demo.launch()
