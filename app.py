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

from dotenv import load_dotenv

load_dotenv()

import gradio as gr  # noqa: E402
from langfuse import get_client, observe  # noqa: E402

from rag_medical.rag_chain import format_context, generate_answer, get_collection, retrieve  # noqa: E402

# Load the vector store once at startup (not per-request)
collection = get_collection()
langfuse = get_client()

# Store trace IDs so we can attach feedback scores to the right trace
last_trace_id = None


@observe(name="gradio_respond")
def respond(message: str, history: list[dict]) -> str:
    """
    Called by Gradio on each user message.

    Args:
        message: The user's latest question
        history: List of {"role": "user"|"assistant", "content": "..."} dicts
    """
    global last_trace_id

    # Retrieve relevant chunks
    retrieved = retrieve(collection, message)

    # Format sources for display
    sources = set(chunk["source"] for chunk in retrieved)
    source_line = f"\n\n---\n*Sources: {', '.join(sources)}*"

    # Generate answer with conversation history for follow-up awareness
    context = format_context(retrieved)
    answer = generate_answer(message, context, history=history)

    # Capture trace ID for feedback scoring
    last_trace_id = langfuse.get_current_trace_id()

    # Flush traces to Langfuse
    langfuse.flush()

    return answer + source_line


def handle_feedback(data: gr.LikeData):
    """Send thumbs up/down feedback to Langfuse as a score on the trace."""
    if last_trace_id is None:
        return

    langfuse.create_score(
        trace_id=last_trace_id,
        name="user_feedback",
        value=1 if data.liked else 0,
    )
    langfuse.flush()


# --- Build the UI ---
with gr.Blocks() as demo:
    chat = gr.ChatInterface(
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
    chat.chatbot.like(handle_feedback, None, None)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
