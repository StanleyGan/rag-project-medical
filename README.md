# RAG Project — Medical Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions about medical literature (currently pancreatic cancer) using document retrieval and OpenAI, with a Gradio web UI.

## How It Works

```
docs/ (.txt, .pdf) → ingest.py (load + chunk) → embed_and_store.py (embed + store in ChromaDB) → rag_chain.py (retrieve + generate)
```

1. **Ingest** — Load `.txt` and `.pdf` files from `docs/`, split into ~500-character chunks
2. **Embed & Store** — Embed chunks with OpenAI `text-embedding-3-small`, store in ChromaDB
3. **Retrieve & Generate** — For each question, find the top-4 most similar chunks and generate a grounded answer with source citations

## Quick Start

```bash
# Install dependencies
make install    # or: uv sync

# Set your OpenAI API key
echo 'OPENAI_API_KEY=sk-...' > .env

# Add documents to docs/ (supports .txt and .pdf)
mkdir -p docs
# cp your-documents.pdf docs/

# Ingest documents into the vector store
make ingest     # or: uv run python run.py --create-db

# Launch the web UI
make serve      # or: uv run python app.py
# → http://localhost:7860
```

## Project Structure

```
├── app.py                          ← Gradio chat UI (main entry point)
├── run.py                          ← Terminal entry point + document ingestion
├── src/rag_medical/                ← Library package
│   ├── config.py                   ← All tunable parameters
│   ├── ingest.py                   ← Document loading (txt + pdf) and chunking
│   ├── embed_and_store.py          ← OpenAI embeddings + ChromaDB storage
│   └── rag_chain.py                ← Retrieval, prompt construction, LLM generation, conversation memory
├── docs/                           ← Your .txt and .pdf documents (gitignored)
├── chroma_db/                      ← Vector store (auto-generated, gitignored)
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Features

- **PDF and text support** — Extracts text from PDFs using PyMuPDF
- **Conversation memory** — Maintains chat context across follow-up questions, with automatic summarization when history exceeds the token limit
- **Source citations** — Every answer cites which documents it used
- **Grounded answers** — The LLM only answers from provided context, reducing hallucination

## Tech Stack

- **Python 3.11+** (managed with [uv](https://docs.astral.sh/uv/))
- **OpenAI** — embeddings (`text-embedding-3-small`) + chat (`gpt-4.1-mini`)
- **ChromaDB** — local vector database
- **Gradio** — chat web UI
- **PyMuPDF** — PDF text extraction
- **LangChain** — text splitters only
- **Langfuse** — observability and tracing (optional)

## Configuration

All settings are centralized in `src/rag_medical/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL` | `gpt-4.1-mini` | OpenAI chat model |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `MAX_HISTORY_TOKENS` | `64000` | Token threshold before conversation summarization |

| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

## Observability (Langfuse)

Every RAG query is traced automatically — retrieval, LLM calls, latency, and token usage. To enable:

1. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Add to your `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

Langfuse is optional — when the env vars are not set, tracing is a no-op and the app works normally.
