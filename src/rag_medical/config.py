"""
Centralized configuration for the RAG pipeline.

All tunable parameters live here so you can experiment without
editing the pipeline modules.
"""

# --- Model ---
MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Retrieval ---
COLLECTION_NAME = "pancreatic_cancer_docs"
PERSIST_DIR = "./chroma_db"
TOP_K = 4

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Memory ---
MAX_HISTORY_TOKENS = 64000
