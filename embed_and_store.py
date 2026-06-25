"""
STEP 2: EMBEDDINGS & VECTOR STORE
==================================

KEY CONCEPT — EMBEDDINGS:
An embedding is a list of numbers (a vector) that captures the *meaning* of text.
Texts with similar meanings have vectors that are close together in vector space.

    "pancreatic cancer treatment" → [0.12, -0.45, 0.78, ...]  (1536 dimensions)
    "therapy for pancreas tumors" → [0.11, -0.44, 0.77, ...]  (very similar!)
    "best pizza in NYC"          → [0.89,  0.23, -0.56, ...] (very different)

We use OpenAI's text-embedding-3-small model — fast, cheap, good quality.

KEY CONCEPT — VECTOR STORE:
A vector store (ChromaDB here) indexes these vectors so you can efficiently
find the k most similar ones to a query. Under the hood it uses approximate
nearest neighbor (ANN) algorithms — you don't need to compare against every
vector, which makes retrieval fast even with millions of chunks.

ChromaDB stores everything locally in a SQLite + hnswlib index.
No server needed — perfect for learning.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# We import from step 1
from ingest import load_documents, chunk_documents

# --- Configuration ---
COLLECTION_NAME = "pancreatic_cancer_docs"
PERSIST_DIR = "./chroma_db"  # Where ChromaDB saves its index to disk


def get_embedding_function():
    """
    Create an OpenAI embedding function for ChromaDB.
    ChromaDB will call this automatically when you add or query documents.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set your OPENAI_API_KEY environment variable:\n"
            "  export OPENAI_API_KEY='sk-...'"
        )
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",  # 1536 dimensions, $0.02/1M tokens
    )


def create_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """
    Embed all chunks and store them in ChromaDB.

    ChromaDB handles embedding automatically — you pass raw text and it
    calls the embedding function for you.
    """
    # PersistentClient saves to disk so you don't re-embed every time
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Delete existing collection if re-running (idempotent)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        # Cosine similarity is standard for text embeddings
        metadata={"hnsw:space": "cosine"},
    )

    # Add chunks in batches (ChromaDB has a batch limit)
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
        print(f"  Embedded & stored batch {i // BATCH_SIZE + 1}")

    print(f"  Total vectors in store: {collection.count()}")
    return collection


# --- Run standalone to test ---
if __name__ == "__main__":
    print("=== Loading & chunking ===")
    docs = load_documents()
    if not docs:
        print("  No documents found in docs/. Add .txt files first.")
        exit(1)

    chunks = chunk_documents(docs)

    print("\n=== Embedding & storing ===")
    collection = create_vector_store(chunks)

    # Quick test: query the store
    print("\n=== Test query ===")
    results = collection.query(
        query_texts=["What are the symptoms of pancreatic cancer?"],
        n_results=3,
    )
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n  Result {i+1} (distance: {results['distances'][0][i]:.3f}):")
        print(f"  {doc[:150]}...")
