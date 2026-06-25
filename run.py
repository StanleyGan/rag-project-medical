"""
RAG MINI PROJECT — Main Entry Point
=====================================

This script runs the full pipeline end-to-end:
  1. Load documents from docs/
  2. Chunk them
  3. Embed chunks and store in ChromaDB
  4. Start an interactive Q&A loop (this is turned off when --create-db is used)

SETUP:
  1. pip install -r requirements.txt
  2. export OPENAI_API_KEY='sk-...'
  3. Put .txt files in the docs/ folder
  4. python run.py

HOW IT ALL FITS TOGETHER (the "RAG" pattern):

    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  Your Docs  │ ──► │  Chunking    │ ──► │  Embedding  │
    │  (.txt)     │     │  (split text │     │  (text →    │
    │             │     │   into ~500  │     │   vectors)  │
    │             │     │   char parts)│     │             │
    └─────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
                         ┌──────────────┐     ┌─────────────┐
                         │  ChromaDB    │ ◄── │  Store      │
                         │  (vector     │     │  vectors    │
                         │   database)  │     │             │
                         └──────┬───────┘     └─────────────┘
                                │
         User asks a question   │  Find similar chunks
                    │           │
                    ▼           ▼
              ┌─────────────────────┐
              │  Retrieval          │
              │  (embed question,   │
              │   find top-k chunks)│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  LLM Generation     │
              │  (GPT-4o-mini +     │
              │   retrieved context)│
              └──────────┬──────────┘
                         │
                         ▼
                   Grounded Answer
                   (with sources!)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ingest import load_documents, chunk_documents
from embed_and_store import create_vector_store
from rag_chain import ask
from dotenv import load_dotenv
load_dotenv()
import argparse


def main():
    parser = argparse.ArgumentParser(description="RAG Mini Project")
    parser.add_argument("--create-db", action="store_true", help="Create the database")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG Mini Project — Pancreatic Cancer Q&A")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n  ERROR: Set your OpenAI API key first:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Step 1: Load
    print("\n📄 Step 1: Loading documents...")
    docs = load_documents()
    if not docs:
        print("  No .txt files found in docs/. Add some and retry.")
        sys.exit(1)

    # Step 2: Chunk
    print("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(docs)

    # Step 3: Embed & store
    print("\n🧮 Step 3: Embedding & storing in ChromaDB...")
    create_vector_store(chunks)

    # Step 4: Interactive Q&A
    if not args.create_db:
        print("\n" + "=" * 60)
        print("  Ready! Ask questions about your documents.")
        print("  Type 'quit' to exit.")
        print("=" * 60)

        while True:
            try:
                question = input("\n❓ Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            answer = ask(question)
            print(f"\n💡 Answer:\n{answer}")


if __name__ == "__main__":
    main()
