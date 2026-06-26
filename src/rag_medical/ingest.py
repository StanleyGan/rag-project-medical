"""
STEP 1: DOCUMENT INGESTION — Loading & Chunking
================================================

KEY CONCEPT: An LLM has a limited context window. You can't paste an entire
textbook into a prompt. Instead, RAG breaks documents into small "chunks" and
only retrieves the most relevant ones at query time.

CHUNKING STRATEGY matters a lot:
- Too small → chunks lack context, answers are shallow
- Too big   → retrieval is imprecise, wastes token budget
- Overlap   → prevents cutting a sentence in half at a boundary

We use RecursiveCharacterTextSplitter because it tries to split on natural
boundaries (paragraphs → sentences → words) before falling back to characters.
"""

import glob
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_medical.config import CHUNK_OVERLAP, CHUNK_SIZE

# PDF support — install with: uv add pymupdf
try:
    import fitz  # PyMuPDF

    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False


def load_pdf(filepath: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.

    PyMuPDF (fitz) is fast and handles most PDFs well, including multi-column
    layouts. For scanned PDFs (images of text), you'd need OCR (e.g. pytesseract)
    on top of this — but most research papers are text-based PDFs.
    """
    if not PDF_SUPPORTED:
        raise ImportError("PyMuPDF is required for PDF support. Install it:\n  uv add pymupdf")
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()


def load_documents(docs_dir: str = "docs") -> list[dict]:
    """Load all .txt and .pdf files from a directory into a list of {text, source} dicts."""
    documents = []

    # Load .txt files
    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        documents.append(
            {
                "text": text,
                "source": os.path.basename(filepath),
            }
        )
        print(f"  Loaded: {os.path.basename(filepath)} ({len(text):,} chars)")

    # Load .pdf files
    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        if not PDF_SUPPORTED:
            print(f"  Skipped: {os.path.basename(filepath)} (install pymupdf for PDF support)")
            continue
        text = load_pdf(filepath)
        if text:
            documents.append(
                {
                    "text": text,
                    "source": os.path.basename(filepath),
                }
            )
            print(f"  Loaded: {os.path.basename(filepath)} ({len(text):,} chars)")
        else:
            print(f"  Skipped: {os.path.basename(filepath)} (no extractable text — may be scanned)")

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split documents into overlapping chunks.

    Args:
        chunk_size:    Target size of each chunk in characters.
        chunk_overlap: How many characters adjacent chunks share.
                       This prevents losing context at boundaries.

    Returns:
        List of {text, source, chunk_id} dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order — paragraph breaks first,
        # then sentences, then words, then characters as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split_text in enumerate(splits):
            chunks.append(
                {
                    "text": split_text,
                    "source": doc["source"],
                    "chunk_id": f"{doc['source']}::chunk_{i}",
                }
            )

    print(f"\n  {len(documents)} document(s) → {len(chunks)} chunks")
    print(f"  Chunk size: ~{chunk_size} chars with {chunk_overlap} overlap")
    return chunks
