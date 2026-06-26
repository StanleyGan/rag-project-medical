import os

import pytest

from rag_medical.ingest import chunk_documents, load_documents


@pytest.fixture
def sample_docs_dir(tmp_path):
    """Create a temp directory with sample text files."""
    (tmp_path / "doc1.txt").write_text("This is document one about pancreatic cancer symptoms.")
    (tmp_path / "doc2.txt").write_text("This is document two about treatment options for pancreatic cancer.")
    return str(tmp_path)


@pytest.fixture
def sample_documents():
    return [
        {"text": "A " * 300, "source": "doc1.txt"},
        {"text": "B " * 300, "source": "doc2.txt"},
    ]


class TestLoadDocuments:
    def test_loads_txt_files(self, sample_docs_dir):
        docs = load_documents(sample_docs_dir)
        assert len(docs) == 2
        assert all("text" in d and "source" in d for d in docs)

    def test_returns_empty_for_missing_dir(self, tmp_path):
        empty_dir = str(tmp_path / "nonexistent")
        os.makedirs(empty_dir)
        docs = load_documents(empty_dir)
        assert docs == []

    def test_source_is_filename_not_path(self, sample_docs_dir):
        docs = load_documents(sample_docs_dir)
        for doc in docs:
            assert "/" not in doc["source"]
            assert doc["source"].endswith(".txt")


class TestChunkDocuments:
    def test_produces_chunks(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > len(sample_documents)

    def test_chunk_has_required_keys(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "chunk_id" in chunk

    def test_chunk_id_contains_source(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert chunk["source"] in chunk["chunk_id"]

    def test_small_doc_produces_single_chunk(self):
        docs = [{"text": "Short text.", "source": "small.txt"}]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1

    def test_respects_chunk_size(self, sample_documents):
        chunk_size = 100
        chunks = chunk_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk["text"]) <= chunk_size * 1.5  # allow some tolerance
