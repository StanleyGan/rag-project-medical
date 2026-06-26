from rag_medical.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_HISTORY_TOKENS,
    MODEL,
    TOP_K,
)


def test_model_is_set():
    assert MODEL and isinstance(MODEL, str)


def test_embedding_model_is_set():
    assert EMBEDDING_MODEL and isinstance(EMBEDDING_MODEL, str)


def test_chunk_overlap_less_than_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_top_k_positive():
    assert TOP_K > 0


def test_max_history_tokens_positive():
    assert MAX_HISTORY_TOKENS > 0
