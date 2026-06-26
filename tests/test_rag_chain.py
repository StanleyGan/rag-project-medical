from rag_medical.rag_chain import _get_text, count_tokens, format_context


class TestGetText:
    def test_plain_string(self):
        assert _get_text("hello world") == "hello world"

    def test_none_returns_empty(self):
        assert _get_text(None) == ""

    def test_empty_string(self):
        assert _get_text("") == ""

    def test_gradio_list_format(self):
        content = [{"text": "hello", "type": "text"}]
        assert _get_text(content) == "hello"

    def test_gradio_list_multiple_parts(self):
        content = [
            {"text": "hello", "type": "text"},
            {"text": "world", "type": "text"},
        ]
        assert _get_text(content) == "hello world"

    def test_gradio_list_skips_non_text(self):
        content = [
            {"text": "hello", "type": "text"},
            {"type": "image"},
        ]
        assert _get_text(content) == "hello"

    def test_empty_list(self):
        assert _get_text([]) == ""


class TestFormatContext:
    def test_single_chunk(self):
        retrieved = [{"text": "some text", "source": "doc.txt", "distance": 0.1}]
        result = format_context(retrieved)
        assert "[Source: doc.txt]" in result
        assert "some text" in result

    def test_multiple_chunks(self):
        retrieved = [
            {"text": "text one", "source": "a.txt", "distance": 0.1},
            {"text": "text two", "source": "b.txt", "distance": 0.2},
        ]
        result = format_context(retrieved)
        assert "[Source: a.txt]" in result
        assert "[Source: b.txt]" in result
        assert "text one" in result
        assert "text two" in result

    def test_empty_retrieved(self):
        assert format_context([]) == ""


class TestCountTokens:
    def test_basic_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = count_tokens(messages)
        assert count > 0

    def test_empty_messages(self):
        assert count_tokens([]) == 0

    def test_none_content(self):
        messages = [{"role": "user", "content": None}]
        count = count_tokens(messages)
        assert count == 4  # just the overhead

    def test_gradio_list_content(self):
        messages = [{"role": "user", "content": [{"text": "hello", "type": "text"}]}]
        count = count_tokens(messages)
        assert count > 4  # overhead + at least 1 token

    def test_more_text_means_more_tokens(self):
        short = [{"role": "user", "content": "Hi"}]
        long = [{"role": "user", "content": "This is a much longer message with many words"}]
        assert count_tokens(long) > count_tokens(short)
