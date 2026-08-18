"""
Pytest tests for the ingestion module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion import (
    load_wikipedia_url,
    fetch_docs,
    chunk_docs,
    save_to_json,
)

MODULE_PATH = "app.ingestion"


# ---------------------------------------------------------------------------
# load_wikipedia_url
# ---------------------------------------------------------------------------


class TestLoadWikipediaUrl:
    def _mock_response(self, html_text="<html>some html</html>", status_ok=True):
        mock_resp = MagicMock()
        mock_resp.text = html_text
        if status_ok:
            mock_resp.raise_for_status.return_value = None
        else:
            mock_resp.raise_for_status.side_effect = Exception("HTTP error")
        return mock_resp

    @patch(f"{MODULE_PATH}.trafilatura.extract")
    @patch(f"{MODULE_PATH}.requests.get")
    def test_success_returns_expected_dict(self, mock_get, mock_extract):
        mock_get.return_value = self._mock_response()
        mock_extract.return_value = "Extracted article text."

        result = load_wikipedia_url(
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            user_agent="test-agent",
        )

        assert result == {
            "page_content": "Extracted article text.",
            "metadata": {
                "source": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "title": "Python (programming language)",
            },
        }

    @patch(f"{MODULE_PATH}.trafilatura.extract")
    @patch(f"{MODULE_PATH}.requests.get")
    def test_builds_correct_api_url_and_headers(self, mock_get, mock_extract):
        mock_get.return_value = self._mock_response()
        mock_extract.return_value = "text"

        load_wikipedia_url("https://en.wikipedia.org/wiki/Dog", user_agent="my-agent")

        called_url = mock_get.call_args.args[0]
        called_kwargs = mock_get.call_args.kwargs
        assert called_url == "https://en.wikipedia.org/api/rest_v1/page/html/Dog"
        assert called_kwargs["headers"]["User-Agent"] == "my-agent"
        assert called_kwargs["headers"]["Accept-Encoding"] == "gzip"
        assert called_kwargs["timeout"] == 30

    @patch(f"{MODULE_PATH}.trafilatura.extract")
    @patch(f"{MODULE_PATH}.requests.get")
    def test_url_encoded_title_is_unquoted(self, mock_get, mock_extract):
        mock_get.return_value = self._mock_response()
        mock_extract.return_value = "text"

        result = load_wikipedia_url(
            "https://en.wikipedia.org/wiki/S%C3%A3o_Paulo", user_agent="agent"
        )

        assert result["metadata"]["title"] == "São Paulo"

    @patch(f"{MODULE_PATH}.trafilatura.extract")
    @patch(f"{MODULE_PATH}.requests.get")
    def test_raises_value_error_when_extraction_fails(self, mock_get, mock_extract):
        mock_get.return_value = self._mock_response()
        mock_extract.return_value = None

        with pytest.raises(ValueError, match="Could not extract text"):
            load_wikipedia_url("https://en.wikipedia.org/wiki/Empty", user_agent="a")

    @patch(f"{MODULE_PATH}.trafilatura.extract")
    @patch(f"{MODULE_PATH}.requests.get")
    def test_raises_value_error_when_extraction_is_empty_string(
        self, mock_get, mock_extract
    ):
        mock_get.return_value = self._mock_response()
        mock_extract.return_value = ""

        with pytest.raises(ValueError):
            load_wikipedia_url("https://en.wikipedia.org/wiki/Empty", user_agent="a")

    @patch(f"{MODULE_PATH}.requests.get")
    def test_propagates_http_errors(self, mock_get):
        mock_get.return_value = self._mock_response(status_ok=False)

        with pytest.raises(Exception, match="HTTP error"):
            load_wikipedia_url("https://en.wikipedia.org/wiki/Cat", user_agent="a")


# ---------------------------------------------------------------------------
# fetch_docs
# ---------------------------------------------------------------------------


class TestFetchDocs:
    @patch(f"{MODULE_PATH}.load_wikipedia_url")
    def test_calls_loader_for_every_url(self, mock_loader):
        mock_loader.side_effect = lambda url, user_agent: {
            "page_content": f"content for {url}",
            "metadata": {"source": url, "title": url},
        }

        urls = [
            "https://en.wikipedia.org/wiki/A",
            "https://en.wikipedia.org/wiki/B",
        ]
        docs = fetch_docs(urls)

        assert mock_loader.call_count == 2
        assert [d["metadata"]["source"] for d in docs] == urls

    @patch(f"{MODULE_PATH}.load_wikipedia_url")
    def test_empty_url_list_returns_empty_list(self, mock_loader):
        assert fetch_docs([]) == []
        mock_loader.assert_not_called()


# ---------------------------------------------------------------------------
# chunk_docs
# ---------------------------------------------------------------------------


class TestChunkDocs:
    def test_chunks_single_document(self):
        docs = [
            {
                "page_content": "word " * 200,  # long enough to force multiple chunks
                "metadata": {
                    "source": "https://en.wikipedia.org/wiki/Foo",
                    "title": "Foo",
                },
            }
        ]

        chunks = chunk_docs(docs, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "https://en.wikipedia.org/wiki/Foo"

    def test_chunk_ids_are_sequential_per_document(self):
        docs = [
            {
                "page_content": "word " * 200,
                "metadata": {
                    "source": "https://en.wikipedia.org/wiki/Foo",
                    "title": "Foo",
                },
            }
        ]

        chunks = chunk_docs(docs, chunk_size=50, chunk_overlap=10)

        expected_ids = [
            f"https://en.wikipedia.org/wiki/Foo_chunk_{i}" for i in range(len(chunks))
        ]
        assert [c.id for c in chunks] == expected_ids

    def test_multiple_documents_are_all_chunked(self):
        docs = [
            {
                "page_content": "word " * 100,
                "metadata": {"source": "https://en.wikipedia.org/wiki/A", "title": "A"},
            },
            {
                "page_content": "word " * 100,
                "metadata": {"source": "https://en.wikipedia.org/wiki/B", "title": "B"},
            },
        ]

        chunks = chunk_docs(docs, chunk_size=50, chunk_overlap=10)

        sources = {c.metadata["source"] for c in chunks}
        assert sources == {
            "https://en.wikipedia.org/wiki/A",
            "https://en.wikipedia.org/wiki/B",
        }

    def test_short_content_produces_single_chunk(self):
        docs = [
            {
                "page_content": "short text",
                "metadata": {
                    "source": "https://en.wikipedia.org/wiki/Short",
                    "title": "Short",
                },
            }
        ]

        chunks = chunk_docs(docs, chunk_size=300, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].page_content == "short text"

    def test_empty_docs_list_returns_empty(self):
        assert chunk_docs([]) == []

    def test_missing_metadata_defaults_to_unknown_source(self):
        docs = [{"page_content": "some content here"}]

        chunks = chunk_docs(docs, chunk_size=300, chunk_overlap=50)

        assert chunks[0].id == "unknown_chunk_0"
        assert chunks[0].metadata == {}


# ---------------------------------------------------------------------------
# save_to_json
# ---------------------------------------------------------------------------


class TestSaveToJson:
    def test_writes_json_file_with_expected_content(self, tmp_path):
        docs = [{"page_content": "hello", "metadata": {"source": "url", "title": "t"}}]
        save_path = tmp_path / "out.json"

        save_to_json(docs, save_path)

        assert save_path.exists()
        with open(save_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == docs

    def test_creates_parent_directories(self, tmp_path):
        docs = [{"page_content": "hello"}]
        save_path = tmp_path / "nested" / "dir" / "out.json"

        save_to_json(docs, save_path)

        assert save_path.exists()

    def test_preserves_unicode_characters(self, tmp_path):
        docs = [{"page_content": "São Paulo café"}]
        save_path = tmp_path / "unicode.json"

        save_to_json(docs, save_path)

        raw = save_path.read_text(encoding="utf-8")
        assert "São Paulo café" in raw  # ensure_ascii=False keeps unicode readable

    def test_overwrites_existing_file(self, tmp_path):
        save_path = tmp_path / "out.json"
        save_to_json([{"page_content": "first"}], save_path)
        save_to_json([{"page_content": "second"}], save_path)

        with open(save_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == [{"page_content": "second"}]
