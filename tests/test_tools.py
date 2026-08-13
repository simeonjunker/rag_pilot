"""
Pytest tests for the tools module.
"""

from unittest.mock import MagicMock

import pytest
import requests

from app.tools import query_wikipedia_db, get_weather

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDoc:
    """Minimal stand-in for a langchain Document."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# query_wikipedia_db
# ---------------------------------------------------------------------------


class TestQueryWikipediaDb:
    def test_formats_single_result(self, monkeypatch):
        fake_docs = [FakeDoc("Some content about cats.", {"source": "Cat"})]
        mock_search = MagicMock(return_value=fake_docs)
        monkeypatch.setattr("app.tools.vector_db.similarity_search", mock_search)

        result = query_wikipedia_db.invoke({"query": "tell me about cats"})

        assert result == "# Source: Cat\n\nSome content about cats."

    def test_formats_multiple_results_joined_with_separator(self, monkeypatch):
        fake_docs = [
            FakeDoc("Content A", {"source": "Doc A"}),
            FakeDoc("Content B", {"source": "Doc B"}),
        ]
        monkeypatch.setattr(
            "app.tools.vector_db.similarity_search", MagicMock(return_value=fake_docs)
        )

        result = query_wikipedia_db.invoke({"query": "anything"})

        expected = (
            "# Source: Doc A\n\nContent A" "\n\n---\n\n" "# Source: Doc B\n\nContent B"
        )
        assert result == expected

    def test_missing_source_metadata_defaults_to_unknown(self, monkeypatch):
        fake_docs = [FakeDoc("No source here", metadata={})]
        monkeypatch.setattr(
            "app.tools.vector_db.similarity_search", MagicMock(return_value=fake_docs)
        )

        result = query_wikipedia_db.invoke({"query": "anything"})

        assert result.startswith("# Source: unknown")

    def test_no_results_returns_empty_string(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.vector_db.similarity_search", MagicMock(return_value=[])
        )

        result = query_wikipedia_db.invoke({"query": "nonexistent topic"})

        assert result == ""

    def test_calls_similarity_search_with_query_and_configured_k(self, monkeypatch):
        mock_search = MagicMock(return_value=[])
        monkeypatch.setattr("app.tools.vector_db.similarity_search", mock_search)
        monkeypatch.setattr("app.tools.RETRIEVAL_K", 7)

        query_wikipedia_db.invoke({"query": "some query"})

        mock_search.assert_called_once_with("some query", k=7)

    def test_propagates_vector_db_errors(self, monkeypatch):
        monkeypatch.setattr(
            "app.tools.vector_db.similarity_search",
            MagicMock(side_effect=RuntimeError("db unavailable")),
        )

        with pytest.raises(RuntimeError, match="db unavailable"):
            query_wikipedia_db.invoke({"query": "anything"})


# ---------------------------------------------------------------------------
# get_weather
# ---------------------------------------------------------------------------


def _make_fake_response(json_data: dict, status_ok: bool = True):
    resp = MagicMock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError("boom")
    return resp


class TestGetWeather:
    def test_formats_current_weather(self, monkeypatch):
        fake_json = {
            "current": {
                "temperature_2m": 21.5,
                "relative_humidity_2m": 55,
                "weather_code": 3,
                "wind_speed_10m": 12.3,
            }
        }
        mock_get = MagicMock(return_value=_make_fake_response(fake_json))
        monkeypatch.setattr("app.tools.requests.get", mock_get)

        result = get_weather.invoke({"latitude": 51.05, "longitude": 13.74})

        assert result == (
            "Temperature: 21.5°C\n"
            "Humidity: 55%\n"
            "Wind: 12.3 km/h\n"
            "Weather code: 3"
        )

    def test_calls_correct_endpoint_and_params(self, monkeypatch):
        fake_json = {
            "current": {
                "temperature_2m": 0,
                "relative_humidity_2m": 0,
                "weather_code": 0,
                "wind_speed_10m": 0,
            }
        }
        mock_get = MagicMock(return_value=_make_fake_response(fake_json))
        monkeypatch.setattr("app.tools.requests.get", mock_get)

        get_weather.invoke({"latitude": 51.05, "longitude": 13.74})

        mock_get.assert_called_once_with(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 51.05,
                "longitude": 13.74,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            },
        )

    def test_raises_on_http_error(self, monkeypatch):
        mock_get = MagicMock(return_value=_make_fake_response({}, status_ok=False))
        monkeypatch.setattr("app.tools.requests.get", mock_get)

        with pytest.raises(requests.HTTPError):
            get_weather.invoke({"latitude": 51.05, "longitude": 13.74})

    def test_raises_on_network_error(self, monkeypatch):
        mock_get = MagicMock(side_effect=requests.ConnectionError("network down"))
        monkeypatch.setattr("app.tools.requests.get", mock_get)

        with pytest.raises(requests.ConnectionError):
            get_weather.invoke({"latitude": 51.05, "longitude": 13.74})

    def test_raises_on_malformed_response_missing_fields(self, monkeypatch):
        # "current" present but missing an expected key
        fake_json = {"current": {"temperature_2m": 20.0}}
        mock_get = MagicMock(return_value=_make_fake_response(fake_json))
        monkeypatch.setattr("app.tools.requests.get", mock_get)

        with pytest.raises(KeyError):
            get_weather.invoke({"latitude": 51.05, "longitude": 13.74})


# ---------------------------------------------------------------------------
# Tool metadata (docstring parsing sanity checks)
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_query_wikipedia_db_has_expected_name_and_args(self):
        assert query_wikipedia_db.name == "query_wikipedia_db"
        assert "query" in query_wikipedia_db.args

    def test_get_weather_has_expected_name_and_args(self):
        assert get_weather.name == "get_weather"
        assert set(get_weather.args.keys()) == {"latitude", "longitude"}
