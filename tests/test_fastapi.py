"""
Pytest tests for the FastAPI chat app.
"""
import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient

MODULE_NAME = "app.api"  # the entrypoint module, importable as app.api


class FakeChatAgent:
    """Stand-in for app.agent.ChatAgent, shared across all tests via the
    imported module's `agent` attribute."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def stream_answer(self, question, thread_id="1"):
        self.calls.append((question, thread_id))
        yield f"echo:{question}:{thread_id}:part1"
        yield "part2"


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    """Import the FastAPI app module with a fake ChatAgent and a temp
    static/index.html, so the module-level side effects succeed safely."""

    # 1. Inject a fake `app.agent` submodule before `app.api` imports from
    #    it. The real `app` package is left alone so `app.api` still
    #    imports normally from disk; only the `app.agent` submodule is
    #    swapped out via the sys.modules cache, which Python's import
    #    system consults before touching the filesystem.
    fake_agent_module = types.ModuleType("app.agent")
    fake_agent_module.ChatAgent = FakeChatAgent

    monkeypatch.setitem(sys.modules, "app.agent", fake_agent_module)

    # 2. Provide static/index.html relative to the cwd used at import time.
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>hi</body></html>")
    monkeypatch.chdir(tmp_path)

    # 3. Force a fresh import so the patched app.agent is actually used.
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)
    yield module

    sys.modules.pop(MODULE_NAME, None)


@pytest.fixture
def client(app_module):
    return TestClient(app_module.app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_static_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "hi" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_chat_streams_agent_output(client):
    response = client.get("/chat", params={"question": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "echo:hello:1:part1part2"


def test_chat_uses_default_thread_id(client, app_module):
    client.get("/chat", params={"question": "hi"})
    assert app_module.agent.calls[-1] == ("hi", "1")


def test_chat_passes_custom_thread_id(client, app_module):
    client.get("/chat", params={"question": "hi", "thread_id": "42"})
    assert app_module.agent.calls[-1] == ("hi", "42")


def test_chat_missing_question_returns_422(client):
    response = client.get("/chat")
    assert response.status_code == 422


def test_chat_multiple_calls_tracked_independently(client, app_module):
    client.get("/chat", params={"question": "a", "thread_id": "1"})
    client.get("/chat", params={"question": "b", "thread_id": "2"})
    assert app_module.agent.calls == [("a", "1"), ("b", "2")]