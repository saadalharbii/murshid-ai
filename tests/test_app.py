"""The deployed page, run end to end with the network faked.

test_murshid.py covers the pipeline without importing Streamlit. These tests
run streamlit_app.py itself through Streamlit's test runner, so a failure that
only shows on the live page - a broken import, a render error, an exception
between the fallbacks - fails here first instead of in front of a visitor.

The fake replaces urllib.request.urlopen, which every API client calls, so the
app's own code runs unchanged down to the socket.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from murshid.messages import TROUBLE_AR, TROUBLE_EN  # noqa: E402

APP = Path(__file__).parent.parent / "streamlit_app.py"
INDEX = Path(__file__).parent.parent / "data" / "index.npz"


def _sse(*texts: str) -> bytes:
    events = [{"type": "content_block_delta", "delta": {"text": text}} for text in texts]
    return b"".join(f"data: {json.dumps(event)}\n".encode() for event in events)


class FakeNetwork:
    """Answers each API the way it answers a healthy request, or fails.

    `mode` is "ok", "down" (every connection reset) or "garbled" (every reply
    an HTML error page, as a proxy in front of an API returns during an outage).
    """

    def __init__(self):
        self.mode = "ok"
        self.urls: list[str] = []
        # A real passage's own vector, so vector search has a true match.
        with np.load(INDEX) as index:
            self.vector = index["vectors"][0].astype(float).tolist()

    def __call__(self, request, timeout=None, context=None):
        url = request.full_url
        self.urls.append(url)
        if self.mode == "down":
            raise ConnectionResetError("connection reset by peer")
        if self.mode == "garbled":
            return io.BytesIO(b"<html><body>502 Bad Gateway</body></html>")

        payload = json.loads(request.data)
        if url.endswith("/embeddings"):
            body = {"data": [{"embedding": self.vector}]}
        elif url.endswith("/rerank"):
            count = min(len(payload["documents"]), payload["top_k"])
            body = {"data": [{"index": i, "relevance_score": 0.9} for i in range(count)]}
        elif "standalone search query" in payload["system"]:
            return io.BytesIO(_sse("How much does it cost to live in Manchester?"))
        else:
            return io.BytesIO(_sse("Students ", "recommend ", "Monzo [1]."))
        return io.BytesIO(json.dumps(body).encode())


@pytest.fixture
def network(monkeypatch):
    """Fake the network and give each test a freshly imported package.

    The app drops and re-imports the murshid package when its files change,
    so the package test_murshid.py imported must be put back afterwards -
    its tests hold references to those module objects.
    """
    import urllib.request

    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "murshid"}
    for name in saved:
        del sys.modules[name]

    fake = FakeNetwork()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("VOYAGE_API_KEY", "test")
    # One attempt per call: retries only add sleeps to a test of what
    # happens once they are exhausted.
    monkeypatch.setenv("QUERY_ATTEMPTS", "1")
    st.cache_resource.clear()

    yield fake

    st.cache_resource.clear()
    for name in [name for name in sys.modules if name.split(".")[0] == "murshid"]:
        del sys.modules[name]
    sys.modules.update(saved)


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def _ask(app: AppTest, question: str) -> AppTest:
    app.chat_input[0].set_value(question).run()
    assert not app.exception, app.exception
    return app


def _text(app: AppTest) -> str:
    return "\n".join(element.value for element in app.markdown)


def _errors(app: AppTest) -> list[str]:
    return [element.value for element in app.error]


class TestPage:
    def test_loads(self, network):
        app = _app()
        assert "MurshidAI" in app.title[0].value
        assert not app.error

    def test_answers_with_sources(self, network):
        app = _ask(_app(), "How do I open a bank account?")
        assert "Monzo [1]" in _text(app)
        assert any("Sources" in expander.label for expander in app.expander)
        assert not app.error

    def test_follow_up_shows_what_was_searched(self, network):
        app = _ask(_app(), "How much does it cost to live in London?")
        app = _ask(app, "What about Manchester?")
        assert "Searched for: How much does it cost to live in Manchester?" in _text(app)


class TestOutages:
    """Whatever fails, a visitor sees a sentence in their language."""

    def test_every_service_down(self, network):
        network.mode = "down"
        app = _ask(_app(), "How do I open a bank account?")
        assert _errors(app) == [TROUBLE_EN]

    def test_every_service_down_in_arabic(self, network):
        network.mode = "down"
        app = _ask(_app(), "كيف أفتح حساب بنكي؟")
        assert _errors(app) == [TROUBLE_AR]

    def test_every_service_garbled(self, network):
        # Search falls back to keywords; the model's garbled reply has no
        # text in it, which must read as a failure, not an empty answer.
        network.mode = "garbled"
        app = _ask(_app(), "How do I open a bank account?")
        assert _errors(app) == [TROUBLE_EN]

    def test_outage_during_a_follow_up(self, network):
        app = _ask(_app(), "How much does it cost to live in London?")
        network.mode = "down"
        app = _ask(app, "What about Manchester?")
        assert _errors(app) == [TROUBLE_EN]
        # The earlier answer is still on the page.
        assert "Monzo [1]" in _text(app)

    def test_recovers_once_services_return(self, network):
        network.mode = "down"
        app = _ask(_app(), "How do I open a bank account?")
        network.mode = "ok"
        app = _ask(app, "How do I open a bank account?")
        assert "Monzo [1]" in _text(app)

    def test_unexpected_failure_is_contained(self, network, monkeypatch):
        # A bug no fallback anticipated, after the page has loaded.
        app = _app()
        rag = sys.modules["murshid.rag"]

        def broken(*args, **kwargs):
            raise AttributeError("a bug nobody anticipated")

        monkeypatch.setattr(rag.RAGPipeline, "retrieve", broken)
        app = _ask(app, "How do I open a bank account?")
        assert _errors(app) == [TROUBLE_EN]


class TestDeploys:
    """Streamlit Cloud deploys by pulling new files into the running process.

    Once, that left the old murshid.messages in memory while the new
    streamlit_app.py imported a name only the new one had, and every visitor
    got an ImportError until the app was rebooted by hand.
    """

    def _stale_messages(self) -> types.ModuleType:
        """murshid.messages as it was before `searched_for` was added."""
        current = sys.modules["murshid.messages"]
        stale = types.ModuleType("murshid.messages")
        for name, value in vars(current).items():
            if name != "searched_for":
                setattr(stale, name, value)
        return stale

    def test_the_failure_reproduces_without_the_guard(self, network):
        # The package still matches the files on disk, so the guard keeps
        # it - and the stale module breaks the page exactly as it did live.
        app = _app()
        sys.modules["murshid.messages"] = self._stale_messages()
        app.run()
        assert app.exception
        assert "searched_for" in app.exception[0].message

    def test_code_changed_on_disk_is_reloaded(self, network):
        app = _app()
        sys.modules["murshid.messages"] = self._stale_messages()
        # As after a pull: the files on disk no longer match what is loaded.
        sys.modules["murshid"]._source_stamp = ("before the pull",)
        app.run()
        assert not app.exception, app.exception
        assert hasattr(sys.modules["murshid.messages"], "searched_for")

        app = _ask(app, "How do I open a bank account?")
        assert "Monzo [1]" in _text(app)
