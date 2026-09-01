"""Minimal Anthropic Messages API client.

Uses urllib rather than the official SDK: the SDK's HTTP stack (httpx2) hangs
indefinitely on some macOS Python installations, while urllib with an explicit
certifi CA bundle is reliable. The surface used here is small and stable, so
the tradeoff is worth the dependency reduction.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from . import config

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeError(RuntimeError):
    """Raised when the Messages API cannot be reached or returns an error."""


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def complete(
    prompt: str,
    system: str,
    model: str | None = None,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> str:
    """Send one message and return Claude's text reply."""
    if not config.ANTHROPIC_API_KEY:
        raise ClaudeError("ANTHROPIC_API_KEY is not set. Add it to .env.")

    payload = json.dumps(
        {
            "model": model or config.CLAUDE_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    request = urllib.request.Request(
        _API_URL,
        data=payload,
        headers={
            "x-api-key": config.ANTHROPIC_API_KEY,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        if exc.code == 401:
            raise ClaudeError("Anthropic rejected the API key (401).") from exc
        if exc.code == 429:
            raise ClaudeError("Rate limited by the Anthropic API (429).") from exc
        raise ClaudeError(f"Anthropic API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ClaudeError(f"Could not reach the Anthropic API: {exc.reason}") from exc

    return "".join(
        block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
    ).strip()
