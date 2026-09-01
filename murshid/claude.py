"""Minimal Anthropic Messages API client.

Uses urllib rather than the official SDK: the SDK's HTTP stack (httpx2) hangs
indefinitely on some macOS Python installations, while urllib with an explicit
certifi CA bundle is reliable. The surface used here is small and stable, so
the tradeoff is worth the dependency reduction.
"""

from __future__ import annotations

import json
import ssl
import time
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

    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                body = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            # Rate limits and server faults are transient; the request is fine.
            if (exc.code == 429 or exc.code >= 500) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            if exc.code == 401:
                raise ClaudeError("The API key was rejected.") from exc
            if exc.code == 429:
                raise ClaudeError("Too many requests right now. Please try again shortly.") from exc
            if exc.code == 400 and b"credit" in exc.read().lower():
                raise ClaudeError("The API account is out of credit.") from exc
            raise ClaudeError(
                f"The language model returned an error ({exc.code}). Please try again."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise ClaudeError("Could not reach the language model.") from exc
    else:
        raise ClaudeError("The language model is busy. Please try again in a moment.")

    return "".join(
        block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
    ).strip()
