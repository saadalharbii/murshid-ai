"""Minimal Anthropic Messages API client.

Uses urllib rather than the official SDK: the SDK's HTTP stack (httpx2) hangs
indefinitely on some macOS Python installations, while urllib with an explicit
certifi CA bundle is reliable. The surface used here is small and stable, so
the tradeoff is worth the dependency reduction.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config
from ._http import TRANSPORT_ERRORS, ssl_context

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeError(RuntimeError):
    """Raised when the Messages API cannot be reached or returns an error."""


# Models that answered 400 to a `temperature` setting. Newer models reject it
# outright, and the model is chosen by environment variable, so the client
# learns which ones do rather than hardcoding a list that goes stale.
_REJECTS_TEMPERATURE: set[str] = set()


def _rejects_temperature(error: urllib.error.HTTPError) -> bool:
    try:
        return error.code == 400 and "temperature" in error.read().decode("utf-8", "replace")
    except Exception:
        return False


def stream(
    prompt: str,
    system: str,
    model: str | None = None,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    temperature: float | None = None,
):
    """Yield Claude's reply incrementally, so callers can render as it arrives."""
    if not config.ANTHROPIC_API_KEY:
        raise ClaudeError("ANTHROPIC_API_KEY is not set. Add it to .env.")

    model = model or config.CLAUDE_MODEL
    if model in _REJECTS_TEMPERATURE:
        temperature = None

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    payload = json.dumps(body).encode()

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
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "error":
                    # Overload and similar failures arrive as an event inside
                    # a 200 response, so ignoring them ended the reply early
                    # and silently.
                    raise ClaudeError("The language model is busy. Please try again.")
                if event.get("type") == "content_block_delta":
                    text = (event.get("delta") or {}).get("text")
                    if text:
                        yield text
    except urllib.error.HTTPError as exc:
        if temperature is not None and _rejects_temperature(exc):
            # Nothing was yielded yet - the status arrives before the stream -
            # so asking again without it is invisible to the caller. Without
            # this, the follow-up rewrite failed on every call under Sonnet 5.
            _REJECTS_TEMPERATURE.add(model)
            yield from stream(prompt, system, model, max_tokens, timeout)
            return
        if exc.code == 401:
            raise ClaudeError("The API key was rejected.") from exc
        if exc.code == 429:
            raise ClaudeError("Too many requests right now. Please try again shortly.") from exc
        raise ClaudeError(
            f"The language model returned an error ({exc.code}). Please try again."
        ) from exc
    except TRANSPORT_ERRORS as exc:
        # Includes a connection dropped mid-reply and a garbled event, which
        # escaped as raw exceptions before and skipped every fallback.
        raise ClaudeError("Could not reach the language model.") from exc


def complete(
    prompt: str,
    system: str,
    model: str | None = None,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    temperature: float | None = None,
) -> str:
    """Return Claude's full reply as a string.

    For callers that need the finished text rather than a stream: the
    evaluation harness, and the follow-up rewrite, whose output has to be
    complete before the search it feeds can start.
    """
    return "".join(
        stream(
            prompt,
            system,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
        )
    )
