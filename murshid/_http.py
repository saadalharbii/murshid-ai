"""Shared HTTP helpers for the Voyage and Anthropic REST clients.

Both clients use urllib rather than their vendor SDKs: the Anthropic SDK's
httpx2 stack hangs indefinitely on some macOS Python installations, and urllib
keeps the deployed dependency set small enough for Streamlit Cloud.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request

_ssl_singleton: ssl.SSLContext | None = None


def ssl_context() -> ssl.SSLContext:
    """An SSL context using certifi's CA bundle, built once and reused.

    Python installations from python.org cannot always read the macOS system
    trust store, which makes HTTPS fail with CERTIFICATE_VERIFY_FAILED even
    where curl succeeds.
    """
    global _ssl_singleton

    if _ssl_singleton is None:
        try:
            import certifi

            _ssl_singleton = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            _ssl_singleton = ssl.create_default_context()

    return _ssl_singleton


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    attempts: int = 4,
    retry_delay: float = 2.0,
    rate_limit_delay: float = 21.0,
) -> dict:
    """POST JSON and return the decoded response, retrying transient failures.

    Retries 429 (rate limit) and 5xx (server fault); both mean the request
    itself was fine. Raises urllib.error.HTTPError for anything else so callers
    can map status codes onto their own error types.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers},
    )

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < attempts - 1:
                time.sleep(rate_limit_delay * (attempt + 1))
                continue
            if exc.code >= 500 and attempt < attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < attempts - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise

    raise RuntimeError("unreachable")
