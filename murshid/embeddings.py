"""Text embeddings via the Voyage AI API.

Voyage is used instead of a local sentence-transformers model so the deployed
app needs no torch (~1GB) and fits in Streamlit Cloud's memory limit. Voyage
also encodes queries and documents differently (`input_type`), which suits
question-to-passage retrieval better than a symmetric paraphrase model.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request

from . import config

_API_URL = "https://api.voyageai.com/v1/embeddings"


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring certifi's CA bundle.

    Python installations from python.org do not always have access to the
    system trust store, which makes HTTPS requests fail with
    CERTIFICATE_VERIFY_FAILED even though curl works.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_MAX_BATCH = 16
_FREE_TIER_DELAY = 21.0  # 3 requests/min ceiling on Voyage's free tier


class EmbeddingError(RuntimeError):
    """Raised when the embedding API cannot be reached or returns an error."""


def _post(texts: list[str], input_type: str, timeout: float) -> list[list[float]]:
    if not config.VOYAGE_API_KEY:
        raise EmbeddingError(
            "VOYAGE_API_KEY is not set. Add it to .env "
            "(free key at https://www.voyageai.com/)."
        )

    payload = json.dumps(
        {"input": texts, "model": config.VOYAGE_MODEL, "input_type": input_type}
    ).encode()

    request = urllib.request.Request(
        _API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                body = json.load(response)
            return [item["embedding"] for item in body["data"]]
        except urllib.error.HTTPError as exc:
            # 429 is the free tier's rate limit; 5xx are transient server faults.
            # Both are worth retrying - the request itself is fine.
            if (exc.code == 429 or exc.code >= 500) and attempt < 5:
                delay = _FREE_TIER_DELAY if exc.code == 429 else 2.0
                time.sleep(delay * (attempt + 1))
                continue
            if exc.code == 401:
                raise EmbeddingError("Voyage rejected the API key.") from exc
            raise EmbeddingError(
                f"The embedding service returned an error ({exc.code}). Please try again."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 5:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise EmbeddingError("Could not reach the embedding service.") from exc

    raise EmbeddingError("The embedding service is busy. Please try again in a moment.")


def embed_query(text: str, timeout: float = 30.0) -> list[float]:
    """Embed a single user question."""
    return _post([text], "query", timeout)[0]


def embed_documents(
    texts: list[str], timeout: float = 120.0, progress=None
) -> list[list[float]]:
    """Embed a list of documents, batching to respect API limits."""
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _MAX_BATCH):
        if start:
            time.sleep(_FREE_TIER_DELAY)  # stay within the free tier's request rate
        batch = texts[start : start + _MAX_BATCH]
        vectors.extend(_post(batch, "document", timeout))
        if progress:
            progress(len(vectors), len(texts))

    return vectors
