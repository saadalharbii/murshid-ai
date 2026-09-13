"""Text embeddings via the Voyage AI API.

Voyage is used instead of a local sentence-transformers model so the deployed
app needs no torch (~1GB) and fits in Streamlit Cloud's memory limit. Voyage
also encodes queries and documents differently (`input_type`), which suits
question-to-passage retrieval better than a symmetric paraphrase model.
"""

from __future__ import annotations

import urllib.error

from . import config
from ._http import post_json

_API_URL = "https://api.voyageai.com/v1/embeddings"


_MAX_BATCH = 128

# Sleep applied only AFTER the API says 429. Voyage's free tier allows 3
# requests/min, so a rate-limited retry has to wait out most of a minute;
# a paid key raises the ceiling to thousands and never reaches this path.
# The delay is deliberately NOT applied pre-emptively between batches: doing
# so throttled ingestion to free-tier speed regardless of the account's real
# limit, which cost ~14 minutes of sleeping on a corpus that embeds in under
# a minute.
_RATE_LIMIT_DELAY = 21.0


class EmbeddingError(RuntimeError):
    """Raised when the embedding API cannot be reached or returns an error."""


def _post(texts: list[str], input_type: str, timeout: float) -> list[list[float]]:
    """Embed a batch, mapping transport failures onto EmbeddingError.

    Retries and backoff live in `_http.post_json`, shared with the reranker,
    so there is one retry policy to reason about rather than two that drift.
    """
    if not config.VOYAGE_API_KEY:
        raise EmbeddingError("Search is not configured.")

    try:
        body = post_json(
            _API_URL,
            {"input": texts, "model": config.VOYAGE_MODEL, "input_type": input_type},
            {"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
            timeout=timeout,
            attempts=6,
            rate_limit_delay=_RATE_LIMIT_DELAY,
        )
    except urllib.error.HTTPError as exc:
        # User-facing text stays generic and vendor-neutral: an end user can
        # act on "search is unavailable", not on a provider's name or an HTTP
        # status. The status is kept on the chained exception for the logs.
        raise EmbeddingError("Search is unavailable right now.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EmbeddingError("Could not reach the search service.") from exc

    return [item["embedding"] for item in body["data"]]


def embed_query(text: str, timeout: float = 30.0) -> list[float]:
    """Embed a single user question."""
    return _post([text], "query", timeout)[0]


def embed_documents(
    texts: list[str], timeout: float = 120.0, progress=None
) -> list[list[float]]:
    """Embed a list of documents, batching to respect API limits.

    Batches are sent back to back. If the account is rate limited the API
    answers 429 and _post waits it out, so throughput matches whatever the key
    actually allows instead of being pinned to the slowest possible tier.
    """
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _MAX_BATCH):
        batch = texts[start : start + _MAX_BATCH]
        vectors.extend(_post(batch, "document", timeout))
        if progress:
            progress(len(vectors), len(texts))

    return vectors
