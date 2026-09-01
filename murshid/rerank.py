"""Reranking via the Voyage AI rerank API.

Cosine similarity over this corpus separates poorly: every chunk mixes several
conversations, so the top-10 scores for a query land within about 0.05 of each
other and the genuinely relevant passage is often far down the list. A cross-
encoder reranker reads each candidate against the query directly and scores it
on an absolute scale, which both reorders the list and gives a usable
confidence signal.

Reranking is always optional: if the API is unavailable the caller keeps the
vector ordering and still answers.
"""

from __future__ import annotations

import urllib.error

from . import config
from ._http import post_json

_API_URL = "https://api.voyageai.com/v1/rerank"
_MAX_DOC_CHARS = 1500


class RerankError(RuntimeError):
    """Raised when the rerank API cannot be reached or returns an error."""


def rerank(query: str, documents: list[str], top_n: int, timeout: float = 30.0) -> list[tuple[int, float]]:
    """Score `documents` against `query`.

    Returns (index, relevance_score) pairs, most relevant first, where index
    refers to the position in the supplied `documents` list.
    """
    if not documents:
        return []

    if not config.VOYAGE_API_KEY:
        raise RerankError("VOYAGE_API_KEY is not set.")

    payload = {
        "query": query,
        "documents": [doc[:_MAX_DOC_CHARS] for doc in documents],
        "model": config.RERANK_MODEL,
        "top_k": min(top_n, len(documents)),
    }

    try:
        body = post_json(
            _API_URL,
            payload,
            {"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
            timeout=timeout,
            # The rerank endpoint has its own rate limit, so a 429 here means
            # waiting out the window rather than failing the whole query.
            attempts=3,
        )
    except urllib.error.HTTPError as exc:
        raise RerankError(f"Voyage rerank returned {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RerankError("Could not reach the rerank service.") from exc

    return [(item["index"], item["relevance_score"]) for item in body["data"]]
