"""Local vector store backed by a prebuilt numpy index.

The corpus is static, so vectors are computed once by `ingest.py` and committed
to the repo as a single .npz file. That removes the external vector database
the project previously depended on, and with it the only piece of the demo
that could silently go offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class Document:
    """A retrieved chunk and its score against the query.

    `score_kind` says what `score` means, because the two stages produce
    different numbers on different scales: "similarity" is the cosine score
    from vector search, "relevance" is the reranker's absolute score. They are
    not comparable, so anything displaying or thresholding a score needs to
    know which it holds.
    """

    __slots__ = ("content", "metadata", "score", "score_kind")

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any],
        score: float,
        score_kind: str = "similarity",
    ):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.score_kind = score_kind


class VectorStore:
    """Cosine-similarity search over an in-memory matrix of embeddings."""

    def __init__(self, vectors: np.ndarray, contents: list[str], metadata: list[dict]):
        if len(vectors) != len(contents) or len(contents) != len(metadata):
            raise ValueError("vectors, contents and metadata must be the same length")

        # Pre-normalise so cosine similarity is a single dot product per query.
        # Widened to float32 first: the index is stored as float16, and
        # normalising in half precision would round away the differences
        # between scores that already sit close together.
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.maximum(norms, 1e-12)
        self._contents = contents
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._contents)

    def year_range(self) -> tuple[int, int] | None:
        """Earliest and latest year in the indexed metadata.

        Read from the index rather than hardcoded, so it stays honest when the
        corpus is resampled - a stated date range that quietly goes stale is
        worse than none.
        """
        years = []
        for metadata in self._metadata:
            stamp = str(metadata.get("date", ""))
            parts = stamp.split(".")
            if len(parts) >= 3:
                try:
                    years.append(int(parts[2].split()[0]))
                except (ValueError, IndexError):
                    continue
        return (min(years), max(years)) if years else None

    @classmethod
    def load(cls, path: Path | None = None) -> "VectorStore":
        """Load the prebuilt index from disk."""
        path = path or config.INDEX_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first: python ingest.py"
            )

        data = np.load(path, allow_pickle=False)

        if "corpus" not in data:
            # Indexes written before the text moved into one UTF-8 blob.
            # Kept readable so an old index still loads until it is rebuilt.
            return cls(
                vectors=data["vectors"],
                contents=[str(c) for c in data["contents"]],
                metadata=[json.loads(m) for m in data["metadata"]],
            )

        corpus = json.loads(data["corpus"].tobytes().decode("utf-8"))
        return cls(
            vectors=data["vectors"],
            contents=corpus["contents"],
            metadata=corpus["metadata"],
        )

    @staticmethod
    def save(
        vectors: list[list[float]],
        contents: list[str],
        metadata: list[dict],
        path: Path | None = None,
    ) -> Path:
        """Write an index to disk, creating parent directories as needed.

        Stored for size, since the file is committed and every rebuild stays
        in git history for good. Vectors are float16: on this corpus the top
        10 results for 50 sampled queries were identical to float32, at half
        the bytes - and zip compression barely touches float vectors, so
        precision is the only lever. Text is one UTF-8 JSON blob rather than
        numpy unicode arrays, which pad every string to the longest one at 4
        bytes a character and held 4 MB of text in 104 MB.
        """
        path = path or config.INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        corpus = json.dumps({"contents": contents, "metadata": metadata}, ensure_ascii=False)
        np.savez_compressed(
            path,
            vectors=np.asarray(vectors, dtype=np.float16),
            corpus=np.frombuffer(corpus.encode("utf-8"), dtype=np.uint8),
        )
        return path

    def search(
        self, query_vector: list[float], top_k: int = 5, threshold: float = 0.0
    ) -> list[Document]:
        """Return the top_k most similar documents above `threshold`."""
        query = np.asarray(query_vector, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)

        scores = self._vectors @ query

        # Partial sort: we only need the top_k, not a full ordering.
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Document(self._contents[i], self._metadata[i], float(scores[i]))
            for i in top
            if scores[i] >= threshold
        ]
