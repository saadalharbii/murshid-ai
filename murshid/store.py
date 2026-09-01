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
    """A retrieved chunk and its similarity to the query."""

    __slots__ = ("content", "metadata", "score")

    def __init__(self, content: str, metadata: dict[str, Any], score: float):
        self.content = content
        self.metadata = metadata
        self.score = score


class VectorStore:
    """Cosine-similarity search over an in-memory matrix of embeddings."""

    def __init__(self, vectors: np.ndarray, contents: list[str], metadata: list[dict]):
        if len(vectors) != len(contents) or len(contents) != len(metadata):
            raise ValueError("vectors, contents and metadata must be the same length")

        # Pre-normalise so cosine similarity is a single dot product per query.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.maximum(norms, 1e-12)
        self._contents = contents
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._contents)

    @classmethod
    def load(cls, path: Path | None = None) -> "VectorStore":
        """Load the prebuilt index from disk."""
        path = path or config.INDEX_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first: python ingest.py"
            )

        data = np.load(path, allow_pickle=False)
        return cls(
            vectors=data["vectors"],
            contents=[str(c) for c in data["contents"]],
            metadata=[json.loads(m) for m in data["metadata"]],
        )

    @staticmethod
    def save(
        vectors: list[list[float]],
        contents: list[str],
        metadata: list[dict],
        path: Path | None = None,
    ) -> Path:
        """Write an index to disk, creating parent directories as needed."""
        path = path or config.INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            vectors=np.asarray(vectors, dtype=np.float32),
            contents=np.asarray(contents, dtype=object).astype("U"),
            metadata=np.asarray([json.dumps(m, ensure_ascii=False) for m in metadata]),
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
