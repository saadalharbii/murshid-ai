"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Secrets
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Models
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-2.5-lite")
EMBEDDING_DIM = 1024

# Retrieval. Vector search casts a wide net and the reranker decides what is
# actually relevant, so the vector threshold stays low on purpose: cosine
# scores here sit close together (chunks average 0.42 similarity to each
# other), which makes them a poor relevance filter. The rerank score is
# absolute and comparable across queries, so it carries the cutoff instead.
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "40"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.40"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Paths
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.npz"
