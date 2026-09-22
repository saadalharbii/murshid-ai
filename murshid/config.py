"""Configuration loaded from environment variables."""

import os
from pathlib import Path

# python-dotenv is a local convenience only. On Streamlit Cloud the secrets
# arrive as real environment variables, so the package need not be installed
# there - and every package in requirements.txt is installed on each cold
# start, which is time the user spends watching a spinner.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Secrets
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Models
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4-large")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-2.5-lite")

# Retrieval. Vector search supplies the candidate pool and the reranker orders
# it. The vector threshold stays low on purpose: cosine scores here sit close
# together (chunks average 0.42 similarity to each other), which makes them a
# poor relevance filter. The rerank score is absolute and comparable across
# queries, so it carries the cutoff instead.
#
# The pool is deliberately narrow. Measured on the 28-question eval, widening
# it monotonically hurts: pool 5 and 10 score MRR 0.86/0.85, pool 40 scores
# 0.79 and drops recall@5 to 0.96. The reranker over-promotes chunks that
# restate the question - for "how much does London cost" it demoted a chunk
# giving actual figures to rank 14 and promoted one that merely asks about
# prices - and a wider net gives it more of those to find. Keeping the pool
# near the number of results retains the reranker's ordering benefit without
# handing it 35 chances to find a plausible-looking non-answer.
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "10"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
# Measured on the 36-question eval, the threshold is a weak filter, not a
# safety net: answerable questions score 0.51-0.84 at rank 1 and unanswerable
# ones 0.37-0.66, so the distributions overlap and no cutoff separates them.
# 0.50 is the best available point - it drops nothing answerable while
# blocking 3 of 8 unanswerable questions. Raising it further starts costing
# real questions (0.55 loses two) for little gain. What actually makes the
# system decline is the system prompt instructing Claude to say when the
# excerpts do not cover a question; this threshold only trims the worst
# matches before they reach it.
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.50"))
# How many times a live question tries each search call. Kept low because
# someone is watching a spinner: with the ingest-grade policy of six attempts
# and waits growing by 21 seconds, a rate-limited question sat for over five
# minutes before showing an error. Offline tools with no one waiting (the
# evals) raise both rather than let a rate limit skew their numbers.
QUERY_ATTEMPTS = int(os.getenv("QUERY_ATTEMPTS", "2"))
QUERY_RATE_LIMIT_DELAY = float(os.getenv("QUERY_RATE_LIMIT_DELAY", "3.0"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))

# Paths
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.npz"
