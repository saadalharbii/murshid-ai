"""Offline retrieval evaluation.

Caveat on the metric: relevance is approximated by keyword overlap, which is
cheap to maintain but blunt. The corpus is one community discussing one broad
subject, so recall saturates easily and the numbers understate reranking - its
clearest effect is pulling genuinely better passages up from deep in the
candidate list, which keyword matching cannot see. Treat `spread` (how well
the top result separates from the fifth) as the more informative signal, and
read the retrieved text when a change looks surprising.

Compares vector-only retrieval against vector+rerank on a fixed question set.
Query embeddings are cached to disk, so re-runs after a code change cost no
embedding calls - important under Voyage's free-tier rate limit. No Claude
calls are made, so running this costs nothing against the Anthropic budget.

    python eval/run_eval.py
    python eval/run_eval.py --no-rerank    # baseline only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from murshid import config  # noqa: E402
from murshid.embeddings import embed_query  # noqa: E402
from murshid.rerank import RerankError, rerank  # noqa: E402
from murshid.store import VectorStore  # noqa: E402

# Above this length a chunk holds several messages, so incidental keyword
# matches become likely and the relevance bar rises.
_LONG_CHUNK = 500

QUESTIONS = Path(__file__).parent / "questions.json"
CACHE = Path(__file__).parent / ".embedding_cache.pkl"


def load_cache() -> dict[str, list[float]]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    return {}


def cached_embedding(question: str, cache: dict) -> list[float]:
    """Embed a question, reusing a cached vector when the model has not changed."""
    key = f"{config.VOYAGE_MODEL}:{question}"
    if key not in cache:
        cache[key] = embed_query(question)
        CACHE.write_bytes(pickle.dumps(cache))
    return cache[key]


def is_relevant(content: str, expect: list[str]) -> bool:
    """A chunk counts as relevant when it matches enough expected terms.

    The bar scales with chunk length, because a fixed one does not survive a
    chunking change. Under the old 500-character blend, two terms were needed:
    a single common word like "سكن" appeared almost everywhere, so one term
    marked nearly every chunk relevant. Conversation-aware chunking cut the
    mean chunk to ~326 characters, and the same two-term rule then scored
    correct answers as misses - the chunk "tuition is 10-16 thousand pounds a
    year and you need about 14 thousand for living" was judged irrelevant to a
    question about London costs because it contained one keyword rather than
    two. Requiring one term from a short, focused chunk and two from a long,
    mixed one measures retrieval instead of chunk size.
    """
    matches = sum(1 for term in expect if term in content)
    return matches >= (2 if len(content) > _LONG_CHUNK else 1)


def evaluate(use_rerank: bool) -> dict:
    store = VectorStore.load()
    questions = json.loads(QUESTIONS.read_text())["questions"]
    cache = load_cache()

    hits_at_5 = 0
    reciprocal_ranks = []
    spreads = []
    misses = []

    for item in questions:
        vector = np.asarray(cached_embedding(item["q"], cache), dtype=np.float32)
        candidates = store.search(
            vector.tolist(),
            top_k=config.RETRIEVE_CANDIDATES if use_rerank else config.TOP_K_RESULTS,
            threshold=0.0,
        )

        if use_rerank and candidates:
            try:
                ranked = rerank(
                    item["q"], [d.content for d in candidates], top_n=config.TOP_K_RESULTS
                )
                candidates = [candidates[i] for i, _ in ranked]
                scores = [s for _, s in ranked]
            except RerankError as exc:
                print(f"  ! rerank unavailable ({exc}); using vector order", file=sys.stderr)
                candidates = candidates[: config.TOP_K_RESULTS]
                scores = [d.score for d in candidates]
        else:
            scores = [d.score for d in candidates]

        top = candidates[: config.TOP_K_RESULTS]
        if scores:
            spreads.append(scores[0] - scores[min(len(scores), config.TOP_K_RESULTS) - 1])

        rank = next(
            (i for i, d in enumerate(top, 1) if is_relevant(d.content, item["expect"])), None
        )
        if rank:
            hits_at_5 += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item["q"])

    n = len(questions)
    return {
        "questions": n,
        "recall@5": hits_at_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "mean_spread": sum(spreads) / len(spreads) if spreads else 0.0,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rerank", action="store_true", help="evaluate vector search alone")
    args = parser.parse_args()

    if args.no_rerank:
        results = {"vector only": evaluate(use_rerank=False)}
    else:
        results = {
            "vector only": evaluate(use_rerank=False),
            "vector + rerank": evaluate(use_rerank=True),
        }

    print(f"\n{'strategy':<18}{'recall@5':>10}{'MRR':>8}{'spread':>9}")
    print("-" * 45)
    for name, r in results.items():
        print(f"{name:<18}{r['recall@5']:>10.2f}{r['mrr']:>8.2f}{r['mean_spread']:>9.3f}")
    print(f"\n{results['vector only']['questions']} questions | model {config.VOYAGE_MODEL}")

    # Name the misses. An aggregate that drops tells you something broke; the
    # list tells you what, and whether the question or the retrieval is at
    # fault - which is usually the question.
    for name, r in results.items():
        if r["misses"]:
            print(f"\nmissed by {name}:")
            for question in r["misses"]:
                print(f"  - {question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
