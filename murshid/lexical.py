"""Keyword search over the corpus, used when the embedding service is down.

Measured against vector search on the eval set, this is a clear step down -
it found a relevant passage for 24 of 28 questions against Voyage's 28, and
English questions suffer most (6 of 10), because the archive is mostly Arabic
and keywords cannot cross languages. It exists for one reason: with it, an
embedding outage degrades answers instead of replacing them with an error
page, and it needs no service, no model and no network to do that.

BM25 with light Arabic normalisation, over whole words plus 4-letter
fragments of them. Arabic builds words around a root with affixes, so افتح
and فتحت, or بنك and بنكي, are different words that share fragments. Matching
fragments too lifted the eval from 21 to 24 of 28 (MRR 0.55 to 0.69); 3-letter
fragments matched too loosely and gained nothing. Per-posting weights are
computed once at build time, so a query is a handful of array additions.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_DIACRITICS = re.compile(r"[ً-ْـ]")
_ALEF = re.compile("[أإآ]")
_WORD = re.compile(r"\w+")
# The definite article and its attached prepositions and conjunctions. Only
# stripped from longer words, so short roots are not cut down to nothing.
_PREFIX = re.compile(r"^(?:و|ف)?(?:بال|كال|لل|ال)")

_K1 = 1.5
_B = 0.75
_FRAGMENT = 4


def tokens(text: str) -> list[str]:
    """Normalise spelling variants students use interchangeably, then split.

    Hamza forms of alef, taa marbuta and alef maqsura are written both ways
    throughout the archive - a search for تأمين otherwise misses تامين.
    """
    text = _DIACRITICS.sub("", text.lower())
    text = _ALEF.sub("ا", text).replace("ة", "ه").replace("ى", "ي")

    out = []
    for word in _WORD.findall(text):
        if len(word) > 4:
            word = _PREFIX.sub("", word)
        if len(word) > 1:
            out.append(word)
    return out


def terms(text: str) -> list[str]:
    """What the index matches on: each word, and its fragments marked apart."""
    out = []
    for word in tokens(text):
        out.append(word)
        if len(word) > _FRAGMENT:
            out.extend(
                f"#{word[i:i + _FRAGMENT]}" for i in range(len(word) - _FRAGMENT + 1)
            )
    return out


class KeywordIndex:
    """BM25 over a fixed list of documents."""

    def __init__(self, contents: list[str]):
        counts = [Counter(terms(text)) for text in contents]
        lengths = np.array([sum(c.values()) for c in counts], dtype=np.float32)
        average = float(lengths.mean()) if len(lengths) else 1.0
        total = len(contents)

        postings: dict[str, tuple[list[int], list[float]]] = {}
        for position, count in enumerate(counts):
            norm = _K1 * (1 - _B + _B * lengths[position] / average)
            for token, frequency in count.items():
                ids, weights = postings.setdefault(token, ([], []))
                ids.append(position)
                weights.append(frequency * (_K1 + 1) / (frequency + norm))

        self._size = total
        self._postings = {
            token: (
                np.array(ids, dtype=np.int32),
                np.array(weights, dtype=np.float32)
                * math.log(1 + (total - len(ids) + 0.5) / (len(ids) + 0.5)),
            )
            for token, (ids, weights) in postings.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """(position, score) pairs for documents sharing words with `query`."""
        scores = np.zeros(self._size, dtype=np.float32)
        for token in set(terms(query)):
            if token in self._postings:
                ids, weights = self._postings[token]
                np.add.at(scores, ids, weights)

        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
