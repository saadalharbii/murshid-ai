"""Tests for parsing, chunking, retrieval, and language detection.

These cover the logic that runs without network access; the Voyage and Claude
calls are exercised by running the app.
"""

from __future__ import annotations

import numpy as np
import pytest

from murshid.rag import detect_language
from murshid.store import VectorStore
from murshid.telegram import TelegramParser


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ما هي أفضل الجامعات؟", "arabic"),
            ("How do I renew my visa?", "english"),
            ("تجديد التأشيرة visa renewal", "arabic"),
            ("Mostly english with كلمة", "english"),
            ("", "english"),
            ("12345 ???", "english"),
        ],
    )
    def test_detects_script(self, text, expected):
        assert detect_language(text) == expected


class TestChunking:
    def _messages(self, count: int, text: str = "a message"):
        return [
            {"content": f"{text} {i}", "metadata": {"author": f"user{i}", "date": f"day{i}"}}
            for i in range(count)
        ]

    def test_groups_messages_into_chunks(self):
        chunks = TelegramParser().chunk(self._messages(50), size=100, overlap=10)
        assert len(chunks) > 1
        assert all(chunk["content"] for chunk in chunks)

    def test_merges_metadata(self):
        chunks = TelegramParser().chunk(self._messages(3), size=10_000, overlap=0)
        metadata = chunks[0]["metadata"]
        assert metadata["message_count"] == 3
        assert "user0" in metadata["authors"]
        assert "day0" in metadata["date_range"]

    def test_empty_input_yields_no_chunks(self):
        assert TelegramParser().chunk([]) == []

    def test_preserves_arabic(self):
        messages = [{"content": "مرحبا بالعالم", "metadata": {"author": "a", "date": "d"}}]
        assert "مرحبا" in TelegramParser().chunk(messages)[0]["content"]


class TestVectorStore:
    @pytest.fixture
    def store(self):
        return VectorStore(
            vectors=np.array([[1.0, 0, 0], [0, 1.0, 0], [0.9, 0.1, 0]], dtype=np.float32),
            contents=["east", "north", "east-ish"],
            metadata=[{"n": 0}, {"n": 1}, {"n": 2}],
        )

    def test_finds_nearest(self, store):
        assert store.search([1.0, 0, 0], top_k=1)[0].content == "east"

    def test_orders_by_similarity(self, store):
        scores = [d.score for d in store.search([1.0, 0, 0], top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_excludes_weak_matches(self, store):
        # "north" is orthogonal to the query; the two eastward vectors are not.
        assert len(store.search([1.0, 0, 0], top_k=3, threshold=0.5)) == 2
        assert len(store.search([1.0, 0, 0], top_k=3, threshold=0.999)) == 1

    def test_top_k_larger_than_corpus(self, store):
        assert len(store.search([1.0, 0, 0], top_k=99)) == 3

    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "index.npz"
        VectorStore.save([[1.0, 0.0], [0.0, 1.0]], ["اختبار", "test"],
                         [{"a": 1}, {"a": 2}], path)
        loaded = VectorStore.load(path)
        assert len(loaded) == 2
        assert loaded.search([1.0, 0.0], top_k=1)[0].content == "اختبار"
        assert loaded.search([0.0, 1.0], top_k=1)[0].metadata == {"a": 2}

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            VectorStore(np.array([[1.0, 0.0]]), ["a", "b"], [{}])

    def test_missing_index_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ingest"):
            VectorStore.load(tmp_path / "absent.npz")


class TestRerankFallback:
    """Reranking is an enhancement, never a dependency: if it fails the
    pipeline must still answer using the vector ordering."""

    def _pipeline(self, monkeypatch, rerank_impl):
        import numpy as np

        from murshid import rag
        from murshid.store import VectorStore

        store = VectorStore(
            vectors=np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
            contents=["first", "second", "third"],
            metadata=[{}, {}, {}],
        )
        monkeypatch.setattr(rag, "embed_query", lambda q: [1.0, 0.0])
        monkeypatch.setattr(rag, "rerank", rerank_impl)
        return rag.RAGPipeline(store=store)

    def test_falls_back_to_vector_order_when_rerank_fails(self, monkeypatch):
        from murshid.rag import RerankError

        def failing(*args, **kwargs):
            raise RerankError("service down")

        pipeline = self._pipeline(monkeypatch, failing)
        _, sources, error = pipeline.retrieve("anything")

        assert error is None
        assert sources, "must still return results when reranking is unavailable"
        assert sources[0].content == "first"

    def test_rerank_reorders_results(self, monkeypatch):
        # Rerank indices address the CANDIDATE list, which vector search has
        # already sorted (first, third, second) - not the store's own order.
        # Promoting candidate 1 over candidate 0 must invert that ordering.
        pipeline = self._pipeline(monkeypatch, lambda *a, **k: [(1, 0.9), (0, 0.8)])
        _, sources, _ = pipeline.retrieve("anything")

        assert [s.content for s in sources] == ["third", "first"]
        assert sources[0].score == 0.9, "rerank score should replace the cosine score"

    def test_drops_results_below_threshold(self, monkeypatch):
        from murshid import config

        pipeline = self._pipeline(
            monkeypatch, lambda *a, **k: [(0, 0.9), (1, config.RERANK_THRESHOLD - 0.1)]
        )
        _, sources, _ = pipeline.retrieve("anything")

        assert len(sources) == 1, "weak matches should be filtered out"


class TestEmbeddingErrorHandling:
    def test_embedding_failure_is_reported_not_raised(self, monkeypatch):
        from murshid import rag
        from murshid.embeddings import EmbeddingError

        def failing(_):
            raise EmbeddingError("no key")

        monkeypatch.setattr(rag, "embed_query", failing)
        pipeline = rag.RAGPipeline(store=object())
        _, sources, error = pipeline.retrieve("anything")

        assert sources == []
        assert "no key" in error


class TestRefusalDetection:
    """The answer eval decides whether Claude declined a question. Getting this
    wrong silently corrupts the metric, and it did during development: quoted
    student chatter was scored as the assistant refusing."""

    def _is_refusal(self, text):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
        from run_answer_eval import is_refusal

        return is_refusal(text)

    def test_empty_answer_is_a_refusal(self):
        assert self._is_refusal("   ")

    def test_opening_decline_is_a_refusal(self):
        assert self._is_refusal("The excerpts do not contain information about Tokyo.")
        assert self._is_refusal("لا تحتوي المقتطفات على معلومات عن أستراليا.")

    def test_quoted_chatter_is_not_a_refusal(self):
        # A real answer that answers the question, then later quotes a student
        # saying nothing had happened yet. The quote sits outside the opening
        # window, so it must not read as the assistant declining.
        answer = (
            "بناءً على النقاشات، المكافأة تُحسب بالتقويم الهجري وهي متأخرة شهر، "
            "فبعض الطلاب ذكروا أنهم ينتظرون تقريباً شهرين حتى تنزل كاملة [4]. "
            "وأحد المشاركين ذكر أن المكافأة تنزل يوم ٢٧ من كل شهر [2]، "
            "بينما قال آخر والله لين الحين ما صار شي ولا يوجد رد [1]"
        )
        assert not self._is_refusal(answer)

    def test_refusal_reaching_its_point_late_is_caught(self):
        # A real Arabic refusal opened with an apology and only named the gap
        # in the following sentence; searching just the first sentence missed it.
        answer = (
            "ما أقدر أساعدك بهذا السؤال 🙏\n\n"
            "المقتطفات المتوفرة لدي تخص تجارب طلاب سعوديين في بريطانيا، "
            "ولا تحتوي على أي معلومات عن أستراليا."
        )
        assert self._is_refusal(answer)

    def test_refusal_that_cites_is_still_a_refusal(self):
        # A good refusal often cites the excerpts to show what they DO cover,
        # so the presence of citations must not rule a refusal out.
        answer = (
            "I don't have any information about the University of Tokyo. "
            "The discussions only cover UK topics [1][2][3]."
        )
        assert self._is_refusal(answer)
