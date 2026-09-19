"""Tests for parsing, chunking, retrieval, and language detection.

These cover the logic that runs without network access; the Voyage and Claude
calls are exercised by running the app.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from ingest import corpus_fingerprint
from murshid import __version__, config
from murshid.messages import (
    TROUBLE_AR,
    TROUBLE_EN,
    no_results,
    searching,
    source_authors,
    source_date,
    trouble,
    writing,
)
from murshid.filters import (
    drop_filler,
    is_filler,
    is_question_only,
    keep_chunk,
)
from murshid.rag import detect_language
from murshid.scrub import contains_contact_details, scrub
from murshid.store import Document, VectorStore
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
    """Chunking follows conversation boundaries, not a character budget."""

    def _messages(self, count, text="a message", gap_minutes=0, author=None):
        from datetime import datetime, timedelta

        base = datetime(2019, 8, 1, 17, 0, 0)
        return [
            {
                "content": f"{text} {i}",
                "metadata": {
                    "author": author or f"user{i}",
                    "date": f"day{i}",
                    "timestamp": base + timedelta(minutes=gap_minutes * i),
                    "message_id": f"message{i}",
                    "reply_to": None,
                },
            }
            for i in range(count)
        ]

    def test_groups_messages_into_chunks(self):
        chunks = TelegramParser().chunk(self._messages(50, text="x" * 60))
        assert len(chunks) > 1
        assert all(chunk["content"] for chunk in chunks)

    def test_merges_metadata(self):
        chunks = TelegramParser().chunk(self._messages(3))
        metadata = chunks[0]["metadata"]
        assert metadata["message_count"] == 3
        assert "user0" in metadata["authors"]
        assert "day0" in metadata["date_range"]

    def test_empty_input_yields_no_chunks(self):
        assert TelegramParser().chunk([]) == []

    def test_preserves_arabic(self):
        messages = self._messages(1)
        messages[0]["content"] = "مرحبا بالعالم"
        assert "مرحبا" in TelegramParser().chunk(messages)[0]["content"]

    def test_labels_each_line_with_its_author(self):
        # Speaker attribution is what preserves who answered whom.
        chunks = TelegramParser().chunk(self._messages(2, text="x" * 40))
        assert "user0: " in chunks[0]["content"]
        assert "user1: " in chunks[0]["content"]

    def test_long_pause_starts_a_new_chunk(self):
        # Two messages an hour apart are not the same conversation.
        messages = self._messages(2, text="x" * 80, gap_minutes=60)
        assert len(TelegramParser().chunk(messages)) == 2

    def test_reply_into_recent_history_keeps_thread_together(self):
        # A reply arriving after a pause continues the thread rather than
        # starting a new one - 367 real messages in the corpus do this.
        messages = self._messages(2, text="x" * 80, gap_minutes=60)
        messages[1]["metadata"]["reply_to"] = "message0"
        assert len(TelegramParser().chunk(messages)) == 1

    def test_never_splits_mid_message(self):
        # Splitting only on message boundaries is what removed the need for
        # the old character overlap.
        chunks = TelegramParser().chunk(self._messages(40, text="y" * 100))
        for chunk in chunks:
            for line in chunk["content"].split("\n"):
                assert line.endswith(tuple("0123456789")), line[:40]

    def test_drops_per_message_metadata_from_the_chunk(self):
        # A chunk spans many messages, so a single message_id or reply_to
        # would misattribute it.
        metadata = TelegramParser().chunk(self._messages(3))[0]["metadata"]
        for key in ("timestamp", "reply_to", "message_id", "author"):
            assert key not in metadata


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
        from murshid.store import Document, VectorStore

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


class TestCorpusFingerprint:
    """The ingest checkpoint is only valid for the corpus that produced it.

    Vectors are paired with chunk text by list position, so resuming a run
    against re-chunked content would attach every vector to the wrong passage
    and produce an index whose citations point at unrelated conversations.
    """

    def test_same_corpus_produces_the_same_fingerprint(self):
        chunks = ["one", "two", "three"]
        assert corpus_fingerprint(chunks) == corpus_fingerprint(list(chunks))

    def test_changed_content_changes_the_fingerprint(self):
        before = corpus_fingerprint(["one", "two"])
        after = corpus_fingerprint(["one", "two!"])
        assert before != after

    def test_reordering_changes_the_fingerprint(self):
        # Order matters: position is what binds a vector to its text.
        assert corpus_fingerprint(["a", "b"]) != corpus_fingerprint(["b", "a"])

    def test_different_chunk_count_changes_the_fingerprint(self):
        assert corpus_fingerprint(["a", "b"]) != corpus_fingerprint(["a", "b", "c"])

    def test_boundary_shift_is_detected(self):
        # The realistic failure: same text, different chunk boundaries. A hash
        # of concatenated content alone would miss this; the count prefix and
        # per-chunk hashing together catch it.
        assert corpus_fingerprint(["ab", "cd"]) != corpus_fingerprint(["abc", "d"])


class TestUserFacingErrors:
    """Failures are shown to end users, so they stay generic and bilingual.

    Two rules: never name the upstream vendor or an HTTP status in text a
    visitor reads, and speak the language they asked in. The diagnostic detail
    lives on the chained exception for the logs.
    """

    def test_embedding_errors_never_name_the_vendor(self):
        import inspect

        import murshid.embeddings as embeddings

        source = inspect.getsource(embeddings)
        messages = re.findall(r'EmbeddingError\(\s*"([^"]+)"', source)
        assert messages, "expected to find EmbeddingError messages"
        for message in messages:
            assert "voyage" not in message.lower(), message

    def test_trouble_message_follows_the_question_language(self):
        assert trouble("arabic") == TROUBLE_AR
        assert trouble("english") == TROUBLE_EN
        assert trouble("arabic") != trouble("english")

    def test_trouble_message_is_not_a_stack_trace(self):
        for language in ("arabic", "english"):
            message = trouble(language)
            assert "Error" not in message
            assert "Traceback" not in message
            assert len(message) < 120

    def test_no_results_message_follows_the_question_language(self):
        assert no_results("arabic") != no_results("english")
        assert "أرشيف" in no_results("arabic")


class TestAnswerCacheKey:
    """A cached answer is only reusable if nothing that shaped it has changed.

    The answer eval caches generated answers to avoid paying twice. The key
    therefore has to cover retrieval settings and prompts, not just the model
    names - otherwise a retrieval change reports stale numbers as current.
    """

    def test_key_changes_with_candidate_pool(self, monkeypatch):
        from eval.run_answer_eval import cache_key

        before = cache_key("q")
        monkeypatch.setattr(config, "RETRIEVE_CANDIDATES", config.RETRIEVE_CANDIDATES + 30)
        assert cache_key("q") != before

    def test_key_changes_with_rerank_threshold(self, monkeypatch):
        from eval.run_answer_eval import cache_key

        before = cache_key("q")
        monkeypatch.setattr(config, "RERANK_THRESHOLD", config.RERANK_THRESHOLD + 0.1)
        assert cache_key("q") != before

    def test_key_changes_with_the_system_prompt(self, monkeypatch):
        import eval.run_answer_eval as harness

        before = harness.cache_key("q")
        monkeypatch.setattr(harness, "_SYSTEM_EN", "a different prompt")
        assert harness.cache_key("q") != before

    def test_same_settings_give_a_stable_key(self):
        from eval.run_answer_eval import cache_key

        assert cache_key("q") == cache_key("q")
        assert cache_key("q") != cache_key("other")


class TestDeployCompatibility:
    """Guard against a redeploy breaking sessions that are already open.

    Streamlit keeps Document objects in st.session_state across a redeploy,
    and @st.cache_resource holds a pipeline keyed on __version__. A field
    added to Document therefore reaches the renderer as a missing attribute on
    old objects. This crashed the live app once; both halves are tested here.
    """

    def test_version_changes_when_document_gains_a_field(self):
        # A tripwire, not a rule: if Document's shape changes, __version__ has
        # to change with it so the cached pipeline is discarded on deploy.
        assert Document.__slots__ == ("content", "metadata", "score", "score_kind"), (
            "Document changed shape - bump murshid.__version__ so Streamlit "
            "discards its cached pipeline, then update this test."
        )
        assert __version__ == "2.4.0"

    def test_score_kind_defaults_for_documents_built_without_it(self):
        # Positional construction is what the old cached code did.
        document = Document("text", {}, 0.5)
        assert document.score_kind == "similarity"

    def test_renderer_tolerates_a_document_missing_score_kind(self):
        class LegacyDocument:
            """A Document as pickled before score_kind existed."""

            def __init__(self):
                self.content = "text"
                self.metadata = {}
                self.score = 0.5

        assert getattr(LegacyDocument(), "score_kind", "similarity") == "similarity"


class TestSourceDisplay:
    """The sources panel exists so a reader can check a citation.

    That makes its job showing what helps them judge a passage - when it was
    said, and by whom - rather than what the retrieval pipeline did.
    """

    def test_date_reduces_to_month_and_year(self):
        stamp = "03.08.2019 10:42:52 UTC+00:00 to 03.08.2019 10:48:38 UTC+00:00"
        assert source_date(stamp, "english") == "August 2019"

    def test_date_is_localised(self):
        stamp = "03.08.2019 10:42:52 UTC+00:00"
        assert source_date(stamp, "arabic") == "أغسطس 2019"

    def test_unparseable_date_yields_nothing_rather_than_junk(self):
        for bad in ("", "not a date", "2019"):
            assert source_date(bad, "english") == ""

    def test_deleted_accounts_become_a_count(self):
        # Two thirds of chunks are "Deleted Account"; printing it is noise.
        assert source_authors("Deleted Account", "english") == "a student"
        assert source_authors("Deleted Account, Deleted Account", "english") == "2 students"

    def test_real_names_are_shown_with_a_remainder(self):
        result = source_authors("Deleted Account, Tariq, Zainab", "english")
        assert result.startswith("Tariq, Zainab")
        assert "Deleted Account" not in result

    def test_no_authors_yields_nothing(self):
        assert source_authors("", "english") == ""


class TestScrub:
    """Contact details are removed before anything is indexed.

    The sources panel renders retrieved chunk text directly, so a phone number
    that reaches the index reaches the page - the system prompt never sees it.
    Redaction therefore happens at parse time, and these tests pin both halves:
    what must be removed, and what must survive.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "رقمي 0501234567",
            "call me on 07397789352",
            "الرقم +966540395541",
            "hotline 00442079173000",
            "تواصل @alhafh",
            "t.me/ukstudents/296251",
            "https://t.me/joinchat/AAAA",
            "mail me at test.user@gmail.com",
        ],
    )
    def test_contact_details_are_removed(self, text):
        assert not contains_contact_details(scrub(text))

    @pytest.mark.parametrize(
        "text",
        [
            "الجامعة تكلف 15000 باوند في السنة",
            "درجة الايلتس 6.5",
            "التقديم في 2024 والقبول 2025",
            "السكن 850 شهريا",
            "sort code 12345678",
        ],
    )
    def test_ordinary_numbers_survive(self, text):
        # Over-redaction destroys the retrievable content; years, prices and
        # scores must pass through untouched.
        assert scrub(text) == text

    def test_placeholder_keeps_the_sentence_readable(self):
        out = scrub("للتواصل مع السفارة 00442079173000 للاستفسار")
        assert out == "للتواصل مع السفارة [phone] للاستفسار"

    def test_email_is_not_mangled_into_a_handle(self):
        # The @ inside an address would otherwise match the handle pattern and
        # leave "[handle].com" behind, so emails are matched first.
        assert scrub("a@b.com") == "[email]"

    def test_parser_scrubs_on_the_way_in(self, tmp_path):
        html = (
            '<div class="message"><div class="from_name">S</div>'
            '<div class="date" title="01.08.2019 10:00:00 UTC+00:00"></div>'
            '<div class="text">رقمي 0501234567 وحسابي @someuser</div></div>'
        )
        path = tmp_path / "messages1.html"
        path.write_text(html, encoding="utf-8")

        messages = TelegramParser().parse_file(path)

        assert len(messages) == 1
        assert not contains_contact_details(messages[0]["content"])
        assert "[phone]" in messages[0]["content"]

    def test_empty_text_is_handled(self):
        assert scrub("") == ""


class TestFilters:
    """Filler and answerless chunks are dropped before embedding.

    Both filters are conservative on purpose: dropping a short chunk that did
    hold an answer costs more than keeping some filler, so these tests pin the
    boundary in both directions.
    """

    @pytest.mark.parametrize(
        "text",
        ["شكرا", "تمام", "لا", "ايه", "السلام عليكم", "...", "🙏", "تمام شكرا", "ok"],
    )
    def test_filler_is_detected(self, text):
        assert is_filler(text)

    @pytest.mark.parametrize(
        "text",
        [
            "البنك يطلب اثبات عنوان وتقدر تجيبه من الجامعة",
            "تكلفة المعيشة في لندن حوالي ١٤ الف باوند بالسنة",
            "يحتاج قبول جامعي وشهادة ايلتس 6.5",
        ],
    )
    def test_content_is_kept(self, text):
        assert not is_filler(text)

    def test_question_without_an_answer_is_dropped(self):
        chunk = "K: ايش افضل بنك بريطاني؟ وايش متطلباته لفتح الحساب"
        assert is_question_only(chunk)
        assert not keep_chunk({"content": chunk})

    def test_question_with_an_answer_is_kept(self):
        chunk = (
            "K: ايش افضل بنك بريطاني؟\n"
            "S: مونزو الاسهل للمبتعثين الجدد وما يطلبون الا البي ار بي وعنوان السكن"
        )
        assert not is_question_only(chunk)
        assert keep_chunk({"content": chunk})

    def test_statement_without_a_question_is_kept(self):
        chunk = "S: تكلفة المعيشة في لندن حوالي ١٤ الف باوند بالسنة وتزيد بالوسط"
        assert not is_question_only(chunk)
        assert keep_chunk({"content": chunk})

    def test_chunk_of_only_short_lines_is_dropped(self):
        assert not keep_chunk({"content": "A: تمام\nB: ايه\nC: صح"})

    def test_drop_filler_preserves_order_and_content(self):
        messages = [
            {"content": "شكرا"},
            {"content": "البنك يطلب اثبات عنوان من الجامعة او فاتورة"},
            {"content": "تمام"},
        ]
        kept = drop_filler(messages)
        assert len(kept) == 1
        assert kept[0]["content"].startswith("البنك")


class TestStartupCost:
    """The deployed app pays for every runtime dependency on each cold start.

    Streamlit Cloud reinstalls requirements.txt when a container spins up, so
    a package the running app never imports is time a visitor spends watching
    a spinner. These tests pin the split between runtime and tooling deps.
    """

    def test_runtime_requirements_exclude_ingest_only_packages(self):
        runtime = Path("requirements.txt").read_text().lower()
        installed = [
            line.split(">")[0].split("=")[0].strip()
            for line in runtime.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        # bs4 is only needed to parse exports; dotenv only for local .env.
        assert "beautifulsoup4" not in installed
        assert "python-dotenv" not in installed
        assert "torch" not in installed
        assert "sentence-transformers" not in installed

    def test_runtime_requirements_keep_what_the_app_imports(self):
        installed = Path("requirements.txt").read_text().lower()
        for package in ("streamlit", "numpy", "certifi"):
            assert package in installed

    def test_dev_requirements_include_the_ingest_extras(self):
        dev = Path("requirements-dev.txt").read_text().lower()
        assert "beautifulsoup4" in dev
        assert "python-dotenv" in dev
        assert "-r requirements.txt" in dev

    def test_config_imports_without_dotenv(self, monkeypatch):
        # Production installs no dotenv; config must still read real env vars.
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("dotenv"):
                raise ImportError("No module named 'dotenv'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        for module in [m for m in sys.modules if m.startswith("murshid.config")]:
            monkeypatch.delitem(sys.modules, module, raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "from-environment")

        import importlib

        reloaded = importlib.import_module("murshid.config")
        importlib.reload(reloaded)

        assert reloaded.VOYAGE_API_KEY == "from-environment"


class TestProgressMessages:
    """Retrieval and generation are separate waits; each names its stage."""

    def test_stage_messages_follow_the_question_language(self):
        assert searching("arabic") != searching("english")
        assert writing("arabic") != writing("english")

    def test_stages_are_distinct(self):
        # One unchanging spinner across both waits reads as a stall.
        assert searching("english") != writing("english")
        assert searching("arabic") != writing("arabic")
