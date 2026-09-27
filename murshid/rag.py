"""Retrieval-augmented question answering over the Telegram corpus."""

from __future__ import annotations

import re
import sys
import threading

from . import config
from .claude import ClaudeError, complete, stream
from .embeddings import EmbeddingError, embed_query
from .lexical import KeywordIndex
from .rerank import RerankError, rerank
from .store import Document, VectorStore

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[a-zA-Z]")

# The instruction lists are deliberately unnumbered. While numbered, Claude
# emitted "[7]" - the number of the "this is student chatter, not official
# guidance" rule - as if it were a citation, producing references to an
# excerpt that did not exist.
_SYSTEM_AR = """أنت "مرشد"، مساعد ذكي يساعد الطلاب السعوديين المبتعثين في بريطانيا.
تجيب على الأسئلة اعتماداً على نقاشات حقيقية من مجموعات الطلاب على تيليجرام.

تعليمات:
- اعتمد على المقتطفات المقدمة فقط، ولا تخترع معلومات
- أشر إلى المصدر بعد كل معلومة باستخدام رقمه، مثل [1] أو [2]، ولا تستخدم إلا الأرقام المعطاة فعلاً
   ولا تضع رقم مصدر بعد تنبيهاتك أو ملاحظاتك الخاصة، فهي ليست من المقتطفات
- إذا اختلف الطلاب في نقطة، اذكر الآراء المختلفة بدل اختيار واحد فقط
- إذا كان المقتطف قديماً، نبّه أن الأنظمة قد تكون تغيّرت
- إذا لم تجد إجابة في المقتطفات، قل ذلك بوضوح ولا تخمّن
- لا تذكر أسماء أشخاص أو أرقام هواتف وردت في النقاشات
- المصدر نقاشات طلاب وليس جهة رسمية، فنبّه المستخدم عند الأسئلة الرسمية
- أجب بالعربية، وكن مختصراً وعملياً"""

_SYSTEM_EN = """You are "Murshid", an assistant for Saudi scholarship students in the UK.
You answer using real discussions from student Telegram groups.

Instructions:
- Rely only on the provided excerpts; never invent details
- Cite the excerpt number after each claim, e.g. [1] or [2]; use only numbers you were given
   Never attach a citation to your own caveats or disclaimers - they are not from the excerpts
- Where students disagree, present the differing views rather than picking one
- If an excerpt is old, note that rules may have changed since
- If the excerpts do not answer the question, say so plainly and do not guess
- Never repeat personal names or phone numbers that appear in the discussions
- The source is student chatter, not an official body - flag this on official matters
- Answer in English, and keep it concise and practical"""


_DEGRADED_AR = (
    "ملاحظة: البحث يعمل الآن بوضع احتياطي محدود وقد تفوته نقاشات ذات صلة. "
    "إذا لم تجب المقتطفات عن السؤال فقل إن البحث محدود حالياً واقترح المحاولة "
    "بعد قليل، ولا تقل إن الأرشيف لا يغطي الموضوع.\n\n"
)
_DEGRADED_EN = (
    "Note: search is running in a limited backup mode and may miss relevant "
    "discussions. If the excerpts do not answer the question, say search is "
    "limited right now and suggest trying again shortly - do not say the "
    "archive lacks the topic.\n\n"
)


_REWRITE_SYSTEM = """You turn the latest message in a chat into a standalone search query for an archive of Saudi student discussions about studying in the UK.

- If the latest message refers back to the chat - "what about Manchester?", "which one is cheapest?", "do they need a letter?", "is that per month?" - rewrite it as a complete question, replacing the reference with what it points to. Words like it, they, that, those, which one, there, هذا, هذي, ذلك, فيها, لها are references
- Otherwise copy it back word for word, even if it is short or vague. Do not add places, details or clarifications to a question that does not refer back
- Keep the language of the latest message
- Reply with the question only: no explanation, no quotation marks"""

# Enough history to resolve "what about X?" and "and the second one?". Older
# turns rarely matter to the latest question and make the rewrite slower.
_HISTORY_TURNS = 2
_HISTORY_CHARS = 500
_CITATION = re.compile(r"\s*\[\d+\]")
_REWRITE_MAX_CHARS = 300


def past_exchanges(messages: list[dict]) -> list[tuple[str, str]]:
    """(question, answer) pairs from the chat log, oldest first.

    A question whose answer failed has no reply after it and is skipped - the
    reader saw an error, so there is nothing for a follow-up to refer to.
    """
    return [
        (asked["content"], answered["content"])
        for asked, answered in zip(messages, messages[1:])
        if asked["role"] == "user" and answered["role"] == "assistant"
    ]


def standalone_question(question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up so it can be searched without the conversation.

    Retrieval sees one question at a time, so "what about Manchester?" after
    a question about rent in London searches the archive for Manchester in
    general. Claude rewrites it first - "how much is rent in Manchester?" -
    and that is what gets searched and answered.

    `history` holds earlier (question, answer) pairs, oldest first. With none
    there is nothing to resolve, so the first question costs no extra call.
    If the rewrite fails the question is used as asked: a follow-up searched
    literally is a weaker answer, not a failed one.
    """
    if not history:
        return question

    turns = []
    for asked, answered in history[-_HISTORY_TURNS:]:
        # Citation markers mean nothing without the excerpts they pointed at.
        answered = _CITATION.sub("", answered)[:_HISTORY_CHARS]
        turns.append(f"User: {asked}\nAssistant: {answered}")
    prompt = "\n\n".join(turns) + f"\n\nLatest message: {question}"

    try:
        # Temperature 0: the same conversation should search the same thing
        # every time. At the default, a borderline follow-up like "is that per
        # month?" was resolved on one run and left as typed on the next.
        rewritten = complete(
            prompt, _REWRITE_SYSTEM, max_tokens=200, timeout=10.0, temperature=0.0
        )
    except ClaudeError as exc:
        print(f"rewrite failed ({exc}); searching the question as asked", file=sys.stderr)
        return question

    # The reply should be one question. Anything past the first line is the
    # model explaining itself, and a reply far longer than any question is
    # not a rewrite at all - searching either would be worse than the original.
    lines = [line for line in rewritten.strip().splitlines() if line.strip()]
    rewritten = lines[0].strip().strip('"«»“”') if lines else ""
    if not rewritten or len(rewritten) > _REWRITE_MAX_CHARS:
        return question
    return rewritten


def detect_language(text: str) -> str:
    """Classify text as 'arabic' or 'english' by script prevalence."""
    arabic = len(_ARABIC.findall(text))
    latin = len(_LATIN.findall(text))

    if arabic == 0 and latin == 0:
        return "english"
    return "arabic" if arabic >= latin else "english"


class RAGPipeline:
    """Embeds a question, retrieves matching chunks, and asks Claude to answer."""

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore.load()

        # The keyword fallback takes seconds to build, which is fine at
        # startup and not fine in the middle of an outage with someone
        # waiting. Built in the background so neither pays for it.
        self._keywords: KeywordIndex | None = None
        self._keywords_lock = threading.Lock()
        threading.Thread(target=self._keyword_index, daemon=True).start()

    def _keyword_index(self) -> KeywordIndex:
        """The fallback index, waiting for the background build if needed."""
        with self._keywords_lock:
            if self._keywords is None:
                self._keywords = KeywordIndex(self.store.contents)
            return self._keywords

    def _build_prompt(self, question: str, sources: list[Document], language: str) -> str:
        blocks = []
        for i, source in enumerate(sources, 1):
            authors = source.metadata.get("authors", "unknown")
            date = source.metadata.get("date_range", "unknown date")
            blocks.append(f"[{i}] ({authors}, {date})\n{source.content}")

        context = "\n\n".join(blocks)

        # Excerpts from the outage fallback are a weaker search, so a gap in
        # them is not evidence of a gap in the archive. Without this, the
        # model told readers the archive had nothing on bank accounts - one
        # of its best-covered topics - because keyword search missed it.
        degraded = any(source.score_kind == "keyword" for source in sources)

        if language == "arabic":
            note = _DEGRADED_AR if degraded else ""
            return f"مقتطفات من نقاشات الطلاب:\n\n{context}\n\n{note}السؤال: {question}"
        note = _DEGRADED_EN if degraded else ""
        return f"Excerpts from student discussions:\n\n{context}\n\n{note}Question: {question}"

    def retrieve(self, question: str) -> tuple[str, list[Document], str | None]:
        """Detect language and fetch context. Returns (language, sources, error).

        Vector search supplies a wide candidate pool and the reranker picks the
        final passages, because cosine similarity barely separates chunks in
        this corpus. If reranking is unavailable the vector ordering is used
        as-is - a worse answer beats no answer.

        The same reasoning covers the embedding service itself: if the
        question cannot be embedded, keyword search stands in so the reader
        still gets an answer.
        """
        language = detect_language(question)

        try:
            query_vector = embed_query(question)
        except EmbeddingError as exc:
            return language, *self._keyword_fallback(question, exc)

        candidates = self.store.search(
            query_vector,
            top_k=config.RETRIEVE_CANDIDATES,
            threshold=config.SIMILARITY_THRESHOLD,
        )

        if not candidates:
            return language, [], None

        return language, self._rerank(question, candidates), None

    def _keyword_fallback(
        self, question: str, cause: EmbeddingError
    ) -> tuple[list[Document], str | None]:
        """Retrieve by keywords when the embedding service is unavailable.

        Reranking is skipped: it runs on the same service that just failed,
        so trying it would only add the reader's wait to the outage.
        """
        print(f"embedding failed ({cause}); falling back to keyword search", file=sys.stderr)
        # A wider net than vector search gets: keywords match people asking
        # the question as readily as people answering it, and the passages
        # are short enough that ten cost Claude little more than five.
        hits = self._keyword_index().search(question, top_k=config.RETRIEVE_CANDIDATES)
        if not hits:
            # Nothing shared a single word with the question. Report the
            # outage rather than claim the archive has no answer - the search
            # that could have found one never ran.
            return [], str(cause)
        return self.store.fetch(hits, score_kind="keyword"), None

    def _rerank(self, question: str, candidates: list[Document]) -> list[Document]:
        """Reorder candidates by cross-encoder relevance, dropping weak matches."""
        try:
            ranked = rerank(
                question,
                [doc.content for doc in candidates],
                top_n=config.TOP_K_RESULTS,
            )
        except RerankError:
            # Fall back to vector order rather than failing the query.
            return candidates[: config.TOP_K_RESULTS]

        sources = []
        for index, score in ranked:
            document = candidates[index]
            # Replace the cosine score with the rerank score, which is absolute
            # and so meaningful to show and to threshold on. The kind changes
            # with it, so the UI does not label a rerank score "similarity".
            document.score = score
            document.score_kind = "relevance"
            if score >= config.RERANK_THRESHOLD:
                sources.append(document)

        return sources

    def stream_answer(self, question: str, language: str, sources: list[Document]):
        """Yield the answer in chunks as Claude generates it."""
        yield from stream(
            prompt=self._build_prompt(question, sources, language),
            system=_SYSTEM_AR if language == "arabic" else _SYSTEM_EN,
        )
