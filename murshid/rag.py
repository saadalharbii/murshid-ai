"""Retrieval-augmented question answering over the Telegram corpus."""

from __future__ import annotations

import re

from . import config
from .claude import stream
from .embeddings import EmbeddingError, embed_query
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

    def _build_prompt(self, question: str, sources: list[Document], language: str) -> str:
        blocks = []
        for i, source in enumerate(sources, 1):
            authors = source.metadata.get("authors", "unknown")
            date = source.metadata.get("date_range", "unknown date")
            blocks.append(f"[{i}] ({authors}, {date})\n{source.content}")

        context = "\n\n".join(blocks)

        if language == "arabic":
            return f"مقتطفات من نقاشات الطلاب:\n\n{context}\n\nالسؤال: {question}"
        return f"Excerpts from student discussions:\n\n{context}\n\nQuestion: {question}"

    def retrieve(self, question: str) -> tuple[str, list[Document], str | None]:
        """Detect language and fetch context. Returns (language, sources, error).

        Vector search supplies a wide candidate pool and the reranker picks the
        final passages, because cosine similarity barely separates chunks in
        this corpus. If reranking is unavailable the vector ordering is used
        as-is - a worse answer beats no answer.
        """
        language = detect_language(question)

        try:
            query_vector = embed_query(question)
        except EmbeddingError as exc:
            return language, [], str(exc)

        candidates = self.store.search(
            query_vector,
            top_k=config.RETRIEVE_CANDIDATES,
            threshold=config.SIMILARITY_THRESHOLD,
        )

        if not candidates:
            return language, [], None

        return language, self._rerank(question, candidates), None

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
            # and so meaningful to show and to threshold on.
            document.score = score
            if score >= config.RERANK_THRESHOLD:
                sources.append(document)

        return sources

    def stream_answer(self, question: str, language: str, sources: list[Document]):
        """Yield the answer in chunks as Claude generates it."""
        yield from stream(
            prompt=self._build_prompt(question, sources, language),
            system=_SYSTEM_AR if language == "arabic" else _SYSTEM_EN,
        )
