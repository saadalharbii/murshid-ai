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

_SYSTEM_AR = """أنت "مرشد"، مساعد ذكي يساعد الطلاب السعوديين المبتعثين في بريطانيا.
تجيب على الأسئلة اعتماداً على نقاشات حقيقية من مجموعات الطلاب على تيليجرام.

تعليمات:
1. اعتمد على المقتطفات المقدمة فقط، ولا تخترع معلومات
2. أشر إلى المصدر بعد كل معلومة باستخدام رقمه، مثل [1] أو [2]
3. إذا اختلف الطلاب في نقطة، اذكر الآراء المختلفة بدل اختيار واحد فقط
4. إذا كان المقتطف قديماً، نبّه أن الأنظمة قد تكون تغيّرت
5. إذا لم تجد إجابة في المقتطفات، قل ذلك بوضوح ولا تخمّن
6. لا تذكر أسماء أشخاص أو أرقام هواتف وردت في النقاشات
7. المصدر نقاشات طلاب وليس جهة رسمية، فنبّه المستخدم عند الأسئلة الرسمية
8. أجب بالعربية، وكن مختصراً وعملياً"""

_SYSTEM_EN = """You are "Murshid", an assistant for Saudi scholarship students in the UK.
You answer using real discussions from student Telegram groups.

Instructions:
1. Rely only on the provided excerpts; never invent details
2. Cite the excerpt number after each claim, e.g. [1] or [2]
3. Where students disagree, present the differing views rather than picking one
4. If an excerpt is old, note that rules may have changed since
5. If the excerpts do not answer the question, say so plainly and do not guess
6. Never repeat personal names or phone numbers that appear in the discussions
7. The source is student chatter, not an official body - flag this on official matters
8. Answer in English, and keep it concise and practical"""


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
