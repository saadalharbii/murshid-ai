"""Retrieval-augmented question answering over the Telegram corpus."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import config
from .claude import ClaudeError, complete
from .embeddings import EmbeddingError, embed_query
from .store import Document, VectorStore

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[a-zA-Z]")

_SYSTEM_AR = """أنت "مرشد"، مساعد ذكي يساعد الطلاب السعوديين المبتعثين في بريطانيا.
تجيب على الأسئلة اعتماداً على نقاشات حقيقية من مجموعات الطلاب على تيليجرام.

تعليمات:
1. اعتمد على السياق المقدم فقط، ولا تخترع معلومات
2. إذا كان السياق لا يحتوي على إجابة، قل ذلك بوضوح
3. المصدر نقاشات طلاب وليس جهة رسمية، فنبّه المستخدم عند الأسئلة الرسمية
4. أجب بالعربية، وكن مختصراً وعملياً"""

_SYSTEM_EN = """You are "Murshid", an assistant for Saudi scholarship students in the UK.
You answer using real discussions from student Telegram groups.

Instructions:
1. Rely only on the provided context; never invent details
2. If the context does not contain the answer, say so plainly
3. The source is student chatter, not an official body - flag this on official matters
4. Answer in English, and keep it concise and practical"""

_NO_CONTEXT_AR = "لم أجد في أرشيف المجموعات ما يجيب على سؤالك. جرّب صياغة أخرى."
_NO_CONTEXT_EN = (
    "I couldn't find anything in the group archive that answers that. "
    "Try rephrasing your question."
)


@dataclass
class Answer:
    """The result of one question, including what it was based on."""

    text: str
    language: str
    sources: list[Document] = field(default_factory=list)
    elapsed: float = 0.0


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

    def answer(self, question: str) -> Answer:
        """Answer a question, returning the text plus the sources used."""
        started = time.time()
        language = detect_language(question)

        try:
            query_vector = embed_query(question)
        except EmbeddingError as exc:
            return Answer(f"Retrieval is unavailable: {exc}", language, [], time.time() - started)

        sources = self.store.search(
            query_vector,
            top_k=config.TOP_K_RESULTS,
            threshold=config.SIMILARITY_THRESHOLD,
        )

        if not sources:
            message = _NO_CONTEXT_AR if language == "arabic" else _NO_CONTEXT_EN
            return Answer(message, language, [], time.time() - started)

        try:
            text = complete(
                prompt=self._build_prompt(question, sources, language),
                system=_SYSTEM_AR if language == "arabic" else _SYSTEM_EN,
            )
        except ClaudeError as exc:
            return Answer(f"Could not generate an answer: {exc}", language, sources,
                          time.time() - started)

        return Answer(text, language, sources, time.time() - started)
