"""
RAG (Retrieval-Augmented Generation) pipeline.
Combines vector search with Claude LLM for answering questions.
"""

from typing import List, Dict, Any, Tuple
import anthropic
from loguru import logger
import time
import re

from .database import db
from .embeddings import get_embedding_model
from .config import settings
from .models import Source


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self):
        """Initialize the RAG pipeline with Claude client and embedding model."""
        self.claude_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.embedding_model = get_embedding_model()
        logger.info("RAG pipeline initialized")

    def detect_language(self, text: str) -> str:
        """
        Detect if text is Arabic, English, or other.

        Args:
            text: Input text

        Returns:
            'arabic', 'english', or 'other'
        """
        # Check for Arabic characters (Unicode range)
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        has_arabic = bool(arabic_pattern.search(text))

        # Check for English characters
        english_pattern = re.compile(r'[a-zA-Z]')
        has_english = bool(english_pattern.search(text))

        # Determine language
        if has_arabic and not has_english:
            return 'arabic'
        elif has_english and not has_arabic:
            return 'english'
        elif has_arabic and has_english:
            # Mixed - count which is more prevalent
            arabic_count = len(arabic_pattern.findall(text))
            english_count = len(english_pattern.findall(text))
            return 'arabic' if arabic_count > english_count else 'english'
        else:
            return 'other'

    def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        threshold: float = 0.5
    ) -> List[Source]:
        """
        Retrieve relevant context from the database.

        Args:
            query: User's query
            top_k: Number of results to retrieve (default from settings)
            threshold: Minimum similarity threshold

        Returns:
            List of Source objects with content and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)

            # Search similar documents
            k = top_k or settings.top_k_results
            results = db.search_similar_documents(
                query_embedding=query_embedding,
                top_k=k,
                threshold=threshold
            )

            # Convert to Source objects
            sources = []
            for result in results:
                source = Source(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    similarity_score=result.get('similarity', 0.0)
                )
                sources.append(source)

            logger.info(f"Retrieved {len(sources)} relevant documents")
            return sources

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def generate_answer(
        self,
        query: str,
        context: List[Source],
        language: str
    ) -> str:
        """
        Generate answer using Claude with retrieved context.

        Args:
            query: User's query
            context: Retrieved context sources
            language: Detected language ('arabic' or 'english')

        Returns:
            Generated answer
        """
        try:
            # Build context string
            context_text = self._build_context_text(context)

            # Build prompt
            system_prompt = self._build_system_prompt(language)
            user_prompt = self._build_user_prompt(query, context_text, language)

            # Call Claude API
            logger.info(f"Calling Claude API (model: {settings.claude_model})")

            response = self.claude_client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract answer
            answer = response.content[0].text

            logger.info("Answer generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _build_context_text(self, sources: List[Source]) -> str:
        """
        Build context text from sources.

        Args:
            sources: List of Source objects

        Returns:
            Formatted context string
        """
        if not sources:
            return "No relevant context found."

        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Context {i}]\n{source.content}\n")

        return "\n".join(context_parts)

    def _build_system_prompt(self, language: str) -> str:
        """
        Build system prompt for Claude.

        Args:
            language: Target language for response

        Returns:
            System prompt string
        """
        if language == 'arabic':
            return """أنت مساعد ذكي متخصص في مساعدة الطلاب السعوديين المبتعثين في بريطانيا.
مهمتك هي الإجابة على أسئلة الطلاب بناءً على المعلومات المتوفرة في السياق المقدم.

تعليمات مهمة:
1. استخدم المعلومات من السياق المقدم للإجابة على السؤال
2. إذا لم تجد معلومات كافية في السياق، قل ذلك بوضوح
3. كن مفيداً ودقيقاً في إجاباتك
4. أجب باللغة العربية فقط
5. إذا كان السؤال غير واضح، اطلب توضيحاً"""

        else:  # English
            return """You are an intelligent assistant specialized in helping Saudi scholarship students in the UK.
Your task is to answer students' questions based on the information provided in the context.

Important instructions:
1. Use information from the provided context to answer the question
2. If you don't find enough information in the context, clearly state that
3. Be helpful and accurate in your responses
4. Respond in English only
5. If the question is unclear, ask for clarification"""

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        language: str
    ) -> str:
        """
        Build user prompt with query and context.

        Args:
            query: User's query
            context: Retrieved context
            language: Target language

        Returns:
            User prompt string
        """
        if language == 'arabic':
            return f"""السياق المتوفر:
{context}

السؤال: {query}

الإجابة:"""

        else:  # English
            return f"""Available context:
{context}

Question: {query}

Answer:"""

    def answer_question(self, query: str) -> Tuple[str, str, List[Source], float]:
        """
        Main method to answer a question using RAG pipeline.

        Args:
            query: User's question

        Returns:
            Tuple of (answer, language, sources, query_time)
        """
        start_time = time.time()

        try:
            # Detect language
            language = self.detect_language(query)

            logger.info(f"Query language detected: {language}")

            # Check if language is supported
            if language not in ['arabic', 'english']:
                unsupported_msg_ar = "عذراً، هذا النظام يدعم فقط اللغة العربية والإنجليزية."
                unsupported_msg_en = "Sorry, this system only supports Arabic and English."
                return (
                    f"{unsupported_msg_ar}\n\n{unsupported_msg_en}",
                    'other',
                    [],
                    time.time() - start_time
                )

            # Retrieve relevant context
            sources = self.retrieve_context(query)

            if not sources:
                no_context_msg = (
                    "لم أجد معلومات ذات صلة بسؤالك في قاعدة البيانات."
                    if language == 'arabic'
                    else "I couldn't find relevant information for your question in the database."
                )
                return (
                    no_context_msg,
                    language,
                    [],
                    time.time() - start_time
                )

            # Generate answer
            answer = self.generate_answer(query, sources, language)

            # Calculate query time
            query_time = time.time() - start_time

            logger.info(f"Query answered in {query_time:.2f}s")

            return answer, language, sources, query_time

        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            error_msg = (
                "عذراً، حدث خطأ أثناء معالجة سؤالك."
                if language == 'arabic'
                else "Sorry, an error occurred while processing your question."
            )
            return error_msg, language, [], time.time() - start_time


# Global RAG pipeline instance
rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the global RAG pipeline instance.

    Returns:
        RAGPipeline instance
    """
    global rag_pipeline

    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()

    return rag_pipeline


if __name__ == "__main__":
    # Test the RAG pipeline
    from dotenv import load_dotenv
    load_dotenv()

    pipeline = RAGPipeline()

    # Test with Arabic query
    query_ar = "ما هي أفضل الجامعات في لندن؟"
    print(f"Query: {query_ar}")
    answer, lang, sources, time_taken = pipeline.answer_question(query_ar)
    print(f"Language: {lang}")
    print(f"Answer: {answer}")
    print(f"Sources: {len(sources)}")
    print(f"Time: {time_taken:.2f}s")

    print("\n" + "="*50 + "\n")

    # Test with English query
    query_en = "What are the best universities in London?"
    print(f"Query: {query_en}")
    answer, lang, sources, time_taken = pipeline.answer_question(query_en)
    print(f"Language: {lang}")
    print(f"Answer: {answer}")
    print(f"Sources: {len(sources)}")
    print(f"Time: {time_taken:.2f}s")
