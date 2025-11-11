"""
Query router for handling Q&A requests.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models import QueryRequest, QueryResponse, Source
from ..rag import get_rag_pipeline

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Answer a user's question using RAG pipeline.

    Args:
        request: QueryRequest with user's question

    Returns:
        QueryResponse with answer, sources, and metadata
    """
    try:
        logger.info(f"Received query: {request.question[:100]}...")

        # Get RAG pipeline
        rag = get_rag_pipeline()

        # Answer question
        answer, language, sources, query_time = rag.answer_question(request.question)

        # Build response
        response = QueryResponse(
            answer=answer,
            language=language,
            sources=sources,
            query_time=query_time
        )

        logger.info(f"Query processed successfully in {query_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
