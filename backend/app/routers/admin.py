"""
Admin router for data management operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from loguru import logger
import tempfile
import os

from ..models import UploadTextRequest, UploadResponse
from ..database import db
from ..embeddings import get_embedding_model
from ..config import settings

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/upload/text", response_model=UploadResponse)
async def upload_text(request: UploadTextRequest):
    """
    Upload text content manually (for FAQs, notes, etc.).

    Args:
        request: UploadTextRequest with content and metadata

    Returns:
        UploadResponse with success status
    """
    try:
        logger.info("Uploading text content...")

        # Get embedding model
        embedding_model = get_embedding_model()

        # Generate embedding
        embedding = embedding_model.encode(request.content)

        # Upload to database
        success = db.insert_document(
            content=request.content,
            embedding=embedding,
            metadata=request.metadata or {'source': 'manual', 'type': 'text'}
        )

        if success:
            return UploadResponse(
                success=True,
                message="Content uploaded successfully",
                documents_processed=1
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload content to database"
            )

    except Exception as e:
        logger.error(f"Error uploading text: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading text: {str(e)}"
        )


@router.post("/upload/html", response_model=UploadResponse)
async def upload_html_files(files: List[UploadFile] = File(...)):
    """
    Upload Telegram HTML export files.

    Args:
        files: List of HTML files to process

    Returns:
        UploadResponse with processing results
    """
    try:
        logger.info(f"Uploading {len(files)} HTML files...")

        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
        from telegram_parser import TelegramParser

        parser = TelegramParser()
        embedding_model = get_embedding_model()

        all_messages = []

        # Process each file
        for file in files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Parse HTML
                messages = parser.parse_html_file(tmp_path)
                all_messages.extend(messages)
                logger.info(f"Parsed {len(messages)} messages from {file.filename}")
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        if not all_messages:
            return UploadResponse(
                success=False,
                message="No messages found in uploaded files",
                documents_processed=0
            )

        # Create chunks
        chunks = parser.chunk_messages(
            all_messages,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )

        # Generate embeddings and upload
        texts = [chunk['content'] for chunk in chunks]
        embeddings = embedding_model.encode_batch(texts)

        # Prepare documents
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                'content': chunk['content'],
                'embedding': embedding,
                'metadata': chunk['metadata']
            })

        # Upload to database
        uploaded = db.insert_documents_batch(documents)

        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(files)} files",
            documents_processed=uploaded
        )

    except Exception as e:
        logger.error(f"Error uploading HTML files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading HTML files: {str(e)}"
        )


@router.get("/stats")
async def get_stats():
    """
    Get database statistics.

    Returns:
        Dictionary with database stats
    """
    try:
        doc_count = db.get_document_count()

        return {
            "total_documents": doc_count,
            "status": "healthy"
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@router.delete("/documents")
async def delete_all_documents():
    """
    Delete all documents from the database.
    WARNING: This is destructive!

    Returns:
        Success message
    """
    try:
        logger.warning("Deleting all documents...")

        success = db.delete_all_documents()

        if success:
            return {
                "success": True,
                "message": "All documents deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete documents"
            )

    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents: {str(e)}"
        )
