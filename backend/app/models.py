"""
Pydantic models for API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for querying the chatbot."""
    question: str = Field(..., min_length=1, description="User's question in Arabic or English")


class Source(BaseModel):
    """Model for a source document chunk."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for chatbot queries."""
    answer: str
    language: str
    sources: List[Source]
    query_time: float


class UploadTextRequest(BaseModel):
    """Request model for uploading text content manually."""
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Response model for upload operations."""
    success: bool
    message: str
    documents_processed: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
