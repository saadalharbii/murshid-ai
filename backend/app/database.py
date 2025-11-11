"""
Database connection and operations using Supabase.
Handles vector storage and retrieval using pgvector.
"""

from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from loguru import logger
from .config import settings


class Database:
    """Database client for Supabase operations."""

    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self.table_name = "documents"
        logger.info("Database client initialized")

    async def create_table_if_not_exists(self):
        """
        Create the documents table with pgvector extension.

        Note: You need to run this SQL in your Supabase SQL editor:

        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Create documents table
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(768),  -- 768 dimensions for multilingual-mpnet
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        -- Create function for similarity search
        CREATE OR REPLACE FUNCTION match_documents(
            query_embedding vector(768),
            match_threshold float,
            match_count int
        )
        RETURNS TABLE (
            id bigint,
            content text,
            metadata jsonb,
            similarity float
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                documents.id,
                documents.content,
                documents.metadata,
                1 - (documents.embedding <=> query_embedding) AS similarity
            FROM documents
            WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
            ORDER BY documents.embedding <=> query_embedding
            LIMIT match_count;
        $$;
        """
        logger.info("Table creation SQL provided. Please run it in Supabase SQL editor.")

    def insert_document(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert a document with its embedding into the database.

        Args:
            content: The text content of the document
            embedding: The vector embedding of the content
            metadata: Optional metadata (source, date, author, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {}
            }

            result = self.client.table(self.table_name).insert(data).execute()
            logger.info(f"Document inserted successfully: {len(content)} chars")
            return True
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False

    def insert_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple documents in a batch.

        Args:
            documents: List of documents, each with 'content', 'embedding', and 'metadata'

        Returns:
            Number of documents successfully inserted
        """
        try:
            result = self.client.table(self.table_name).insert(documents).execute()
            count = len(documents)
            logger.info(f"Batch inserted {count} documents")
            return count
        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
            return 0

    def search_similar_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: The query vector embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of similar documents with content, metadata, and similarity score
        """
        try:
            # Call the match_documents function
            result = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': top_k
                }
            ).execute()

            logger.info(f"Found {len(result.data)} similar documents")
            return result.data
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the database.

        Returns:
            Total document count
        """
        try:
            result = self.client.table(self.table_name).select(
                "id", count="exact"
            ).execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def delete_all_documents(self) -> bool:
        """
        Delete all documents from the database.
        WARNING: This is destructive!

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all rows
            result = self.client.table(self.table_name).delete().neq('id', 0).execute()
            logger.warning("All documents deleted from database")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False


# Global database instance
db = Database()
