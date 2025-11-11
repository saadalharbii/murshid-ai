"""
Data ingestion script for processing Telegram HTML exports
and uploading them to Supabase with embeddings.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db
from app.embeddings import get_embedding_model
from app.config import settings
from telegram_parser import TelegramParser


def setup_logging():
    """Configure logging."""
    logger.add(
        "ingestion.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO"
    )


def ingest_telegram_data(directory_path: str, batch_size: int = 50):
    """
    Main ingestion function.

    Args:
        directory_path: Path to directory containing Telegram HTML exports
        batch_size: Number of documents to process in each batch
    """
    logger.info("=" * 50)
    logger.info("Starting Telegram data ingestion")
    logger.info("=" * 50)

    # Initialize parser and embedding model
    logger.info("Initializing parser and embedding model...")
    parser = TelegramParser()
    embedding_model = get_embedding_model()

    # Parse all messages
    logger.info(f"Parsing Telegram messages from: {directory_path}")
    messages = parser.parse_directory(directory_path)

    if not messages:
        logger.error("No messages found. Exiting.")
        return

    logger.info(f"Parsed {len(messages)} messages successfully")

    # Create chunks
    logger.info("Creating text chunks...")
    chunks = parser.chunk_messages(
        messages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )

    logger.info(f"Created {len(chunks)} chunks")

    # Process in batches
    total_uploaded = 0
    failed_uploads = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        try:
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in batch]

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = embedding_model.encode_batch(texts, batch_size=32)

            # Prepare documents for insertion
            documents = []
            for chunk, embedding in zip(batch, embeddings):
                documents.append({
                    'content': chunk['content'],
                    'embedding': embedding,
                    'metadata': chunk['metadata']
                })

            # Upload to database
            logger.info("Uploading to database...")
            uploaded = db.insert_documents_batch(documents)

            if uploaded > 0:
                total_uploaded += uploaded
                logger.info(f"✓ Batch {batch_num} uploaded successfully ({uploaded} documents)")
            else:
                failed_uploads += len(batch)
                logger.error(f"✗ Batch {batch_num} failed to upload")

        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            failed_uploads += len(batch)

    # Summary
    logger.info("=" * 50)
    logger.info("Ingestion Complete")
    logger.info("=" * 50)
    logger.info(f"Total chunks processed: {len(chunks)}")
    logger.info(f"Successfully uploaded: {total_uploaded}")
    logger.info(f"Failed uploads: {failed_uploads}")

    # Verify database count
    doc_count = db.get_document_count()
    logger.info(f"Total documents in database: {doc_count}")


def ingest_text_content(content: str, metadata: dict = None):
    """
    Ingest a single text content (for manual FAQs/notes).

    Args:
        content: Text content to ingest
        metadata: Optional metadata dictionary
    """
    logger.info("Ingesting text content...")

    # Get embedding model
    embedding_model = get_embedding_model()

    # Generate embedding
    embedding = embedding_model.encode(content)

    # Upload to database
    success = db.insert_document(
        content=content,
        embedding=embedding,
        metadata=metadata or {'source': 'manual', 'type': 'text'}
    )

    if success:
        logger.info("✓ Content uploaded successfully")
    else:
        logger.error("✗ Failed to upload content")

    return success


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Setup logging
    setup_logging()

    # Get Telegram export directory
    telegram_dir = os.getenv('TELEGRAM_EXPORT_DIR', 'ChatExport_2025-10-26')

    # Check if directory exists
    if not os.path.exists(telegram_dir):
        logger.error(f"Directory not found: {telegram_dir}")
        logger.info("Please set TELEGRAM_EXPORT_DIR in .env or provide correct path")
        return

    # Run ingestion
    ingest_telegram_data(telegram_dir, batch_size=50)


if __name__ == "__main__":
    # For command line usage
    import argparse

    parser_cli = argparse.ArgumentParser(description='Ingest Telegram data into MurshidAI')
    parser_cli.add_argument(
        '--dir',
        type=str,
        help='Path to Telegram export directory',
        default='ChatExport_2025-10-26'
    )
    parser_cli.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing',
        default=50
    )
    parser_cli.add_argument(
        '--text',
        type=str,
        help='Ingest a single text content instead of directory',
        default=None
    )

    args = parser_cli.parse_args()

    # Load environment
    load_dotenv()
    setup_logging()

    if args.text:
        # Ingest single text
        ingest_text_content(args.text)
    else:
        # Ingest Telegram directory
        ingest_telegram_data(args.dir, args.batch_size)
