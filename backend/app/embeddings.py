"""
Embeddings generation using Sentence Transformers.
Supports multilingual text (Arabic and English).
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
from loguru import logger
from .config import settings


class EmbeddingModel:
    """Wrapper for Sentence Transformers embedding model."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the Sentence Transformers model.
                       Defaults to settings.embedding_model
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar for batch encoding

        Returns:
            Single embedding vector or list of embedding vectors
        """
        try:
            # Convert single string to list
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]

            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )

            # Convert to list format
            embeddings_list = embeddings.tolist()

            # Return single embedding if input was single text
            if is_single:
                return embeddings_list[0]

            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with progress tracking.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            embeddings_list = embeddings.tolist()
            logger.info(f"Generated {len(embeddings_list)} embeddings")

            return embeddings_list

        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (e.g., 768 for multilingual-mpnet)
        """
        return self.model.get_sentence_embedding_dimension()


# Global embedding model instance
embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the global embedding model instance.

    Returns:
        EmbeddingModel instance
    """
    global embedding_model

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    return embedding_model


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()

    # Test with Arabic text
    arabic_text = "مرحبا، كيف حالك؟"
    arabic_embedding = model.encode(arabic_text)
    print(f"Arabic embedding dimension: {len(arabic_embedding)}")

    # Test with English text
    english_text = "Hello, how are you?"
    english_embedding = model.encode(english_text)
    print(f"English embedding dimension: {len(english_embedding)}")

    # Test batch encoding
    texts = [
        "أحتاج مساعدة في الدراسة",
        "I need help with my studies",
        "ما هي أفضل الجامعات في بريطانيا؟",
        "What are the best universities in Britain?"
    ]

    batch_embeddings = model.encode_batch(texts)
    print(f"\nBatch encoding: {len(batch_embeddings)} embeddings generated")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
