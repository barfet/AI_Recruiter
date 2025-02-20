"""Factory for creating embedding model instances."""

from enum import Enum
from typing import Optional

from src.data.embeddings.base import EmbeddingModelInterface, EmbeddingError
from src.data.embeddings.sentence_transformer import SentenceTransformerModel
from src.data.embeddings.openai import OpenAIModel

class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"

def create_embedding_model(
    model_type: EmbeddingModelType,
    model_name: Optional[str] = None
) -> EmbeddingModelInterface:
    """Create an embedding model instance.
    
    Args:
        model_type: Type of embedding model to create
        model_name: Optional model name override
        
    Returns:
        EmbeddingModelInterface: Initialized embedding model
        
    Raises:
        EmbeddingError: If model type is not supported
    """
    if model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
        return SentenceTransformerModel(model_name=model_name)
    elif model_type == EmbeddingModelType.OPENAI:
        return OpenAIModel(model_name=model_name)
    else:
        raise EmbeddingError(f"Unsupported embedding model type: {model_type}")

# Singleton instance
_embedding_model_instance: Optional[EmbeddingModelInterface] = None

def get_embedding_model() -> EmbeddingModelInterface:
    """Get or create the singleton embedding model instance.
    
    Returns:
        EmbeddingModelInterface: Global embedding model instance
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = create_embedding_model(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER
        )
    return _embedding_model_instance 