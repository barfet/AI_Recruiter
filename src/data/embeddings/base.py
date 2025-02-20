"""Base interface for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class EmbeddingModelInterface(ABC):
    """Abstract base class for embedding models.
    
    This interface defines the common operations that all embedding model implementations
    must support, ensuring consistency across different backends (OpenAI, SentenceTransformers, etc.).
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text into a vector.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Vector representation
            
        Raises:
            EmbeddingError: If embedding fails
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts into vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Matrix of vector representations
            
        Raises:
            EmbeddingError: If embedding fails
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            int: Embedding dimension
        """
        pass

class EmbeddingError(Exception):
    """Base exception for embedding operations."""
    pass 