"""SentenceTransformer implementation of the embedding model interface."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer as STModel

from src.core.config import get_config
from src.data.embeddings.base import EmbeddingError, EmbeddingModelInterface

logger = logging.getLogger(__name__)

class SentenceTransformerModel(EmbeddingModelInterface):
    """SentenceTransformer-based embedding model implementation."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize SentenceTransformer model.
        
        Args:
            model_name: Optional model name override
        """
        try:
            self.config = get_config().embedding
            
            # Initialize model
            cache_dir = Path(self.config.CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.model = STModel(
                model_name or self.config.MODEL_NAME,
                cache_folder=str(cache_dir)
            )
            
            # Set max sequence length
            self.model.max_seq_length = self.config.MAX_LENGTH
            
            # Move to specified device
            if self.config.DEVICE != "cpu":
                self.model = self.model.to(self.config.DEVICE)
                
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {str(e)}")
            raise EmbeddingError(f"Model initialization failed: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        try:
            return self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise EmbeddingError(f"Text embedding failed: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts into vectors."""
        try:
            # Process in batches
            embeddings = []
            batch_size = self.config.BATCH_SIZE
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size
                )
                embeddings.append(batch_embeddings)
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
            
            return np.vstack(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {str(e)}")
            raise EmbeddingError(f"Batch embedding failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.config.EMBEDDING_DIM 