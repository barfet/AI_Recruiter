"""OpenAI implementation of the embedding model interface."""

import logging
from typing import List, Optional
import time

import numpy as np
from openai import OpenAI, RateLimitError

from src.core.config import get_config
from src.data.embeddings.base import EmbeddingError, EmbeddingModelInterface


logger = logging.getLogger(__name__)

class OpenAIModel(EmbeddingModelInterface):
    """OpenAI-based embedding model implementation."""
    
    # OpenAI's text-embedding-ada-002 dimension
    EMBEDDING_DIM = 1536
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize OpenAI model.
        
        Args:
            model_name: Optional model name override
        """
        try:
            self.config = get_config()
            
            if not self.config.OPENAI_API_KEY:
                raise EmbeddingError("OpenAI API key not found in configuration")
            
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.model_name = model_name or "text-embedding-ada-002"
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {str(e)}")
            raise EmbeddingError(f"Model initialization failed: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        try:
            response = self._call_api_with_retry(
                texts=[text],
                max_retries=self.config.MAX_RETRIES,
                retry_delay=self.config.RETRY_DELAY
            )
            return np.array(response[0])
            
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise EmbeddingError(f"Text embedding failed: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts into vectors."""
        try:
            # Process in batches
            embeddings = []
            batch_size = 100  # OpenAI recommended batch size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._call_api_with_retry(
                    texts=batch,
                    max_retries=self.config.MAX_RETRIES,
                    retry_delay=self.config.RETRY_DELAY
                )
                embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % (batch_size * 5) == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {str(e)}")
            raise EmbeddingError(f"Batch embedding failed: {str(e)}")
    
    def _call_api_with_retry(
        self,
        texts: List[str],
        max_retries: int,
        retry_delay: float
    ) -> List[List[float]]:
        """Call OpenAI API with retry logic.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            List[List[float]]: List of embeddings
            
        Raises:
            EmbeddingError: If API call fails after retries
        """
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [data.embedding for data in response.data]
                
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise EmbeddingError(f"Rate limit exceeded after {max_retries} retries")
                logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                raise EmbeddingError(f"API call failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.EMBEDDING_DIM 