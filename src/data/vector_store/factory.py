"""Factory for creating vector store instances."""

from typing import Optional

from src.core.config import get_config, VectorStoreType
from src.data.vector_store.base import VectorStoreInterface
from src.data.vector_store.faiss_store import FAISSStore
from src.data.vector_store.chroma_store import ChromaStore

def create_vector_store(store_type: Optional[VectorStoreType] = None) -> VectorStoreInterface:
    """Create a vector store instance based on configuration.
    
    Args:
        store_type: Optional override for store type from config
        
    Returns:
        VectorStoreInterface: Initialized vector store
        
    Raises:
        ValueError: If store type is not supported
    """
    if store_type is None:
        store_type = get_config().vector_store.STORE_TYPE
        
    if store_type == VectorStoreType.FAISS:
        return FAISSStore()
    elif store_type == VectorStoreType.CHROMA:
        return ChromaStore()
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

# Singleton instance
_vector_store_instance: Optional[VectorStoreInterface] = None

def get_vector_store() -> VectorStoreInterface:
    """Get or create the singleton vector store instance.
    
    Returns:
        VectorStoreInterface: Global vector store instance
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = create_vector_store()
    return _vector_store_instance 