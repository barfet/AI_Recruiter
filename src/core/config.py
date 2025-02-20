"""Configuration settings for the application"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enum import Enum
from pydantic import Field

# Load environment variables from .env file
load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "dataset"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    INDEXES_DIR: Path = DATA_DIR / "indexes"

    # Job posting directories
    JOB_POSTINGS_DIR: Path = RAW_DATA_DIR / "jobs"
    RESUME_DIR: Path = RAW_DATA_DIR / "resumes"

    # API Keys
    OPENAI_API_KEY: str
    LANGSMITH_API_KEY: Optional[str] = None

    # Vector DB settings
    VECTOR_DB_HOST: Optional[str] = None
    VECTOR_DB_API_KEY: Optional[str] = None

    # Model settings
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.7 
    LLM_CONFIG: Dict[str, Any] = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE
    }

    # Vector search settings
    VECTOR_SIMILARITY_TOP_K: int = 5

    # Processing settings
    MAX_JOBS: int = 1000  # Maximum number of jobs to process
    MAX_CANDIDATES: int = 100  # Maximum number of candidates to process

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }


# Create global settings instance
settings = Settings()

# Ensure required directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
settings.JOB_POSTINGS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESUME_DIR.mkdir(parents=True, exist_ok=True)


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    FAISS = "faiss"
    CHROMA = "chroma"

class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models and parameters."""
    
    # Model configuration
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 512
    DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # Dimension of embeddings
    EMBEDDING_DIM: int = 384  # matches all-MiniLM-L6-v2
    
    # Cache settings
    CACHE_DIR: str = ".cache/embeddings"
    USE_CACHE: bool = True

class VectorStoreConfig(BaseSettings):
    """Configuration for vector stores."""
    
    # Vector store type
    STORE_TYPE: VectorStoreType = VectorStoreType.CHROMA
    
    # Common settings
    PERSIST_DIRECTORY: str = ".data/vector_store"
    
    # FAISS specific settings
    FAISS_INDEX_TYPE: str = "Flat"  # or "IVF", "HNSW" etc.
    FAISS_METRIC_TYPE: str = "cosine"  # or "l2", "inner_product"
    
    # Chroma specific settings
    CHROMA_COLLECTION_NAME: str = "ai_recruiter"
    CHROMA_PERSIST_DIRECTORY: str = ".data/chroma"

class Config(BaseSettings):
    """Main configuration class combining all settings."""
    
    # Vector store settings
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # Embedding settings
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Error handling
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # seconds
    
    class Config:
        """Pydantic config."""
        env_prefix = "AI_RECRUITER_"
        case_sensitive = True

# Global config instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config: Global configuration instance
    """
    return config

def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration values.
    
    Args:
        updates: Dictionary of configuration updates
        
    Example:
        >>> update_config({"embedding": {"MODEL_NAME": "new-model"}})
    """
    global config
    for key, value in updates.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                current = getattr(config, key)
                for k, v in value.items():
                    setattr(current, k, v)
            else:
                setattr(config, key, value)
