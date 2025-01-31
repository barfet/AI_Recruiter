from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    INDEXES_DIR: Path = DATA_DIR / "indexes"
    
    # Original data paths
    JOB_POSTINGS_DIR: Path = DATA_DIR / "job-postings"
    RESUME_DIR: Path = DATA_DIR / "resume"
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    
    # Vector DB settings
    VECTOR_DB_HOST: Optional[str] = os.getenv("VECTOR_DB_HOST")
    VECTOR_DB_API_KEY: Optional[str] = os.getenv("VECTOR_DB_API_KEY")
    
    # Model settings
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.0
    
    # Processing settings
    MAX_JOBS: int = 1000  # Maximum number of jobs to process
    MAX_CANDIDATES: int = 100  # Maximum number of candidates to process
    
    class Config:
        env_file = ".env"

# Create global settings instance
settings = Settings()

# Ensure required directories exist
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True) 