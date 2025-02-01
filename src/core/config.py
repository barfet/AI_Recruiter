"""Configuration settings for the application"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

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
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.0

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
