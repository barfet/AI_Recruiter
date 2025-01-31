"""Configuration settings for the application"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Vector search settings
VECTOR_SIMILARITY_TOP_K = 5

# Original data paths
JOB_POSTINGS_DIR = DATA_DIR / "raw" / "jobs"
RESUME_DIR = DATA_DIR / "raw" / "resumes/data/data"

# API Keys
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Vector DB settings
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST")
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")

# Model settings
LLM_MODEL = OPENAI_MODEL
LLM_TEMPERATURE = 0.0

# Processing settings
MAX_JOBS = 1000  # Maximum number of jobs to process
MAX_CANDIDATES = 100  # Maximum number of candidates to process

# Create global settings instance
settings = {
    "PROJECT_ROOT": BASE_DIR,
    "DATA_DIR": DATA_DIR,
    "RAW_DATA_DIR": DATA_DIR / "raw",
    "PROCESSED_DATA_DIR": PROCESSED_DATA_DIR,
    "INDEXES_DIR": INDEXES_DIR,
    "JOB_POSTINGS_DIR": JOB_POSTINGS_DIR,
    "RESUME_DIR": RESUME_DIR,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "LANGSMITH_API_KEY": LANGSMITH_API_KEY,
    "VECTOR_DB_HOST": VECTOR_DB_HOST,
    "VECTOR_DB_API_KEY": VECTOR_DB_API_KEY,
    "EMBEDDING_MODEL": EMBEDDING_MODEL,
    "LLM_MODEL": LLM_MODEL,
    "LLM_TEMPERATURE": LLM_TEMPERATURE,
    "MAX_JOBS": MAX_JOBS,
    "MAX_CANDIDATES": MAX_CANDIDATES,
    "VECTOR_SIMILARITY_TOP_K": VECTOR_SIMILARITY_TOP_K
}

# Ensure required directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True) 