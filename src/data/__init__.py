"""
Data management package
"""

from pathlib import Path
from src.core.config import settings
from src.core.logging import setup_logger

logger = setup_logger(__name__)


def init_data_directories() -> None:
    """Initialize all required data directories"""
    try:
        # Create main data directories
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)

        # Create job posting directories
        settings.JOB_POSTINGS_DIR.mkdir(parents=True, exist_ok=True)
        (settings.JOB_POSTINGS_DIR / "companies").mkdir(parents=True, exist_ok=True)
        (settings.JOB_POSTINGS_DIR / "jobs").mkdir(parents=True, exist_ok=True)
        (settings.JOB_POSTINGS_DIR / "mappings").mkdir(parents=True, exist_ok=True)

        # Create resume directories
        settings.RESUME_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Data directories initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing data directories: {str(e)}")
        raise
