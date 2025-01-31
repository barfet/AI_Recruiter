from pathlib import Path
from src.core.config import settings
from src.core.logging import setup_logger

logger = setup_logger(__name__)

def init_data_directories():
    """Initialize data directories if they don't exist"""
    try:
        # Create main data directories
        settings.DATA_DIR.mkdir(exist_ok=True)
        settings.RAW_DATA_DIR.mkdir(exist_ok=True)
        settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        settings.INDEXES_DIR.mkdir(exist_ok=True)
        
        logger.info("Data directories initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing data directories: {str(e)}")
        raise 