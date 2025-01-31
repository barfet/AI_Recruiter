import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .config import settings

# Create logs directory
LOGS_DIR = settings.PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """Set up logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    file_handler = RotatingFileHandler(
        LOGS_DIR / f"{name}.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create main application logger
logger = setup_logger('ai_recruiter') 