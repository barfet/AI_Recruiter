"""Logging configuration for the application"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Set up a logger with the given name"""
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        # Set log level
        logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        
        # Prevent the logger from propagating to the root logger
        logger.propagate = False
    
    return logger

# Create main application logger
logger = setup_logger('ai_recruiter') 