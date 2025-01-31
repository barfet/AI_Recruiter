from pathlib import Path
import logging
from typing import List, Dict, Any

from src.core.config import settings
from src.core.logging import setup_logger
from src.data.managers.job import JobManager
from src.data.managers.candidate import CandidateManager
from src.data import init_data_directories

logger = setup_logger(__name__)

def process_data() -> None:
    """Process job postings and candidate resumes"""
    try:
        # Initialize data directories
        init_data_directories()
        
        # Process job postings
        logger.info("Processing job postings...")
        job_manager = JobManager()
        jobs = job_manager.process_jobs()
        logger.info(f"Successfully processed {len(jobs)} job postings")
        
        # Process candidate resumes
        logger.info("Processing candidate resumes...")
        candidate_manager = CandidateManager()
        candidates = candidate_manager.process_candidates()
        logger.info(f"Successfully processed {len(candidates)} candidate profiles")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_data() 