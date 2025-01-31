from typing import List, Dict, Any
from pathlib import Path

from src.core.logging import setup_logger
from src.core.exceptions import DataIngestionError
from src.data.managers.job_manager import JobManager
from src.data.managers.candidate_manager import CandidateManager
from src.data import init_data_directories

logger = setup_logger(__name__)

def ingest_data() -> None:
    """Main function to ingest and process all data"""
    try:
        # Initialize data directories
        init_data_directories()
        
        # Process job data
        logger.info("Processing job data...")
        job_manager = JobManager()
        processed_jobs = job_manager.load_and_process_jobs()
        logger.info(f"Successfully processed {len(processed_jobs)} job postings")
        
        # Process candidate data
        logger.info("Processing candidate data...")
        candidate_manager = CandidateManager()
        processed_candidates = candidate_manager.load_and_process_candidates()
        logger.info(f"Successfully processed {len(processed_candidates)} candidate profiles")
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise DataIngestionError(f"Data ingestion failed: {str(e)}")
        
if __name__ == "__main__":
    ingest_data() 