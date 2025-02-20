"""Main data processing script."""

import logging
from pathlib import Path

from src.core.config import get_config
from src.data.embeddings.manager import EmbeddingManager
from src.data.managers.candidate import CandidateManager
from src.data.managers.job import JobManager

logger = logging.getLogger(__name__)

def process_data(
    force_clean: bool = False,
    force_embeddings: bool = False
) -> None:
    """Process all data through the pipeline.
    
    This function orchestrates the entire data processing pipeline:
    1. Load and clean job data
    2. Load and clean candidate data
    3. Create embeddings for jobs and candidates
    4. Store everything in the vector database
    
    Args:
        force_clean: Whether to force reprocessing of cleaned data
        force_embeddings: Whether to force recreation of embeddings
    """
    try:
        config = get_config()
        
        # Initialize managers
        job_manager = JobManager()
        candidate_manager = CandidateManager()
        embedding_manager = EmbeddingManager()
        
        # Process jobs
        logger.info("Processing jobs...")
        jobs_file = config.PROCESSED_DATA_DIR / "structured_jobs.json"
        if force_clean or not jobs_file.exists():
            job_manager.process_jobs()
        
        # Process candidates
        logger.info("Processing candidates...")
        candidates_file = config.PROCESSED_DATA_DIR / "structured_candidates.json"
        if force_clean or not candidates_file.exists():
            candidate_manager.process_candidates()
        
        # Create embeddings
        logger.info("Creating embeddings...")
        if force_embeddings:
            # Create embeddings for jobs and candidates
            embedding_manager.create_job_embeddings(jobs_file)
            embedding_manager.create_candidate_embeddings(candidates_file)
            logger.info("Successfully created all embeddings")
        else:
            # Only create embeddings for new or updated data
            # This would require tracking changes, which is a future enhancement
            logger.info("Skipping embeddings creation (use force_embeddings=True to recreate)")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    process_data()
