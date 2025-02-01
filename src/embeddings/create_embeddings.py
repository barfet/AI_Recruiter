from src.core.logging import setup_logger
from src.core.exceptions import EmbeddingError
from src.embeddings.manager import EmbeddingManager
from src.data import init_data_directories

logger = setup_logger(__name__)


def create_embeddings() -> None:
    """Create embeddings for jobs and candidates"""
    try:
        # Initialize data directories
        init_data_directories()

        # Initialize embedding manager
        embedding_manager = EmbeddingManager()

        # Create job embeddings
        logger.info("Creating job embeddings...")
        embedding_manager.create_job_embeddings()

        # Create candidate embeddings
        logger.info("Creating candidate embeddings...")
        embedding_manager.create_candidate_embeddings()

        logger.info("Successfully created all embeddings")

    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise EmbeddingError(f"Failed to create embeddings: {str(e)}")


if __name__ == "__main__":
    create_embeddings()
