from typing import List, Dict, Any, Optional

from src.core.logging import setup_logger
from src.core.exceptions import EmbeddingError
from src.embeddings.manager import EmbeddingManager

logger = setup_logger(__name__)


class SearchManager:
    """Manager for search operations"""

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.job_vectorstore = self.embedding_manager.load_embeddings("jobs")
        self.candidate_vectorstore = self.embedding_manager.load_embeddings(
            "candidates"
        )

    def search_jobs(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for jobs matching the query"""
        try:
            if not self.job_vectorstore:
                raise EmbeddingError("Job embeddings not found")

            results = self.embedding_manager.similarity_search(
                self.job_vectorstore, query, k=k
            )

            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in results:
                    metadata = result["metadata"]
                    match = True
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                return filtered_results

            return results

        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            raise

    def search_candidates(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for candidates matching the query"""
        try:
            if not self.candidate_vectorstore:
                raise EmbeddingError("Candidate embeddings not found")

            results = self.embedding_manager.similarity_search(
                self.candidate_vectorstore, query, k=k
            )

            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in results:
                    metadata = result["metadata"]
                    match = True
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                return filtered_results

            return results

        except Exception as e:
            logger.error(f"Error searching candidates: {str(e)}")
            raise

    def match_job_candidates(
        self, job_id: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Find candidates matching a specific job"""
        try:
            if not self.job_vectorstore or not self.candidate_vectorstore:
                raise EmbeddingError("Job or candidate embeddings not found")

            # Get job details
            job_results = self.search_jobs(f"job_id:{job_id}", k=1)
            if not job_results:
                raise ValueError(f"Job with ID {job_id} not found")

            job = job_results[0]["metadata"]

            # Create search query from job details
            search_query = f"""
            Looking for candidates with:
            - Skills in: {', '.join(job['skills'])}
            - Experience level: {job['experience_level'] or 'Any'}
            - Location: {job['location']}
            - Role: {job['title']}
            """

            # Search for matching candidates
            return self.search_candidates(search_query, k=k, filters=filters)

        except Exception as e:
            logger.error(f"Error matching candidates to job: {str(e)}")
            raise
