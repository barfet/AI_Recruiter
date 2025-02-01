from typing import Dict, Any

from src.core.logging import setup_logger
from src.embeddings.manager import EmbeddingManager

logger = setup_logger(__name__)


def format_job_result(result: Dict[str, Any]) -> str:
    """Format job search result for display"""
    metadata = result["metadata"]
    return f"""
    Job ID: {metadata['job_id']}
    Title: {metadata['title']}
    Company: {metadata['company']}
    Location: {metadata['location']}
    Skills: {', '.join(metadata['skills'])}
    Score: {result['score']:.3f}
    """


def format_candidate_result(result: Dict[str, Any]) -> str:
    """Format candidate search result for display"""
    metadata = result["metadata"]
    return f"""
    Resume ID: {metadata['resume_id']}
    Name: {metadata.get('name', 'Anonymous')}
    Location: {metadata.get('location', 'Not specified')}
    Skills: {', '.join(metadata.get('skills', []))}
    Score: {result['score']:.3f}
    """


def test_job_search():
    """Test semantic search for jobs"""
    manager = EmbeddingManager()
    vectorstore = manager.load_embeddings("jobs")
    if not vectorstore:
        logger.error("No job embeddings found. Please create embeddings first.")
        return

    # Test queries covering different aspects
    test_queries = [
        "Senior software engineer with Python and AWS experience",
        "Entry level data scientist with machine learning",
        "Remote frontend developer React TypeScript",
        "DevOps engineer with Kubernetes and CI/CD",
        "Product manager with agile experience",
    ]

    logger.info("Testing job search functionality...")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        try:
            results = manager.similarity_search(vectorstore, query, k=3)
            for result in results:
                print(format_job_result(result))
        except Exception as e:
            logger.error(f"Error testing job search: {str(e)}")


def test_candidate_search():
    """Test candidate search functionality"""
    logger.info("Testing candidate search functionality...")

    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.load_embeddings("candidates")

    # Test queries
    test_queries = [
        "Experienced software architect with cloud expertise",
        "Fresh graduate in computer science",
        "Full stack developer with React and Node.js",
        "Data engineer with big data experience",
        "Technical lead with team management skills",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = embedding_manager.similarity_search(vectorstore, query)

        for result in results:
            metadata = result["metadata"]
            score = 1 - (result["score"] / 2)  # Normalize score to 0-1 range

            logger.info(
                f"""
    Resume ID: {metadata['resume_id']}
    Name: {metadata.get('name', 'Anonymous')}
    Industry: {metadata.get('industry', 'Not specified')}
    Skills: {', '.join(metadata.get('skills', []))}
    Location: {metadata.get('location', 'Not specified')}
    Experience: {len(metadata.get('experience', []))} roles
    Education: {len(metadata.get('education', []))} degrees
    Match Score: {score:.2f}
    """
            )


def test_cross_matching():
    """Test matching between jobs and candidates"""
    manager = EmbeddingManager()
    jobs_store = manager.load_embeddings("jobs")
    candidates_store = manager.load_embeddings("candidates")

    if not jobs_store or not candidates_store:
        logger.error(
            "Missing embeddings. Please create both job and candidate embeddings first."
        )
        return

    # Get a sample job posting
    job_results = manager.similarity_search(jobs_store, "Senior software engineer", k=1)
    if not job_results:
        logger.error("No job found for testing")
        return

    job = job_results[0]
    logger.info("\nMatching candidates for job:")
    print(format_job_result(job))

    # Find matching candidates
    try:
        results = manager.similarity_search(candidates_store, job["document"], k=3)
        logger.info("Top matching candidates:")
        for result in results:
            print(format_candidate_result(result))
    except Exception as e:
        logger.error(f"Error testing cross matching: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting semantic search tests...")

    # Run all tests
    test_job_search()
    test_candidate_search()
    test_cross_matching()

    logger.info("Semantic search tests completed")
