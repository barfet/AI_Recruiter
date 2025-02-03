import asyncio
import pytest
from typing import Dict

from src.agent.test_agent import parse_response
from src.agent.agent import RecruitingAgent
from src.services.job_discovery import JobDiscoveryService
from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager
from src.core.logging import setup_logger

logger = setup_logger(__name__)

@pytest.fixture
async def store():
    """Fixture for ChromaStore with actual data."""
    store = ChromaStore(persist_directory=".chroma", is_persistent=True)
    async with store as s:
        yield s

@pytest.fixture
def job_manager():
    """Fixture for JobManager."""
    return JobManager()

async def get_test_candidate() -> Dict:
    """Get a test candidate using the existing agent."""
    agent = RecruitingAgent(temperature=0.3)
    response = await agent.run("Find a candidate with Python and AWS experience")
    candidate_data = parse_response(response)
    
    # Get actual candidate data from ChromaDB
    async with ChromaStore(persist_directory=".chroma", is_persistent=True) as store:
        if resume_id := candidate_data.get("resume_id"):
            result = await store.get_candidate_by_id(resume_id)
            if result:
                metadata = result["metadata"]
                return {
                    "experience": metadata.get("experience", []),
                    "skills": metadata.get("skills", []),
                    "education": metadata.get("education", []),
                    "industry": metadata.get("industry", "")
                }
    
    # Fallback to sample data if no real data found
    return {
        "experience": [
            "Software Engineer with 5 years of experience",
            "Python development and AWS cloud services",
            "Led development teams and implemented CI/CD"
        ],
        "skills": [
            "Python",
            "AWS",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Team Leadership"
        ]
    }

@pytest.mark.asyncio
async def test_job_discovery_integration(store, job_manager):
    """Test job discovery integration with existing agent and data."""
    try:
        # Get test candidate
        candidate_profile = await get_test_candidate()
        logger.info(f"\nTest candidate profile: {candidate_profile}")
        
        # Initialize service
        service = JobDiscoveryService(store, job_manager)
        
        # Find matching jobs
        matches = await service.find_matching_jobs(candidate_profile, limit=3)
        
        # Validate results
        assert matches, "Should find at least one match"
        
        # Log detailed results
        logger.info("\n=== Job Discovery Results ===")
        for i, match in enumerate(matches, 1):
            logger.info(f"\nMatch {i}:")
            logger.info(f"Job: {match.title} at {match.company}")
            logger.info(f"Location: {match.location}")
            logger.info(f"Match Score: {match.match_score:.2f}")
            logger.info(f"Matching Skills: {', '.join(match.matching_skills)}")
            logger.info(f"Skills to Develop: {', '.join(match.missing_skills)}")
            logger.info("\nRelevance Explanation:")
            logger.info(match.relevance_explanation)
            
            if match.detailed_analysis:
                logger.info("\nDetailed Analysis:")
                logger.info(f"Candidate Summary: {match.detailed_analysis['candidate_summary']}")
                logger.info(f"Job Analysis: {match.detailed_analysis['job_analysis']}")
                logger.info(f"Skills Gap Analysis: {match.detailed_analysis['skills_gap_analysis']}")
                logger.info(f"Interview Strategy: {match.detailed_analysis['interview_strategy']}")
        
        # Validate match quality
        best_match = matches[0]
        assert best_match.match_score > 0.5, "Best match should have a good score"
        assert best_match.matching_skills, "Should have matching skills"
        assert best_match.detailed_analysis, "Should have detailed analysis"
        
    except Exception as e:
        logger.error(f"Error in job discovery integration test: {str(e)}")
        raise

async def main():
    """Run the integration test."""
    logger.info("Starting job discovery integration test...")
    
    async with ChromaStore(persist_directory=".chroma", is_persistent=True) as store:
        job_manager = JobManager()
        await test_job_discovery_integration(store, job_manager)
    
    logger.info("Job discovery integration test completed")

if __name__ == "__main__":
    asyncio.run(main()) 