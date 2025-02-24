"""Base class for integration tests."""

import pytest
from typing import AsyncGenerator
from src.agent.agent import RecruitingAgent
from src.vector_store.chroma_store import ChromaStore
from src.services.job_discovery import JobDiscoveryService
from src.data.managers.job import JobManager
from src.embeddings.manager import EmbeddingManager

class BaseIntegrationTest:
    """Base class for integration tests with common setup."""

    @pytest.fixture(autouse=True)
    async def setup(self) -> AsyncGenerator[None, None]:
        """Setup test environment."""
        # Initialize core services
        self.store = ChromaStore()
        self.job_manager = JobManager()
        self.agent = RecruitingAgent()
        self.job_service = JobDiscoveryService(store=self.store)
        self.embedding_manager = EmbeddingManager()

        # Setup test data
        await self._setup_test_data()
        
        yield
        
        # Cleanup
        await self._cleanup_test_data()

    async def _setup_test_data(self) -> None:
        """Setup test data in the vector store."""
        # Add sample jobs
        self.test_jobs = [
            {
                "id": "test_job_1",
                "title": "Senior Python Developer",
                "description": "Looking for a Python expert with AWS experience",
                "skills": ["Python", "AWS", "Docker"],
                "requirements": "5+ years Python experience, AWS certification preferred"
            },
            {
                "id": "test_job_2",
                "title": "ML Engineer",
                "description": "AI/ML role with focus on NLP",
                "skills": ["Python", "PyTorch", "NLP"],
                "requirements": "3+ years ML experience, strong NLP background"
            }
        ]
        
        # Add sample candidates
        self.test_candidates = [
            {
                "id": "test_candidate_1",
                "name": "John Doe",
                "skills": ["Python", "AWS", "Kubernetes"],
                "experience": "5 years in software development"
            },
            {
                "id": "test_candidate_2",
                "name": "Jane Smith",
                "skills": ["Python", "PyTorch", "Deep Learning"],
                "experience": "3 years in ML"
            }
        ]

        # Store test data
        for job in self.test_jobs:
            # Create embedding
            job_text = f"{job['title']} {job['description']} {' '.join(job['skills'])}"
            job_embedding = await self.embedding_manager.get_embedding(job_text)
            
            # Store in vector store
            await self.store.add_job(
                job_id=job["id"],
                title=job["title"],
                description=job["description"],
                skills=job["skills"],
                requirements=job["requirements"],
                embedding=job_embedding
            )
            
        for candidate in self.test_candidates:
            # Create embedding
            candidate_text = f"{candidate['name']} {candidate['experience']} {' '.join(candidate['skills'])}"
            candidate_embedding = await self.embedding_manager.get_embedding(candidate_text)
            
            # Store in vector store
            await self.store.add_candidate(
                resume_id=candidate["id"],
                name=candidate["name"],
                skills=candidate["skills"],
                experience=candidate["experience"],
                embedding=candidate_embedding
            )

    async def _cleanup_test_data(self) -> None:
        """Cleanup test data after tests."""
        # Remove test data
        for job in self.test_jobs:
            await self.store.delete_job(job["id"])
        for candidate in self.test_candidates:
            await self.store.delete_candidate(candidate["id"])

    async def cleanup(self) -> None:
        """Cleanup resources after tests."""
        await self._cleanup_test_data() 