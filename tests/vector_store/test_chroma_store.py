"""Tests for ChromaDB vector store."""

import pytest
from typing import Dict
from src.vector_store.chroma_store import ChromaStore

@pytest.mark.asyncio
class TestChromaStore:
    """Test suite for ChromaStore."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test environment."""
        self.store = ChromaStore()
        
        # Add test data
        await self.store.add_job(
            job_id="123",
            title="Senior Python Developer",
            description="Looking for a Python expert",
            skills=["Python", "AWS", "Docker"]
        )
        
        await self.store.add_candidate(
            resume_id="456",
            name="John Doe",
            skills=["Python", "AWS", "Kubernetes"],
            experience=[{"title": "Developer", "duration": "3 years"}]
        )
        
        yield
        
        # Cleanup
        await self.store.delete_job("123")
        await self.store.delete_candidate("456")
    
    async def test_job_retrieval(self) -> None:
        """Test job retrieval by ID."""
        job = await self.store.get_job_by_id("123")
        
        assert isinstance(job, dict)
        assert job["title"] == "Senior Python Developer"
        assert "Python" in job["skills"]
        
    async def test_candidate_search(self) -> None:
        """Test candidate search functionality."""
        results = await self.store.search_candidates(
            query="Python developer with AWS experience"
        )
        
        assert len(results) > 0
        assert isinstance(results[0], dict)
        assert "score" in results[0]
        
    async def test_batch_operations(self) -> None:
        """Test batch add and delete operations."""
        # Add multiple jobs
        jobs = [
            {
                "job_id": "batch1",
                "title": "ML Engineer",
                "description": "AI/ML role",
                "skills": ["Python", "ML", "TensorFlow"]
            },
            {
                "job_id": "batch2",
                "title": "DevOps Engineer",
                "description": "Infrastructure role",
                "skills": ["AWS", "Docker", "Kubernetes"]
            }
        ]
        
        for job in jobs:
            await self.store.add_job(**job)
            
        # Verify additions
        for job in jobs:
            result = await self.store.get_job_by_id(job["job_id"])
            assert result is not None
            assert result["title"] == job["title"]
            
        # Cleanup
        for job in jobs:
            await self.store.delete_job(job["job_id"])
            
        # Verify deletions
        for job in jobs:
            result = await self.store.get_job_by_id(job["job_id"])
            assert result is None
            
    async def test_error_handling(self) -> None:
        """Test error handling scenarios."""
        # Test non-existent job
        result = await self.store.get_job_by_id("nonexistent")
        assert result is None
        
        # Test invalid ID format
        result = await self.store.get_job_by_id("")
        assert result is None 