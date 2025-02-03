"""Tests for job discovery service."""

import pytest
from typing import Dict
from src.services.job_discovery import JobDiscoveryService
from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager

@pytest.mark.asyncio
class TestJobDiscoveryService:
    """Test suite for JobDiscoveryService."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test dependencies."""
        self.store = ChromaStore()
        self.job_manager = JobManager()
        self.service = JobDiscoveryService(store=self.store, job_manager=self.job_manager)
        yield
    
    async def test_detailed_analysis(
        self,
        sample_job_data: Dict,
        sample_candidate_data: Dict
    ) -> None:
        """Test detailed analysis generation."""
        analysis = await self.service._get_detailed_analysis(
            candidate_info=sample_candidate_data,
            job_info=sample_job_data
        )
        
        assert isinstance(analysis, dict)
        assert "candidate_summary" in analysis
        assert "skill_analysis" in analysis
        assert "experience_analysis" in analysis
        assert "recommendations" in analysis
        
        # Check candidate summary
        assert "name" in analysis["candidate_summary"]
        assert "total_experience" in analysis["candidate_summary"]
        assert "relevant_experience" in analysis["candidate_summary"]
        
        # Check skill analysis
        assert "match_score" in analysis["skill_analysis"]
        assert "matching_skills" in analysis["skill_analysis"]
        assert "missing_skills" in analysis["skill_analysis"]
        assert "additional_skills" in analysis["skill_analysis"]
        
        # Check experience analysis
        assert "total_years" in analysis["experience_analysis"]
        assert "relevant_positions" in analysis["experience_analysis"]
        assert "experience_summary" in analysis["experience_analysis"]
        
        # Check recommendations
        assert "next_steps" in analysis["recommendations"]
        assert "development_areas" in analysis["recommendations"]
        assert "strengths" in analysis["recommendations"]
        
    async def test_match_scoring(self) -> None:
        """Test match score calculation."""
        score = await self.service._calculate_match_score(
            job_skills=["Python", "AWS", "Docker"],
            candidate_skills=["Python", "AWS", "Kubernetes"]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
    async def test_recommendation_generation(self) -> None:
        """Test recommendation generation."""
        recommendations = await self.service._generate_recommendations(
            match_score=0.8,
            missing_skills=["Kubernetes"],
            extra_skills=["Jenkins"]
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(r, str) for r in recommendations) 