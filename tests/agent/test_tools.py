"""Tests for AI tools."""

import pytest
from typing import Dict
from src.agent.tools import SkillAnalysisTool, SkillAnalysisInput
import json
from src.vector_store.chroma_store import ChromaStore

@pytest.mark.asyncio
class TestSkillAnalysisTool:
    """Test suite for SkillAnalysisTool."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test environment."""
        self.store = ChromaStore()
        self.tool = SkillAnalysisTool(store=self.store)
        
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
    
    async def test_skill_matching(
        self,
        sample_job_data: Dict,
        sample_candidate_data: Dict
    ) -> None:
        """Test skill matching functionality."""
        result = await self.tool._arun(
            job_id="123",
            resume_id="456",
            analysis_type="match"
        )
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        data = result_dict["data"]
        assert isinstance(data["match_score"], float)
        assert isinstance(data["exact_matches"], list)
        assert isinstance(data["semantic_matches"], list)
        assert isinstance(data["missing_skills"], list)
        assert isinstance(data["additional_skills"], list)
        
    async def test_score_calculation(self) -> None:
        """Test score calculation logic."""
        # Test exact skill matches
        job_skills = ["Python", "AWS", "Docker"]
        candidate_skills = ["Python", "AWS", "Docker"]
        
        scores = await self.tool._calculate_match_scores(job_skills, candidate_skills)
        assert max(scores.values()) == 1.0
        
        # Test partial matches
        candidate_skills = ["Python", "AWS"]
        scores = await self.tool._calculate_match_scores(job_skills, candidate_skills)
        assert max(scores.values()) >= 0.6  # Account for semantic matching
        
        # Test no matches
        candidate_skills = ["Java", "React"]
        scores = await self.tool._calculate_match_scores(job_skills, candidate_skills)
        assert max(scores.values()) == 0.0
        
    async def test_semantic_matching(self) -> None:
        """Test semantic matching capabilities."""
        # Add test data with semantic variations
        await self.store.add_job(
            job_id="789",
            title="ML Engineer",
            description="AI/ML role",
            skills=["Python", "Machine Learning", "Deep Learning"]
        )
        
        await self.store.add_candidate(
            resume_id="101",
            name="Jane Smith",
            skills=["Python", "ML", "Neural Networks"],
            experience=[{"title": "ML Engineer", "duration": "2 years"}]
        )
        
        result = await self.tool._arun(
            job_id="789",
            resume_id="101",
            analysis_type="match"
        )
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["data"]["match_score"] > 60  # Should recognize semantic similarities
        
        # Cleanup
        await self.store.delete_job("789")
        await self.store.delete_candidate("101")
        
    async def test_error_handling(self) -> None:
        """Test error handling scenarios."""
        # Test with missing required fields
        result = await self.tool._arun()
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
        assert "error" in result_dict
        
        # Test with invalid job/resume IDs
        result = await self.tool._arun(
            job_id="invalid",
            resume_id="invalid",
            analysis_type="match"
        )
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
        assert "error" in result_dict 