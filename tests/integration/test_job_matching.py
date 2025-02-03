"""Tests for job matching functionality."""

import pytest
from typing import Dict
from src.services.job_discovery import JobDiscoveryService
from tests.integration.base import BaseIntegrationTest

@pytest.mark.integration
class TestJobMatching(BaseIntegrationTest):
    """Test suite for job matching functionality."""
    
    async def test_end_to_end_matching(self) -> None:
        """Test end-to-end job matching process."""
        # 1. Search for jobs
        jobs = await self.job_service.search_jobs(
            query="Python developer with AWS experience"
        )
        assert len(jobs) > 0
        job = jobs[0]
        
        # 2. Search for candidates
        candidates = await self.job_service.find_matching_candidates(
            job_id=job["id"]
        )
        assert len(candidates) > 0
        candidate = candidates[0]
        
        # 3. Get detailed match analysis
        analysis = await self.job_service.get_match_analysis(
            job_id=job["id"],
            candidate_id=candidate["id"]
        )
        
        # Verify analysis structure
        assert "match_analysis" in analysis
        assert "skill_match_score" in analysis["match_analysis"]
        assert "semantic_match_score" in analysis["match_analysis"]
        assert "combined_score" in analysis["match_analysis"]
        assert "matching_skills" in analysis["match_analysis"]
        assert "missing_skills" in analysis["match_analysis"]
        assert "additional_skills" in analysis["match_analysis"]
        
        # Verify scores are within expected range
        assert 0 <= analysis["match_analysis"]["skill_match_score"] <= 100
        assert 0 <= analysis["match_analysis"]["semantic_match_score"] <= 100
        assert 0 <= analysis["match_analysis"]["combined_score"] <= 100
        
    async def test_skill_based_filtering(self) -> None:
        """Test filtering candidates based on required skills."""
        # Add ML-specific test data
        ml_job = {
            "job_id": "ml_test",
            "title": "ML Engineer",
            "description": "Looking for an ML expert",
            "skills": ["Python", "Machine Learning", "Deep Learning", "PyTorch"]
        }
        await self.store.add_job(**ml_job)
        
        ml_candidate = {
            "resume_id": "ml_candidate",
            "name": "Jane ML",
            "skills": ["Python", "Machine Learning", "Neural Networks", "PyTorch"],
            "experience": [{"title": "ML Engineer", "duration": "2 years"}]
        }
        await self.store.add_candidate(**ml_candidate)
        
        try:
            # Find ML-specific jobs
            ml_jobs = await self.job_service.search_jobs(
                query="Machine Learning Engineer"
            )
            assert any("ML" in job["title"] for job in ml_jobs)
            
            # Get matching candidates for the specific ML job
            candidates = await self.job_service.find_matching_candidates(
                job_id="ml_test"
            )
            
            # Verify ML skills in matched candidates
            assert len(candidates) > 0
            for candidate in candidates:
                # Get all skills in lowercase for comparison
                candidate_skills = set(s.lower() for s in candidate["skills"])
                required_skills = ["pytorch", "ml", "deep learning", "machine learning", "ai"]
                
                # Check if any required skill matches (including semantic matches)
                has_matching_skill = False
                for skill in required_skills:
                    if skill in candidate_skills:
                        has_matching_skill = True
                        break
                    # Check for semantic matches
                    for candidate_skill in candidate_skills:
                        if self.job_service._are_skills_semantically_similar(skill, candidate_skill):
                            has_matching_skill = True
                            break
                    if has_matching_skill:
                        break
                
                assert has_matching_skill, f"No matching skills found in candidate skills: {candidate_skills}"
                
        finally:
            # Cleanup test data
            await self.store.delete_job("ml_test")
            await self.store.delete_candidate("ml_candidate")
        
    async def test_semantic_matching(self) -> None:
        """Test semantic matching capabilities."""
        # Search for candidates with semantic variations
        candidates = await self.job_service.find_matching_candidates(
            job_id=self.test_jobs[1]["id"]  # ML Engineer job
        )
        
        assert len(candidates) > 0
        top_candidate = candidates[0]
        
        # Verify semantic matching works
        assert top_candidate["match_score"] > 0.5
        assert len(top_candidate["matching_skills"]) > 0 