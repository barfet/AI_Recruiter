"""Tests for skill analysis functionality."""

import pytest
from src.agent.tools import SkillAnalysisTool
from tests.unit.base import BaseUnitTest
import json

class TestSkillAnalysisTool(BaseUnitTest):
    """Test suite for skill analysis tool."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        super().setup_method()
        self.tool = SkillAnalysisTool()
        self.test_skills = {
            "required": ["Python", "AWS", "Docker"],
            "candidate": ["Python", "AWS Lambda", "Kubernetes"]
        }
        
    async def test_skill_matching(self) -> None:
        """Test basic skill matching functionality."""
        result = await self.tool._arun({
            "required_skills": self.test_skills["required"],
            "candidate_skills": self.test_skills["candidate"]
        })
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"
        assert len(result_dict["matching_skills"]) > 0
        
    async def test_score_calculation(self) -> None:
        """Test skill score calculation."""
        # Test exact match
        exact_match = await self.tool._arun({
            "required_skills": ["Python"],
            "candidate_skills": ["Python"]
        })
        exact_match_dict = json.loads(exact_match)
        assert exact_match_dict["skill_match_score"] == 100.0
        
        # Test partial match
        partial_match = await self.tool._arun({
            "required_skills": ["Python", "AWS", "Docker"],
            "candidate_skills": ["Python"]
        })
        partial_match_dict = json.loads(partial_match)
        assert partial_match_dict["skill_match_score"] == 33.33333333333333
        
        # Test no match
        no_match = await self.tool._arun({
            "required_skills": ["Rust"],
            "candidate_skills": ["Python"]
        })
        no_match_dict = json.loads(no_match)
        assert no_match_dict["skill_match_score"] == 0.0
        
    async def test_semantic_matching(self) -> None:
        """Test semantic matching capabilities."""
        result = await self.tool._arun({
            "required_skills": ["Machine Learning"],
            "candidate_skills": ["Deep Learning", "Neural Networks", "AI"]
        })
        result_dict = json.loads(result)
        assert result_dict["semantic_match_score"] >= 60.0
        
    async def test_error_handling(self) -> None:
        """Test error handling for invalid inputs."""
        # Test empty skills
        result = await self.tool._arun({})
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
        assert "required_skills" in result_dict["error"]
            
        # Test invalid skill format
        result = await self.tool._arun({
            "required_skills": ["Python"],
            "candidate_skills": 123  # type: ignore
        })
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
        assert "invalid" in result_dict["error"].lower() 