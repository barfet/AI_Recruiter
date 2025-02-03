"""Integration tests for interview process."""

import pytest
from typing import List, Dict
from .base import BaseIntegrationTest

@pytest.mark.integration
class TestInterviewFlow(BaseIntegrationTest):
    """Test complete interview process flow."""

    async def test_question_generation(self) -> None:
        """Test interview question generation based on job requirements."""
        # Get a test job
        job = self.test_jobs[0]
        
        # Generate questions
        questions = await self.agent.generate_interview_questions(
            job_id=job["id"],
            question_types=["technical", "behavioral"]
        )
        
        assert len(questions) > 0
        assert all(q.get("type") in ["technical", "behavioral"] for q in questions)
        assert all(q.get("question") and q.get("expected_answer") for q in questions)

    async def test_response_evaluation(self) -> None:
        """Test candidate response evaluation."""
        # Get test data
        job = self.test_jobs[0]
        candidate = self.test_candidates[0]
        
        # Generate a question
        questions = await self.agent.generate_interview_questions(
            job_id=job["id"],
            question_types=["technical"]
        )
        question = questions[0]
        
        # Test response evaluation
        evaluation = await self.agent.evaluate_response(
            job_id=job["id"],
            question=question["question"],
            response="I would use AWS Lambda for serverless computing and Docker for containerization",
            expected_answer=question["expected_answer"]
        )
        
        # Verify evaluation structure
        assert isinstance(evaluation, dict)
        assert "score" in evaluation
        assert "strengths" in evaluation
        assert "improvements" in evaluation
        assert "technical_accuracy" in evaluation
        assert "communication" in evaluation
        
        # Verify score range
        assert 0 <= evaluation["score"] <= 100

    async def test_interview_feedback_generation(self) -> None:
        """Test comprehensive interview feedback generation."""
        # Simulate complete interview
        interview_data = {
            "job_id": self.test_jobs[0]["id"],
            "candidate_id": self.test_candidates[0]["id"],
            "responses": [
                {
                    "question": "Explain containerization",
                    "response": "Containerization is a lightweight alternative to VMs...",
                    "score": 85
                },
                {
                    "question": "Describe your experience with AWS",
                    "response": "I've worked extensively with Lambda and S3...",
                    "score": 90
                }
            ]
        }
        
        feedback = await self.agent.generate_interview_feedback(
            interview_data=interview_data
        )
        
        # Verify feedback structure
        assert "overall_score" in feedback
        assert "Key strengths" in feedback
        assert isinstance(feedback["Key strengths"], list)
        assert len(feedback["Key strengths"]) > 0
        assert "Areas for development" in feedback
        assert "Hiring recommendation" in feedback 