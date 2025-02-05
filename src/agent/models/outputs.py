"""Output models for recruiting tools."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class StandardizedOutput(BaseModel):
    """Standardized output format for all tools."""
    
    status: str = Field(..., description="Status of the operation (success/error)")
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = Field(None, description="Output data")
    error: Optional[str] = Field(None, description="Error message if status is error")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)


class SkillMatchOutput(BaseModel):
    """Output format for skill matching operations."""
    
    match_score: float = Field(..., description="Overall match score (0-100)")
    exact_matches: List[str] = Field(default_factory=list, description="List of exact skill matches")
    semantic_matches: List[tuple[str, str, float]] = Field(
        default_factory=list, 
        description="List of semantic matches with scores"
    )
    missing_skills: List[str] = Field(default_factory=list, description="Required skills not found")
    additional_skills: List[str] = Field(default_factory=list, description="Extra skills not required")


class InterviewQuestionOutput(BaseModel):
    """Output format for interview questions."""
    
    question: str = Field(..., description="The interview question")
    difficulty: str = Field(..., description="Question difficulty (easy/medium/hard)")
    skill_focus: str = Field(..., description="Primary skill being assessed")
    expected_signals: List[str] = Field(
        default_factory=list, 
        description="Expected signals in a good answer"
    )
    follow_ups: List[str] = Field(
        default_factory=list, 
        description="Follow-up questions"
    )


class InterviewFeedbackOutput(BaseModel):
    """Output format for interview feedback."""
    
    overall_score: float = Field(..., description="Overall interview score (0-100)")
    technical_score: float = Field(..., description="Technical competency score (0-100)")
    communication_score: float = Field(..., description="Communication score (0-100)")
    strengths: List[str] = Field(default_factory=list, description="Key strengths demonstrated")
    improvements: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendation: str = Field(..., description="Hiring recommendation")
    notes: Optional[str] = Field(None, description="Additional notes") 