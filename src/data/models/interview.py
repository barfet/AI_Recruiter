from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

class InterviewQuestion(BaseModel):
    """Model representing a single interview question."""
    
    text: str = Field(..., description="The actual question text")
    difficulty: str = Field(..., description="Question difficulty level")
    expected_signals: List[str] = Field(
        ..., 
        description="Signals/indicators to look for in the answer"
    )

    @validator("difficulty")
    def validate_difficulty(cls, v):
        if v not in ["easy", "medium", "hard"]:
            raise ValueError("Difficulty must be one of: easy, medium, hard")
        return v

class ResponseEvaluation(BaseModel):
    """Model representing the evaluation of a candidate's response."""
    
    score: float = Field(..., description="Overall score for the response", ge=0, le=100)
    strengths: List[str] = Field(..., description="Key strengths demonstrated")
    improvements: List[str] = Field(..., description="Areas for improvement")
    technical_accuracy: float = Field(..., description="Technical accuracy score", ge=0, le=100)
    communication_score: float = Field(..., description="Communication effectiveness score", ge=0, le=100)
    notes: Optional[str] = Field(None, description="Additional evaluation notes")

class InterviewFeedback(BaseModel):
    """Model representing comprehensive interview feedback."""
    
    overall_score: float = Field(..., description="Overall interview score", ge=0, le=100)
    technical_score: float = Field(..., description="Technical competency score", ge=0, le=100)
    communication_score: float = Field(..., description="Communication effectiveness score", ge=0, le=100)
    strengths: List[str] = Field(..., description="Key strengths demonstrated")
    improvements: List[str] = Field(..., description="Areas for improvement")
    recommendation: str = Field(..., description="Clear hiring recommendation")
    notes: Optional[str] = Field(None, description="Additional feedback notes")

class InterviewPhase(BaseModel):
    """Model representing a single phase of the interview."""
    
    priority: int = Field(..., description="Priority order of this phase")
    skill_focus: str = Field(..., description="The skill being assessed in this phase")
    reason: str = Field(..., description="Reason for including this phase")
    time_allocation: int = Field(
        ..., 
        description="Time allocated for this phase in minutes",
        ge=10
    )
    questions: List[InterviewQuestion] = Field(
        ..., 
        description="List of questions for this phase"
    )

class InterviewStrategy(BaseModel):
    """Model representing the complete interview strategy."""
    
    total_time: int = Field(
        ..., 
        description="Total interview duration in minutes",
        ge=30
    )
    phases: List[InterviewPhase] = Field(
        ..., 
        description="Ordered list of interview phases"
    )
    notes: Optional[str] = Field(
        None, 
        description="Additional notes or guidance for the interviewer"
    )

    @validator("phases")
    def validate_phases(cls, v, values):
        if not v:
            raise ValueError("Strategy must have at least one phase")
            
        # Validate total time matches sum of phase times
        if "total_time" in values:
            total_phase_time = sum(phase.time_allocation for phase in v)
            if total_phase_time != values["total_time"]:
                raise ValueError(
                    f"Sum of phase times ({total_phase_time}) must equal total time ({values['total_time']})"
                )
            
        # Validate priorities are sequential
        priorities = [phase.priority for phase in v]
        if priorities != list(range(1, len(v) + 1)):
            raise ValueError("Phase priorities must be sequential starting from 1")
            
        return v 