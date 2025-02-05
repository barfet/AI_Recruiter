"""Input models for recruiting tools."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Base input for search operations."""
    
    query: str = Field(..., description="Search query string")
    limit: int = Field(default=5, description="Maximum number of results to return")
    filters: Dict[str, str] = Field(default_factory=dict, description="Search filters")


class JobSearchInput(SearchInput):
    """Input for job search operations."""
    
    location: Optional[str] = Field(None, description="Job location filter")
    experience_level: Optional[str] = Field(None, description="Required experience level")
    job_type: Optional[str] = Field(None, description="Job type (full-time, contract, etc)")


class CandidateSearchInput(SearchInput):
    """Input for candidate search operations."""
    
    skills: Optional[List[str]] = Field(None, description="Required skills")
    experience_years: Optional[int] = Field(None, description="Minimum years of experience")
    industry: Optional[str] = Field(None, description="Industry preference")


class SkillMatchInput(BaseModel):
    """Input for skill matching operations."""
    
    job_id: str = Field(..., description="ID of the job posting")
    resume_id: str = Field(..., description="ID of the candidate resume")
    required_skills: List[str] = Field(..., description="List of required skills")
    preferred_skills: Optional[List[str]] = Field(None, description="List of preferred skills")


class InterviewQuestionInput(BaseModel):
    """Input for interview question generation."""
    
    job_id: str = Field(..., description="ID of the job posting")
    skill_focus: str = Field(..., description="Skill to focus on")
    difficulty: str = Field(default="medium", description="Question difficulty level")
    question_count: int = Field(default=3, description="Number of questions to generate")
    question_type: str = Field(default="technical", description="Type of questions to generate")


class ResponseEvaluationInput(BaseModel):
    """Input for evaluating interview responses."""
    
    job_id: str = Field(..., description="ID of the job posting")
    resume_id: str = Field(..., description="ID of the candidate resume")
    question: str = Field(..., description="Interview question")
    response: str = Field(..., description="Candidate's response")
    skill_focus: str = Field(..., description="Skill being evaluated")


class InterviewFeedbackInput(BaseModel):
    """Input for generating interview feedback."""
    
    job_id: str = Field(..., description="ID of the job posting")
    resume_id: str = Field(..., description="ID of the candidate resume")
    responses: Dict[str, str] = Field(..., description="Dictionary of question-response pairs")
    evaluations: Dict[str, Dict] = Field(..., description="Dictionary of question-evaluation pairs")
    interview_notes: Optional[str] = Field(None, description="Additional interview notes") 