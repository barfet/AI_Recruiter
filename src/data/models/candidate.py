from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

class Education(BaseModel):
    """Education information model"""
    institution: str
    degree: str
    field_of_study: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class Experience(BaseModel):
    """Work experience model"""
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str
    skills: List[str] = Field(default_factory=list)
    location: Optional[str] = None

class CandidateProfile(BaseModel):
    """Candidate profile model"""
    candidate_id: str
    name: str = ""  # Default to empty string if not found
    email: Optional[EmailStr] = None  # Make email optional
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: str
    skills: List[str] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    desired_role: Optional[str] = None
    desired_salary: Optional[float] = None
    desired_location: Optional[str] = None
    remote_preference: Optional[bool] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None 