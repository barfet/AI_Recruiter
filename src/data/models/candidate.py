from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr


class Education(BaseModel):
    """Education entry model"""

    degree: str
    institution: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class Experience(BaseModel):
    """Work experience entry model"""

    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""


class CandidateProfile(BaseModel):
    """Candidate profile model"""

    resume_id: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    industry: str
    name: str = ""  # Default to empty string if not found
    summary: str = ""  # Default to empty string if not found
    languages: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    desired_role: Optional[str] = None
    desired_salary: Optional[float] = None
    desired_location: Optional[str] = None
    remote_preference: Optional[bool] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
