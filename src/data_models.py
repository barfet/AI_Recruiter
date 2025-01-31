from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class WorkExperience(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: List[str] = []  # Bullet points of responsibilities/achievements

class Education(BaseModel):
    degree: str
    institution: str
    location: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    major: Optional[str] = None
    minor: Optional[str] = None

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None

class CertificationOrAward(BaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

class JobPosting(BaseModel):
    job_id: str
    title: str
    description: str
    requirements: str
    company: str
    location: str
    benefits: List[str] = []
    industries: List[str] = []
    skills: List[str] = []
    salary: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    work_type: Optional[str] = None
    experience_level: Optional[str] = None
    remote_allowed: Optional[bool] = None

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    contact_info: ContactInfo
    summary: Optional[str] = None
    experience: List[WorkExperience]
    education: List[Education]
    skills: List[str]
    certifications: List[CertificationOrAward] = []
    awards: List[CertificationOrAward] = []
    languages: List[str] = []
    industry: str
    raw_text: str  # First 1000 chars of resume text for context

# Validation helper
def validate_json(data: dict, model: BaseModel):
    return model(**data) 