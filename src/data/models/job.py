from typing import List, Optional
from pydantic import BaseModel, Field


class EmployeeCount(BaseModel):
    """Employee count range for a company"""

    min: Optional[int] = None
    max: Optional[int] = None


class CompanyInfo(BaseModel):
    """Company information"""

    industries: List[str] = Field(default_factory=list)
    specialities: List[str] = Field(default_factory=list)
    employee_count: Optional[EmployeeCount] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None


class Salary(BaseModel):
    """Salary information"""

    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    currency: str = "USD"
    compensation_type: Optional[str] = None


class JobPosting(BaseModel):
    """Job posting model"""

    job_id: str
    title: str
    description: str
    requirements: str = ""
    company: str
    location: str
    benefits: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    salary: Optional[Salary] = None
    company_info: Optional[CompanyInfo] = None
    work_type: Optional[str] = None
    experience_level: Optional[str] = None
    remote_allowed: Optional[bool] = None
