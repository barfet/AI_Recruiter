from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.services.job_discovery import JobDiscoveryService, JobMatch
from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager
from src.core.dependencies import get_chroma_store, get_job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])

class CandidateProfile(BaseModel):
    """Input model for candidate profile"""
    experience: List[str]
    skills: List[str]
    preferred_location: str = None
    preferred_remote: bool = None
    preferred_industries: List[str] = None

class DetailedAnalysis(BaseModel):
    """Detailed job match analysis from the agent chain"""
    candidate_summary: str
    job_analysis: str
    skills_gap_analysis: str
    interview_strategy: str

class JobMatchResponse(BaseModel):
    """Response model for job matches"""
    job_id: str
    title: str
    company: str
    location: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    relevance_explanation: str
    detailed_analysis: Optional[DetailedAnalysis] = None

    class Config:
        from_attributes = True

@router.post("/discover", response_model=List[JobMatchResponse])
async def discover_jobs(
    profile: CandidateProfile,
    limit: int = 10,
    store: ChromaStore = Depends(get_chroma_store),
    job_manager: JobManager = Depends(get_job_manager)
) -> List[JobMatchResponse]:
    """
    Discover matching jobs based on candidate profile.
    
    Returns a list of jobs ordered by match relevance, including:
    - Basic job details (title, company, location)
    - Match score and skill analysis
    - Matching and missing skills
    - Relevance explanation
    - Detailed analysis from agent chain including:
        - Candidate summary
        - Job analysis
        - Skills gap analysis
        - Interview strategy
    """
    try:
        service = JobDiscoveryService(store, job_manager)
        matches = await service.find_matching_jobs(
            candidate_profile=profile.dict(),
            limit=limit
        )
        return [JobMatchResponse.from_orm(match) for match in matches]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error discovering jobs: {str(e)}"
        ) 