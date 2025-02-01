from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json

from src.core.logging import setup_logger
from src.embeddings.manager import EmbeddingManager

logger = setup_logger(__name__)


class SearchJobsInput(BaseModel):
    """Input for the SearchJobsTool"""

    query: str = Field(..., description="The search query for finding jobs")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters to apply to the search results"
    )


class SearchCandidatesInput(BaseModel):
    """Input for the SearchCandidatesTool"""

    query: str = Field(..., description="The search query for finding candidates")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters to apply to the search results"
    )


class MatchJobCandidatesInput(BaseModel):
    """Input for the MatchJobCandidatesTool"""

    job_id: str = Field(..., description="The ID of the job to match candidates with")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters to apply to the search results"
    )


class SearchJobsTool(BaseTool):
    """Tool for searching job postings"""

    name: str = "search_jobs"
    description: str = (
        "Search for job postings using semantic search. "
        "Input should be a description of the job you're looking for."
    )

    def __init__(self):
        super().__init__()
        self.embedding_manager = EmbeddingManager()
        self.vectorstore = self.embedding_manager.load_embeddings("jobs")

    def _run(self, query: str) -> str:
        """Run job search"""
        try:
            results = self.embedding_manager.similarity_search(
                self.vectorstore, query, k=5
            )
            return json.dumps(
                [
                    {
                        "job_id": r["metadata"]["job_id"],
                        "title": r["metadata"]["title"],
                        "company": r["metadata"]["company"],
                        "location": r["metadata"]["location"],
                        "score": r["score"],
                    }
                    for r in results
                ],
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)


class SearchCandidatesTool(BaseTool):
    """Tool for searching candidate profiles"""

    name: str = "search_candidates"
    description: str = (
        "Search for candidate profiles using semantic search. "
        "Input should be a description of the candidate you're looking for."
    )

    def __init__(self):
        super().__init__()
        self.embedding_manager = EmbeddingManager()
        self.vectorstore = self.embedding_manager.load_embeddings("candidates")

    def _run(self, query: str) -> str:
        """Run candidate search"""
        try:
            results = self.embedding_manager.similarity_search(
                self.vectorstore, query, k=5
            )
            return json.dumps(
                [
                    {
                        "resume_id": r["metadata"]["resume_id"],
                        "name": r["metadata"].get("name", "Anonymous"),
                        "industry": r["metadata"]["industry"],
                        "skills": r["metadata"].get("skills", []),
                        "score": r["score"],
                    }
                    for r in results
                ],
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error searching candidates: {str(e)}")
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)


class MatchJobCandidatesTool(BaseTool):
    """Tool for matching jobs with candidates"""

    name: str = "match_job_candidates"
    description: str = (
        "Match a job posting with potential candidates or vice versa. "
        "Input should be a job ID or resume ID."
    )

    def __init__(self):
        super().__init__()
        self.embedding_manager = EmbeddingManager()
        self.jobs_store = self.embedding_manager.load_embeddings("jobs")
        self.candidates_store = self.embedding_manager.load_embeddings("candidates")

    def _run(self, query: str) -> str:
        """Run matching"""
        try:
            # Parse input to determine if it's a job ID or resume ID
            input_type = "job" if query.startswith("job_") else "resume"

            if input_type == "job":
                # Find candidates for a job
                job_id = query
                job = next(
                    (j for j in self.jobs_store if j["metadata"]["job_id"] == job_id),
                    None,
                )
                if not job:
                    return f"Job {job_id} not found"

                # Use job description to find matching candidates
                results = self.embedding_manager.similarity_search(
                    self.candidates_store, job["document"], k=5
                )
                return json.dumps(
                    {
                        "job": {
                            "job_id": job_id,
                            "title": job["metadata"]["title"],
                            "company": job["metadata"]["company"],
                        },
                        "matching_candidates": [
                            {
                                "resume_id": r["metadata"]["resume_id"],
                                "name": r["metadata"].get("name", "Anonymous"),
                                "skills": r["metadata"].get("skills", []),
                                "score": r["score"],
                            }
                            for r in results
                        ],
                    },
                    indent=2,
                )
            else:
                # Find jobs for a candidate
                resume_id = query
                candidate = next(
                    (
                        c
                        for c in self.candidates_store
                        if c["metadata"]["resume_id"] == resume_id
                    ),
                    None,
                )
                if not candidate:
                    return f"Candidate {resume_id} not found"

                # Use candidate profile to find matching jobs
                results = self.embedding_manager.similarity_search(
                    self.jobs_store, candidate["document"], k=5
                )
                return json.dumps(
                    {
                        "candidate": {
                            "resume_id": resume_id,
                            "name": candidate["metadata"].get("name", "Anonymous"),
                            "skills": candidate["metadata"].get("skills", []),
                        },
                        "matching_jobs": [
                            {
                                "job_id": r["metadata"]["job_id"],
                                "title": r["metadata"]["title"],
                                "company": r["metadata"]["company"],
                                "score": r["score"],
                            }
                            for r in results
                        ],
                    },
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error matching: {str(e)}")
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)
