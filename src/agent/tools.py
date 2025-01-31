from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from src.core.logging import setup_logger
from src.search.search_manager import SearchManager

logger = setup_logger(__name__)

class SearchJobsInput(BaseModel):
    """Input for the SearchJobsTool"""
    query: str = Field(..., description="The search query for finding jobs")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to the search results"
    )

class SearchCandidatesInput(BaseModel):
    """Input for the SearchCandidatesTool"""
    query: str = Field(..., description="The search query for finding candidates")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to the search results"
    )

class MatchJobCandidatesInput(BaseModel):
    """Input for the MatchJobCandidatesTool"""
    job_id: str = Field(..., description="The ID of the job to match candidates with")
    k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to the search results"
    )

class SearchJobsTool(BaseTool):
    """Tool for searching job postings"""
    
    name = "search_jobs"
    description = "Search for job postings based on a query"
    args_schema = SearchJobsInput
    
    def __init__(self):
        super().__init__()
        self.search_manager = SearchManager()
        
    def _run(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        """Run the tool"""
        try:
            results = self.search_manager.search_jobs(query, k=k, filters=filters)
            return self._format_job_results(results)
        except Exception as e:
            logger.error(f"Error in SearchJobsTool: {str(e)}")
            return f"Error searching jobs: {str(e)}"
            
    def _format_job_results(self, results: List[Dict[str, Any]]) -> str:
        """Format job search results"""
        if not results:
            return "No matching jobs found."
            
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_job = f"""
            Job ID: {metadata['job_id']}
            Title: {metadata['title']}
            Company: {metadata['company']}
            Location: {metadata['location']}
            Skills: {', '.join(metadata['skills'])}
            Experience Level: {metadata['experience_level'] or 'Not specified'}
            Remote: {'Yes' if metadata['remote_allowed'] else 'No'}
            Similarity Score: {result['score']:.2f}
            """
            formatted_results.append(formatted_job)
            
        return "\n".join(formatted_results)

class SearchCandidatesTool(BaseTool):
    """Tool for searching candidate profiles"""
    
    name = "search_candidates"
    description = "Search for candidates based on a query"
    args_schema = SearchCandidatesInput
    
    def __init__(self):
        super().__init__()
        self.search_manager = SearchManager()
        
    def _run(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        """Run the tool"""
        try:
            results = self.search_manager.search_candidates(query, k=k, filters=filters)
            return self._format_candidate_results(results)
        except Exception as e:
            logger.error(f"Error in SearchCandidatesTool: {str(e)}")
            return f"Error searching candidates: {str(e)}"
            
    def _format_candidate_results(self, results: List[Dict[str, Any]]) -> str:
        """Format candidate search results"""
        if not results:
            return "No matching candidates found."
            
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_candidate = f"""
            Candidate ID: {metadata['candidate_id']}
            Name: {metadata['name']}
            Location: {metadata['location'] or 'Not specified'}
            Skills: {', '.join(metadata['skills'])}
            Experience: {len(metadata['experience'])} positions
            Education: {len(metadata['education'])} degrees
            Languages: {', '.join(metadata['languages'])}
            Desired Role: {metadata['desired_role'] or 'Not specified'}
            Similarity Score: {result['score']:.2f}
            """
            formatted_results.append(formatted_candidate)
            
        return "\n".join(formatted_results)

class MatchJobCandidatesTool(BaseTool):
    """Tool for matching candidates to a specific job"""
    
    name = "match_job_candidates"
    description = "Find candidates that match a specific job posting"
    args_schema = MatchJobCandidatesInput
    
    def __init__(self):
        super().__init__()
        self.search_manager = SearchManager()
        
    def _run(self, job_id: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        """Run the tool"""
        try:
            results = self.search_manager.match_job_candidates(job_id, k=k, filters=filters)
            return self._format_match_results(results)
        except Exception as e:
            logger.error(f"Error in MatchJobCandidatesTool: {str(e)}")
            return f"Error matching candidates: {str(e)}"
            
    def _format_match_results(self, results: List[Dict[str, Any]]) -> str:
        """Format match results"""
        if not results:
            return "No matching candidates found."
            
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_match = f"""
            Candidate ID: {metadata['candidate_id']}
            Name: {metadata['name']}
            Location: {metadata['location'] or 'Not specified'}
            Skills: {', '.join(metadata['skills'])}
            Experience: {len(metadata['experience'])} positions
            Education: {len(metadata['education'])} degrees
            Match Score: {result['score']:.2f}
            """
            formatted_results.append(formatted_match)
            
        return "\n".join(formatted_results) 