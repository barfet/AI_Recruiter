"""Search tools for the recruiting agent."""
from typing import Dict, Any, List, Optional
from pydantic import Field

from src.agent.tools.base import BaseRecruitingTool
from src.agent.models.inputs import JobSearchInput, CandidateSearchInput
from src.agent.models.outputs import StandardizedOutput
from src.embeddings.manager import EmbeddingManager
from src.vector_store.chroma_store import ChromaStore
from src.data.models.job import JobPosting
from src.data.models.candidate import CandidateProfile


class BaseSearchTool(BaseRecruitingTool):
    """Base class for search tools."""
    
    collection_name: str = Field(...)
    store: Optional[ChromaStore] = Field(default_factory=ChromaStore)
    embedding_manager: EmbeddingManager = Field(default_factory=EmbeddingManager)
    _vectorstore: Optional[Dict[str, Any]] = None
    
    async def _ensure_vectorstore(self):
        """Ensure vector store is loaded."""
        if not self._vectorstore:
            self._vectorstore = await self.embedding_manager.load_embeddings(self.collection_name)

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a search result."""
        score = 1 - (result["score"] / 2) if "score" in result else 0
        return {
            "score": float(score),
            **result.get("metadata", {})
        }


class JobSearchTool(BaseSearchTool):
    """Tool for searching jobs."""
    
    name: str = "search_jobs"
    description: str = "Search for jobs based on query"
    collection_name: str = Field(default="jobs")
    args_schema: type[JobSearchInput] = JobSearchInput
    
    async def _arun(
        self, 
        query: str,
        limit: int = 5,
        filters: Dict[str, str] = None,
        **kwargs
    ) -> str:
        """Run job search."""
        try:
            # Try direct ChromaDB search first
            if filters and "job_id" in filters:
                # Direct ID lookup
                job_id = filters["job_id"].replace(":", "")
                result = await self.store.get_job_by_id(job_id)
                if result:
                    results = [{
                        "metadata": result,
                        "score": 1.0
                    }]
                else:
                    results = []
            else:
                # Full text search
                results = await self.store.search_jobs(query, limit=limit)
            
            if not results:
                return StandardizedOutput(
                    status="success",
                    data={"message": "No matching jobs found"},
                    metadata={"query": query}
                ).to_json()
            
            # Format results
            formatted_results = [self._format_result(r) for r in results]
            
            return StandardizedOutput(
                status="success",
                data=formatted_results,
                metadata={
                    "query": query,
                    "total_results": len(formatted_results)
                }
            ).to_json()
            
        except Exception as e:
            self.logger.error(f"Error in job search: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json()


class CandidateSearchTool(BaseSearchTool):
    """Tool for searching candidates."""
    
    name: str = "search_candidates" 
    description: str = "Search for candidates based on query"
    collection_name: str = Field(default="candidates")
    args_schema: type[CandidateSearchInput] = CandidateSearchInput
    
    async def _arun(
        self, 
        query: str,
        limit: int = 5,
        filters: Dict[str, str] = None,
        skills: List[str] = None,
        **kwargs
    ) -> str:
        """Run candidate search."""
        try:
            await self._ensure_vectorstore()
            
            # Build search query with filters and skills
            search_parts = [query]
            if filters:
                search_parts.extend(f"{k}:{v}" for k, v in filters.items())
            if skills:
                search_parts.append(f"skills:{','.join(skills)}")
            
            full_query = " ".join(search_parts)

            # Perform search
            results = await self.embedding_manager.similarity_search(
                self._vectorstore,
                full_query,
                k=limit
            )
            
            if not results:
                return StandardizedOutput(
                    status="success",
                    data={"message": "No matching candidates found"},
                    metadata={"query": full_query}
                ).to_json()
            
            # Format results
            formatted_results = [self._format_result(r) for r in results]
            
            return StandardizedOutput(
                status="success",
                data=formatted_results,
                metadata={
                    "query": full_query,
                    "total_results": len(formatted_results)
                }
            ).to_json()
            
        except Exception as e:
            self.logger.error(f"Error in candidate search: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json() 