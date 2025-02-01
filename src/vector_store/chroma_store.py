import os
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class ChromaStore:
    def __init__(self, persist_directory: str = ".chroma", is_persistent: bool = True):
        """Initialize ChromaStore with persistence directory."""
        self.persist_directory = persist_directory
        self.is_persistent = is_persistent
        self.embeddings = OpenAIEmbeddings()
        self.client = None
        self.jobs_collection = None
        self.candidates_collection = None

    async def __aenter__(self):
        """Initialize async resources."""
        if self.is_persistent:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True)
            )
        else:
            self.client = chromadb.Client(Settings(
                is_persistent=False,
                allow_reset=True
            ))
            
        self.jobs_collection = self.client.get_or_create_collection("jobs")
        self.candidates_collection = self.client.get_or_create_collection("candidates")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if self.client:
            self.client.reset()

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any list values in metadata to strings."""
        return {
            key: ", ".join(value) if isinstance(value, list) else value
            for key, value in metadata.items()
        }

    async def add_job(self, job_id: str, job_data: Dict[str, Any]) -> None:
        """Add a job posting to the vector store."""
        text_to_embed = f"{job_data['title']} {job_data['description']} {job_data['requirements']}"
        metadata = self._prepare_metadata(job_data)
        self.jobs_collection.add(
            documents=[text_to_embed],
            metadatas=[metadata],
            ids=[job_id]
        )
    
    async def add_candidate(self, candidate_id: str, candidate_data: Dict[str, Any]) -> None:
        """Add a candidate profile to the vector store."""
        text_to_embed = f"{' '.join(candidate_data['experience'])} {' '.join(candidate_data['skills'])}"
        metadata = self._prepare_metadata(candidate_data)
        self.candidates_collection.add(
            documents=[text_to_embed],
            metadatas=[metadata],
            ids=[candidate_id]
        )
    
    async def search_jobs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for jobs based on a query."""
        results = self.jobs_collection.query(
            query_texts=[query],
            n_results=limit
        )
        return [
            {"id": id, "metadata": metadata, "score": score}
            for id, metadata, score in zip(
                results["ids"][0], 
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    async def search_candidates(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for candidates based on a query."""
        results = self.candidates_collection.query(
            query_texts=[query],
            n_results=limit
        )
        return [
            {"id": id, "metadata": metadata, "score": score}
            for id, metadata, score in zip(
                results["ids"][0], 
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    async def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by its ID."""
        try:
            result = self.jobs_collection.get(ids=[job_id])
            if result["ids"]:
                return {"id": job_id, "metadata": result["metadatas"][0]}
            return None
        except Exception:
            return None
        
    async def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Get a candidate by their ID."""
        try:
            result = self.candidates_collection.get(ids=[candidate_id])
            if result["ids"]:
                return {"id": candidate_id, "metadata": result["metadatas"][0]}
            return None
        except Exception:
            return None