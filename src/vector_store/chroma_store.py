import os
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from src.core.logging import setup_logger

logger = setup_logger(__name__)

class ChromaStore:
    def __init__(self, persist_directory: str = ".chroma", is_persistent: bool = True):
        """Initialize ChromaStore with persistence directory."""
        self.persist_directory = persist_directory
        self.is_persistent = is_persistent
        self.embeddings = OpenAIEmbeddings()
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
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
            
        # Initialize collections with embedding function
        self.jobs_collection = self.client.get_or_create_collection(
            "jobs",
            embedding_function=self.embedding_function
        )
        self.candidates_collection = self.client.get_or_create_collection(
            "candidates",
            embedding_function=self.embedding_function
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        # Don't reset the client on exit to maintain data between sessions
        pass

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Convert metadata values to strings for ChromaDB storage."""
        prepared = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif key in ["skills", "experience", "education"]:
                if isinstance(value, (list, tuple)):
                    # Join with spaces for better parsing later
                    prepared[key] = " ".join(str(v).strip() for v in value if v is not None)
                else:
                    # Handle string input by splitting on commas and spaces
                    items = []
                    for item in str(value).replace(",", " ").split():
                        item = item.strip()
                        if item and item not in items:  # Avoid duplicates
                            items.append(item)
                    prepared[key] = " ".join(items)
            else:
                prepared[key] = str(value)
        return prepared

    def _parse_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        """Parse metadata from ChromaDB storage back to Python types."""
        parsed = {}
        for key, value in metadata.items():
            if not value:
                continue
            if key in ["skills", "experience", "education"]:
                # Split on both commas and spaces, then clean up
                items = []
                for item in value.replace(",", " ").split():
                    item = item.strip()
                    if item and item not in items:  # Avoid duplicates
                        items.append(item)
                parsed[key] = items
            elif key == "remote_allowed":
                # Parse boolean
                parsed[key] = value.lower() in ["true", "1", "yes"]
            else:
                parsed[key] = value
        return parsed

    async def add_job(self, job_id: str, job_data: Dict[str, Any]) -> None:
        """Add a job posting to the vector store."""
        try:
            # Prepare document text
            text_to_embed = f"""
            Title: {job_data.get('title', '')}
            Company: {job_data.get('company', '')}
            Description: {job_data.get('description', '')}
            Requirements: {job_data.get('requirements', '')}
            Skills: {', '.join(job_data.get('skills', []))}
            """
            
            # Prepare metadata
            metadata = self._prepare_metadata(job_data)
            
            # Add to collection
            self.jobs_collection.add(
                documents=[text_to_embed],
                metadatas=[metadata],
                ids=[str(job_id)]
            )
        except Exception as e:
            logger.error(f"Error adding job {job_id}: {str(e)}")
            raise
    
    async def add_candidate(self, candidate_id: str, candidate_data: Dict[str, Any]) -> None:
        """Add a candidate profile to the vector store."""
        try:
            # Prepare document text
            text_to_embed = f"""
            Experience: {' '.join(candidate_data.get('experience', []))}
            Skills: {' '.join(candidate_data.get('skills', []))}
            Education: {' '.join(candidate_data.get('education', []))}
            Industry: {candidate_data.get('industry', '')}
            """
            
            # Prepare metadata
            metadata = self._prepare_metadata(candidate_data)
            
            # Add to collection
            self.candidates_collection.add(
                documents=[text_to_embed],
                metadatas=[metadata],
                ids=[str(candidate_id)]
            )
        except Exception as e:
            logger.error(f"Error adding candidate {candidate_id}: {str(e)}")
            raise
    
    async def search_jobs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for jobs based on a query."""
        try:
            results = self.jobs_collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "distances", "documents"]
            )
            
            if not results["ids"][0]:
                return []
                
            return [
                {
                    "id": id,
                    "metadata": self._parse_metadata(metadata),
                    "score": score,
                    "content": doc
                }
                for id, metadata, score, doc in zip(
                    results["ids"][0], 
                    results["metadatas"][0],
                    results["distances"][0],
                    results["documents"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            return []
    
    async def search_candidates(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for candidates based on a query."""
        try:
            results = self.candidates_collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "distances", "documents"]
            )
            
            if not results["ids"][0]:
                return []
                
            return [
                {
                    "id": id,
                    "metadata": self._parse_metadata(metadata),
                    "score": score,
                    "content": doc
                }
                for id, metadata, score, doc in zip(
                    results["ids"][0], 
                    results["metadatas"][0],
                    results["distances"][0],
                    results["documents"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching candidates: {str(e)}")
            return []
    
    async def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by its ID."""
        try:
            result = self.jobs_collection.get(
                ids=[str(job_id)],
                include=["metadatas", "documents"]
            )
            if result["ids"]:
                return {
                    "id": job_id,
                    "metadata": self._parse_metadata(result["metadatas"][0]),
                    "content": result["documents"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {str(e)}")
            return None
        
    async def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Get a candidate by their ID."""
        try:
            result = self.candidates_collection.get(
                ids=[str(candidate_id)],
                include=["metadatas", "documents"]
            )
            if result["ids"]:
                return {
                    "id": candidate_id,
                    "metadata": self._parse_metadata(result["metadatas"][0]),
                    "content": result["documents"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting candidate {candidate_id}: {str(e)}")
            return None