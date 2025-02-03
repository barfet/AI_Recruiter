import os
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from src.core.logging import setup_logger
import json
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = setup_logger(__name__)

class ChromaStore:
    """Vector store implementation using ChromaDB."""
    
    def __init__(self, is_persistent: bool = True) -> None:
        """Initialize ChromaStore."""
        self.is_persistent = is_persistent
        self.client = None
        self.jobs_collection = None
        self.candidates_collection = None
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collections."""
        try:
            if self.is_persistent:
                self.client = chromadb.PersistentClient(path="./chroma_db")
            else:
                self.client = chromadb.Client()

            # Initialize embedding function
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Create or get collections
            self.jobs_collection = self.client.get_or_create_collection(
                name="jobs",
                embedding_function=embedding_func
            )
            
            self.candidates_collection = self.client.get_or_create_collection(
                name="candidates",
                embedding_function=embedding_func
            )

        except Exception as e:
            logger.error(f"Error initializing ChromaStore: {str(e)}")
            raise

    async def __aenter__(self) -> "ChromaStore":
        """Initialize collections on enter."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources on exit."""
        if self.client:
            self.client = None
            self.jobs_collection = None
            self.candidates_collection = None

    async def _batch_embed(self, documents: List[str]) -> List[List[float]]:
        """Batch embed documents using the embedding model."""
        try:
            embeddings = []
            for doc in documents:
                embedding = self.embedding_model.embed_query(doc)  # Not async
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error batch embedding documents: {str(e)}")
            raise
            
    async def add_job(
        self,
        job_id: str,
        title: str,
        description: str,
        skills: List[str],
        **kwargs: Any
    ) -> None:
        """Add a job to the vector store."""
        try:
            document = f"""
            Title: {title}
            Description: {description}
            Skills: {', '.join(skills)}
            """
            
            metadata = {
                "job_id": job_id,
                "title": title,
                "skills": json.dumps(skills),  # Store as JSON string
                **kwargs
            }
            
            self.jobs_collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[job_id]
            )
            logger.info(f"Successfully added job {job_id}")
        except Exception as e:
            logger.error(f"Error adding job {job_id}: {str(e)}")
            raise
            
    async def add_candidate(
        self,
        resume_id: str,
        name: str,
        skills: List[str],
        experience: List[Dict],
        **kwargs: Any
    ) -> None:
        """Add a candidate to the vector store."""
        try:
            document = f"""
            Experience: {json.dumps(experience)}
            Skills: {', '.join(skills)}
            Education: {kwargs.get('education', '')}
            Industry: {kwargs.get('industry', '')}
            """
            
            metadata = {
                "resume_id": resume_id,
                "name": name,
                "skills": json.dumps(skills),  # Store as JSON string
                "experience": json.dumps(experience),
                **{k: str(v) if isinstance(v, (list, dict)) else v for k, v in kwargs.items()}
            }
            
            self.candidates_collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[resume_id]
            )
            logger.info(f"Successfully added candidate {resume_id}")
        except Exception as e:
            logger.error(f"Error adding candidate {resume_id}: {str(e)}")
            raise
            
    async def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Get job by ID."""
        try:
            result = self.jobs_collection.get(
                ids=[job_id],
                include=["metadatas", "documents"]
            )
            
            if result["ids"]:
                metadata = result["metadatas"][0]
                # Convert skills JSON string back to list
                if "skills" in metadata:
                    try:
                        metadata["skills"] = json.loads(metadata["skills"])
                    except json.JSONDecodeError:
                        # Fallback for old format
                        metadata["skills"] = [s.strip() for s in metadata["skills"].split(",")]
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    **metadata
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {str(e)}")
            return None
            
    async def get_candidate_by_id(self, resume_id: str) -> Optional[Dict]:
        """Get candidate by ID."""
        try:
            result = self.candidates_collection.get(
                ids=[resume_id],
                include=["metadatas", "documents"]
            )
            
            if not result["ids"]:
                logger.error(f"No candidate found with ID {resume_id}")
                return None

            metadata = result["metadatas"][0]
            
            # Convert skills from JSON string to list
            if "skills" in metadata:
                try:
                    if isinstance(metadata["skills"], str):
                        try:
                            metadata["skills"] = json.loads(metadata["skills"])
                        except json.JSONDecodeError:
                            # Fallback for comma-separated string
                            metadata["skills"] = [s.strip() for s in metadata["skills"].split(",")]
                except Exception as e:
                    logger.error(f"Error parsing skills for candidate {resume_id}: {str(e)}")
                    metadata["skills"] = []

            # Convert experience from JSON string to list
            if "experience" in metadata:
                try:
                    if isinstance(metadata["experience"], str):
                        try:
                            metadata["experience"] = json.loads(metadata["experience"])
                        except json.JSONDecodeError:
                            metadata["experience"] = []
                except Exception as e:
                    logger.error(f"Error parsing experience for candidate {resume_id}: {str(e)}")
                    metadata["experience"] = []

            # Ensure skills is always a list
            if not isinstance(metadata.get("skills"), list):
                metadata["skills"] = []

            # Ensure experience is always a list
            if not isinstance(metadata.get("experience"), list):
                metadata["experience"] = []

            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": metadata,
                **metadata  # Also spread metadata for backward compatibility
            }
            
        except Exception as e:
            logger.error(f"Error getting candidate {resume_id}: {str(e)}")
            return None
            
    async def search_jobs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for jobs using semantic similarity."""
        try:
            results = self.jobs_collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            # Check if we have valid results
            if not results or not results.get("ids") or not results["ids"][0]:
                logger.info("No jobs found for query")
                return []
                
            jobs = []
            for i, job_id in enumerate(results["ids"][0]):
                try:
                    metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                    document = results["documents"][0][i] if results.get("documents") and results["documents"][0] else ""
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else None
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1 - (distance / 2) if distance is not None else 0
                    
                    # Ensure we have the job_id in the metadata
                    if "job_id" not in metadata:
                        metadata["job_id"] = job_id
                    
                    # Convert skills from JSON string to list if needed
                    if "skills" in metadata:
                        try:
                            if isinstance(metadata["skills"], str):
                                try:
                                    metadata["skills"] = json.loads(metadata["skills"])
                                except json.JSONDecodeError:
                                    metadata["skills"] = [s.strip() for s in metadata["skills"].split(",")]
                        except Exception as e:
                            logger.error(f"Error parsing skills for job {job_id}: {str(e)}")
                            metadata["skills"] = []
                    
                    jobs.append({
                        "id": job_id,
                        "job_id": metadata.get("job_id", job_id),
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "skills": metadata.get("skills", []),
                        "metadata": metadata,
                        "document": document,
                        "score": similarity
                    })
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {str(e)}")
                    continue
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            return []
            
    async def search_candidates(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for candidates using semantic similarity."""
        try:
            results = self.candidates_collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            # Check if we have valid results
            if not results or not results.get("ids") or not results["ids"][0]:
                logger.info("No candidates found for query")
                return []
                
            candidates = []
            for i, candidate_id in enumerate(results["ids"][0]):
                try:
                    metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                    document = results["documents"][0][i] if results.get("documents") and results["documents"][0] else ""
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else None
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1 - (distance / 2) if distance is not None else 0
                    
                    # Ensure we have the resume_id in the metadata
                    if "resume_id" not in metadata:
                        metadata["resume_id"] = candidate_id
                    
                    candidates.append({
                        "id": candidate_id,
                        "resume_id": metadata.get("resume_id", candidate_id),
                        "metadata": metadata,
                        "document": document,
                        "score": similarity
                    })
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_id}: {str(e)}")
                    continue
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error searching candidates: {str(e)}")
            return []

    async def delete_job(self, job_id: str) -> None:
        """Delete a job by ID."""
        try:
            self.jobs_collection.delete(
                ids=[job_id]
            )
            logger.info(f"Successfully deleted job {job_id}")
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {str(e)}")
            raise

    async def delete_candidate(self, resume_id: str) -> None:
        """Delete a candidate by ID."""
        try:
            self.candidates_collection.delete(
                ids=[resume_id]
            )
            logger.info(f"Successfully deleted candidate {resume_id}")
        except Exception as e:
            logger.error(f"Error deleting candidate {resume_id}: {str(e)}")
            raise