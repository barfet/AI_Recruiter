"""Embedding manager for handling text-to-vector transformations."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.core.config import get_config
from src.data.embeddings.factory import get_embedding_model
from src.data.vector_store.factory import get_vector_store


logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manager class for handling embeddings and vector storage."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.config = get_config()
        self.model = get_embedding_model()
        self.vector_store = get_vector_store()
    
    def create_job_embeddings(self, jobs_file: Union[str, Path]) -> None:
        """Create and store embeddings for job postings.
        
        Args:
            jobs_file: Path to JSON file containing job data
            
        Raises:
            FileNotFoundError: If jobs file doesn't exist
            ValueError: If jobs file is invalid
            EmbeddingError: If embedding creation fails
        """
        try:
            jobs_path = Path(jobs_file)
            if not jobs_path.exists():
                raise FileNotFoundError(f"Jobs file not found: {jobs_file}")
            
            with open(jobs_path, "r") as f:
                jobs = json.load(f)
            
            logger.info(f"Processing {len(jobs)} jobs for embedding creation")
            
            # Prepare texts for batch embedding
            texts = []
            for job in jobs:
                text = self._prepare_job_text(job)
                texts.append(text)
            
            # Create embeddings in batch
            embeddings = self.model.embed_texts(texts)
            
            # Store in vector store
            for job, text, embedding in zip(jobs, texts, embeddings):
                try:
                    self.vector_store.add_job(
                        job_id=job["id"],
                        embedding=embedding,
                        metadata=job,
                        text=text
                    )
                except Exception as e:
                    logger.error(f"Failed to store job {job['id']}: {str(e)}")
                    continue
            
            # Persist changes
            self.vector_store.persist()
            logger.info("Successfully created and stored job embeddings")
            
        except Exception as e:
            logger.error(f"Failed to create job embeddings: {str(e)}")
            raise
    
    def create_candidate_embeddings(self, candidates_file: Union[str, Path]) -> None:
        """Create and store embeddings for candidate profiles.
        
        Args:
            candidates_file: Path to JSON file containing candidate data
            
        Raises:
            FileNotFoundError: If candidates file doesn't exist
            ValueError: If candidates file is invalid
            EmbeddingError: If embedding creation fails
        """
        try:
            candidates_path = Path(candidates_file)
            if not candidates_path.exists():
                raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
            
            with open(candidates_path, "r") as f:
                candidates = json.load(f)
            
            logger.info(f"Processing {len(candidates)} candidates for embedding creation")
            
            # Prepare texts for batch embedding
            texts = []
            for candidate in candidates:
                text = self._prepare_candidate_text(candidate)
                texts.append(text)
            
            # Create embeddings in batch
            embeddings = self.model.embed_texts(texts)
            
            # Store in vector store
            for candidate, text, embedding in zip(candidates, texts, embeddings):
                try:
                    self.vector_store.add_candidate(
                        candidate_id=candidate["id"],
                        embedding=embedding,
                        metadata=candidate,
                        text=text
                    )
                except Exception as e:
                    logger.error(f"Failed to store candidate {candidate['id']}: {str(e)}")
                    continue
            
            # Persist changes
            self.vector_store.persist()
            logger.info("Successfully created and stored candidate embeddings")
            
        except Exception as e:
            logger.error(f"Failed to create candidate embeddings: {str(e)}")
            raise
    
    def search_similar_jobs(
        self,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for jobs similar to the query text.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar jobs with scores and metadata
            
        Raises:
            EmbeddingError: If search fails
        """
        try:
            # Create query embedding
            query_embedding = self.model.embed_text(query_text)
            
            # Search in vector store
            return self.vector_store.search_jobs(
                query_embedding=query_embedding,
                k=k,
                filter_metadata=filter_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to search similar jobs: {str(e)}")
            raise
    
    def search_similar_candidates(
        self,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for candidates similar to the query text.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar candidates with scores and metadata
            
        Raises:
            EmbeddingError: If search fails
        """
        try:
            # Create query embedding
            query_embedding = self.model.embed_text(query_text)
            
            # Search in vector store
            return self.vector_store.search_candidates(
                query_embedding=query_embedding,
                k=k,
                filter_metadata=filter_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to search similar candidates: {str(e)}")
            raise
    
    def _prepare_job_text(self, job: Dict[str, Any]) -> str:
        """Prepare job data for embedding.
        
        Args:
            job: Job posting data
            
        Returns:
            str: Formatted text for embedding
        """
        parts = []
        
        if "title" in job:
            parts.append(f"Title: {job['title']}")
        if "description" in job:
            parts.append(f"Description: {job['description']}")
        if "requirements" in job:
            if isinstance(job["requirements"], list):
                parts.append("Requirements: " + ". ".join(job["requirements"]))
            else:
                parts.append(f"Requirements: {job['requirements']}")
        if "skills" in job:
            if isinstance(job["skills"], list):
                parts.append("Skills: " + ", ".join(job["skills"]))
            else:
                parts.append(f"Skills: {job['skills']}")
        if "location" in job:
            parts.append(f"Location: {job['location']}")
        
        return "\n".join(parts)
    
    def _prepare_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Prepare candidate data for embedding.
        
        Args:
            candidate: Candidate profile data
            
        Returns:
            str: Formatted text for embedding
        """
        parts = []
        
        if "summary" in candidate:
            parts.append(f"Summary: {candidate['summary']}")
        if "experience" in candidate:
            if isinstance(candidate["experience"], list):
                experiences = []
                for exp in candidate["experience"]:
                    if isinstance(exp, dict):
                        exp_parts = []
                        if "title" in exp:
                            exp_parts.append(exp["title"])
                        if "company" in exp:
                            exp_parts.append(f"at {exp['company']}")
                        if "description" in exp:
                            exp_parts.append(exp["description"])
                        experiences.append(" ".join(exp_parts))
                    else:
                        experiences.append(str(exp))
                parts.append("Experience: " + ". ".join(experiences))
            else:
                parts.append(f"Experience: {candidate['experience']}")
        if "skills" in candidate:
            if isinstance(candidate["skills"], list):
                parts.append("Skills: " + ", ".join(candidate["skills"]))
            else:
                parts.append(f"Skills: {candidate['skills']}")
        if "education" in candidate:
            if isinstance(candidate["education"], list):
                education = []
                for edu in candidate["education"]:
                    if isinstance(edu, dict):
                        edu_parts = []
                        if "degree" in edu:
                            edu_parts.append(edu["degree"])
                        if "institution" in edu:
                            edu_parts.append(f"from {edu['institution']}")
                        education.append(" ".join(edu_parts))
                    else:
                        education.append(str(edu))
                parts.append("Education: " + ". ".join(education))
            else:
                parts.append(f"Education: {candidate['education']}")
        
        return "\n".join(parts)
