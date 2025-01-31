from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os

from src.core.config import settings
from src.core.logging import setup_logger
from src.core.exceptions import EmbeddingError
from src.data.models.job import JobPosting
from src.data.models.candidate import CandidateProfile

logger = setup_logger(__name__)

class EmbeddingManager:
    """Manager for creating and managing embeddings"""
    
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )
        self.data_dir = settings.DATA_DIR
        self.index_dir = settings.INDEXES_DIR
        
    def _prepare_job_text(self, job: JobPosting) -> str:
        """Prepare job posting text for embedding"""
        return f"""
        Title: {job.title}
        Company: {job.company}
        Location: {job.location}
        Description: {job.description}
        Requirements: {job.requirements}
        Skills: {', '.join(job.skills)}
        Work Type: {job.work_type or 'Not specified'}
        Experience Level: {job.experience_level or 'Not specified'}
        Remote Allowed: {'Yes' if job.remote_allowed else 'No'}
        """
        
    def _prepare_candidate_text(self, candidate: CandidateProfile) -> str:
        """Prepare candidate profile text for embedding"""
        experience_text = "\n".join([
            f"- {exp.title} at {exp.company}: {exp.description}"
            for exp in candidate.experience
        ])
        
        education_text = "\n".join([
            f"- {edu.degree} in {edu.field_of_study} from {edu.institution}"
            for edu in candidate.education
        ])
        
        return f"""
        Name: {candidate.name}
        Location: {candidate.location or 'Not specified'}
        Summary: {candidate.summary}
        Skills: {', '.join(candidate.skills)}
        Experience:\n{experience_text}
        Education:\n{education_text}
        Languages: {', '.join(candidate.languages)}
        Desired Role: {candidate.desired_role or 'Not specified'}
        Desired Location: {candidate.desired_location or 'Not specified'}
        Remote Preference: {'Yes' if candidate.remote_preference else 'No'}
        """
        
    def create_job_embeddings(self, force: bool = False) -> FAISS:
        """Create embeddings for job postings"""
        try:
            index_path = self.index_dir / "jobs_index"
            if not force and index_path.exists():
                logger.info("Job embeddings already exist. Use force=True to recreate.")
                return self.load_embeddings("jobs")
            
            # Load processed job data
            with open(self.data_dir / "processed/structured_jobs.json", "r") as f:
                jobs_data = json.load(f)
            
            jobs = [JobPosting(**job) for job in jobs_data]
            texts = [self._prepare_job_text(job) for job in jobs]
            metadatas = [job.dict() for job in jobs]
            
            # Create FAISS index
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save index
            vectorstore.save_local(str(index_path))
            logger.info(f"Successfully created embeddings for {len(jobs)} job postings")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating job embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to create job embeddings: {str(e)}")
            
    def create_candidate_embeddings(self, force: bool = False) -> FAISS:
        """Create embeddings for candidate profiles"""
        try:
            index_path = self.index_dir / "candidates_index"
            if not force and index_path.exists():
                logger.info("Candidate embeddings already exist. Use force=True to recreate.")
                return self.load_embeddings("candidates")
            
            # Load processed candidate data
            with open(self.data_dir / "processed/structured_candidates.json", "r") as f:
                candidates_data = json.load(f)
            
            candidates = [CandidateProfile(**candidate) for candidate in candidates_data]
            texts = [self._prepare_candidate_text(candidate) for candidate in candidates]
            metadatas = [candidate.dict() for candidate in candidates]
            
            # Create FAISS index
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save index
            vectorstore.save_local(str(index_path))
            logger.info(f"Successfully created embeddings for {len(candidates)} candidate profiles")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating candidate embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to create candidate embeddings: {str(e)}")
            
    def load_embeddings(self, index_name: str) -> Optional[FAISS]:
        """Load existing embeddings from disk"""
        try:
            index_path = self.index_dir / f"{index_name}_index"
            if not index_path.exists():
                logger.warning(f"No existing index found at {index_path}")
                return None
                
            vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings
            )
            logger.info(f"Successfully loaded {index_name} embeddings")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to load embeddings: {str(e)}")
            
    def similarity_search(
        self,
        vectorstore: FAISS,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store"""
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "document": result[0].page_content,
                    "metadata": result[0].metadata,
                    "score": result[1]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise EmbeddingError(f"Failed to perform similarity search: {str(e)}") 