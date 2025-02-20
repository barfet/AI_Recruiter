"""Base interface for vector stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

class VectorStoreInterface(ABC):
    """Abstract base class for vector stores.
    
    This interface defines the common operations that all vector store implementations
    must support, ensuring consistency across different backends (FAISS, Chroma, etc.).
    """
    
    @abstractmethod
    def add_job(
        self, 
        job_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        text: str
    ) -> None:
        """Add a job posting to the vector store.
        
        Args:
            job_id: Unique identifier for the job
            embedding: Vector representation of the job
            metadata: Additional job information
            text: Original text representation
            
        Raises:
            VectorStoreError: If addition fails
        """
        pass
    
    @abstractmethod
    def add_candidate(
        self,
        candidate_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        text: str
    ) -> None:
        """Add a candidate profile to the vector store.
        
        Args:
            candidate_id: Unique identifier for the candidate
            embedding: Vector representation of the candidate
            metadata: Additional candidate information
            text: Original text representation
            
        Raises:
            VectorStoreError: If addition fails
        """
        pass
    
    @abstractmethod
    def search_jobs(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar jobs.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching jobs with scores and metadata
            
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abstractmethod
    def search_candidates(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar candidates.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching candidates with scores and metadata
            
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abstractmethod
    def update_job(
        self,
        job_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> None:
        """Update an existing job posting.
        
        Args:
            job_id: Job identifier to update
            embedding: New vector representation (optional)
            metadata: New metadata (optional)
            text: New text representation (optional)
            
        Raises:
            VectorStoreError: If update fails
            KeyError: If job_id not found
        """
        pass
    
    @abstractmethod
    def update_candidate(
        self,
        candidate_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> None:
        """Update an existing candidate profile.
        
        Args:
            candidate_id: Candidate identifier to update
            embedding: New vector representation (optional)
            metadata: New metadata (optional)
            text: New text representation (optional)
            
        Raises:
            VectorStoreError: If update fails
            KeyError: If candidate_id not found
        """
        pass
    
    @abstractmethod
    def delete_job(self, job_id: str) -> None:
        """Delete a job posting.
        
        Args:
            job_id: Job identifier to delete
            
        Raises:
            VectorStoreError: If deletion fails
            KeyError: If job_id not found
        """
        pass
    
    @abstractmethod
    def delete_candidate(self, candidate_id: str) -> None:
        """Delete a candidate profile.
        
        Args:
            candidate_id: Candidate identifier to delete
            
        Raises:
            VectorStoreError: If deletion fails
            KeyError: If candidate_id not found
        """
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk.
        
        Raises:
            VectorStoreError: If persistence fails
        """
        pass

class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass 