"""Chroma implementation of the vector store interface."""

import os
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from src.core.logging import setup_logger
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from pathlib import Path
from src.core.config import get_config
from src.data.vector_store.base import VectorStoreInterface, VectorStoreError

logger = setup_logger(__name__)

class ChromaStore(VectorStoreInterface):
    """Chroma-based vector store implementation."""
    
    def __init__(self):
        """Initialize Chroma vector store."""
        self.config = get_config().vector_store
        self.embedding_dim = get_config().embedding.EMBEDDING_DIM
        
        # Initialize Chroma client
        persist_dir = Path(self.config.CHROMA_PERSIST_DIRECTORY)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get collections for jobs and candidates
        self.jobs_collection = self.client.get_or_create_collection(
            name=f"{self.config.CHROMA_COLLECTION_NAME}_jobs",
            metadata={"type": "jobs"}
        )
        
        self.candidates_collection = self.client.get_or_create_collection(
            name=f"{self.config.CHROMA_COLLECTION_NAME}_candidates",
            metadata={"type": "candidates"}
        )
    
    def add_job(
        self,
        job_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        text: str
    ) -> None:
        """Add a job posting to the vector store."""
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise VectorStoreError(
                    f"Invalid embedding dimension: {embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )
            
            # Add to Chroma
            self.jobs_collection.add(
                ids=[job_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                documents=[text]
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add job {job_id}: {str(e)}")
    
    def add_candidate(
        self,
        candidate_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        text: str
    ) -> None:
        """Add a candidate profile to the vector store."""
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise VectorStoreError(
                    f"Invalid embedding dimension: {embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )
            
            # Add to Chroma
            self.candidates_collection.add(
                ids=[candidate_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                documents=[text]
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add candidate {candidate_id}: {str(e)}")
    
    def search_jobs(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar jobs."""
        try:
            if query_embedding.shape[0] != self.embedding_dim:
                raise VectorStoreError(
                    f"Invalid query embedding dimension: {query_embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )
            
            # Convert filter_metadata to Chroma where clause if provided
            where = self._convert_filter_to_where(filter_metadata) if filter_metadata else None
            
            # Search in Chroma
            results = self.jobs_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "score": float(results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                    "text": results["documents"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search jobs: {str(e)}")
    
    def search_candidates(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar candidates."""
        try:
            if query_embedding.shape[0] != self.embedding_dim:
                raise VectorStoreError(
                    f"Invalid query embedding dimension: {query_embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )
            
            # Convert filter_metadata to Chroma where clause if provided
            where = self._convert_filter_to_where(filter_metadata) if filter_metadata else None
            
            # Search in Chroma
            results = self.candidates_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "score": float(results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                    "text": results["documents"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search candidates: {str(e)}")
    
    def update_job(
        self,
        job_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> None:
        """Update an existing job posting."""
        try:
            # Check if job exists
            existing = self.jobs_collection.get(
                ids=[job_id],
                include=["metadatas", "documents", "embeddings"]
            )
            if not existing["ids"]:
                raise KeyError(f"Job {job_id} not found")
            
            # Prepare update data
            update_data = {
                "ids": [job_id]
            }
            
            if embedding is not None:
                if embedding.shape[0] != self.embedding_dim:
                    raise VectorStoreError(
                        f"Invalid embedding dimension: {embedding.shape[0]}, "
                        f"expected {self.embedding_dim}"
                    )
                update_data["embeddings"] = [embedding.tolist()]
            
            if metadata is not None:
                new_metadata = existing["metadatas"][0].copy()
                new_metadata.update(metadata)
                update_data["metadatas"] = [new_metadata]
            
            if text is not None:
                update_data["documents"] = [text]
            
            # Update in Chroma
            self.jobs_collection.update(**update_data)
            
        except KeyError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to update job {job_id}: {str(e)}")
    
    def update_candidate(
        self,
        candidate_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> None:
        """Update an existing candidate profile."""
        try:
            # Check if candidate exists
            existing = self.candidates_collection.get(
                ids=[candidate_id],
                include=["metadatas", "documents", "embeddings"]
            )
            if not existing["ids"]:
                raise KeyError(f"Candidate {candidate_id} not found")
            
            # Prepare update data
            update_data = {
                "ids": [candidate_id]
            }
            
            if embedding is not None:
                if embedding.shape[0] != self.embedding_dim:
                    raise VectorStoreError(
                        f"Invalid embedding dimension: {embedding.shape[0]}, "
                        f"expected {self.embedding_dim}"
                    )
                update_data["embeddings"] = [embedding.tolist()]
            
            if metadata is not None:
                new_metadata = existing["metadatas"][0].copy()
                new_metadata.update(metadata)
                update_data["metadatas"] = [new_metadata]
            
            if text is not None:
                update_data["documents"] = [text]
            
            # Update in Chroma
            self.candidates_collection.update(**update_data)
            
        except KeyError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to update candidate {candidate_id}: {str(e)}")
    
    def delete_job(self, job_id: str) -> None:
        """Delete a job posting."""
        try:
            # Check if job exists
            existing = self.jobs_collection.get(
                ids=[job_id],
                include=["metadatas"]
            )
            if not existing["ids"]:
                raise KeyError(f"Job {job_id} not found")
            
            # Delete from Chroma
            self.jobs_collection.delete(ids=[job_id])
            
        except KeyError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to delete job {job_id}: {str(e)}")
    
    def delete_candidate(self, candidate_id: str) -> None:
        """Delete a candidate profile."""
        try:
            # Check if candidate exists
            existing = self.candidates_collection.get(
                ids=[candidate_id],
                include=["metadatas"]
            )
            if not existing["ids"]:
                raise KeyError(f"Candidate {candidate_id} not found")
            
            # Delete from Chroma
            self.candidates_collection.delete(ids=[candidate_id])
            
        except KeyError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to delete candidate {candidate_id}: {str(e)}")
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        # Chroma automatically persists changes, so this is a no-op
        pass
    
    def _convert_filter_to_where(self, filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filter metadata to Chroma where clause.
        
        Args:
            filter_metadata: Filter criteria
            
        Returns:
            Dict[str, Any]: Chroma where clause
        """
        where = {}
        for key, value in filter_metadata.items():
            if isinstance(value, (list, set)):
                # For list values, create an OR condition
                where[key] = {"$in": list(value)}
            else:
                # For scalar values, create an equality condition
                where[key] = {"$eq": value}
        return where