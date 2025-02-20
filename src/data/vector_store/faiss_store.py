"""FAISS implementation of the vector store interface."""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.vector_store.base import VectorStoreInterface, VectorStoreError
from src.core.config import get_config

class FAISSStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""
    
    def __init__(self):
        """Initialize FAISS vector store."""
        self.config = get_config().vector_store
        self.embedding_dim = get_config().embedding.EMBEDDING_DIM
        
        # Initialize separate indices for jobs and candidates
        self.job_index = self._create_index()
        self.candidate_index = self._create_index()
        
        # Store metadata separately since FAISS only stores vectors
        self.job_metadata: Dict[str, Dict[str, Any]] = {}
        self.candidate_metadata: Dict[str, Dict[str, Any]] = {}
        
        # ID mappings (string ID to FAISS internal ID)
        self.job_id_map: Dict[str, int] = {}
        self.candidate_id_map: Dict[str, int] = {}
        
        # Load existing data if available
        self._load_persisted_data()
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index based on configuration.
        
        Returns:
            faiss.Index: Initialized FAISS index
        """
        if self.config.FAISS_INDEX_TYPE == "Flat":
            if self.config.FAISS_METRIC_TYPE == "cosine":
                return faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            else:
                return faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
        else:
            raise VectorStoreError(f"Unsupported index type: {self.config.FAISS_INDEX_TYPE}")
    
    def add_job(
        self,
        job_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        text: str
    ) -> None:
        """Add a job posting to the vector store."""
        try:
            # Normalize for cosine similarity if needed
            if self.config.FAISS_METRIC_TYPE == "cosine":
                embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            faiss_id = len(self.job_id_map)
            self.job_index.add(embedding.reshape(1, -1))
            
            # Store metadata and mapping
            self.job_metadata[job_id] = {"metadata": metadata, "text": text}
            self.job_id_map[job_id] = faiss_id
            
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
            # Normalize for cosine similarity if needed
            if self.config.FAISS_METRIC_TYPE == "cosine":
                embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            faiss_id = len(self.candidate_id_map)
            self.candidate_index.add(embedding.reshape(1, -1))
            
            # Store metadata and mapping
            self.candidate_metadata[candidate_id] = {"metadata": metadata, "text": text}
            self.candidate_id_map[candidate_id] = faiss_id
            
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
            # Normalize query if using cosine similarity
            if self.config.FAISS_METRIC_TYPE == "cosine":
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS
            D, I = self.job_index.search(query_embedding.reshape(1, -1), k)
            
            # Convert results
            results = []
            reverse_map = {v: k for k, v in self.job_id_map.items()}
            
            for score, idx in zip(D[0], I[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                job_id = reverse_map[idx]
                job_data = self.job_metadata[job_id]
                
                # Apply metadata filtering
                if filter_metadata:
                    if not self._matches_filter(job_data["metadata"], filter_metadata):
                        continue
                
                results.append({
                    "id": job_id,
                    "score": float(score),
                    "metadata": job_data["metadata"],
                    "text": job_data["text"]
                })
            
            return results
            
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
            # Normalize query if using cosine similarity
            if self.config.FAISS_METRIC_TYPE == "cosine":
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS
            D, I = self.candidate_index.search(query_embedding.reshape(1, -1), k)
            
            # Convert results
            results = []
            reverse_map = {v: k for k, v in self.candidate_id_map.items()}
            
            for score, idx in zip(D[0], I[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                candidate_id = reverse_map[idx]
                candidate_data = self.candidate_metadata[candidate_id]
                
                # Apply metadata filtering
                if filter_metadata:
                    if not self._matches_filter(candidate_data["metadata"], filter_metadata):
                        continue
                
                results.append({
                    "id": candidate_id,
                    "score": float(score),
                    "metadata": candidate_data["metadata"],
                    "text": candidate_data["text"]
                })
            
            return results
            
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
        if job_id not in self.job_id_map:
            raise KeyError(f"Job {job_id} not found")
            
        try:
            # Update metadata/text if provided
            if metadata is not None or text is not None:
                current_data = self.job_metadata[job_id]
                if metadata is not None:
                    current_data["metadata"].update(metadata)
                if text is not None:
                    current_data["text"] = text
            
            # Update embedding if provided
            if embedding is not None:
                # For FAISS, we need to remove and re-add with new embedding
                old_faiss_id = self.job_id_map[job_id]
                new_faiss_id = len(self.job_id_map)
                
                # Add new embedding
                if self.config.FAISS_METRIC_TYPE == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                self.job_index.add(embedding.reshape(1, -1))
                
                # Update mapping
                self.job_id_map[job_id] = new_faiss_id
                
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
        if candidate_id not in self.candidate_id_map:
            raise KeyError(f"Candidate {candidate_id} not found")
            
        try:
            # Update metadata/text if provided
            if metadata is not None or text is not None:
                current_data = self.candidate_metadata[candidate_id]
                if metadata is not None:
                    current_data["metadata"].update(metadata)
                if text is not None:
                    current_data["text"] = text
            
            # Update embedding if provided
            if embedding is not None:
                # For FAISS, we need to remove and re-add with new embedding
                old_faiss_id = self.candidate_id_map[candidate_id]
                new_faiss_id = len(self.candidate_id_map)
                
                # Add new embedding
                if self.config.FAISS_METRIC_TYPE == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                self.candidate_index.add(embedding.reshape(1, -1))
                
                # Update mapping
                self.candidate_id_map[candidate_id] = new_faiss_id
                
        except Exception as e:
            raise VectorStoreError(f"Failed to update candidate {candidate_id}: {str(e)}")
    
    def delete_job(self, job_id: str) -> None:
        """Delete a job posting."""
        if job_id not in self.job_id_map:
            raise KeyError(f"Job {job_id} not found")
            
        try:
            # Remove from metadata and mapping
            del self.job_metadata[job_id]
            del self.job_id_map[job_id]
            
            # Note: FAISS doesn't support true deletion
            # The vector remains in the index but becomes inaccessible
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete job {job_id}: {str(e)}")
    
    def delete_candidate(self, candidate_id: str) -> None:
        """Delete a candidate profile."""
        if candidate_id not in self.candidate_id_map:
            raise KeyError(f"Candidate {candidate_id} not found")
            
        try:
            # Remove from metadata and mapping
            del self.candidate_metadata[candidate_id]
            del self.candidate_id_map[candidate_id]
            
            # Note: FAISS doesn't support true deletion
            # The vector remains in the index but becomes inaccessible
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete candidate {candidate_id}: {str(e)}")
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        try:
            persist_dir = Path(self.config.PERSIST_DIRECTORY)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS indices
            faiss.write_index(self.job_index, str(persist_dir / "jobs.index"))
            faiss.write_index(self.candidate_index, str(persist_dir / "candidates.index"))
            
            # Save metadata and mappings
            with open(persist_dir / "job_metadata.json", "w") as f:
                json.dump(self.job_metadata, f)
            with open(persist_dir / "candidate_metadata.json", "w") as f:
                json.dump(self.candidate_metadata, f)
            with open(persist_dir / "job_id_map.json", "w") as f:
                json.dump(self.job_id_map, f)
            with open(persist_dir / "candidate_id_map.json", "w") as f:
                json.dump(self.candidate_id_map, f)
                
        except Exception as e:
            raise VectorStoreError(f"Failed to persist vector store: {str(e)}")
    
    def _load_persisted_data(self) -> None:
        """Load persisted data if available."""
        persist_dir = Path(self.config.PERSIST_DIRECTORY)
        
        if not persist_dir.exists():
            return
            
        try:
            # Load FAISS indices if they exist
            job_index_path = persist_dir / "jobs.index"
            if job_index_path.exists():
                self.job_index = faiss.read_index(str(job_index_path))
                
            candidate_index_path = persist_dir / "candidates.index"
            if candidate_index_path.exists():
                self.candidate_index = faiss.read_index(str(candidate_index_path))
            
            # Load metadata and mappings
            if (persist_dir / "job_metadata.json").exists():
                with open(persist_dir / "job_metadata.json", "r") as f:
                    self.job_metadata = json.load(f)
            if (persist_dir / "candidate_metadata.json").exists():
                with open(persist_dir / "candidate_metadata.json", "r") as f:
                    self.candidate_metadata = json.load(f)
            if (persist_dir / "job_id_map.json").exists():
                with open(persist_dir / "job_id_map.json", "r") as f:
                    self.job_id_map = json.load(f)
            if (persist_dir / "candidate_id_map.json").exists():
                with open(persist_dir / "candidate_id_map.json", "r") as f:
                    self.candidate_id_map = json.load(f)
                    
        except Exception as e:
            raise VectorStoreError(f"Failed to load persisted data: {str(e)}")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria.
        
        Args:
            metadata: Item metadata
            filter_metadata: Filter criteria
            
        Returns:
            bool: True if metadata matches all filter criteria
        """
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, set)):
                if not isinstance(metadata[key], (list, set)):
                    return False
                if not set(value).issubset(set(metadata[key])):
                    return False
            elif metadata[key] != value:
                return False
        return True 