"""Tests for vector store implementations."""

import json
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any

from src.core.config import get_config, update_config, VectorStoreType
from src.data.vector_store.factory import create_vector_store, get_vector_store
from src.data.vector_store.base import VectorStoreError

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Test data fixture."""
    return {
        "jobs": [
            {
                "id": "job1",
                "title": "Software Engineer",
                "description": "Python developer position",
                "skills": ["Python", "Django", "PostgreSQL"],
                "location": "New York"
            },
            {
                "id": "job2",
                "title": "Data Scientist",
                "description": "ML engineer position",
                "skills": ["Python", "TensorFlow", "PyTorch"],
                "location": "San Francisco"
            }
        ],
        "candidates": [
            {
                "id": "candidate1",
                "summary": "Experienced Python developer",
                "skills": ["Python", "Django", "React"],
                "location": "New York"
            },
            {
                "id": "candidate2",
                "summary": "ML engineer with 5 years experience",
                "skills": ["Python", "TensorFlow", "PyTorch"],
                "location": "Remote"
            }
        ]
    }

@pytest.fixture
def test_embeddings() -> Dict[str, Any]:
    """Test embeddings fixture."""
    return {
        "jobs": {
            "job1": np.random.rand(384),  # Matches MODEL_NAME dimension
            "job2": np.random.rand(384)
        },
        "candidates": {
            "candidate1": np.random.rand(384),
            "candidate2": np.random.rand(384)
        }
    }

def test_faiss_store_creation():
    """Test FAISS store initialization."""
    update_config({"vector_store": {"STORE_TYPE": VectorStoreType.FAISS}})
    store = create_vector_store()
    assert store is not None
    assert store.__class__.__name__ == "FAISSStore"

def test_chroma_store_creation():
    """Test Chroma store initialization."""
    update_config({"vector_store": {"STORE_TYPE": VectorStoreType.CHROMA}})
    store = create_vector_store()
    assert store is not None
    assert store.__class__.__name__ == "ChromaStore"

def test_invalid_store_type():
    """Test invalid store type handling."""
    with pytest.raises(ValueError):
        create_vector_store("invalid_type")

def test_singleton_pattern():
    """Test vector store singleton pattern."""
    store1 = get_vector_store()
    store2 = get_vector_store()
    assert store1 is store2

@pytest.mark.parametrize("store_type", [VectorStoreType.FAISS, VectorStoreType.CHROMA])
def test_job_operations(store_type: VectorStoreType, test_data: Dict[str, Any], test_embeddings: Dict[str, Any]):
    """Test job operations in vector store.
    
    Args:
        store_type: Vector store type to test
        test_data: Test data fixture
        test_embeddings: Test embeddings fixture
    """
    update_config({"vector_store": {"STORE_TYPE": store_type}})
    store = create_vector_store()
    
    # Test adding jobs
    for job in test_data["jobs"]:
        store.add_job(
            job_id=job["id"],
            embedding=test_embeddings["jobs"][job["id"]],
            metadata=job,
            text=f"{job['title']}\n{job['description']}"
        )
    
    # Test searching jobs
    results = store.search_jobs(
        query_embedding=test_embeddings["jobs"]["job1"],
        k=2
    )
    assert len(results) > 0
    assert "id" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    
    # Test metadata filtering
    filtered_results = store.search_jobs(
        query_embedding=test_embeddings["jobs"]["job1"],
        k=2,
        filter_metadata={"location": "New York"}
    )
    assert len(filtered_results) > 0
    assert all(r["metadata"]["location"] == "New York" for r in filtered_results)
    
    # Test updating job
    store.update_job(
        job_id="job1",
        metadata={"location": "Remote"}
    )
    updated_results = store.search_jobs(
        query_embedding=test_embeddings["jobs"]["job1"],
        k=1
    )
    assert updated_results[0]["metadata"]["location"] == "Remote"
    
    # Test deleting job
    store.delete_job("job1")
    with pytest.raises(KeyError):
        store.delete_job("job1")  # Should raise KeyError for deleted job

@pytest.mark.parametrize("store_type", [VectorStoreType.FAISS, VectorStoreType.CHROMA])
def test_candidate_operations(store_type: VectorStoreType, test_data: Dict[str, Any], test_embeddings: Dict[str, Any]):
    """Test candidate operations in vector store.
    
    Args:
        store_type: Vector store type to test
        test_data: Test data fixture
        test_embeddings: Test embeddings fixture
    """
    update_config({"vector_store": {"STORE_TYPE": store_type}})
    store = create_vector_store()
    
    # Test adding candidates
    for candidate in test_data["candidates"]:
        store.add_candidate(
            candidate_id=candidate["id"],
            embedding=test_embeddings["candidates"][candidate["id"]],
            metadata=candidate,
            text=candidate["summary"]
        )
    
    # Test searching candidates
    results = store.search_candidates(
        query_embedding=test_embeddings["candidates"]["candidate1"],
        k=2
    )
    assert len(results) > 0
    assert "id" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    
    # Test metadata filtering
    filtered_results = store.search_candidates(
        query_embedding=test_embeddings["candidates"]["candidate1"],
        k=2,
        filter_metadata={"location": "New York"}
    )
    assert len(filtered_results) > 0
    assert all(r["metadata"]["location"] == "New York" for r in filtered_results)
    
    # Test updating candidate
    store.update_candidate(
        candidate_id="candidate1",
        metadata={"location": "Remote"}
    )
    updated_results = store.search_candidates(
        query_embedding=test_embeddings["candidates"]["candidate1"],
        k=1
    )
    assert updated_results[0]["metadata"]["location"] == "Remote"
    
    # Test deleting candidate
    store.delete_candidate("candidate1")
    with pytest.raises(KeyError):
        store.delete_candidate("candidate1")  # Should raise KeyError for deleted candidate

@pytest.mark.parametrize("store_type", [VectorStoreType.FAISS, VectorStoreType.CHROMA])
def test_persistence(store_type: VectorStoreType, test_data: Dict[str, Any], test_embeddings: Dict[str, Any], tmp_path: Path):
    """Test vector store persistence.
    
    Args:
        store_type: Vector store type to test
        test_data: Test data fixture
        test_embeddings: Test embeddings fixture
        tmp_path: Temporary directory path
    """
    # Configure persistence directory
    persist_dir = tmp_path / "vector_store"
    update_config({
        "vector_store": {
            "STORE_TYPE": store_type,
            "PERSIST_DIRECTORY": str(persist_dir)
        }
    })
    
    # Create and populate store
    store = create_vector_store()
    
    for job in test_data["jobs"]:
        store.add_job(
            job_id=job["id"],
            embedding=test_embeddings["jobs"][job["id"]],
            metadata=job,
            text=f"{job['title']}\n{job['description']}"
        )
    
    # Persist data
    store.persist()
    
    # Create new store instance
    new_store = create_vector_store()
    
    # Verify data was loaded
    results = new_store.search_jobs(
        query_embedding=test_embeddings["jobs"]["job1"],
        k=1
    )
    assert len(results) > 0
    assert results[0]["id"] == "job1"

def test_error_handling(test_data: Dict[str, Any], test_embeddings: Dict[str, Any]):
    """Test error handling in vector store operations."""
    store = create_vector_store()
    
    # Test adding with invalid embedding dimension
    with pytest.raises(VectorStoreError):
        store.add_job(
            job_id="invalid",
            embedding=np.random.rand(100),  # Wrong dimension
            metadata={},
            text=""
        )
    
    # Test updating non-existent items
    with pytest.raises(KeyError):
        store.update_job("nonexistent")
    with pytest.raises(KeyError):
        store.update_candidate("nonexistent")
    
    # Test deleting non-existent items
    with pytest.raises(KeyError):
        store.delete_job("nonexistent")
    with pytest.raises(KeyError):
        store.delete_candidate("nonexistent") 