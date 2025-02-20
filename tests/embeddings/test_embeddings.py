"""Tests for embedding models and manager."""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.core.config import get_config, update_config
from src.data.embeddings.base import EmbeddingError
from src.data.embeddings.factory import (
    create_embedding_model,
    get_embedding_model,
    EmbeddingModelType
)
from src.data.embeddings.manager import EmbeddingManager

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
def temp_data_files(tmp_path: Path, test_data: Dict[str, Any]) -> Dict[str, Path]:
    """Create temporary data files."""
    jobs_file = tmp_path / "jobs.json"
    candidates_file = tmp_path / "candidates.json"
    
    with open(jobs_file, "w") as f:
        json.dump(test_data["jobs"], f)
    with open(candidates_file, "w") as f:
        json.dump(test_data["candidates"], f)
    
    return {
        "jobs_file": jobs_file,
        "candidates_file": candidates_file
    }

def test_sentence_transformer_model():
    """Test SentenceTransformer model."""
    model = create_embedding_model(EmbeddingModelType.SENTENCE_TRANSFORMER)
    
    # Test single text embedding
    text = "This is a test"
    embedding = model.embed_text(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (model.dimension,)
    
    # Test batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = model.embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.dimension)

@pytest.mark.skipif(
    not get_config().OPENAI_API_KEY,
    reason="OpenAI API key not configured"
)
def test_openai_model():
    """Test OpenAI model."""
    model = create_embedding_model(EmbeddingModelType.OPENAI)
    
    # Test single text embedding
    text = "This is a test"
    embedding = model.embed_text(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (model.dimension,)
    
    # Test batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = model.embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.dimension)

def test_invalid_model_type():
    """Test invalid model type handling."""
    with pytest.raises(EmbeddingError):
        create_embedding_model("invalid_type")

def test_singleton_pattern():
    """Test embedding model singleton pattern."""
    model1 = get_embedding_model()
    model2 = get_embedding_model()
    assert model1 is model2

def test_embedding_manager(temp_data_files: Dict[str, Path]):
    """Test embedding manager."""
    manager = EmbeddingManager()
    
    # Test job embeddings creation
    manager.create_job_embeddings(temp_data_files["jobs_file"])
    
    # Test candidate embeddings creation
    manager.create_candidate_embeddings(temp_data_files["candidates_file"])
    
    # Test job search
    results = manager.search_similar_jobs(
        query_text="Python developer with web experience",
        k=2
    )
    assert len(results) > 0
    assert all(isinstance(r["score"], float) for r in results)
    
    # Test candidate search
    results = manager.search_similar_candidates(
        query_text="Machine learning engineer",
        k=2
    )
    assert len(results) > 0
    assert all(isinstance(r["score"], float) for r in results)
    
    # Test metadata filtering
    filtered_results = manager.search_similar_jobs(
        query_text="Python developer",
        k=2,
        filter_metadata={"location": "New York"}
    )
    assert len(filtered_results) > 0
    assert all(r["metadata"]["location"] == "New York" for r in filtered_results)

def test_error_handling():
    """Test error handling in embedding operations."""
    manager = EmbeddingManager()
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        manager.create_job_embeddings("nonexistent.json")
    
    # Test invalid JSON
    with pytest.raises(json.JSONDecodeError):
        manager.create_job_embeddings(__file__)  # Try to load Python file as JSON

def test_text_preparation(test_data: Dict[str, Any]):
    """Test text preparation for embedding."""
    manager = EmbeddingManager()
    
    # Test job text preparation
    job_text = manager._prepare_job_text(test_data["jobs"][0])
    assert "Software Engineer" in job_text
    assert "Python" in job_text
    assert "New York" in job_text
    
    # Test candidate text preparation
    candidate_text = manager._prepare_candidate_text(test_data["candidates"][0])
    assert "Python developer" in candidate_text
    assert "Django" in candidate_text
    assert "New York" in candidate_text

def test_batch_processing(temp_data_files: Dict[str, Path]):
    """Test batch processing of embeddings."""
    # Create a larger dataset
    large_data = {
        "jobs": [
            {
                "id": f"job{i}",
                "title": f"Job {i}",
                "description": f"Description {i}",
                "skills": ["Python"],
                "location": "Test"
            }
            for i in range(100)
        ]
    }
    
    large_file = temp_data_files["jobs_file"].parent / "large_jobs.json"
    with open(large_file, "w") as f:
        json.dump(large_data["jobs"], f)
    
    # Process in batches
    manager = EmbeddingManager()
    manager.create_job_embeddings(large_file)
    
    # Verify all jobs were processed
    results = manager.search_similar_jobs("Python", k=100)
    assert len(results) == 100 