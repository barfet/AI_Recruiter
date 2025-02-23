"""Integration tests for the complete data ingestion and embedding pipeline."""

import pytest
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.core.config import get_config
from src.data.process import process_all, process_jobs, process_candidates, create_embeddings
from src.data.embeddings.manager import EmbeddingManager
from src.data.vector_store.factory import get_vector_store
from tests.unit.test_job_processing import test_job_data, test_csv_files
from tests.unit.test_candidate_processing import test_pdf_content, test_pdf_file

@pytest.fixture
def test_environment(tmp_path: Path, test_csv_files: Dict[str, Path], test_pdf_file: Path):
    """Set up test environment with all necessary files."""
    # Create data directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()
    processed_dir = data_dir / "processed"
    processed_dir.mkdir()
    
    # Copy job files to raw directory
    for name, file_path in test_csv_files.items():
        dest = raw_dir / file_path.name
        dest.write_bytes(file_path.read_bytes())
    
    # Copy candidate file to raw directory
    candidates_dir = raw_dir / "candidates"
    candidates_dir.mkdir()
    dest = candidates_dir / test_pdf_file.name
    dest.write_bytes(test_pdf_file.read_bytes())
    
    # Update config to use test directories
    config = get_config()
    config.RAW_DATA_DIR = raw_dir
    config.PROCESSED_DATA_DIR = processed_dir
    
    return {
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir
    }

def test_full_candidate_pipeline_end_to_end(test_environment: Dict[str, Path]):
    """Test the complete candidate pipeline from PDF to embeddings."""
    # Process candidates
    candidates = process_candidates(force=True)
    assert len(candidates) > 0
    
    # Create embeddings
    create_embeddings("candidates", force=True)
    
    # Verify embeddings in vector store
    store = get_vector_store()
    results = store.search_candidates(
        "Python developer",
        k=1
    )
    
    assert len(results) == 1
    assert results[0]["score"] > 0.5  # Should have high relevance
    assert "Python" in results[0]["metadata"]["skills"]

def test_full_job_pipeline_end_to_end(test_environment: Dict[str, Path]):
    """Test the complete job pipeline from CSVs to embeddings."""
    # Process jobs
    jobs = process_jobs(force=True)
    assert len(jobs) == 2  # From test data
    
    # Create embeddings
    create_embeddings("jobs", force=True)
    
    # Verify embeddings in vector store
    store = get_vector_store()
    results = store.search_jobs(
        "Python developer position",
        k=2
    )
    
    assert len(results) == 2
    # First result should be the Python developer job
    assert "Python" in results[0]["metadata"]["title"]
    assert results[0]["score"] > results[1]["score"]  # Better match should score higher

def test_process_data_main(test_environment: Dict[str, Path]):
    """Test the main process_data orchestrator."""
    # Run full pipeline
    process_all(
        force_clean=True,
        force_embeddings=True,
        batch_size=2
    )
    
    processed_dir = test_environment["processed_dir"]
    
    # Verify processed files exist
    assert (processed_dir / "structured_jobs.json").exists()
    assert (processed_dir / "structured_candidates.json").exists()
    
    # Verify vector store has data
    store = get_vector_store()
    
    # Check jobs were embedded
    job_results = store.search_jobs("developer", k=10)
    assert len(job_results) > 0
    
    # Check candidates were embedded
    candidate_results = store.search_candidates("developer", k=10)
    assert len(candidate_results) > 0
    
    # Verify metadata is preserved
    assert all("id" in r["metadata"] for r in job_results)
    assert all("id" in r["metadata"] for r in candidate_results)

def test_incremental_processing(test_environment: Dict[str, Path]):
    """Test incremental processing of new data."""
    # Initial processing
    process_all()
    
    # Add new job
    new_job = pd.DataFrame({
        "job_id": ["job3"],
        "title": ["Frontend Developer"],
        "description": ["React developer needed"],
        "location": ["Remote"],
        "company_id": ["comp1"]
    })
    
    new_job_file = test_environment["raw_dir"] / "new_postings.csv"
    new_job.to_csv(new_job_file, index=False)
    
    # Process only new data
    process_jobs(force=False)
    create_embeddings("jobs", force=False)
    
    # Verify new job is searchable
    store = get_vector_store()
    results = store.search_jobs("React frontend", k=1)
    assert len(results) == 1
    assert "Frontend" in results[0]["metadata"]["title"]

def test_error_handling_and_recovery(test_environment: Dict[str, Path]):
    """Test error handling and recovery in the pipeline."""
    # Test invalid PDF handling
    invalid_pdf = test_environment["raw_dir"] / "candidates" / "invalid.pdf"
    invalid_pdf.write_bytes(b"Not a PDF")
    
    # Should continue processing valid files
    process_all()
    
    # Verify valid data was processed
    store = get_vector_store()
    assert len(store.search_jobs("any", k=1)) > 0
    assert len(store.search_candidates("any", k=1)) > 0
    
    # Test invalid embedding handling
    with pytest.raises(Exception):
        create_embeddings("invalid_type")  # Should raise on invalid type 