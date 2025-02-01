import pytest
import pytest_asyncio
import shutil
import os
from .chroma_store import ChromaStore

@pytest_asyncio.fixture(scope="function")
async def chroma_store():
    """Create a test instance of ChromaStore."""
    store = ChromaStore(persist_directory=".chroma_test", is_persistent=False)
    await store.__aenter__()  # Initialize async resources
    yield store
    await store.__aexit__(None, None, None)  # Cleanup async resources

@pytest.mark.asyncio
async def test_add_and_get_job(chroma_store):
    """Test adding and retrieving a job."""
    job_data = {
        "title": "Senior Software Engineer",
        "description": "We are looking for a senior engineer with Python experience",
        "requirements": "5+ years Python, AWS, Kubernetes",
        "company": "TechCorp",
        "location": "Remote"
    }

    # Add job
    await chroma_store.add_job("job123", job_data)
    
    # Get job
    result = await chroma_store.get_job_by_id("job123")
    assert result is not None
    assert result["id"] == "job123"
    assert result["metadata"]["title"] == job_data["title"]

@pytest.mark.asyncio
async def test_add_and_get_candidate(chroma_store):
    """Test adding and retrieving a candidate."""
    candidate_data = {
        "name": "John Doe",
        "experience": ["Senior Python Developer at TechCorp", "AWS Solutions Architect"],
        "skills": ["Python", "AWS", "Docker"],
        "education": "BSc Computer Science"
    }

    # Add candidate
    await chroma_store.add_candidate("cand123", candidate_data)
    
    # Get candidate
    result = await chroma_store.get_candidate_by_id("cand123")
    assert result is not None
    assert result["id"] == "cand123"
    assert result["metadata"]["name"] == candidate_data["name"]
    assert result["metadata"]["experience"] == ", ".join(candidate_data["experience"])
    assert result["metadata"]["skills"] == ", ".join(candidate_data["skills"])

@pytest.mark.asyncio
async def test_search_jobs(chroma_store):
    """Test searching for jobs."""
    # Add multiple jobs
    jobs = [
        {
            "id": "job1",
            "data": {
                "title": "Python Developer",
                "description": "Python and AWS development",
                "requirements": "Python, AWS"
            }
        },
        {
            "id": "job2",
            "data": {
                "title": "Frontend Developer",
                "description": "React and TypeScript development",
                "requirements": "React, TypeScript"
            }
        }
    ]

    for job in jobs:
        await chroma_store.add_job(job["id"], job["data"])

    # Search for Python jobs
    results = await chroma_store.search_jobs("Python developer")
    assert len(results) > 0
    assert any("Python" in result["metadata"]["title"] for result in results)

@pytest.mark.asyncio
async def test_search_candidates(chroma_store):
    """Test searching for candidates."""
    # Add multiple candidates
    candidates = [
        {
            "id": "cand1",
            "data": {
                "name": "John Doe",
                "experience": ["Python Developer", "AWS Architect"],
                "skills": ["Python", "AWS", "Docker"],
                "education": "BSc CS"
            }
        },
        {
            "id": "cand2",
            "data": {
                "name": "Jane Smith",
                "experience": ["Frontend Developer", "UI Designer"],
                "skills": ["React", "TypeScript", "CSS"],
                "education": "BSc Design"
            }
        }
    ]

    for candidate in candidates:
        await chroma_store.add_candidate(candidate["id"], candidate["data"])

    # Search for Python developers
    results = await chroma_store.search_candidates("Python developer")
    assert len(results) > 0
    assert any("Python" in result["metadata"]["skills"] for result in results)

@pytest.mark.asyncio
async def test_nonexistent_job(chroma_store):
    """Test retrieving a nonexistent job."""
    result = await chroma_store.get_job_by_id("nonexistent")
    assert result is None

@pytest.mark.asyncio
async def test_nonexistent_candidate(chroma_store):
    """Test retrieving a nonexistent candidate."""
    result = await chroma_store.get_candidate_by_id("nonexistent")
    assert result is None 