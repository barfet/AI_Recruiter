"""Test configuration and fixtures."""

import os
import pytest
from typing import AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test")

from langchain.chat_models import ChatOpenAI
from src.core.config import settings
from src.agent.agent import RecruitingAgent
from src.vector_store.chroma_store import ChromaStore
from src.data.managers.candidate import CandidateManager
from src.data.managers.job import JobManager

@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    mock = MagicMock(spec=ChatOpenAI)
    mock.temperature = 0.0
    return mock

@pytest.fixture
def mock_chroma() -> MagicMock:
    """Create a mock ChromaDB instance."""
    return MagicMock(spec=ChromaStore)

@pytest.fixture
async def recruiting_agent(mock_llm) -> AsyncGenerator[RecruitingAgent, None]:
    """Create a test recruiting agent."""
    agent = RecruitingAgent(temperature=0.0)
    agent.llm = mock_llm
    yield agent

@pytest.fixture
def sample_job_data() -> Dict:
    """Sample job posting data."""
    return {
        "id": "test_job_1",
        "title": "Senior Software Engineer",
        "company": "TechCorp",
        "location": "Remote",
        "description": "Looking for a senior engineer with Python and AWS experience",
        "requirements": ["Python", "AWS", "Microservices"],
        "skills": ["Python", "AWS", "Docker", "Kubernetes"]
    }

@pytest.fixture
def sample_candidate_data() -> Dict:
    """Sample candidate data."""
    return {
        "id": "test_resume_1",
        "name": "John Doe",
        "experience": [
            {
                "title": "Software Engineer",
                "company": "ABC Corp",
                "duration": "3 years",
                "description": "Developed microservices using Python and AWS"
            }
        ],
        "skills": ["Python", "AWS", "Docker", "REST APIs"]
    } 