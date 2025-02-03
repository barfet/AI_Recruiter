from typing import AsyncGenerator
from fastapi import Depends

from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager
from src.core.config import settings

async def get_chroma_store() -> AsyncGenerator[ChromaStore, None]:
    """Dependency provider for ChromaStore."""
    store = ChromaStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        is_persistent=True
    )
    async with store as s:
        yield s

def get_job_manager() -> JobManager:
    """Dependency provider for JobManager."""
    return JobManager() 