import asyncio
import pickle
import faiss
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.vector_store.chroma_store import ChromaStore
from src.core.logging import setup_logger
from src.agent.tools import extract_skills
from src.embeddings.manager import EmbeddingManager

logger = setup_logger(__name__)

def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load FAISS index and metadata from directory."""
    index_path = Path(index_dir)
    
    # Load FAISS index
    index = faiss.read_index(str(index_path / "index.faiss"))
    
    # Load metadata
    with open(index_path / "index.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def clean_metadata(data: Dict) -> Dict:
    """Clean metadata for ChromaDB storage."""
    cleaned = {}
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, (list, tuple)):
            # Join lists into comma-separated strings
            cleaned[key] = ", ".join(str(v) for v in value if v is not None)
        elif isinstance(value, (dict, bool, int, float)):
            # Convert other types to strings
            cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)
    return cleaned

async def migrate_jobs(store: ChromaStore):
    """Migrate jobs from FAISS to ChromaDB."""
    logger.info("Starting jobs migration...")
    
    try:
        # Load FAISS index
        embedding_manager = EmbeddingManager()
        jobs_store = embedding_manager.load_embeddings("jobs")
        
        if not jobs_store:
            logger.error("No jobs index found")
            return
            
        logger.info(f"Loaded jobs index with {len(jobs_store.docstore._dict)} entries")
        
        # Migrate each job
        for i, (doc_id, doc) in enumerate(jobs_store.docstore._dict.items()):
            try:
                job_data = doc.metadata
                
                # Extract skills if not present
                if not job_data.get("skills"):
                    text_to_analyze = f"{job_data.get('description', '')} {job_data.get('requirements', '')}"
                    job_data["skills"] = extract_skills(text_to_analyze)
                
                # Add to ChromaDB
                await store.add_job(
                    job_id=doc_id,
                    job_data=job_data
                )
                logger.info(f"Migrated job {i}: {job_data.get('title', '')}")
            except Exception as e:
                logger.error(f"Error migrating job {i}: {str(e)}")
                continue
                
        logger.info("Jobs migration completed")
    except Exception as e:
        logger.error(f"Error in jobs migration: {str(e)}")
        raise

async def migrate_candidates(store: ChromaStore):
    """Migrate candidates from FAISS to ChromaDB."""
    logger.info("Starting candidates migration...")
    
    try:
        # Load FAISS index
        embedding_manager = EmbeddingManager()
        candidates_store = embedding_manager.load_embeddings("candidates")
        
        if not candidates_store:
            logger.error("No candidates index found")
            return
            
        logger.info(f"Loaded candidates index with {len(candidates_store.docstore._dict)} entries")
        
        # Migrate each candidate
        for i, (doc_id, doc) in enumerate(candidates_store.docstore._dict.items()):
            try:
                candidate_data = doc.metadata
                
                # Extract skills if not present
                if not candidate_data.get("skills"):
                    text_to_analyze = f"{' '.join(candidate_data.get('experience', []))} {' '.join(candidate_data.get('education', []))}"
                    candidate_data["skills"] = extract_skills(text_to_analyze)
                
                # Add to ChromaDB
                await store.add_candidate(
                    candidate_id=doc_id,
                    candidate_data=candidate_data
                )
                logger.info(f"Migrated candidate {i}")
            except Exception as e:
                logger.error(f"Error migrating candidate {i}: {str(e)}")
                continue
                
        logger.info("Candidates migration completed")
    except Exception as e:
        logger.error(f"Error in candidates migration: {str(e)}")
        raise

async def main():
    """Run the full migration."""
    logger.info("Starting FAISS to ChromaDB migration...")
    
    try:
        # Initialize ChromaDB
        async with ChromaStore(persist_directory=".chroma", is_persistent=True) as store:
            # Migrate jobs first
            await migrate_jobs(store)
            
            # Then migrate candidates
            await migrate_candidates(store)
            
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 