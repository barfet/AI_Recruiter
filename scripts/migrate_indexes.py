import asyncio
import pickle
import faiss
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.vector_store.chroma_store import ChromaStore
from src.core.logging import setup_logger
from src.embeddings.manager import EmbeddingManager
from src.services.skill_normalization import SkillNormalizer

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

async def migrate_jobs(store: ChromaStore, data_path: Path) -> None:
    """Migrate job data to the new format."""
    skill_normalizer = SkillNormalizer()
    
    with open(data_path / "jobs.json") as f:
        jobs = json.load(f)
    
    for job in jobs:
        # Extract and normalize skills
        skills = skill_normalizer.extract_skills(job["description"])
        job["skills"] = skills
        
        # Add to store
        await store.add_job(
            job_id=job["id"],
            title=job["title"],
            description=job["description"],
            skills=skills
        )
        
    logger.info(f"Migrated {len(jobs)} jobs")

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
            await migrate_jobs(store, Path("data"))
            
            # Then migrate candidates
            await migrate_candidates(store)
            
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 