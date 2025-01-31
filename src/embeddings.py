from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict, Any
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Initialize embedding manager with OpenAI embeddings"""
        self.embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.stores = {
            'jobs': None,
            'candidates': None
        }
        self.data_dir = Path(__file__).parent.parent / "data"
        self.index_dir = self.data_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def _prepare_job_text(self, job: Dict[str, Any]) -> str:
        """Convert job posting to searchable text"""
        sections = [
            job['title'],
            job['description'],
            job['requirements'],
            f"Company: {job['company']}",
            f"Location: {job['location']}",
            f"Skills: {', '.join(job['skills'])}",
            f"Industries: {', '.join(job['industries'])}",
            f"Benefits: {', '.join(job['benefits'])}"
        ]
        return "\n".join(sections)
    
    def _prepare_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Convert candidate profile to searchable text"""
        sections = [
            f"Name: {candidate['name']}",
            f"Summary: {candidate.get('summary', '')}",
            "Experience:",
            *[f"- {exp['title']} at {exp['company']}" for exp in candidate['experience']],
            "Education:",
            *[f"- {edu['degree']} from {edu['institution']}" for edu in candidate['education']],
            f"Skills: {', '.join(candidate['skills'])}",
            f"Industry: {candidate['industry']}"
        ]
        return "\n".join(sections)
    
    def create_job_embeddings(self) -> None:
        """Create and save job embeddings"""
        # Load job data
        with open(self.data_dir / "processed/structured_jobs.json") as f:
            jobs = json.load(f)
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        
        for job in jobs:
            texts.append(self._prepare_job_text(job))
            metadatas.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location']
            })
        
        # Create FAISS index
        self.stores['jobs'] = FAISS.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas
        )
        
        # Save index
        self.stores['jobs'].save_local(str(self.index_dir / "jobs"))
    
    def create_candidate_embeddings(self) -> None:
        """Create and save candidate embeddings"""
        # Load candidate data
        with open(self.data_dir / "processed/structured_candidates.json") as f:
            candidates = json.load(f)
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        
        for candidate in candidates:
            texts.append(self._prepare_candidate_text(candidate))
            metadatas.append({
                'candidate_id': candidate['candidate_id'],
                'name': candidate['name'],
                'industry': candidate['industry']
            })
        
        # Create FAISS index
        self.stores['candidates'] = FAISS.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas
        )
        
        # Save index
        self.stores['candidates'].save_local(str(self.index_dir / "candidates"))
    
    def load_embeddings(self) -> None:
        """Load existing embeddings from disk"""
        # Load job embeddings if they exist
        job_index = self.index_dir / "jobs"
        if job_index.exists():
            self.stores['jobs'] = FAISS.load_local(
                str(job_index),
                self.embedder
            )
        
        # Load candidate embeddings if they exist
        candidate_index = self.index_dir / "candidates"
        if candidate_index.exists():
            self.stores['candidates'] = FAISS.load_local(
                str(candidate_index),
                self.embedder
            )
    
    def similarity_search(self, query: str, store_type: str, k: int = 5) -> List[Dict]:
        """Search for similar documents in the specified store"""
        if store_type not in self.stores or self.stores[store_type] is None:
            raise ValueError(f"Store {store_type} not initialized")
        
        return self.stores[store_type].similarity_search_with_score(query, k=k)

if __name__ == "__main__":
    # Example usage
    manager = EmbeddingManager()
    
    # Create embeddings if needed
    manager.create_job_embeddings()
    manager.create_candidate_embeddings()
    
    # Or load existing embeddings
    # manager.load_embeddings() 