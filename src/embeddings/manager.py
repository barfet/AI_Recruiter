from typing import List, Dict, Any, Optional
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.core.config import settings
from src.core.logging import setup_logger
from src.core.exceptions import EmbeddingError
from src.data.models.job import JobPosting
from src.data.models.candidate import CandidateProfile

logger = setup_logger(__name__)


class LocalEmbeddings(Embeddings):
    """Local embeddings using Sentence Transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text"""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class EmbeddingManager:
    """Manager for creating and managing embeddings"""

    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.data_dir = settings.DATA_DIR
        self.index_dir = settings.INDEXES_DIR

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise EmbeddingError(f"Failed to get embedding: {str(e)}")

    def _prepare_job_text(self, job: JobPosting) -> str:
        """Prepare job posting text for embedding"""
        return f"""
        Title: {job.title}
        Company: {job.company}
        Location: {job.location}
        Description: {job.description}
        Requirements: {job.requirements}
        Skills: {', '.join(job.skills)}
        Work Type: {job.work_type or 'Not specified'}
        Experience Level: {job.experience_level or 'Not specified'}
        Remote Allowed: {'Yes' if job.remote_allowed else 'No'}
        """

    def _prepare_candidate_text(self, candidate: CandidateProfile) -> str:
        """Prepare candidate profile text for embedding"""
        experience_text = "\n".join(
            [
                f"- {exp.title} at {exp.company}: {exp.description}"
                for exp in candidate.experience
            ]
        )

        education_text = "\n".join(
            [f"- {edu.degree} from {edu.institution}" for edu in candidate.education]
        )

        return f"""
        Name: {candidate.name}
        Location: {candidate.location or 'Not specified'}
        Summary: {candidate.summary}
        Skills: {', '.join(candidate.skills)}
        Experience:\n{experience_text}
        Education:\n{education_text}
        Languages: {', '.join(candidate.languages)}
        Industry: {candidate.industry}
        Certifications: {', '.join(candidate.certifications)}
        Desired Role: {candidate.desired_role or 'Not specified'}
        Desired Location: {candidate.desired_location or 'Not specified'}
        Remote Preference: {'Yes' if candidate.remote_preference else 'No'}
        """

    def create_job_embeddings(self, force: bool = False, batch_size: int = 50) -> FAISS:
        """Create embeddings for job postings"""
        try:
            index_path = self.index_dir / "jobs_index"
            if not force and index_path.exists():
                logger.info("Job embeddings already exist. Use force=True to recreate.")
                return self.load_embeddings("jobs")

            # Load processed job data
            with open(self.data_dir / "processed/structured_jobs.json", "r") as f:
                jobs_data = json.load(f)

            jobs = [JobPosting(**job) for job in jobs_data]
            vectorstore = None

            # Process in batches
            for i in range(0, len(jobs), batch_size):
                batch = jobs[i:i + batch_size]
                texts = [self._prepare_job_text(job) for job in batch]
                metadatas = [job.dict() for job in batch]

                # Create or merge FAISS index
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(
                        texts=texts, embedding=self.embeddings, metadatas=metadatas
                    )
                else:
                    vectorstore.add_texts(texts=texts, metadatas=metadatas)

                logger.info(
                    f"Processed batch {i // batch_size + 1} "
                    f"({i + len(batch)}/{len(jobs)} jobs)"
                )

            # Save index
            vectorstore.save_local(str(index_path))
            logger.info(f"Successfully created embeddings for {len(jobs)} job postings")
            return vectorstore

        except Exception as e:
            logger.error(f"Error creating job embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to create job embeddings: {str(e)}")

    def create_candidate_embeddings(
        self, force: bool = False, batch_size: int = 50
    ) -> FAISS:
        """Create embeddings for candidate profiles"""
        try:
            index_path = self.index_dir / "candidates_index"
            if not force and index_path.exists():
                logger.info(
                    "Candidate embeddings already exist. Use force=True to recreate."
                )
                return self.load_embeddings("candidates")

            # Load processed candidate data
            with open(self.data_dir / "processed/structured_candidates.json", "r") as f:
                candidates_data = json.load(f)

            candidates = [
                CandidateProfile(**candidate) for candidate in candidates_data
            ]
            vectorstore = None

            # Process in batches
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                texts = [self._prepare_candidate_text(candidate) for candidate in batch]
                metadatas = [candidate.dict() for candidate in batch]

                # Create or merge FAISS index
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(
                        texts=texts, embedding=self.embeddings, metadatas=metadatas
                    )
                else:
                    vectorstore.add_texts(texts=texts, metadatas=metadatas)

                logger.info(
                    f"Processed batch {i // batch_size + 1} "
                    f"({i + len(batch)}/{len(candidates)} candidates)"
                )

            # Save index
            vectorstore.save_local(str(index_path))
            logger.info(
                f"Successfully created embeddings for {len(candidates)} candidate profiles"
            )
            return vectorstore

        except Exception as e:
            logger.error(f"Error creating candidate embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to create candidate embeddings: {str(e)}")

    def load_embeddings(self, index_name: str) -> Optional[FAISS]:
        """Load existing embeddings from disk"""
        try:
            index_path = self.index_dir / f"{index_name}_index"
            if not index_path.exists():
                logger.warning(f"No existing index found at {index_path}")
                return None

            vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,  # Safe since we created these files
            )
            logger.info(f"Successfully loaded {index_name} embeddings")
            return vectorstore

        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to load embeddings: {str(e)}")

    async def similarity_search(self, index, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search on the index."""
        try:
            if not index:
                logger.warning("No index provided for similarity search")
                return []
                
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search the index
            results = index.search(query_embedding, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
