from typing import List, Set, Tuple, Dict
import json
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer, util
import torch

from src.core.logging import setup_logger

logger = setup_logger(__name__)

class SkillMatcher:
    """Service for intelligent skill matching using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.skill_cache: Dict[str, torch.Tensor] = {}
        
    def calculate_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity score between two skills."""
        try:
            # Get embeddings
            emb1 = self._get_embedding(skill1)
            emb2 = self._get_embedding(skill2)
            
            # Calculate cosine similarity
            similarity = float(util.pytorch_cos_sim(emb1, emb2))
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity between {skill1} and {skill2}: {str(e)}")
            return 0.0
        
    def _get_embedding(self, skill: str) -> torch.Tensor:
        """Get embedding for a skill, using cache if available."""
        skill = skill.lower().strip()
        if skill not in self.skill_cache:
            self.skill_cache[skill] = self.model.encode(skill, convert_to_tensor=True)
        return self.skill_cache[skill]
        
    def are_skills_similar(
        self,
        skill1: str,
        skill2: str,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """Check if two skills are semantically similar."""
        try:
            # Get embeddings
            emb1 = self._get_embedding(skill1)
            emb2 = self._get_embedding(skill2)
            
            # Calculate cosine similarity
            similarity = float(util.pytorch_cos_sim(emb1, emb2))
            
            return similarity > threshold, similarity
        except Exception as e:
            logger.error(f"Error comparing skills {skill1} and {skill2}: {str(e)}")
            return False, 0.0
            
    def find_matching_skills(
        self,
        required_skills: List[str],
        candidate_skills: List[str],
        threshold: float = 0.7
    ) -> Dict[str, List]:
        """Find matching skills between required and candidate skills."""
        try:
            # Normalize skills
            required = [s.lower().strip() for s in required_skills]
            candidate = [s.lower().strip() for s in candidate_skills]
            
            # Initialize results
            exact_matches = []
            semantic_matches = []
            missing_skills = []
            
            # Find exact and semantic matches
            for req_skill in required:
                if req_skill in candidate:
                    exact_matches.append(req_skill)
                    continue
                    
                # Look for semantic matches
                best_match = None
                best_score = 0
                
                for cand_skill in candidate:
                    is_similar, score = self.are_skills_similar(req_skill, cand_skill, threshold)
                    if is_similar and score > best_score:
                        best_match = (req_skill, cand_skill, score)
                        best_score = score
                        
                if best_match:
                    semantic_matches.append(best_match)
                else:
                    missing_skills.append(req_skill)
                    
            return {
                "exact_matches": exact_matches,
                "semantic_matches": semantic_matches,
                "missing_skills": missing_skills
            }
            
        except Exception as e:
            logger.error(f"Error finding matching skills: {str(e)}")
            return {
                "exact_matches": [],
                "semantic_matches": [],
                "missing_skills": required_skills
            }
            
    def calculate_match_score(
        self,
        required_skills: List[str],
        candidate_skills: List[str],
        threshold: float = 0.7,
        semantic_weight: float = 0.8
    ) -> Dict:
        """Calculate match score between required and candidate skills."""
        try:
            matches = self.find_matching_skills(required_skills, candidate_skills, threshold)
            
            total_required = len(required_skills)
            if total_required == 0:
                return {
                    "match_score": 100.0,
                    "exact_score": 100.0,
                    "semantic_score": 0.0,
                    **matches
                }
                
            exact_count = len(matches["exact_matches"])
            semantic_count = len(matches["semantic_matches"])
            
            exact_score = (exact_count / total_required) * 100
            semantic_score = (semantic_count / total_required) * 100
            
            # Calculate weighted score
            weighted_semantic = semantic_count * semantic_weight
            final_score = ((exact_count + weighted_semantic) / total_required) * 100
            
            return {
                "match_score": final_score,
                "exact_score": exact_score,
                "semantic_score": semantic_score,
                **matches
            }
            
        except Exception as e:
            logger.error(f"Error calculating match score: {str(e)}")
            return {
                "match_score": 0.0,
                "exact_score": 0.0,
                "semantic_score": 0.0,
                "exact_matches": [],
                "semantic_matches": [],
                "missing_skills": required_skills
            } 