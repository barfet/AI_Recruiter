"""Service for skill normalization and variation detection."""
from typing import Dict, List, Set, Optional
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging
import json
import os

from src.core.logging import setup_logger

logger = setup_logger(__name__)

class SkillNormalizer:
    """Service for normalizing skills and detecting variations using embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        load_cached: bool = True,
        cache_file: str = "data/skill_clusters.json"
    ):
        """Initialize the skill normalizer.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Threshold for considering skills similar
            load_cached: Whether to load cached skill clusters
            cache_file: Path to the cache file for skill clusters
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.skill_cache: Dict[str, torch.Tensor] = {}
        self.skill_clusters: Dict[str, List[str]] = {}
        self.canonical_forms: Dict[str, str] = {}
        self.cache_file = cache_file
        
        if load_cached and os.path.exists(cache_file):
            self._load_cached_clusters()
            
    def _get_embedding(self, skill: str) -> torch.Tensor:
        """Get embedding for a skill, using cache if available."""
        skill = skill.lower().strip()
        if skill not in self.skill_cache:
            self.skill_cache[skill] = self.model.encode(skill, convert_to_tensor=True)
        return self.skill_cache[skill]
        
    def find_variations(self, skill: str) -> List[str]:
        """Find variations of a given skill."""
        skill = skill.lower().strip()
        
        # Check if we already know this skill's cluster
        if skill in self.canonical_forms:
            canonical = self.canonical_forms[skill]
            return self.skill_clusters.get(canonical, [skill])
            
        # Get embedding for the skill
        emb = self._get_embedding(skill)
        
        # Find similar skills from existing clusters
        variations = [skill]
        for canonical, cluster in self.skill_clusters.items():
            canonical_emb = self._get_embedding(canonical)
            similarity = float(util.pytorch_cos_sim(emb, canonical_emb))
            
            if similarity > self.similarity_threshold:
                self.canonical_forms[skill] = canonical
                if skill not in cluster:
                    cluster.append(skill)
                return cluster
                
        # If no existing cluster found, create new one
        self.skill_clusters[skill] = variations
        self.canonical_forms[skill] = skill
        return variations
        
    def normalize_skill(self, skill: str) -> str:
        """Get the canonical form of a skill."""
        skill = skill.lower().strip()
        return self.canonical_forms.get(skill, skill)
        
    def build_clusters(self, skills: List[str]) -> None:
        """Build skill clusters from a list of skills using DBSCAN clustering."""
        if not skills:
            return
            
        # Get embeddings for all skills
        embeddings = []
        skills = [s.lower().strip() for s in skills]
        
        for skill in skills:
            emb = self._get_embedding(skill)
            embeddings.append(emb.cpu().numpy())
            
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=2,
            metric="cosine"
        ).fit(embeddings)
        
        # Build clusters
        clusters = defaultdict(list)
        for skill, label in zip(skills, clustering.labels_):
            if label != -1:  # Not noise
                clusters[label].append(skill)
                
        # Update skill clusters and canonical forms
        for cluster in clusters.values():
            # Use the most common/shortest skill as canonical
            canonical = min(cluster, key=len)
            self.skill_clusters[canonical] = cluster
            
            for skill in cluster:
                self.canonical_forms[skill] = canonical
                
        self._save_clusters()
        
    def _save_clusters(self) -> None:
        """Save skill clusters to cache file."""
        try:
            data = {
                "clusters": self.skill_clusters,
                "canonical_forms": self.canonical_forms
            }
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving skill clusters: {str(e)}")
            
    def _load_cached_clusters(self) -> None:
        """Load skill clusters from cache file."""
        try:
            with open(self.cache_file) as f:
                data = json.load(f)
                self.skill_clusters = data["clusters"]
                self.canonical_forms = data["canonical_forms"]
        except Exception as e:
            logger.error(f"Error loading skill clusters: {str(e)}")
            
    def extract_skills(self, text: str) -> Set[str]:
        """Extract and normalize skills from text."""
        # First pass: extract skills using basic keyword matching
        words = set(text.lower().split())
        skills = set()
        
        # Look for single-word skills
        for word in words:
            if word in self.canonical_forms:
                skills.add(self.normalize_skill(word))
                
        # Look for multi-word skills
        for canonical in self.skill_clusters:
            if canonical in text.lower():
                skills.add(canonical)
                
        return skills
        
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize a list of skills to their canonical forms."""
        return [self.normalize_skill(skill) for skill in skills] 