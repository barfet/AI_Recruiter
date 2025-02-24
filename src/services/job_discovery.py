from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging
from datetime import datetime

from src.services.skill_normalization import SkillNormalizer
from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager
from src.agent.agent import RecruitingAgent
from src.agent.chains import CandidateJobMatchChain
from src.agent.test_agent import parse_response
from src.core.logging import setup_logger
from src.services.skill_matching import SkillMatcher

logger = setup_logger(__name__)

@dataclass
class JobMatch:
    """Represents a job match with relevance details"""
    job_id: str
    title: str
    company: str
    location: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    relevance_explanation: str
    detailed_analysis: Dict

class JobDiscoveryService:
    """Service for discovering relevant job opportunities"""
    
    def __init__(self, store: Any) -> None:
        """Initialize the service with a store."""
        self.store = store
        self.skill_matcher = SkillMatcher()
        self.skill_normalizer = SkillNormalizer()
        self.agent = RecruitingAgent()
        self.match_chain = CandidateJobMatchChain()
        
        # Build initial skill clusters from known variations
        initial_skills = [
            "Python", "python", "py",
            "Machine Learning", "ML", "machine learning", "ml",
            "Deep Learning", "DL", "deep learning", "dl",
            "PyTorch", "pytorch", "torch",
            "Neural Networks", "neural networks", "nn",
            "Artificial Intelligence", "AI", "artificial intelligence", "ai",
            "TensorFlow", "tensorflow", "tf",
            "Natural Language Processing", "NLP", "nlp",
            "Computer Vision", "CV", "cv",
            "Data Science", "data science", "DS", "ds"
        ]
        self.skill_normalizer.build_clusters(initial_skills)

    def _extract_job_id(self, response: str) -> Optional[str]:
        """Extract job ID from agent response."""
        try:
            parsed = parse_response(response)
            return parsed.get("job_id")
        except Exception as e:
            logger.error(f"Error parsing agent response: {str(e)}")
            return None
        
    def _get_skill_variations(self, skill: str) -> List[str]:
        """Get common variations of a skill."""
        return self.skill_normalizer.find_variations(skill)

    def _calculate_skill_match(
        self,
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> Tuple[List[str], List[str], float]:
        """Calculate skill match between candidate and job."""
        # Normalize skills
        candidate_skills_norm = self.skill_normalizer.normalize_skills(candidate_skills)
        job_skills_norm = self.skill_normalizer.normalize_skills(job_skills)
        
        # Find matching and missing skills with variations
        matching = []
        missing = []
        
        for job_skill in job_skills_norm:
            job_variations = self.skill_normalizer.find_variations(job_skill)
            matched = False
            
            for candidate_skill in candidate_skills_norm:
                candidate_variations = self.skill_normalizer.find_variations(candidate_skill)
                if any(jv in candidate_variations for jv in job_variations):
                    matching.append(job_skill)
                    matched = True
                    break
                    
            if not matched:
                missing.append(job_skill)
        
        # Calculate match score (0-1)
        if not job_skills:
            return matching, missing, 1.0
            
        match_score = len(matching) / len(job_skills)
        return matching, missing, match_score
        
    async def _calculate_match_score(
        self,
        job_skills: List[str],
        candidate_skills: List[str]
    ) -> float:
        """Calculate match score between job and candidate skills."""
        _, _, score = self._calculate_skill_match(candidate_skills, job_skills)
        return score
        
    async def _generate_recommendations(
        self,
        match_score: float,
        missing_skills: List[str],
        extra_skills: List[str]
    ) -> List[str]:
        """Generate recommendations based on match analysis."""
        recommendations = []
        
        if match_score >= 0.8:
            recommendations.append("Strong match - proceed with interview process")
        elif match_score >= 0.6:
            recommendations.append("Good match - consider technical assessment")
        else:
            recommendations.append("Consider additional skill development")
            
        if missing_skills:
            recommendations.append(f"Focus on acquiring: {', '.join(missing_skills)}")
            
        if extra_skills:
            recommendations.append("Highlight transferable skills from additional expertise")
            
        return recommendations
        
    async def _get_detailed_analysis(
        self,
        candidate_info: Dict,
        job_info: Dict
    ) -> Dict:
        """Get detailed analysis of job-candidate match."""
        try:
            # Extract skills
            job_skills = job_info["skills"].split(", ") if isinstance(job_info["skills"], str) else job_info["skills"]
            candidate_skills = candidate_info["skills"].split(", ") if isinstance(candidate_info["skills"], str) else candidate_info["skills"]

            # Calculate skill match
            matching_skills = set(job_skills) & set(candidate_skills)
            missing_skills = set(job_skills) - set(candidate_skills)
            additional_skills = set(candidate_skills) - set(job_skills)
            match_score = len(matching_skills) / len(job_skills) if job_skills else 0.0

            # Analyze experience
            experience = candidate_info.get("experience", [])
            total_years = sum(float(exp.get("duration", "0").split()[0]) for exp in experience)
            relevant_experience = [
                exp for exp in experience 
                if any(skill.lower() in exp.get("description", "").lower() for skill in job_skills)
            ]

            return {
                "candidate_summary": {
                    "name": candidate_info.get("name", ""),
                    "total_experience": f"{total_years} years",
                    "relevant_experience": len(relevant_experience),
                    "education": candidate_info.get("education", "Not specified"),
                    "industry": candidate_info.get("industry", "Not specified")
                },
                "skill_analysis": {
                    "match_score": match_score,
                    "matching_skills": list(matching_skills),
                    "missing_skills": list(missing_skills),
                    "additional_skills": list(additional_skills)
                },
                "experience_analysis": {
                    "total_years": total_years,
                    "relevant_positions": len(relevant_experience),
                    "experience_summary": [
                        {
                            "title": exp.get("title", ""),
                            "duration": exp.get("duration", ""),
                            "relevance": "High" if any(skill.lower() in exp.get("description", "").lower() for skill in matching_skills) else "Low"
                        }
                        for exp in experience
                    ]
                },
                "recommendations": {
                    "next_steps": [
                        "Proceed to technical interview" if match_score > 0.7 else "Consider additional screening",
                        "Focus on verifying claimed skills" if match_score > 0.5 else "Assess willingness to learn new skills"
                    ],
                    "development_areas": list(missing_skills),
                    "strengths": list(matching_skills)
                }
            }
        except Exception as e:
            logger.error(f"Error getting detailed analysis: {str(e)}")
            return {}

    async def find_matching_jobs(
        self,
        candidate_profile: Dict,
        limit: int = 10
    ) -> List[JobMatch]:
        """Find matching jobs for a candidate profile."""
        try:
            skill_normalizer = SkillNormalizer()
            # Use the agent to find relevant jobs
            query = f"Find jobs matching a candidate with experience: {' '.join(candidate_profile['experience'])} and skills: {', '.join(candidate_profile['skills'])}"
            jobs_response = await self.agent.run(query)
            
            # Extract job ID from agent response
            job_id = self._extract_job_id(jobs_response)
            if job_id:
                logger.info(f"Agent found job ID: {job_id}")
                # Get the specific job if found by agent
                job_result = await self.store.get_job_by_id(job_id)
                if job_result:
                    semantic_matches = [job_result]
                else:
                    semantic_matches = []
            
            # Get additional jobs from vector store
            vector_matches = await self.store.search_jobs(query, limit=limit)
            
            # Combine results, removing duplicates
            seen_ids = set()
            all_matches = []
            
            # Add agent-found job first if exists
            if job_id and semantic_matches:
                all_matches.extend(semantic_matches)
                seen_ids.add(job_id)
            
            # Add vector search results
            for match in vector_matches:
                if match["id"] not in seen_ids:
                    all_matches.append(match)
                    seen_ids.add(match["id"])
            
            matches = []
            for match in all_matches:
                try:
                    job_data = match.get("metadata", {})
                    if not job_data:
                        continue
                    
                    # Extract skills from job data
                    job_skills = job_data.get("skills", [])
                    if isinstance(job_skills, str):
                        # Split on both commas and spaces, then clean up
                        items = []
                        for item in job_skills.replace(",", " ").split():
                            item = item.strip()
                            if item and item not in items:  # Avoid duplicates
                                items.append(item)
                        job_skills = items
                    elif not job_skills:  # If empty list or None
                        # Extract skills from description and requirements
                        text_to_analyze = f"{job_data.get('description', '')} {job_data.get('requirements', '')}"
                        job_skills = skill_normalizer.extract_skills(text_to_analyze)
                    
                    # Calculate skill match
                    matching_skills, missing_skills, skill_score = self._calculate_skill_match(
                        candidate_profile["skills"],
                        job_skills
                    )
                    
                    # Get detailed analysis
                    detailed_analysis = await self._get_detailed_analysis(
                        candidate_profile,
                        job_data
                    )
                    
                    # Combine semantic and skill scores
                    # ChromaDB returns cosine distance, convert to similarity
                    semantic_score = match.get("score", 0) or 0
                    # Adjust weights to prioritize skill match more heavily
                    combined_score = (semantic_score * 0.7) + (skill_score * 0.3)
                    
                    # Generate explanation using the detailed analysis
                    explanation = detailed_analysis.get('skills_gap_analysis', '')
                    if not explanation:
                        # Fallback to basic explanation
                        explanation = f"Match score: {combined_score:.2f}. "
                        explanation += f"Matching skills: {', '.join(matching_skills[:3])}. "
                        if missing_skills:
                            explanation += f"Consider developing: {', '.join(missing_skills[:3])}."
                    
                    matches.append(JobMatch(
                        job_id=match.get("id", ""),
                        title=job_data.get("title", ""),
                        company=job_data.get("company", ""),
                        location=job_data.get("location", ""),
                        match_score=combined_score,
                        matching_skills=matching_skills,
                        missing_skills=missing_skills,
                        relevance_explanation=explanation,
                        detailed_analysis=detailed_analysis
                    ))
                except Exception as e:
                    logger.error(f"Error processing match: {str(e)}")
                    continue
            
            # Sort by combined score
            matches.sort(key=lambda x: x.match_score, reverse=True)
            return matches
            
        except Exception as e:
            logger.error(f"Error finding matching jobs: {str(e)}")
            raise 

    async def search_jobs(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for jobs using semantic similarity."""
        try:
            results = await self.store.search_jobs(query=query, limit=limit)
            
            # Process and format results
            formatted_results = []
            for result in results:
                job_data = await self.store.get_job_by_id(result["id"])
                if job_data:
                    # Normalize skills
                    skills = job_data.get("skills", [])
                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(",")]
                    formatted_results.append({
                        "id": result["id"],
                        "title": job_data.get("title", ""),
                        "description": job_data.get("description", ""),
                        "skills": skills,
                        "score": result["score"]
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching jobs: {str(e)}")
            return []

    async def find_matching_candidates(self, job_id: str) -> List[Dict[str, Any]]:
        """Find matching candidates for a specific job."""
        try:
            # Get job details
            job = await self.store.get_job_by_id(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return []

            # Extract job data from metadata if needed
            job_data = job.get("metadata", {}) or job
            if not job_data:
                logger.error("No job data found")
                return []

            # Ensure we have required fields
            title = job_data.get("title")
            skills = job_data.get("skills", [])
            if not title or not skills:
                logger.error("Job missing required fields (title or skills)")
                return []

            # Normalize skills if they're in string format
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills)
                except json.JSONDecodeError:
                    skills = [s.strip() for s in skills.split(",")]
            job_skills = [s.lower() for s in skills]

            # Construct search query from job requirements
            description = job_data.get("description", "")
            search_query = f"{title} {description} {' '.join(job_skills)}"
            
            # Search for candidates
            candidates = await self.store.search_candidates(search_query)
            if not candidates:
                logger.info("No candidates found matching the search query")
                return []

            # Process and score candidates
            results = []
            for candidate in candidates:
                try:
                    # Get candidate ID from the search result
                    candidate_id = candidate.get("id") or candidate.get("resume_id")
                    if not candidate_id:
                        # Try getting it from metadata
                        metadata = candidate.get("metadata", {})
                        candidate_id = metadata.get("resume_id") or metadata.get("id")
                        if not candidate_id:
                            logger.error("No candidate ID found in search result")
                            continue

                    # Get full candidate data
                    full_candidate = await self.store.get_candidate_by_id(candidate_id)
                    if not full_candidate:
                        logger.error(f"Could not retrieve full candidate data for {candidate_id}")
                        continue

                    # Extract candidate data, trying different possible locations
                    candidate_data = full_candidate.get("metadata", {}) or full_candidate

                    # Get candidate skills
                    candidate_skills = candidate_data.get("skills", [])
                    if not candidate_skills:
                        logger.error(f"No skills found for candidate {candidate_id}")
                        continue

                    # Normalize skills
                    if isinstance(candidate_skills, str):
                        try:
                            candidate_skills = json.loads(candidate_skills)
                        except json.JSONDecodeError:
                            candidate_skills = [s.strip() for s in candidate_skills.split(",")]
                    candidate_skills = [skill.lower() for skill in candidate_skills]

                    # Use SkillMatcher to analyze match
                    match_results = self.skill_matcher.calculate_match_score(
                        required_skills=job_skills,
                        candidate_skills=candidate_skills,
                        threshold=0.7,
                        semantic_weight=0.8
                    )

                    # Add candidate to results with match information
                    results.append({
                        "id": candidate_id,
                        "name": candidate_data.get("name", "Unknown"),
                        "skills": candidate_skills,  # Use normalized skills
                        "experience": candidate_data.get("experience", []),
                        "match_score": match_results["match_score"],
                        "matching_skills": match_results["exact_matches"],
                        "semantic_matches": [(req, cand, score) for req, cand, score in match_results["semantic_matches"]],
                        "missing_skills": match_results["missing_skills"]
                    })

                except Exception as e:
                    logger.error(f"Error processing candidate: {str(e)}")
                    continue

            # Sort by match score
            results.sort(key=lambda x: x["match_score"], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Error finding matching candidates: {str(e)}")
            return []

    async def get_match_analysis(self, job_id: str, candidate_id: str) -> Dict:
        """Get detailed analysis of the match between a job and a candidate."""
        try:
            # Get job and candidate data
            job = await self.store.get_job_by_id(job_id)
            candidate = await self.store.get_candidate_by_id(candidate_id)

            if not job or not candidate:
                raise ValueError("Job or candidate not found")

            # Extract job and candidate data from metadata if needed
            job_data = job.get("metadata", job)
            candidate_data = candidate.get("metadata", candidate)

            # Normalize skills
            job_skills = job_data.get("skills", [])
            if isinstance(job_skills, str):
                job_skills = [s.strip() for s in job_skills.split(",")]

            candidate_skills = candidate_data.get("skills", [])
            if isinstance(candidate_skills, str):
                candidate_skills = [s.strip() for s in candidate_skills.split(",")]

            # Use SkillMatcher to analyze match
            match_results = self.skill_matcher.calculate_match_score(
                required_skills=job_skills,
                candidate_skills=candidate_skills,
                threshold=0.7,
                semantic_weight=0.8
            )

            # Get additional skills (candidate skills not matched to any job skills)
            matched_skills = set(match_results["exact_matches"]) | {m[1] for m in match_results["semantic_matches"]}
            additional_skills = [s for s in candidate_skills if s not in matched_skills]

            return {
                "match_analysis": {
                    "skill_match_score": match_results["exact_score"],
                    "semantic_match_score": match_results["semantic_score"],
                    "combined_score": match_results["match_score"],
                    "matching_skills": match_results["exact_matches"],
                    "semantic_matches": [(req, cand, score) for req, cand, score in match_results["semantic_matches"]],
                    "missing_skills": match_results["missing_skills"],
                    "additional_skills": additional_skills
                },
                "candidate_info": {
                    "name": candidate_data.get("name", "Unknown"),
                    "experience": candidate_data.get("experience", []),
                    "education": candidate_data.get("education", "Not specified")
                },
                "job_info": {
                    "title": job_data.get("title", "Unknown"),
                    "company": job_data.get("company", "Unknown"),
                    "location": job_data.get("location", "Not specified")
                }
            }

        except Exception as e:
            logger.error(f"Error getting match analysis: {str(e)}")
            return {}

    def _are_skills_semantically_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are semantically similar using SkillNormalizer."""
        # Normalize both skills
        norm1 = self.skill_normalizer.normalize_skill(skill1)
        norm2 = self.skill_normalizer.normalize_skill(skill2)
        
        # Check if they normalize to the same canonical form
        if norm1 == norm2:
            return True
            
        # Check if they belong to the same variation cluster
        variations1 = self.skill_normalizer.find_variations(skill1)
        variations2 = self.skill_normalizer.find_variations(skill2)
        
        # Check for any overlap in variations
        return bool(set(variations1) & set(variations2))