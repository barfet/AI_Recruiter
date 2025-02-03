from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from src.vector_store.chroma_store import ChromaStore
from src.data.managers.job import JobManager
from src.agent.agent import RecruitingAgent
from src.agent.chains import CandidateJobMatchChain
from src.agent.test_agent import parse_response
from src.core.logging import setup_logger

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
    
    def __init__(self, store: ChromaStore, job_manager: JobManager):
        self.store = store
        self.job_manager = job_manager
        self.agent = RecruitingAgent(temperature=0.3)
        self.match_chain = CandidateJobMatchChain()

    def _extract_job_id(self, response: str) -> Optional[str]:
        """Extract job ID from agent response."""
        try:
            parsed = parse_response(response)
            return parsed.get("job_id")
        except Exception as e:
            logger.error(f"Error parsing agent response: {str(e)}")
            return None
        
    def _calculate_skill_match(
        self,
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> Tuple[List[str], List[str], float]:
        """Calculate skill match between candidate and job."""
        # Normalize skills to lowercase for comparison
        candidate_skills_norm = [s.lower() for s in candidate_skills]
        job_skills_norm = [s.lower() for s in job_skills]
        
        # Find matching and missing skills
        matching = [s for s in job_skills if s.lower() in candidate_skills_norm]
        missing = [s for s in job_skills if s.lower() not in candidate_skills_norm]
        
        # Calculate match score (0-1)
        if not job_skills:
            return matching, missing, 1.0
            
        match_score = len(matching) / len(job_skills)
        return matching, missing, match_score

    async def _get_detailed_analysis(
        self,
        candidate_info: Dict,
        job_info: Dict
    ) -> Dict:
        """Get detailed analysis using the match chain."""
        try:
            # Format info for the chain
            formatted_job = f"""
            Title: {job_info.get('title', '')}
            Company: {job_info.get('company', '')}
            Location: {job_info.get('location', '')}
            Description: {job_info.get('description', '')}
            Requirements: {job_info.get('requirements', '')}
            Skills: {', '.join(job_info.get('skills', []))}
            """
            
            formatted_candidate = f"""
            Experience: {' '.join(candidate_info['experience'])}
            Skills: {', '.join(candidate_info['skills'])}
            """
            
            # Run the match chain
            result = await self.match_chain.run(
                candidate_info=formatted_candidate,
                job_info=formatted_job
            )
            
            return {
                'candidate_summary': result.get('candidate_summary', ''),
                'job_analysis': result.get('job_analysis', ''),
                'skills_gap_analysis': result.get('skills_gap_analysis', ''),
                'interview_strategy': result.get('interview_strategy', '')
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
                        job_skills = extract_skills(text_to_analyze)
                    
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