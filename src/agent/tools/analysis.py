"""Analysis tools for the recruiting agent."""
from typing import List, Dict, Any, Set, Optional
from pydantic import BaseModel, Field

from src.agent.tools.base import BaseRecruitingTool
from src.agent.models.outputs import StandardizedOutput, SkillMatchOutput
from src.services.skill_matching import SkillMatcher
from src.services.skill_normalization import SkillNormalizer


class SkillAnalysisInput(BaseModel):
    """Input schema for skill analysis."""
    job_id: str = Field(..., description="ID of the job posting")
    resume_id: str = Field(..., description="ID of the candidate resume")
    analysis_type: str = Field(
        "match",
        description="Type of analysis to perform (match, requirements, skills)"
    )


class SkillAnalysisTool(BaseRecruitingTool):
    """Tool for analyzing and matching skills."""
    
    name: str = "skill_analysis"
    description: str = "Analyze and match skills between jobs and candidates"
    args_schema: type[SkillAnalysisInput] = SkillAnalysisInput
    
    skill_normalizer: SkillNormalizer = Field(default_factory=SkillNormalizer)
    skill_matcher: SkillMatcher = Field(default_factory=SkillMatcher)
    
    async def _calculate_match_scores(
        self,
        source_skills: List[str],
        target_skills: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate match scores between skills."""
        # Normalize skills
        normalized_source = self.skill_normalizer.normalize_skills(source_skills)
        normalized_target = self.skill_normalizer.normalize_skills(target_skills)
        
        # Calculate match scores
        scores = await self.skill_matcher.calculate_match_scores(
            normalized_source,
            normalized_target,
            weights=weights
        )
        
        return scores

    def _calculate_semantic_matches(
        self,
        job_skills: Set[str],
        candidate_skills: Set[str],
        exact_matches: Set[str]
    ) -> List[tuple[str, str, float]]:
        """Calculate semantic matches between skills.
        
        Args:
            job_skills: Set of required job skills
            candidate_skills: Set of candidate skills
            exact_matches: Set of exact matches to exclude
            
        Returns:
            List of tuples (job_skill, candidate_skill, similarity_score)
        """
        semantic_matches = []
        remaining_job_skills = job_skills - exact_matches
        remaining_candidate_skills = candidate_skills - exact_matches
        
        for job_skill in remaining_job_skills:
            best_match = None
            best_score = 0.0
            
            for candidate_skill in remaining_candidate_skills:
                score = self.skill_matcher.calculate_similarity(job_skill, candidate_skill)
                if score > best_score:
                    best_score = score
                    best_match = candidate_skill
            
            if best_score >= 0.65:  # Threshold for semantic matches
                semantic_matches.append((job_skill, best_match, best_score))
        
        return semantic_matches

    async def _arun(self, **kwargs) -> str:
        """Run skill analysis.
        
        Args:
            **kwargs: Parameters defined in SkillAnalysisInput
            
        Returns:
            JSON string containing analysis results
        """
        try:
            params = SkillAnalysisInput(**kwargs)
            
            if params.analysis_type == "requirements":
                # Get job requirements
                from src.agent.tools.search import JobSearchTool
                job_tool = JobSearchTool()
                job_result = await job_tool._arun(f"job_id:{params.job_id}", limit=1)
                job_data = StandardizedOutput.parse_raw(job_result)
                
                if job_data.status == "error":
                    return StandardizedOutput(
                        status="error",
                        error=job_data.error
                    ).json()
                
                return StandardizedOutput(
                    status="success",
                    data={
                        "title": job_data.data[0].get("title", ""),
                        "skills": job_data.data[0].get("skills", []),
                        "requirements": job_data.data[0].get("requirements", [])
                    }
                ).json()
            
            elif params.analysis_type == "skills":
                # Get candidate skills
                from src.agent.tools.search import CandidateSearchTool
                candidate_tool = CandidateSearchTool()
                candidate_result = await candidate_tool._arun(f"resume_id:{params.resume_id}", limit=1)
                candidate_data = StandardizedOutput.parse_raw(candidate_result)
                
                if candidate_data.status == "error":
                    return StandardizedOutput(
                        status="error",
                        error=candidate_data.error
                    ).json()
                
                return StandardizedOutput(
                    status="success",
                    data={
                        "skills": candidate_data.data[0].get("skills", []),
                        "experience": candidate_data.data[0].get("experience", [])
                    }
                ).json()
            
            else:  # Default to match analysis
                # Get job data
                from src.agent.tools.search import JobSearchTool
                job_tool = JobSearchTool()
                job_result = await job_tool._arun(f"job_id:{params.job_id}", limit=1)
                job_data = StandardizedOutput.parse_raw(job_result)
                
                if job_data.status == "error":
                    return StandardizedOutput(
                        status="error",
                        error=job_data.error
                    ).json()
                
                required_skills = job_data.data[0].get("skills", [])
                preferred_skills = job_data.data[0].get("preferred_skills", [])
                
                # Get candidate data
                from src.agent.tools.search import CandidateSearchTool
                candidate_tool = CandidateSearchTool()
                candidate_result = await candidate_tool._arun(f"resume_id:{params.resume_id}", limit=1)
                candidate_data = StandardizedOutput.parse_raw(candidate_result)
                
                if candidate_data.status == "error":
                    return StandardizedOutput(
                        status="error",
                        error=candidate_data.error
                    ).json()
                
                # Normalize skills
                required_skills_set = set(self.skill_normalizer.normalize_skills(required_skills))
                preferred_skills_set = set(self.skill_normalizer.normalize_skills(preferred_skills))
                candidate_skills_set = set(self.skill_normalizer.normalize_skills(candidate_data.data[0].get("skills", [])))
                
                # Calculate matches
                required_exact_matches = required_skills_set & candidate_skills_set
                preferred_exact_matches = preferred_skills_set & candidate_skills_set
                
                required_semantic_matches = self._calculate_semantic_matches(
                    required_skills_set,
                    candidate_skills_set,
                    required_exact_matches
                )
                
                preferred_semantic_matches = self._calculate_semantic_matches(
                    preferred_skills_set,
                    candidate_skills_set,
                    preferred_exact_matches
                )

                # Calculate scores
                if not required_skills_set:
                    required_score = 0
                else:
                    exact_score = len(required_exact_matches) / len(required_skills_set)
                    semantic_score = len(required_semantic_matches) / len(required_skills_set) * 0.8
                    required_score = (exact_score + semantic_score) * 100
                
                if preferred_skills_set:
                    preferred_score = (
                        (len(preferred_exact_matches) + len(preferred_semantic_matches) * 0.8) / 
                        len(preferred_skills_set)
                    ) * 100
                    # Calculate overall score (70% required, 30% preferred)
                    overall_score = required_score * 0.7 + preferred_score * 0.3
                else:
                    # If no preferred skills and all required skills match exactly, score is 100%
                    overall_score = 100.0 if len(required_exact_matches) == len(required_skills_set) and len(required_semantic_matches) == 0 else required_score

                # Get missing and additional skills
                matched_required = required_exact_matches | {m[0] for m in required_semantic_matches}
                missing_required = required_skills_set - matched_required
                
                matched_preferred = preferred_exact_matches | {m[0] for m in preferred_semantic_matches}
                missing_preferred = preferred_skills_set - matched_preferred
                
                all_matched = matched_required | matched_preferred
                additional_skills = candidate_skills_set - all_matched
                
                # Create output
                output = SkillMatchOutput(
                    match_score=overall_score,
                    exact_matches=list(required_exact_matches | preferred_exact_matches),
                    semantic_matches=required_semantic_matches + preferred_semantic_matches,
                    missing_skills=list(missing_required) + [f"(preferred) {s}" for s in missing_preferred],
                    additional_skills=list(additional_skills)
                )
                
                return StandardizedOutput(
                    status="success",
                    data=output.model_dump(),
                    metadata={
                        "job_id": params.job_id,
                        "resume_id": params.resume_id,
                        "required_score": required_score,
                        "preferred_score": preferred_score
                    }
                ).json()

        except Exception as e:
            self.logger.error(f"Error in skill analysis: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).json() 