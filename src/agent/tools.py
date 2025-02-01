from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, Tool, StructuredTool
import json
import numpy as np

from src.core.logging import setup_logger
from src.embeddings.manager import EmbeddingManager

logger = setup_logger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class SearchJobsInput(BaseModel):
    """Input for the SearchJobsTool"""
    query: str = Field(..., description="The search query for finding jobs")


class SearchCandidatesInput(BaseModel):
    """Input for the SearchCandidatesTool"""
    query: str = Field(..., description="The search query for finding candidates")


class MatchJobCandidatesInput(BaseModel):
    """Input for the MatchJobCandidatesTool"""
    query: str = Field(..., description="The job_id or resume_id to match")


class SkillAnalysisInput(BaseModel):
    """Input for the SkillAnalysisTool"""
    job_id: str = Field(..., description="The ID of the job to analyze")
    resume_id: str = Field(..., description="The ID of the resume to analyze")


class InterviewQuestionsInput(BaseModel):
    """Input for the InterviewQuestionGenerator"""
    job_id: str = Field(..., description="The ID of the job to generate questions for")
    resume_id: Optional[str] = Field(None, description="Optional resume ID for candidate-specific questions")


def search_jobs(query: str) -> str:
    """Search for job postings"""
    try:
        embedding_manager = EmbeddingManager()
        vectorstore = embedding_manager.load_embeddings("jobs")
        results = embedding_manager.similarity_search(vectorstore, query, k=5)
        
        if not results:
            return "No matching jobs found"
            
        # Convert scores to similarity scores (1 is best match, 0 is worst)
        for r in results:
            r["score"] = 1 - (r["score"] / 2)  # Convert distance to similarity
            
        return json.dumps(
            [
                {
                    "job_id": r["metadata"]["job_id"],
                    "title": r["metadata"]["title"],
                    "company": r["metadata"]["company"],
                    "location": r["metadata"]["location"],
                    "score": float(r["score"]),
                }
                for r in results
            ],
            indent=2,
            cls=NumpyJSONEncoder,
        )
    except Exception as e:
        logger.error(f"Error searching jobs: {str(e)}")
        return f"Error: {str(e)}"


def search_candidates(query: str) -> str:
    """Search for candidate profiles"""
    try:
        embedding_manager = EmbeddingManager()
        vectorstore = embedding_manager.load_embeddings("candidates")
        results = embedding_manager.similarity_search(vectorstore, query, k=5)
        
        if not results:
            return "No matching candidates found"
            
        # Convert scores to similarity scores (1 is best match, 0 is worst)
        for r in results:
            r["score"] = 1 - (r["score"] / 2)  # Convert distance to similarity
            
        return json.dumps(
            [
                {
                    "resume_id": r["metadata"]["resume_id"],
                    "name": r["metadata"].get("name", "Anonymous"),
                    "industry": r["metadata"]["industry"],
                    "skills": r["metadata"].get("skills", []),
                    "score": float(r["score"]),
                }
                for r in results
            ],
            indent=2,
            cls=NumpyJSONEncoder,
        )
    except Exception as e:
        logger.error(f"Error searching candidates: {str(e)}")
        return f"Error: {str(e)}"


def match_job_candidates(query: str) -> str:
    """Match jobs with candidates"""
    try:
        embedding_manager = EmbeddingManager()
        jobs_store = embedding_manager.load_embeddings("jobs")
        candidates_store = embedding_manager.load_embeddings("candidates")

        # Parse input to determine if it's a job ID or resume ID
        input_type = "job" if query.startswith("job_") else "resume"

        if input_type == "job":
            # Find candidates for a job
            job_id = query
            job_results = embedding_manager.similarity_search(
                jobs_store,
                f"job_id:{job_id}",
                k=1
            )
            
            if not job_results:
                return f"Job {job_id} not found"
                
            job = job_results[0]

            # Use job description to find matching candidates
            results = embedding_manager.similarity_search(
                candidates_store, job["document"], k=5
            )
            
            if not results:
                return "No matching candidates found"
            
            # Convert scores to similarity scores
            for r in results:
                r["score"] = 1 - (r["score"] / 2)
                
            return json.dumps(
                {
                    "job": {
                        "job_id": job_id,
                        "title": job["metadata"]["title"],
                        "company": job["metadata"]["company"],
                    },
                    "matching_candidates": [
                        {
                            "resume_id": r["metadata"]["resume_id"],
                            "name": r["metadata"].get("name", "Anonymous"),
                            "skills": r["metadata"].get("skills", []),
                            "score": float(r["score"]),
                        }
                        for r in results
                    ],
                },
                indent=2,
                cls=NumpyJSONEncoder,
            )
        else:
            # Find jobs for a candidate
            resume_id = query
            candidate_results = embedding_manager.similarity_search(
                candidates_store,
                f"resume_id:{resume_id}",
                k=1
            )
            
            if not candidate_results:
                return f"Candidate {resume_id} not found"
                
            candidate = candidate_results[0]

            # Use candidate profile to find matching jobs
            results = embedding_manager.similarity_search(
                jobs_store, candidate["document"], k=5
            )
            
            if not results:
                return "No matching jobs found"
            
            # Convert scores to similarity scores
            for r in results:
                r["score"] = 1 - (r["score"] / 2)
                
            return json.dumps(
                {
                    "candidate": {
                        "resume_id": resume_id,
                        "name": candidate["metadata"].get("name", "Anonymous"),
                        "skills": candidate["metadata"].get("skills", []),
                    },
                    "matching_jobs": [
                        {
                            "job_id": r["metadata"]["job_id"],
                            "title": r["metadata"]["title"],
                            "company": r["metadata"]["company"],
                            "score": float(r["score"]),
                        }
                        for r in results
                    ],
                },
                indent=2,
                cls=NumpyJSONEncoder,
            )
    except Exception as e:
        logger.error(f"Error matching: {str(e)}")
        return f"Error: {str(e)}"


def analyze_skills(input_data: SkillAnalysisInput) -> str:
    """Analyze skill match between a job and a candidate"""
    try:
        job_id = input_data.job_id
        resume_id = input_data.resume_id
        
        if not job_id or not resume_id:
            return "Error: Both job_id and resume_id are required"
        
        embedding_manager = EmbeddingManager()
        jobs_store = embedding_manager.load_embeddings("jobs")
        candidates_store = embedding_manager.load_embeddings("candidates")
        
        # Find job and candidate
        job_results = embedding_manager.similarity_search(
            jobs_store,
            f"job_id:{job_id}",
            k=1
        )
        
        candidate_results = embedding_manager.similarity_search(
            candidates_store,
            f"resume_id:{resume_id}",
            k=1
        )
        
        if not job_results or not candidate_results:
            return "Error: Job or candidate not found"
        
        job = job_results[0]
        candidate = candidate_results[0]
        
        # Get skills
        job_skills = job["metadata"].get("skills", [])
        candidate_skills = candidate["metadata"].get("skills", [])
        
        # Analyze skill match
        job_skills_set = set(s.lower() for s in job_skills)
        candidate_skills_set = set(s.lower() for s in candidate_skills)
        
        matching_skills = job_skills_set.intersection(candidate_skills_set)
        missing_skills = job_skills_set - candidate_skills_set
        additional_skills = candidate_skills_set - job_skills_set
        
        match_percentage = len(matching_skills) / len(job_skills_set) * 100 if job_skills_set else 0
        
        result = {
            "job": {
                "job_id": job_id,
                "title": job["metadata"]["title"],
                "company": job["metadata"]["company"],
                "required_skills": job_skills
            },
            "candidate": {
                "resume_id": resume_id,
                "name": candidate["metadata"].get("name", "Anonymous"),
                "skills": candidate_skills
            },
            "analysis": {
                "match_percentage": round(match_percentage, 2),
                "matching_skills": list(matching_skills),
                "missing_skills": list(missing_skills),
                "additional_skills": list(additional_skills)
            }
        }
        
        return json.dumps(result, indent=2, cls=NumpyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error analyzing skills: {str(e)}")
        return f"Error: {str(e)}"


def generate_questions(input_data: InterviewQuestionsInput) -> str:
    """Generate interview questions"""
    try:
        job_id = input_data.job_id
        resume_id = input_data.resume_id
        
        if not job_id:
            return "Error: job_id is required"
        
        embedding_manager = EmbeddingManager()
        jobs_store = embedding_manager.load_embeddings("jobs")
        candidates_store = embedding_manager.load_embeddings("candidates")
        
        # Find job
        job_results = embedding_manager.similarity_search(
            jobs_store,
            f"job_id:{job_id}",
            k=1
        )
        
        if not job_results:
            return "Error: Job not found"
            
        job = job_results[0]
        
        # Get job details
        job_title = job["metadata"]["title"]
        required_skills = job["metadata"].get("skills", [])
        requirements = job["metadata"].get("requirements", [])
        
        # Generate skill questions
        skill_questions = []
        for skill in required_skills:
            skill_questions.extend([
                f"Can you describe a project where you used {skill}?",
                f"What's the most challenging problem you've solved using {skill}?",
                f"How do you stay updated with the latest developments in {skill}?"
            ])
        skill_questions = skill_questions[:5]  # Limit to top 5
        
        # Generate experience questions
        base_questions = [
            f"What interests you about this {job_title} position?",
            "How do you handle tight deadlines and pressure?",
            "Describe a challenging project you led and its outcome.",
            "How do you approach learning new technologies?",
            "Tell me about a time you had to resolve a conflict in your team."
        ]
        
        requirement_questions = [
            f"How does your experience align with our requirement for {req}?"
            for req in requirements[:3]
        ]
        
        result = {
            "job": {
                "job_id": job_id,
                "title": job_title,
                "company": job["metadata"]["company"]
            },
            "questions": {
                "technical_skills": skill_questions,
                "experience_and_behavioral": base_questions + requirement_questions
            }
        }
        
        # If resume_id provided, add candidate-specific questions
        if resume_id:
            candidate_results = embedding_manager.similarity_search(
                candidates_store,
                f"resume_id:{resume_id}",
                k=1
            )
            
            if candidate_results:
                candidate = candidate_results[0]
                candidate_skills = candidate["metadata"].get("skills", [])
                missing_skills = set(required_skills) - set(candidate_skills)
                
                result["questions"]["candidate_specific"] = [
                    f"I notice you haven't listed {skill}. Do you have any experience with it?" 
                    for skill in list(missing_skills)[:3]
                ]
        
        return json.dumps(result, indent=2, cls=NumpyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return f"Error: {str(e)}"


def search_jobs_structured(input_data: SearchJobsInput) -> str:
    """Search for job postings with structured input"""
    return search_jobs(input_data.query)

def search_candidates_structured(input_data: SearchCandidatesInput) -> str:
    """Search for candidate profiles with structured input"""
    return search_candidates(input_data.query)

def match_job_candidates_structured(input_data: MatchJobCandidatesInput) -> str:
    """Match jobs with candidates with structured input"""
    return match_job_candidates(input_data.query)

# Create tool instances
SearchJobsTool = Tool.from_function(
    func=search_jobs,
    name="search_jobs",
    description="Search for job postings using semantic search. Input should be a description of the job you're looking for.",
)

SearchCandidatesTool = Tool.from_function(
    func=search_candidates,
    name="search_candidates",
    description="Search for candidate profiles using semantic search. Input should be a description of the candidate you're looking for.",
)

MatchJobCandidatesTool = Tool.from_function(
    func=match_job_candidates,
    name="match_job_candidates",
    description="Match a job posting with potential candidates or vice versa. Input should be a job ID or resume ID.",
)

SkillAnalysisTool = Tool.from_function(
    func=lambda input_str: analyze_skills(SkillAnalysisInput(
        job_id=input_str.split()[0],
        resume_id=input_str.split()[1]
    )),
    name="analyze_skills",
    description="Analyze the skill match between a candidate and a job posting. Input should be a job_id followed by a resume_id, separated by a space.",
)

InterviewQuestionGenerator = Tool.from_function(
    func=lambda input_str: generate_questions(InterviewQuestionsInput(
        job_id=input_str.split()[0],
        resume_id=input_str.split()[1] if len(input_str.split()) > 1 else None
    )),
    name="generate_questions",
    description="Generate tailored interview questions based on job requirements and candidate profile. Input should be a job_id optionally followed by a resume_id, separated by a space.",
)
