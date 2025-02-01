from typing import Dict, Any, Optional, List, Set
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


class StandardizedOutput(BaseModel):
    """Standardized output format for all tools"""
    fit_score: float = Field(..., description="Match score between 0 and 100")
    candidate_strengths: List[str] = Field(default_factory=list, description="List of candidate's strengths")
    candidate_weaknesses: List[str] = Field(default_factory=list, description="List of candidate's weaknesses")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    recommendations: List[str] = Field(default_factory=list, description="Detailed recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.dict(), indent=2, cls=NumpyJSONEncoder)


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
        extra_skills = candidate_skills_set - job_skills_set
        
        # Calculate match score
        if not job_skills_set:
            match_score = 0
        else:
            match_score = (len(matching_skills) / len(job_skills_set)) * 100

        # Prepare strengths and weaknesses
        strengths = [
            f"Has required skill: {skill}" for skill in matching_skills
        ] + [
            f"Additional valuable skill: {skill}" for skill in extra_skills
        ]

        weaknesses = [
            f"Missing required skill: {skill}" for skill in missing_skills
        ]

        # Prepare recommendations
        recommendations = []
        if missing_skills:
            recommendations.append(f"Focus on acquiring these skills: {', '.join(missing_skills)}")
        if matching_skills:
            recommendations.append("Highlight matching skills in application")
        if extra_skills:
            recommendations.append("Emphasize transferable skills from additional expertise")

        # Prepare next steps
        next_steps = []
        if match_score >= 70:
            next_steps.extend([
                "Proceed with technical interview",
                "Focus on practical experience with matching skills"
            ])
        elif match_score >= 50:
            next_steps.extend([
                "Consider initial screening call",
                "Assess learning potential for missing skills"
            ])
        else:
            next_steps.extend([
                "Recommend additional skill development",
                "Consider alternative positions"
            ])

        # Create standardized output
        output = StandardizedOutput(
            fit_score=match_score,
            candidate_strengths=strengths,
            candidate_weaknesses=weaknesses,
            next_steps=next_steps,
            recommendations=recommendations,
            metadata={
                "job_id": job_id,
                "resume_id": resume_id,
                "job_title": job["metadata"].get("title"),
                "matching_skills": list(matching_skills),
                "missing_skills": list(missing_skills),
                "extra_skills": list(extra_skills)
            }
        )
        
        return output.to_json()

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

class SkillAnalysisTool(BaseTool):
    name: str = "analyze_skills"
    description: str = "Analyze skill match between a job and candidate"

    def _run(self, query: str) -> str:
        try:
            job_id, resume_id = query.split()
            
            # Use real embeddings data
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
            
            # Extract skills from job and candidate text
            job_skills = extract_skills(job["document"].lower())
            candidate_skills = extract_skills(candidate["document"].lower())
            
            # Calculate matching skills
            matching_skills = job_skills & candidate_skills
            missing_skills = job_skills - candidate_skills
            extra_skills = candidate_skills - job_skills
            
            # Calculate fit score
            fit_score = len(matching_skills) / len(job_skills) * 100 if job_skills else 0
            
            # Create standardized output
            output = StandardizedOutput(
                fit_score=fit_score,
                candidate_strengths=[
                    f"Matching skill: {skill}" for skill in matching_skills
                ] + [
                    f"Additional valuable skill: {skill}" for skill in extra_skills
                ],
                candidate_weaknesses=[
                    f"Missing required skill: {skill}" for skill in missing_skills
                ],
                next_steps=[
                    "Schedule technical interview" if fit_score >= 70 else "Recommend additional skill development",
                    "Consider alternative positions" if fit_score < 50 else "Proceed with hiring process"
                ],
                recommendations=[
                    "Focus on acquiring these skills: " + ", ".join(missing_skills) if missing_skills else "Strong skill match - proceed with interview",
                    "Emphasize transferable skills from additional expertise" if extra_skills else "Core skills align well with position"
                ],
                metadata={
                    "job_id": job_id,
                    "resume_id": resume_id,
                    "job_title": job["metadata"].get("title"),
                    "matching_skills": list(matching_skills),
                    "missing_skills": list(missing_skills),
                    "extra_skills": list(extra_skills)
                }
            )
            
            return output.to_json()
            
        except Exception as e:
            logger.error(f"Error in skill analysis: {str(e)}")
            return f"Error: {str(e)}"

InterviewQuestionGenerator = Tool.from_function(
    func=lambda input_str: generate_questions(InterviewQuestionsInput(
        job_id=input_str.split()[0],
        resume_id=input_str.split()[1] if len(input_str.split()) > 1 else None
    )),
    name="generate_questions",
    description="Generate tailored interview questions based on job requirements and candidate profile. Input should be a job_id optionally followed by a resume_id, separated by a space.",
)

def extract_skills(text: str) -> Set[str]:
    """Extract skills from text using simple keyword matching."""
    # Common variations of skills
    skill_variations = {
        'python': ['python', 'py', 'python3'],
        'aws': ['aws', 'amazon web services', 'aws cloud'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
        'java': ['java', 'java8', 'java11'],
        'docker': ['docker', 'containerization'],
        'kubernetes': ['kubernetes', 'k8s'],
        'sql': ['sql', 'mysql', 'postgresql', 'postgres'],
        'nosql': ['nosql', 'mongodb', 'dynamodb'],
        'react': ['react', 'reactjs', 'react.js'],
        'angular': ['angular', 'angularjs', 'angular2+'],
        'vue': ['vue', 'vuejs', 'vue.js'],
        'node': ['node', 'nodejs', 'node.js'],
        'express': ['express', 'expressjs'],
        'django': ['django'],
        'flask': ['flask'],
        'spring': ['spring', 'spring boot'],
        'git': ['git', 'github', 'gitlab'],
        'ci/cd': ['ci/cd', 'jenkins', 'gitlab ci', 'github actions'],
        'devops': ['devops', 'sre'],
        'agile': ['agile', 'scrum', 'kanban'],
        'rest': ['rest', 'restful', 'rest api'],
        'graphql': ['graphql'],
        'microservices': ['microservices', 'microservice'],
        'cloud': ['cloud', 'cloud computing'],
        'linux': ['linux', 'unix'],
        'api': ['api', 'apis', 'rest api'],
        'go': ['go', 'golang'],
        'rust': ['rust'],
        'c++': ['c++', 'cpp'],
        'c#': ['c#', 'csharp', '.net'],
        '.net': ['.net', 'dotnet', 'asp.net'],
        'php': ['php'],
        'ruby': ['ruby', 'rails', 'ruby on rails'],
        'scala': ['scala'],
        'swift': ['swift', 'ios'],
        'kotlin': ['kotlin', 'android'],
        'typescript': ['typescript', 'ts']
    }
    
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()
    
    # Find all skills
    found_skills = set()
    for main_skill, variations in skill_variations.items():
        if any(var in text for var in variations):
            found_skills.add(main_skill)
    
    return found_skills
