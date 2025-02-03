"""Multi-step chains for complex recruiting workflows"""

from typing import Dict, List, Optional, Any
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
import logging

from src.core.config import settings
from src.core.logging import setup_logger
from src.agent.prompts import (
    create_dynamic_job_prompt,
    create_skill_analysis_prompt,
    create_interview_prompt,
    get_system_message,
    JOB_MATCH_EXAMPLES,
    SKILL_ANALYSIS_EXAMPLES
)

logger = setup_logger(__name__)


class CandidateJobMatchChain:
    """Chain for comprehensive candidate-job matching analysis"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the chain."""
        self.llm = ChatOpenAI(model_name=model_name)
        
        # Matching prompt
        self.match_prompt = ChatPromptTemplate.from_template(
            """Analyze the match between the following job and candidate:
            
            Job Description: {job_description}
            Required Skills: {required_skills}
            
            Candidate Profile: {candidate_profile}
            Candidate Skills: {candidate_skills}
            
            Provide a detailed analysis of the match in JSON format with the following fields:
            - match_score (0-100)
            - strengths
            - weaknesses
            - recommendations"""
        )
        
        # Create chain
        self.match_chain = (
            self.match_prompt 
            | self.llm 
            | StrOutputParser()
        )

    async def analyze_match(
        self,
        job_description: str,
        required_skills: List[str],
        candidate_profile: str,
        candidate_skills: List[str]
    ) -> Dict[str, Any]:
        """Analyze the match between a job and candidate."""
        try:
            response = await self.match_chain.ainvoke({
                "job_description": job_description,
                "required_skills": ", ".join(required_skills),
                "candidate_profile": candidate_profile,
                "candidate_skills": ", ".join(candidate_skills)
            })
            
            # Parse response as JSON
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing match: {str(e)}")
            return {}


class InterviewWorkflowChain:
    """Chain for managing the interview process workflow"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the chain."""
        self.llm = ChatOpenAI(model_name=model_name)
        
        # Question generation prompt
        self.question_prompt = ChatPromptTemplate.from_template(
            """Given the following job information and candidate background, generate relevant interview questions.
            
            Job Info: {job_info}
            Candidate Info: {candidate_info}
            Focus Areas: {focus_areas}
            Difficulty: {difficulty}
            
            Generate 5 technical interview questions that will help assess the candidate's fit for this role.
            Format your response as a JSON array of questions."""
        )
        
        # Create chain
        self.question_chain = (
            self.question_prompt 
            | self.llm 
            | StrOutputParser()
        )

    async def generate_questions(
        self,
        job_info: str,
        candidate_info: str = "",
        focus_areas: List[str] = None,
        difficulty: str = "intermediate"
    ) -> List[str]:
        """Generate interview questions."""
        try:
            response = await self.question_chain.ainvoke({
                "job_info": job_info,
                "candidate_info": candidate_info,
                "focus_areas": focus_areas or ["Technical", "Problem Solving"],
                "difficulty": difficulty
            })
            
            # Parse response as JSON
            questions = json.loads(response)
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    async def evaluate_response(
        self,
        question: str,
        answer: str,
        job_requirements: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate a candidate's response with custom focus areas"""
        try:
            # Update prompt with specific focus areas if provided
            if focus_areas:
                prompt = create_skill_analysis_prompt(focus_areas=focus_areas)
                self.response_eval_chain = prompt | self.llm
            
            result = await self.response_eval_chain.ainvoke({
                "job_requirements": job_requirements,
                "candidate_skills": answer,
                "few_shot_examples": SKILL_ANALYSIS_EXAMPLES,
                "focus_area_info": ""
            })
            return {"response_evaluation": result}
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise

    async def generate_feedback(
        self,
        evaluations: List[Dict],
        focus_areas: Optional[List[str]] = None
    ) -> Dict:
        """Generate comprehensive feedback with custom focus areas"""
        try:
            # Update prompt with specific focus areas if provided
            if focus_areas:
                prompt = create_skill_analysis_prompt(focus_areas=focus_areas)
                self.feedback_gen_chain = prompt | self.llm
            
            # Format evaluations as skills for analysis
            eval_text = "\n".join(
                f"Q{i+1}: {eval_['response_evaluation']}"
                for i, eval_ in enumerate(evaluations)
            )
            
            result = await self.feedback_gen_chain.ainvoke({
                "job_requirements": "Overall interview performance",
                "candidate_skills": eval_text,
                "few_shot_examples": SKILL_ANALYSIS_EXAMPLES,
                "focus_area_info": ""
            })
            return {"feedback": result}
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            raise 