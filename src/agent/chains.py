"""Multi-step chains for complex recruiting workflows"""

from typing import Dict, List, Optional
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0
        )
        self._init_chains()

    def _init_chains(self):
        """Initialize all component chains"""
        # 1. Candidate Summary Chain with dynamic context
        candidate_summary_prompt = create_dynamic_job_prompt(context={"role_type": "technical"})
        self.candidate_summary_chain = candidate_summary_prompt | self.llm

        # 2. Job Analysis Chain with focus areas
        job_analysis_prompt = create_skill_analysis_prompt(
            focus_areas=["Technical Skills", "Experience", "Culture Fit"]
        )
        self.job_analysis_chain = job_analysis_prompt | self.llm

        # 3. Skills Gap Analysis Chain
        skills_gap_prompt = create_skill_analysis_prompt(
            focus_areas=["Required Skills", "Nice-to-Have Skills", "Missing Skills"]
        )
        self.skills_gap_chain = skills_gap_prompt | self.llm

        # 4. Interview Strategy Chain
        interview_strategy_prompt = create_interview_prompt(
            difficulty="advanced",
            question_type="technical"
        )
        self.interview_strategy_chain = interview_strategy_prompt | self.llm

    async def run(self, candidate_info: str, job_info: str) -> Dict:
        """Run the full analysis chain"""
        try:
            # Run each chain in sequence
            candidate_summary = await self.candidate_summary_chain.ainvoke({
                "query": job_info,
                "few_shot_examples": JOB_MATCH_EXAMPLES,
                "context_info": ""
            })
            
            job_analysis = await self.job_analysis_chain.ainvoke({
                "job_requirements": job_info,
                "candidate_skills": candidate_info,
                "few_shot_examples": SKILL_ANALYSIS_EXAMPLES,
                "focus_area_info": ""
            })
            
            skills_gap = await self.skills_gap_chain.ainvoke({
                "job_requirements": job_info,
                "candidate_skills": candidate_info,
                "few_shot_examples": SKILL_ANALYSIS_EXAMPLES,
                "focus_area_info": ""
            })
            
            interview_strategy = await self.interview_strategy_chain.ainvoke({
                "topic": "system design",
                "context": "scalability and performance",
                "few_shot_examples": JOB_MATCH_EXAMPLES
            })
            
            return {
                "candidate_summary": candidate_summary,
                "job_analysis": job_analysis,
                "skills_gap_analysis": skills_gap,
                "interview_strategy": interview_strategy
            }
        except Exception as e:
            logger.error(f"Error running match chain: {str(e)}")
            raise


class InterviewWorkflowChain:
    """Chain for managing the interview process workflow"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0
        )
        self._init_chains()

    def _init_chains(self):
        """Initialize interview workflow chains"""
        # 1. Question Generation Chain with dynamic difficulty
        question_gen_prompt = create_interview_prompt(
            difficulty="intermediate",
            question_type="technical"
        )
        self.question_gen_chain = question_gen_prompt | self.llm

        # 2. Response Evaluation Chain
        response_eval_prompt = create_skill_analysis_prompt(
            focus_areas=["Technical Accuracy", "Communication", "Problem Solving"]
        )
        self.response_eval_chain = response_eval_prompt | self.llm

        # 3. Feedback Generation Chain
        feedback_gen_prompt = create_skill_analysis_prompt(
            focus_areas=["Strengths", "Areas for Improvement", "Overall Fit"]
        )
        self.feedback_gen_chain = feedback_gen_prompt | self.llm

    async def generate_questions(
        self,
        job_info: str,
        candidate_info: str,
        focus_areas: List[str],
        difficulty: str = "intermediate"
    ) -> Dict:
        """Generate interview questions with dynamic difficulty"""
        try:
            # Create a new prompt with the specified difficulty
            prompt = create_interview_prompt(
                difficulty=difficulty,
                question_type="technical" if "Technical" in focus_areas else "behavioral"
            )
            
            # Create a new chain with the updated prompt
            chain = prompt | self.llm
            
            # Format topic and context based on focus areas
            is_technical = "Technical" in focus_areas
            topic = "system design" if is_technical else "leadership experience"
            context = "scalability and performance" if is_technical else "team management and collaboration"
            
            result = await chain.ainvoke({
                "topic": topic,
                "context": context,
                "few_shot_examples": JOB_MATCH_EXAMPLES
            })
            return {"interview_questions": result}
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise

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