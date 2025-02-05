"""Agent for recruiting tasks."""

import json
import logging
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.core.logging import setup_logger
from src.agent.tools.search import JobSearchTool, CandidateSearchTool
from src.agent.tools.analysis import SkillAnalysisTool
from src.agent.tools.interview import (
    InterviewQuestionTool,
    ResponseEvaluationTool,
    FeedbackGenerationTool
)
from src.agent.chains import CandidateJobMatchChain, InterviewWorkflowChain
from src.core.config import settings
from src.vector_store.chroma_store import ChromaStore
from langchain.chains import LLMChain

logger = setup_logger(__name__)


class AgentOutput(BaseModel):
    """Standardized output for agent responses"""
    output: str = Field(..., description="The agent's response")


class RecruitingAgent:
    """Agent for handling recruiting tasks"""

    def __init__(self) -> None:
        """Initialize the recruiting agent."""
        self.llm = ChatOpenAI(**settings.LLM_CONFIG)
        self.memory = ConversationBufferMemory()
        self.store = ChromaStore()
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized chains
        self.question_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["job_requirements", "question_types"],
                template="""Based on the following job requirements, generate relevant interview questions:
                
                Job Requirements:
                {job_requirements}
                
                Question Types: {question_types}
                
                Generate 5 specific and detailed questions that will help assess the candidate's fit for this role.
                Format each question on a new line starting with a number."""
            )
        )
        
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["candidate_response", "job_requirements", "question"],
                template="""Evaluate the candidate's response against the job requirements:
                
                Job Requirements:
                {job_requirements}
                
                Question Asked:
                {question}
                
                Candidate's Response:
                {candidate_response}
                
                Provide a detailed evaluation of how well the response demonstrates the required skills and experience.
                Include specific strengths and areas for improvement."""
            )
        )
        
        self.feedback_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["evaluation", "job_requirements"],
                template="""Based on the evaluation and job requirements, provide constructive feedback:
                
                Evaluation:
                {evaluation}
                
                Job Requirements:
                {job_requirements}
                
                Provide specific, actionable feedback that will help the candidate improve and better align with the role requirements."""
            )
        )
        
        # Initialize tool instances
        self.job_search = JobSearchTool()
        self.candidate_search = CandidateSearchTool()
        self.skill_analysis = SkillAnalysisTool()
        self.question_generator = InterviewQuestionTool()
        self.response_evaluator = ResponseEvaluationTool()
        self.feedback_generator = FeedbackGenerationTool()
        
        # Initialize tools for agent
        self.tools = [
            Tool(
                name="search_jobs",
                description="Search for job postings using semantic search. Input should be a description of the job you're looking for.",
                func=self.job_search._arun,
                args_schema=self.job_search.args_schema
            ),
            Tool(
                name="search_candidates",
                description="Search for candidate profiles using semantic search. Input should be a description of the candidate you're looking for.",
                func=self.candidate_search._arun,
                args_schema=self.candidate_search.args_schema
            ),
            Tool(
                name="analyze_skills",
                description="Analyze skill match between a job and a candidate. Input should be a JSON string with job_id and resume_id.",
                func=self.skill_analysis._arun,
                args_schema=self.skill_analysis.args_schema
            ),
            Tool(
                name="generate_interview_questions",
                description="Generate interview questions based on job requirements. Input should be a JSON string with job_id and optional resume_id.",
                func=self.question_generator._arun,
                args_schema=self.question_generator.args_schema
            ),
            Tool(
                name="evaluate_response",
                description="Evaluate a candidate's response to an interview question. Input should be a JSON string with resume_id, job_id, response, and question.",
                func=self.response_evaluator._arun,
                args_schema=self.response_evaluator.args_schema
            ),
            Tool(
                name="generate_feedback",
                description="Generate comprehensive interview feedback. Input should be a JSON string with job_id, resume_id, responses, and evaluations.",
                func=self.feedback_generator._arun,
                args_schema=self.feedback_generator.args_schema
            ),
            Tool(
                name="detailed_match_analysis",
                description="Perform a detailed match analysis between a candidate and a job. Input should be job_id and resume_id separated by space.",
                func=self._run_match_analysis,
                args_schema=self.skill_analysis.args_schema
            ),
            Tool(
                name="full_interview_workflow",
                description="Run a complete interview workflow including question generation and evaluation. Input should be job_id and resume_id separated by space.",
                func=self._run_interview_workflow,
                args_schema=self.question_generator.args_schema
            )
        ]

        system_message = SystemMessagePromptTemplate.from_template(
            """You are an AI recruiting assistant. Your role is to help with:
            1. Finding relevant job postings based on requirements
            2. Finding qualified candidates based on job requirements
            3. Matching candidates with suitable job opportunities
            4. Analyzing skill matches between candidates and jobs
            5. Generating tailored interview questions
            6. Running complete interview workflows

            RESPONSE FORMAT RULES:
            1. For job searches:
               - ALWAYS start with [Job ID: <id>] on a new line for EACH job
               - Include job title, company, and location
               - Explain why it's a good match
            
            2. For candidate searches:
               - ALWAYS start with [Resume ID: <id>] on a new line for EACH candidate
               - Include skills and industry
               - Explain why they're a good match
            
            3. For skill analysis:
               - Use the analyze_skills tool with job_id and resume_id separated by a space
               - Present the match percentage prominently
               - List matching and missing skills clearly
            
            4. For interview questions:
               - Use the generate_questions tool with job_id (and optionally resume_id) separated by a space
               - Organize questions by category (technical, behavioral, etc.)
               - Ensure questions are specific to the job requirements
            
            5. For detailed match analysis:
               - Use the detailed_match_analysis tool for comprehensive analysis
               - Review all aspects: skills, experience, and qualifications
               - Provide specific recommendations
            
            6. For interview workflows:
               - Use the full_interview_workflow tool for end-to-end process
               - Follow up on areas needing clarification
               - Provide structured feedback and recommendations

            ERROR HANDLING:
            - If a tool call fails, clearly state what went wrong
            - If IDs are not found, ask for alternative IDs
            - If inputs are missing, specify exactly what's needed

            IMPORTANT: 
            - NEVER skip the ID format specifications
            - ALWAYS validate tool inputs before calling
            - Keep responses concise but informative
            - Focus on the most relevant matches first"""
        )

        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        } | prompt | self.llm.bind(functions=[t.dict() for t in self.tools]) | OpenAIFunctionsAgentOutputParser()

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

    async def _run_match_analysis(self, input_str: str) -> str:
        """Run a detailed match analysis between a job and a candidate."""
        try:
            job_id, resume_id = input_str.split()
            
            # Get job requirements using skill analysis tool
            job_data = await self.skill_analysis._arun({
                "job_id": job_id,
                "resume_id": resume_id,
                "analysis_type": "requirements"
            })
            
            # Analyze skills
            skill_analysis = await self.skill_analysis._arun({
                "job_id": job_id,
                "resume_id": resume_id,
                "analysis_type": "match"
            })
            
            return json.dumps({
                "job_data": json.loads(job_data),
                "skill_analysis": json.loads(skill_analysis)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error in match analysis: {str(e)}")
            return f"Error: {str(e)}"

    async def _run_interview_workflow(self, input_str: str) -> str:
        """Run a complete interview workflow."""
        try:
            job_id, resume_id = input_str.split()
            
            # Get job requirements
            job_data = await self.skill_analysis._arun({
                "job_id": job_id,
                "resume_id": resume_id,
                "analysis_type": "requirements"
            })
            job_data = json.loads(job_data)
            
            # Generate questions
            questions = await self.question_generator._arun({
                "job_id": job_id,
                "phase": "technical",
                "focus_skills": job_data.get("skills", [])
            })
            
            return json.dumps({
                "job_data": job_data,
                "questions": json.loads(questions)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error in interview workflow: {str(e)}")
            return f"Error: {str(e)}"

    async def generate_interview_questions(
        self,
        job_id: str,
        question_types: List[str]
    ) -> List[Dict[str, str]]:
        """Generate interview questions based on job requirements."""
        try:
            # Get job data first
            job_result = await self.job_search._arun({"query": f"id:{job_id}", "limit": 1})
            job_data = json.loads(job_result)
            
            if not job_data.get("data"):
                raise ValueError(f"No job found with ID: {job_id}")
            
            job = job_data["data"][0] if isinstance(job_data["data"], list) else job_data["data"]
            
            # Extract requirements and skills
            requirements = {
                "title": job.get("title", ""),
                "description": job.get("description", ""),
                "skills": job.get("skills", []),
                "experience": job.get("experience", "")
            }
            
            all_questions = []
            for question_type in question_types:
                # Use question chain with updated prompt
                result = await self.question_chain.ainvoke({
                    "job_requirements": json.dumps(requirements, indent=2),
                    "question_types": question_type
                })
                
                # Parse questions from result
                try:
                    questions = json.loads(result["text"])
                    if isinstance(questions, list):
                        all_questions.extend(questions)
                    elif isinstance(questions, dict):
                        all_questions.append(questions)
                except (json.JSONDecodeError, KeyError):
                    # If not JSON, try to parse as text
                    questions = result["text"].split("\n")
                    questions = [q.strip() for q in questions if q.strip()]
                    all_questions.extend([{"question": q} for q in questions])
            
            return all_questions
            
        except Exception as e:
            self.logger.error(f"Error generating interview questions: {str(e)}")
            return []

    async def run(self, input_str: str) -> AgentOutput:
        """Run the agent with the given input."""
        try:
            result = await self.agent_executor.ainvoke({"input": input_str})
            return AgentOutput(output=result["output"])
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return AgentOutput(output=f"Error: {str(e)}")

    def reset_memory(self) -> None:
        """Reset the agent's memory"""
        self.memory.clear()

    async def evaluate_response(
        self,
        job_id: str,
        question: str,
        response: str,
        expected_signals: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate a candidate's response to an interview question."""
        try:
            evaluation = await self.response_evaluator._arun({
                "job_id": job_id,
                "question": question,
                "response": response,
                "expected_signals": expected_signals or []
            })
            
            return json.loads(evaluation)

        except Exception as e:
            self.logger.error(f"Error evaluating response: {str(e)}")
            return {
                "score": 0,
                "strengths": [],
                "improvements": [],
                "technical_accuracy": 0,
                "communication": 0
            }
    
    async def generate_interview_feedback(self, interview_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive feedback for an interview."""
        try:
            feedback = await self.feedback_generator._arun({
                "job_id": interview_data["job_id"],
                "resume_id": interview_data.get("resume_id"),
                "responses": interview_data["responses"],
                "evaluations": interview_data.get("evaluations", {})
            })
            
            result = json.loads(feedback)
            
            # Ensure required fields are present
            if not result.get("overall_score"):
                scores = [r.get("score", 0) for r in interview_data.get("evaluations", {}).values()]
                result["overall_score"] = sum(scores) / len(scores) if scores else 0
                
            if not result.get("Key strengths"):
                result["Key strengths"] = ["No specific strengths identified"]
            if not result.get("Areas for development"):
                result["Areas for development"] = ["No specific areas for development identified"]
            
            return result

        except Exception as e:
            self.logger.error(f"Error generating feedback: {str(e)}")
            return {
                "overall_score": 0,
                "Key strengths": ["Unable to identify strengths due to processing error"],
                "Areas for development": ["Unable to identify areas for development due to processing error"],
                "Hiring recommendation": "Not recommended"
            }
