from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
import json

from src.core.logging import setup_logger
from src.agent.tools import (
    SearchJobsTool,
    SearchCandidatesTool,
    MatchJobCandidatesTool,
    SkillAnalysisTool,
    InterviewQuestionGenerator,
)
from src.agent.chains import CandidateJobMatchChain, InterviewWorkflowChain
from src.core.config import settings
from src.vector_store.chroma_store import ChromaStore

logger = setup_logger(__name__)


class AgentOutput(BaseModel):
    """Standardized output for agent responses"""
    output: str = Field(..., description="The agent's response")


class RecruitingAgent:
    """Agent for handling recruiting tasks"""

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        """Initialize the recruiting agent."""
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory()
        self.store = ChromaStore()
        
        # Initialize chains
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
                input_variables=["question", "response", "job_requirements"],
                template="""Evaluate the following interview response based on the job requirements:
                
                Question: {question}
                Response: {response}
                Job Requirements: {job_requirements}
                
                Provide a detailed evaluation including:
                1. Score (0-100)
                2. Key strengths demonstrated
                3. Areas for improvement
                4. Technical accuracy
                5. Communication clarity
                
                Format the response as a JSON object with these fields."""
            )
        )
        
        self.feedback_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["job_requirements", "interview_data"],
                template="""Generate comprehensive interview feedback based on the following data:
                
                Job Requirements:
                {job_requirements}
                
                Interview Responses:
                {interview_data}
                
                Provide detailed feedback including:
                1. Overall assessment
                2. Technical competency
                3. Cultural fit
                4. Key strengths
                5. Areas for development
                6. Hiring recommendation
                
                Format the response as a JSON object with these fields."""
            )
        )
        
        # Initialize tools
        self.tools = [
            SearchJobsTool,
            SearchCandidatesTool,
            MatchJobCandidatesTool,
            SkillAnalysisTool(),
            InterviewQuestionGenerator,
            Tool(
                name="detailed_match_analysis",
                func=self._run_match_analysis,
                description="Perform a detailed match analysis between a candidate and a job. Input should be job_id and resume_id separated by space."
            ),
            Tool(
                name="full_interview_workflow",
                func=self._run_interview_workflow,
                description="Run a complete interview workflow including question generation and evaluation. Input should be job_id and resume_id separated by space."
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

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

    async def _run_match_analysis(self, input_str: str) -> str:
        """Run detailed match analysis"""
        try:
            job_id, resume_id = input_str.split()
            # Get job and candidate info
            job_info = await self.tools[0].func(f"job_id:{job_id}")
            candidate_info = await self.tools[1].func(f"resume_id:{resume_id}")
            
            # Run the match chain
            result = await self.match_chain.run(
                candidate_info=candidate_info,
                job_info=job_info
            )
            
            return f"""
            DETAILED MATCH ANALYSIS
            ----------------------
            {result['candidate_summary']}
            
            JOB REQUIREMENTS
            ---------------
            {result['job_analysis']}
            
            SKILLS GAP ANALYSIS
            ------------------
            {result['skills_gap_analysis']}
            
            INTERVIEW STRATEGY
            -----------------
            {result['interview_strategy']}
            """
        except Exception as e:
            logger.error(f"Error in match analysis: {str(e)}")
            return f"Error performing match analysis: {str(e)}"

    async def _run_interview_workflow(self, input_str: str) -> str:
        """Run complete interview workflow"""
        try:
            parts = input_str.split()
            job_id = parts[0]
            resume_id = parts[1] if len(parts) > 1 else None
            
            # Get job and candidate info
            job_info = await self.tools[0].func(f"job_id:{job_id}")
            candidate_info = await self.tools[1].func(f"resume_id:{resume_id}") if resume_id else ""
            
            # Generate questions
            focus_areas = ["Technical Skills", "Problem Solving", "Experience", "Culture Fit"]
            questions_result = await self.interview_chain.generate_questions(
                job_info=job_info,
                candidate_info=candidate_info,
                focus_areas=focus_areas
            )
            
            return f"""
            INTERVIEW WORKFLOW PLAN
            ---------------------
            {questions_result['interview_questions']}
            
            Note: Use these questions as a guide for the interview.
            After each response, use the evaluate_response function to assess the answer.
            """
        except Exception as e:
            logger.error(f"Error in interview workflow: {str(e)}")
            return f"Error setting up interview workflow: {str(e)}"

    async def run(self, query: str) -> str:
        """Run the agent with the given query"""
        try:
            response = await self.agent_executor.ainvoke({"input": query})
            return response.get("output", "No response generated")
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"I encountered an error: {str(e)}"

    def reset_memory(self) -> None:
        """Reset the agent's memory"""
        self.memory.clear()

    async def generate_interview_questions(
        self,
        job_id: str,
        question_types: List[str]
    ) -> List[Dict[str, str]]:
        """Generate interview questions based on job requirements."""
        try:
            # Get job details
            job = await self.store.get_job_by_id(job_id)
            if not job:
                raise ValueError(f"Job with ID {job_id} not found")
                
            # Prepare job requirements
            requirements = f"""
            Title: {job.get('title', '')}
            Skills: {', '.join(job.get('skills', []))}
            Description: {job.get('description', '')}
            """
            
            # Generate questions
            response = await self.question_chain.arun(
                job_requirements=requirements,
                question_types=", ".join(question_types)
            )
            
            # Parse response into structured questions
            questions = []
            current_type = question_types[0]  # Default to first type
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line indicates question type
                if any(qt.lower() in line.lower() for qt in question_types):
                    current_type = next(qt for qt in question_types if qt.lower() in line.lower())
                    continue
                    
                # Parse question if it starts with a number
                if any(c.isdigit() for c in line):
                    # Remove leading number and dot
                    question_text = line.lstrip("0123456789. ")
                    questions.append({
                        "type": current_type,
                        "question": question_text,
                        "expected_answer": "Candidate should demonstrate relevant knowledge and experience"
                    })
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating interview questions: {str(e)}")
            raise
    
    async def evaluate_response(
        self,
        job_id: str,
        question: str,
        response: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a candidate's interview response."""
        try:
            # Get job details
            job = await self.store.get_job_by_id(job_id)
            if not job:
                raise ValueError(f"Job with ID {job_id} not found")
                
            # Prepare job requirements
            requirements = f"""
            Title: {job.get('title', '')}
            Skills: {', '.join(job.get('skills', []))}
            Description: {job.get('description', '')}
            """
            
            # Add expected answer if provided
            if expected_answer:
                requirements += f"\nExpected Answer: {expected_answer}"
            
            # Evaluate response
            result = await self.evaluation_chain.arun(
                question=question,
                response=response,
                job_requirements=requirements
            )
            
            # Parse result into structured format
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    # Convert string result to structured format
                    result = {
                        "score": 0,
                        "strengths": [],
                        "improvements": [],
                        "technical_accuracy": "Poor",
                        "communication": "Poor"
                    }
            
            # Ensure all required fields are present
            if "Score" in result:
                result["score"] = result.pop("Score")
            if "Key strengths demonstrated" in result:
                result["strengths"] = result.pop("Key strengths demonstrated")
            if "Areas for improvement" in result:
                result["improvements"] = result.pop("Areas for improvement")
            if "Communication clarity" in result:
                result["communication"] = result.pop("Communication clarity")
            
            # Add missing fields with defaults if needed
            result.setdefault("score", 0)
            result.setdefault("strengths", [])
            result.setdefault("improvements", [])
            result.setdefault("technical_accuracy", "Poor")
            result.setdefault("communication", "Poor")
            
            # Ensure score is a number between 0 and 100
            if isinstance(result["score"], str):
                try:
                    result["score"] = float(result["score"].rstrip("%"))
                except (ValueError, AttributeError):
                    result["score"] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise
    
    async def generate_interview_feedback(
        self,
        interview_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive interview feedback."""
        try:
            # Get job details
            job = await self.store.get_job_by_id(interview_data["job_id"])
            if not job:
                raise ValueError(f"Job with ID {interview_data['job_id']} not found")
                
            # Prepare job requirements
            requirements = f"""
            Title: {job.get('title', '')}
            Skills: {', '.join(job.get('skills', []))}
            Description: {job.get('description', '')}
            """
            
            # Calculate overall score from responses
            response_scores = [r.get("score", 0) for r in interview_data["responses"]]
            overall_score = sum(response_scores) / len(response_scores) if response_scores else 0
            
            # Generate feedback
            feedback_str = await self.feedback_chain.arun(
                job_requirements=requirements,
                interview_data=str(interview_data["responses"])
            )
            
            # Parse feedback and ensure proper structure
            try:
                feedback = json.loads(feedback_str)
            except json.JSONDecodeError:
                # If feedback is a string, convert it to structured format
                feedback = {
                    "Key strengths": [],
                    "Areas for development": [],
                    "Hiring recommendation": "Not recommended"
                }
            
            # Ensure Key strengths is a list
            if "Key strengths" in feedback and not isinstance(feedback["Key strengths"], list):
                feedback["Key strengths"] = [feedback["Key strengths"]]
            elif "Key strengths" not in feedback:
                feedback["Key strengths"] = []
            
            # Add overall score
            feedback["overall_score"] = overall_score
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating interview feedback: {str(e)}")
            raise
