from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

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

logger = setup_logger(__name__)


class AgentOutput(BaseModel):
    """Standardized output for agent responses"""
    output: str = Field(..., description="The agent's response")


class RecruitingAgent:
    """Agent for handling recruiting tasks"""

    def __init__(self, temperature: float = 0.7):
        """Initialize the agent with tools and LLM"""
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL, 
            temperature=temperature,
            callbacks=None  # Exclude callbacks from serialization
        )
        
        # Initialize chains
        self.match_chain = CandidateJobMatchChain(llm=self.llm)
        self.interview_chain = InterviewWorkflowChain(llm=self.llm)
        
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

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            input_key="input"
        )

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
