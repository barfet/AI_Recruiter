from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage

from src.core.logging import setup_logger
from src.agent.tools import (
    SearchJobsTool,
    SearchCandidatesTool,
    MatchJobCandidatesTool,
    SkillAnalysisTool,
    InterviewQuestionGenerator,
)
from src.core.config import settings

logger = setup_logger(__name__)


class RecruitingAgent:
    """Agent for handling recruiting tasks"""

    def __init__(self, temperature: float = 0.7):
        """Initialize the agent with tools and LLM"""
        self.tools = [
            SearchJobsTool,
            SearchCandidatesTool,
            MatchJobCandidatesTool,
            SkillAnalysisTool,
            InterviewQuestionGenerator,
        ]

        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=temperature)

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
        )

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
