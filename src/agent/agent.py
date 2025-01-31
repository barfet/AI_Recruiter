from typing import List, Dict, Any
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

from src.core.logging import setup_logger
from src.agent.tools import SearchJobsTool, SearchCandidatesTool, MatchJobCandidatesTool
from src.core.config import OPENAI_MODEL

logger = setup_logger(__name__)

class RecruitingAgent:
    """Agent for handling recruiting tasks"""
    
    def __init__(self, temperature: float = 0.7):
        """Initialize the agent with tools and LLM"""
        self.tools = [
            SearchJobsTool(),
            SearchCandidatesTool(),
            MatchJobCandidatesTool()
        ]
        
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=temperature
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message="""You are an AI recruiting assistant. Your role is to help with:
            1. Finding relevant job postings based on requirements
            2. Finding qualified candidates based on job requirements
            3. Matching candidates with suitable job opportunities
            
            Be concise but informative in your responses. Focus on the most relevant matches 
            and explain why they are good fits. Always consider both technical skills and 
            other factors like location and experience level when making recommendations."""
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )
    
    async def run(self, query: str) -> str:
        """Run the agent with the given query"""
        try:
            response = await self.agent_executor.arun(
                input=query
            )
            return response
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"I encountered an error: {str(e)}"

    def reset_memory(self) -> None:
        """Reset the agent's memory"""
        self.memory.clear() 