from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory

from src.core.config import settings
from src.core.logging import setup_logger
from src.agent.prompts import AgentPromptTemplate
from src.agent.tools import (
    SearchJobsTool,
    SearchCandidatesTool,
    MatchJobCandidatesTool
)

logger = setup_logger(__name__)

class RecruitingAgent:
    """Main recruiting agent class"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE,
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize tools
        self.tools = [
            SearchJobsTool(),
            SearchCandidatesTool(),
            MatchJobCandidatesTool()
        ]
        
        # Initialize prompt template
        self.prompt = AgentPromptTemplate(
            tools=[tool.description for tool in self.tools]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            prompt=self.prompt,
            tools=self.tools,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        # Initialize agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
        
    async def run(self, query: str) -> str:
        """Run the agent with a query"""
        try:
            response = await self.agent_executor.arun(input=query)
            return response
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"Error: {str(e)}"
            
    def reset_memory(self) -> None:
        """Reset the agent's memory"""
        self.memory.clear() 