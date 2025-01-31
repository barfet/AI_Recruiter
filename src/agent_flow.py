from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv
from embeddings import EmbeddingManager

# Load environment variables
load_dotenv()

class SearchJobsTool(BaseTool):
    name = "search_jobs"
    description = "Search for job postings using semantic search"
    
    def __init__(self, embedding_manager: EmbeddingManager):
        super().__init__()
        self.embedding_manager = embedding_manager
    
    def _run(self, query: str) -> str:
        """Run job search"""
        results = self.embedding_manager.similarity_search(query, 'jobs')
        return json.dumps([{
            'job_id': r[0].metadata['job_id'],
            'title': r[0].metadata['title'],
            'company': r[0].metadata['company'],
            'location': r[0].metadata['location'],
            'score': r[1]
        } for r in results])
    
    def _arun(self, query: str):
        """Async implementation"""
        raise NotImplementedError("Async not implemented")

class SearchCandidatesTool(BaseTool):
    name = "search_candidates"
    description = "Search for candidate profiles using semantic search"
    
    def __init__(self, embedding_manager: EmbeddingManager):
        super().__init__()
        self.embedding_manager = embedding_manager
    
    def _run(self, query: str) -> str:
        """Run candidate search"""
        results = self.embedding_manager.similarity_search(query, 'candidates')
        return json.dumps([{
            'candidate_id': r[0].metadata['candidate_id'],
            'name': r[0].metadata['name'],
            'industry': r[0].metadata['industry'],
            'score': r[1]
        } for r in results])
    
    def _arun(self, query: str):
        """Async implementation"""
        raise NotImplementedError("Async not implemented")

class AgentPromptTemplate(StringPromptTemplate):
    """Template for agent prompts"""
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        
        # Generate the prompt
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
        
        # Set up the base template
        template = """You are an AI recruiter assistant. Your goal is to help match candidates with jobs and provide insights.

Previous actions and observations:
{thoughts}

Current task: {input}

Available tools:
{tools}

Think through what you need to do step by step. Then use the appropriate tool.
Response should be in this format:
Thought: your thought process
Action: tool name
Action Input: input for the tool

or, if you have a final answer:
Thought: your thought process
Final Answer: your final response"""
        
        # Fill in the template
        return template.format(
            thoughts=thoughts,
            input=kwargs["input"],
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in kwargs["tools"]])
        )

class RecruitingAgent:
    """Main agent class for recruiting tasks"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.load_embeddings()
        
        # Initialize tools
        self.tools = [
            SearchJobsTool(self.embedding_manager),
            SearchCandidatesTool(self.embedding_manager)
        ]
        
        # Set up the prompt template
        self.prompt = AgentPromptTemplate(
            template="",  # Template is defined in the format method
            input_variables=["input", "intermediate_steps", "tools"]
        )
        
        # Initialize the agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=None,  # We'll implement this later
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        # Set up the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def run(self, query: str) -> str:
        """Run the agent on a query"""
        return self.agent_executor.run(query)

if __name__ == "__main__":
    # Example usage
    agent = RecruitingAgent()
    result = agent.run("Find software engineering jobs in San Francisco") 