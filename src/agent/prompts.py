from typing import Dict, List
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, Field

# Template for the agent's system message
SYSTEM_TEMPLATE = """You are an AI Recruiting Assistant that helps match jobs with candidates and vice versa.
You have access to a database of job postings and candidate profiles.
You can search through these using semantic search to find the best matches.

Your goal is to help users find:
1. Relevant job postings based on their criteria
2. Qualified candidates for specific job positions
3. Best matches between jobs and candidates

When responding:
- Be concise but informative
- Focus on the most relevant matches
- Consider both hard skills and soft requirements
- Explain your reasoning for suggested matches
- Provide specific details from the profiles/postings

Available tools:
{tools}

To use a tool, respond with:
```
Thought: Consider what tool to use and why
Action: The tool to use
Action Input: The input for the tool
```

After using a tool, respond with:
```
Observation: The result of the tool
Thought: Consider the next step
```

Begin!"""

class AgentPromptTemplate(StringPromptTemplate):
    """Template for agent prompts"""
    
    def __init__(self, tools: List[str]):
        super().__init__(template="", input_variables=["input", "chat_history", "intermediate_steps"])
        self.tools = tools
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        chat_history = kwargs.get("chat_history", [])
        
        # Format chat history
        history = ""
        for message in chat_history:
            if message.type == "human":
                history += f"\nHuman: {message.content}"
            else:
                history += f"\nAssistant: {message.content}"
        
        # Format intermediate steps
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
        
        # Base template
        template = """You are an AI recruiter assistant. Your goal is to help match candidates with jobs and provide insights.

Previous conversation:
{history}

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
            history=history,
            thoughts=thoughts,
            input=kwargs["input"],
            tools="\n".join(self.tools)
        ) 