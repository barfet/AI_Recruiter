from typing import List, Dict, Optional
from langchain.prompts import StringPromptTemplate, PromptTemplate, FewShotPromptTemplate

# Template for the agent's system message
SYSTEM_TEMPLATE = """You are an AI Recruiting Assistant that helps match jobs with candidates.
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
        super().__init__(
            template="", input_variables=["input", "chat_history", "intermediate_steps"]
        )
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
        template = (
            "You are an AI recruiter assistant. Your goal is to help match "
            "candidates with jobs and provide insights.\n\n"
            "Previous conversation:\n{history}\n\n"
            "Previous actions and observations:\n{thoughts}\n\n"
            "Current task: {input}\n\n"
            "Available tools:\n{tools}\n\n"
            "Think through what you need to do step by step. Then use the appropriate tool.\n"
            "Response should be in this format:\n"
            "Thought: your thought process\n"
            "Action: tool name\n"
            "Action Input: input for the tool\n\n"
            "or, if you have a final answer:\n"
            "Thought: your thought process\n"
            "Final Answer: your final response"
        )

        # Fill in the template
        return template.format(
            history=history,
            thoughts=thoughts,
            input=kwargs["input"],
            tools="\n".join(self.tools),
        )

# Few-shot examples for job matching
JOB_MATCH_EXAMPLES = [
    {
        "query": "Senior Python Developer with AWS",
        "response": """[Job ID: 3904095574] Senior Software Engineer at TechCorp
- Perfect match: Strong Python and AWS focus
- Relevant cloud architecture experience
- Matches team size and culture

[Job ID: 3895208319] Backend Engineer at StartupX
- Good match: Python-first stack
- Some AWS requirements
- Different seniority level"""
    },
    {
        "query": "ML Engineer with PyTorch",
        "response": """[Job ID: 3904407835] ML Engineer at AI Labs
- Exact match: PyTorch is primary framework
- Relevant research background needed
- Matching compensation range

[Job ID: 3904095123] Data Scientist at DataCo
- Partial match: Uses PyTorch occasionally
- More focus on analysis than engineering
- Different core responsibilities"""
    }
]

# Few-shot examples for skill analysis
SKILL_ANALYSIS_EXAMPLES = [
    {
        "job_requirements": "Python, AWS, Kubernetes, CI/CD",
        "candidate_skills": "Python, Docker, AWS, Jenkins",
        "analysis": """Match Score: 85%
Strong Matches:
- Python: Expert level in both
- AWS: Extensive cloud experience
Partial Matches:
- CI/CD: Jenkins experience translates well
Gaps:
- Kubernetes: Missing but has Docker foundation
Recommendations:
- Focus on container orchestration training"""
    }
]

def create_dynamic_job_prompt(context: Optional[Dict] = None) -> PromptTemplate:
    """Create a dynamic job search prompt based on context"""
    
    example_prompt = PromptTemplate(
        input_variables=["query", "response"],
        template="""Query: {query}\nResponse: {response}"""
    )
    
    # Base template
    template = """You are an expert recruiter. Find the most relevant jobs based on the query.
    
{context_info}

{few_shot_examples}

Query: {query}
Response: """

    # Add context-specific information
    context_info = ""
    if context:
        if context.get("industry"):
            context_info += f"\nIndustry Focus: {context['industry']}"
        if context.get("experience_level"):
            context_info += f"\nExperience Level: {context['experience_level']}"
        if context.get("location"):
            context_info += f"\nLocation Preference: {context['location']}"

    return PromptTemplate(
        template=template,
        input_variables=["query", "few_shot_examples", "context_info"],
        partial_variables={"context_info": context_info}
    )

def create_skill_analysis_prompt(
    focus_areas: Optional[List[str]] = None
) -> PromptTemplate:
    """Create a skill analysis prompt with optional focus areas"""
    
    example_prompt = PromptTemplate(
        input_variables=["job_requirements", "candidate_skills", "analysis"],
        template="""Job Requirements: {job_requirements}
Candidate Skills: {candidate_skills}
Analysis: {analysis}"""
    )
    
    template = """Analyze the match between job requirements and candidate skills.
    
{focus_area_info}

{few_shot_examples}

Job Requirements: {job_requirements}
Candidate Skills: {candidate_skills}
Response: """

    # Add focus areas if specified
    focus_area_info = ""
    if focus_areas:
        focus_area_info = "Focus Areas:\n" + "\n".join(f"- {area}" for area in focus_areas)

    return PromptTemplate(
        template=template,
        input_variables=["job_requirements", "candidate_skills", "few_shot_examples", "focus_area_info"],
        partial_variables={"focus_area_info": focus_area_info}
    )

# Interview question templates with difficulty levels
INTERVIEW_TEMPLATES = {
    "technical": {
        "basic": """Generate a basic technical question about {topic}.
Consider the problem: {context}
Use examples from: {few_shot_examples}""",
        "intermediate": """Generate an intermediate technical question about {topic}.
Consider the problem: {context}
Use examples from: {few_shot_examples}""",
        "advanced": """Generate an advanced technical question about {topic}.
Consider the problem: {context}
Use examples from: {few_shot_examples}"""
    },
    "behavioral": {
        "basic": """Generate a basic behavioral question about {topic}.
Consider the context: {context}
Use examples from: {few_shot_examples}""",
        "intermediate": """Generate an intermediate behavioral question about {topic}.
Consider the context: {context}
Use examples from: {few_shot_examples}""",
        "advanced": """Generate an advanced behavioral question about {topic}.
Consider the context: {context}
Use examples from: {few_shot_examples}"""
    }
}

def create_interview_prompt(
    difficulty: str = "intermediate",
    question_type: str = "technical"
) -> PromptTemplate:
    """Create an interview question prompt based on difficulty and type"""
    
    template = INTERVIEW_TEMPLATES[question_type][difficulty]
    
    return PromptTemplate(
        template=template,
        input_variables=["topic", "context", "few_shot_examples"]
    )

# Dynamic system message templates
SYSTEM_MESSAGES = {
    "default": """You are an AI recruiting assistant. Help with:
- Finding relevant jobs and candidates
- Analyzing skill matches
- Generating interview questions
- Providing structured feedback""",
    
    "technical": """You are a technical recruiter specializing in:
- Engineering role evaluation
- Technical skill assessment
- System design questions
- Coding challenge review""",
    
    "executive": """You are an executive recruiter focusing on:
- Leadership role evaluation
- Strategic thinking assessment
- Team building capabilities
- Executive presence analysis"""
}

def get_system_message(role_type: str = "default") -> str:
    """Get appropriate system message based on role type"""
    return SYSTEM_MESSAGES.get(role_type, SYSTEM_MESSAGES["default"])
