"""Tool implementations for the recruiting agent."""

from src.agent.tools.base import BaseRecruitingTool
from src.agent.tools.search import (
    JobSearchTool,
    CandidateSearchTool,
    JobSearchInput,
    CandidateSearchInput
)
from src.agent.tools.analysis import (
    SkillAnalysisTool,
    SkillAnalysisInput
)
from src.agent.tools.interview import (
    InterviewQuestionTool,
    ResponseEvaluationTool,
    FeedbackGenerationTool,
    InterviewQuestionInput,
    ResponseEvaluationInput,
    InterviewFeedbackInput
)

# Registry of all available tools
TOOL_REGISTRY = {
    "search_jobs": JobSearchTool,
    "search_candidates": CandidateSearchTool,
    "skill_analysis": SkillAnalysisTool,
    "generate_questions": InterviewQuestionTool,
    "evaluate_response": ResponseEvaluationTool,
    "generate_feedback": FeedbackGenerationTool
}

__all__ = [
    # Search tools
    "JobSearchTool",
    "CandidateSearchTool",
    "JobSearchInput",
    "CandidateSearchInput",
    
    # Analysis tools
    "SkillAnalysisTool",
    "SkillAnalysisInput",
    
    # Interview tools
    "InterviewQuestionTool",
    "ResponseEvaluationTool",
    "FeedbackGenerationTool",
    "InterviewQuestionInput",
    "ResponseEvaluationInput",
    "InterviewFeedbackInput",
    
    # Tool registry
    "get_tool"
]

def get_tool(tool_name: str) -> BaseRecruitingTool:
    """Get a tool instance by name.
    
    Args:
        tool_name: Name of the tool to get
        
    Returns:
        Tool instance
        
    Raises:
        ValueError: If tool_name is not found in registry
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool_name} not found in registry")
    
    return TOOL_REGISTRY[tool_name]() 