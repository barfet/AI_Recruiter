from fastapi import APIRouter, HTTPException
from src.api.models import QueryRequest, QueryResponse
from src.agent.agent import RecruitingAgent
from src.core.logging import setup_logger

logger = setup_logger(__name__)
router = APIRouter()
agent = RecruitingAgent()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a query using the recruiting agent"""
    try:
        response = await agent.run(request.query)
        return QueryResponse(
            response=response,
            success=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            response="",
            success=False,
            error=str(e)
        )

@router.post("/reset")
async def reset_agent():
    """Reset the agent's memory"""
    try:
        agent.reset_memory()
        return {"message": "Agent memory reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting agent memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset agent memory: {str(e)}"
        ) 