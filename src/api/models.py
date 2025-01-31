from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for agent queries"""
    query: str = Field(..., description="The query to process")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to the search"
    )

class QueryResponse(BaseModel):
    """Response model for agent queries"""
    response: str = Field(..., description="The agent's response")
    success: bool = Field(..., description="Whether the query was successful")
    error: Optional[str] = Field(None, description="Error message if query failed") 