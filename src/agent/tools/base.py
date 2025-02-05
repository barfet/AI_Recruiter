"""Base classes for recruiting tools."""
import logging
from typing import Any, Optional, TypeVar, Generic, Dict
from abc import ABC, abstractmethod
import json

from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.core.logging import setup_logger
from src.agent.models.outputs import StandardizedOutput
from src.core.config import settings

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

class BaseRecruitingTool(BaseModel, Generic[T, R], ABC):
    """Base class for all recruiting tools."""
    
    name: str
    description: str
    args_schema: type[T]
    return_schema: type[R] = StandardizedOutput
    
    llm: Optional[ChatOpenAI] = Field(
        default_factory=lambda: ChatOpenAI(**settings.LLM_CONFIG)
    )
    logger: Optional[logging.Logger] = None
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, *args, **kwargs):
        """Initialize the tool."""
        super().__init__(*args, **kwargs)
        if self.logger is None:
            self.logger = setup_logger(self.__class__.__name__)
    
    async def _handle_errors(self, coro: Any) -> str:
        """Handle errors in async operations.
        
        Args:
            coro: Coroutine to execute
            
        Returns:
            JSON string response
        """
        try:
            result = await coro
            if isinstance(result, (dict, str)):
                return StandardizedOutput(
                    status="success",
                    data=result
                ).to_json()
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json()

    @abstractmethod
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Run the tool asynchronously."""
        raise NotImplementedError("Async run not implemented")

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Run the tool synchronously."""
        raise NotImplementedError("Run not implemented")


class BaseChainTool(BaseRecruitingTool[T, R]):
    """Base class for tools that use LLM chains."""
    
    chain: Optional[LLMChain] = Field(default=None)
    
    def __init__(
        self,
        chain_template: str,
        input_variables: list[str],
        llm: Optional[ChatOpenAI] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the chain tool.
        
        Args:
            chain_template: Template string for the chain
            input_variables: List of input variable names
            llm: Optional LLM instance
            logger: Optional logger instance
        """
        super().__init__(llm=llm, logger=logger)
        self.chain = self._create_chain(chain_template, input_variables)

    def _create_chain(self, template: str, input_variables: list[str]) -> LLMChain:
        """Create an LLM chain.
        
        Args:
            template: Prompt template string
            input_variables: List of input variable names
            
        Returns:
            Configured LLMChain instance
        """
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        )

    async def _run_chain(self, **kwargs) -> str:
        """Run the chain with the given inputs."""
        try:
            # Run chain
            result = await self.chain.ainvoke(kwargs)
            text = result.content if hasattr(result, 'content') else str(result)
            
            # Clean up the text
            text = text.strip()
            
            # Remove any leading/trailing quotes and whitespace
            text = text.strip('"').strip()
            
            # Remove any leading/trailing brackets if they're unmatched
            if text.count('[') != text.count(']'):
                text = text.strip('[]')
            
            # Try to fix common JSON formatting issues
            text = text.replace('\n', ' ')  # Remove newlines in the middle of JSON
            text = ' '.join(text.split())  # Normalize whitespace
            
            # If the text starts with a number and period (like "1."), it's probably not JSON
            if text.split()[0].rstrip('.').isdigit():
                # Convert numbered list to JSON array
                questions = []
                current_question = {}
                
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # If line starts with a number, it's a new question
                    if line.split('.')[0].isdigit():
                        if current_question:
                            questions.append(current_question)
                        current_question = {"question": line.split('.', 1)[1].strip()}
                    elif "Expected" in line or "Signal" in line:
                        signals = line.split(':', 1)[1].strip() if ':' in line else line
                        current_question["expected_signals"] = [s.strip() for s in signals.split(',')]
                    elif "Follow-up" in line:
                        follow_ups = line.split(':', 1)[1].strip() if ':' in line else line
                        current_question["follow_ups"] = [f.strip() for f in follow_ups.split(',')]
                
                if current_question:
                    questions.append(current_question)
                
                if questions:
                    return StandardizedOutput(
                        status="success",
                        data=questions
                    ).to_json()
            
            try:
                # Try to parse as JSON
                data = json.loads(text)
                return StandardizedOutput(
                    status="success",
                    data=data
                ).to_json()
            except json.JSONDecodeError as e:
                # If it's not valid JSON, try to fix common issues
                if '[' not in text and '{' in text:
                    # Single object, wrap in array
                    try:
                        data = json.loads(f"[{text}]")
                        return StandardizedOutput(
                            status="success",
                            data=data
                        ).to_json()
                    except:
                        pass
                
                # Return as raw text if all JSON parsing fails
                return StandardizedOutput(
                    status="success",
                    data=text
                ).to_json()
                
        except Exception as e:
            self.logger.error(f"Error running chain: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json()