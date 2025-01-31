from typing import Any, Dict, List, Optional, Type, TypeVar
from pathlib import Path
import json
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import setup_logger
from src.core.exceptions import DataIngestionError

T = TypeVar('T', bound=BaseModel)
logger = setup_logger(__name__)

class BaseManager:
    """Base class for data managers"""
    
    def __init__(self, model: Type[T]):
        self.model = model
        self.processed_data_dir = settings.PROCESSED_DATA_DIR
        self.raw_data_dir = settings.RAW_DATA_DIR
        
    def save_processed_data(self, data: List[T], filename: str) -> None:
        """Save processed data to JSON file"""
        try:
            output_path = self.processed_data_dir / filename
            with open(output_path, 'w') as f:
                json.dump([item.dict() for item in data], f, indent=2)
            logger.info(f"Saved {len(data)} items to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {str(e)}")
            raise
            
    def load_processed_data(self, filename: str) -> List[T]:
        """Load processed data from JSON file"""
        try:
            input_path = self.processed_data_dir / filename
            with open(input_path, 'r') as f:
                data = json.load(f)
            return [self.model(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise
            
    def validate_data(self, data: Dict[str, Any]) -> T:
        """Validate data against the model schema"""
        try:
            return self.model(**data)
        except Exception as e:
            logger.warning(f"Validation error: {str(e)}")
            raise
        
    def load_json_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise DataIngestionError(f"Failed to load data from {file_path}")
            
    def save_json_data(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Save data to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise DataIngestionError(f"Failed to save data to {file_path}")
            
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw data before validation (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement process_raw_data") 