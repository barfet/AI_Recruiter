"""Custom exceptions for the application"""


class AIRecruiterException(Exception):
    """Base exception for AI Recruiter application"""

    pass


class DataIngestionError(Exception):
    """Raised when there is an error during data ingestion"""

    pass


class DataProcessingError(Exception):
    """Raised when there is an error during data processing"""

    pass


class EmbeddingError(Exception):
    """Raised when there is an error with embeddings"""

    pass


class ModelError(AIRecruiterException):
    """Raised when there's an error with ML models"""

    pass


class ConfigurationError(AIRecruiterException):
    """Raised when there's a configuration error"""

    pass


class ValidationError(Exception):
    """Raised when there is a validation error"""

    pass


class APIError(AIRecruiterException):
    """Raised when there's an API-related error"""

    pass


class ProcessingError(Exception):
    """Raised when there is an error processing data"""

    pass


class SearchError(Exception):
    """Raised when there is an error performing search"""

    pass


class AgentError(Exception):
    """Raised when there is an error with the agent"""

    pass
