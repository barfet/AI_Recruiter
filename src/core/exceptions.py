class AIRecruiterException(Exception):
    """Base exception for AI Recruiter application"""
    pass

class DataIngestionError(AIRecruiterException):
    """Raised when there's an error during data ingestion"""
    pass

class EmbeddingError(AIRecruiterException):
    """Raised when there's an error during embedding creation or loading"""
    pass

class ModelError(AIRecruiterException):
    """Raised when there's an error with ML models"""
    pass

class ConfigurationError(AIRecruiterException):
    """Raised when there's a configuration error"""
    pass

class ValidationError(AIRecruiterException):
    """Raised when data validation fails"""
    pass

class APIError(AIRecruiterException):
    """Raised when there's an API-related error"""
    pass 