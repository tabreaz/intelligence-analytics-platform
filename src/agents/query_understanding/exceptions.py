# src/agents/query_understanding/exceptions.py
"""
Custom exceptions for query understanding
"""
from typing import Optional, Dict, Any


class QueryUnderstandingError(Exception):
    """Base exception for query understanding errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(QueryUnderstandingError):
    """Input validation error"""

    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(
            f"Validation error for {field}: {message}",
            {"field": field, "value": value}
        )
        self.field = field
        self.value = value


class ClassificationError(QueryUnderstandingError):
    """Error during query classification"""
    pass


class LocationExtractionError(QueryUnderstandingError):
    """Error during location extraction"""
    pass


class SessionError(QueryUnderstandingError):
    """Error with session management"""
    pass


class StorageError(QueryUnderstandingError):
    """Error storing data in database"""
    pass


class LLMError(QueryUnderstandingError):
    """Error with LLM interaction"""

    def __init__(self, message: str, model: str = None, prompt_length: int = None):
        super().__init__(
            message,
            {"model": model, "prompt_length": prompt_length}
        )


class ParsingError(QueryUnderstandingError):
    """Error parsing responses"""

    def __init__(self, message: str, raw_response: str = None):
        # Truncate response for logging
        truncated = raw_response[:200] + "..." if raw_response and len(raw_response) > 200 else raw_response
        super().__init__(
            message,
            {"raw_response": truncated}
        )
