# src/agents/query_understanding/error_handler.py
"""
Error handling utilities for query understanding
"""
import asyncio
import functools
from typing import Callable, Dict, Any, TypeVar, Optional

from src.core.logger import get_logger
from .exceptions import (
    QueryUnderstandingError, ValidationError, ParsingError
)

logger = get_logger(__name__)

T = TypeVar('T')


def handle_errors(
        error_return_value: Optional[Any] = None,
        log_level: str = "error",
        include_traceback: bool = True
):
    """
    Decorator to handle errors consistently
    
    Args:
        error_return_value: Value to return on error (None by default)
        log_level: Logging level for errors
        include_traceback: Whether to log full traceback
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except QueryUnderstandingError as e:
                # Our custom exceptions - log with details
                getattr(logger, log_level)(
                    f"{func.__name__} failed: {str(e)}",
                    extra={"details": e.details},
                    exc_info=include_traceback
                )
                return error_return_value
            except Exception as e:
                # Unexpected exceptions - always log as error
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                return error_return_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except QueryUnderstandingError as e:
                # Our custom exceptions - log with details
                getattr(logger, log_level)(
                    f"{func.__name__} failed: {str(e)}",
                    extra={"details": e.details},
                    exc_info=include_traceback
                )
                return error_return_value
            except Exception as e:
                # Unexpected exceptions - always log as error
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                return error_return_value

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ErrorContext:
    """Context manager for error handling with cleanup"""

    def __init__(
            self,
            operation: str,
            cleanup_func: Optional[Callable] = None,
            reraise: bool = True
    ):
        self.operation = operation
        self.cleanup_func = cleanup_func
        self.reraise = reraise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error during {self.operation}: {exc_val}",
                exc_info=True
            )

            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    if asyncio.iscoroutinefunction(self.cleanup_func):
                        await self.cleanup_func()
                    else:
                        self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")

            # Reraise if configured
            return not self.reraise

        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error during {self.operation}: {exc_val}",
                exc_info=True
            )

            # Run cleanup if provided
            if self.cleanup_func and not asyncio.iscoroutinefunction(self.cleanup_func):
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")

            # Reraise if configured
            return not self.reraise

        return False


def convert_exception(from_type: type, to_type: type):
    """Decorator to convert one exception type to another"""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except from_type as e:
                raise to_type(str(e)) from e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except from_type as e:
                raise to_type(str(e)) from e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def format_error_response(
        error: Exception,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format error into standardized response
    
    Returns:
        Dictionary with error details
    """
    response = {
        "status": "error",
        "error": {
            "message": str(error),
            "type": type(error).__name__
        }
    }

    if query_id:
        response["query_id"] = query_id

    if session_id:
        response["session_id"] = session_id

    # Add specific error details based on type
    if isinstance(error, QueryUnderstandingError):
        response["error"]["details"] = error.details

        if isinstance(error, ValidationError):
            response["error"]["field"] = error.field
            response["error"]["invalid_value"] = error.value
        elif isinstance(error, ParsingError):
            # Don't include raw response in user-facing error
            response["error"]["details"] = {"parsing_failed": True}

    return response
