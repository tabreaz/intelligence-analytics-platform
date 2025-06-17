# src/agents/time_parser/exceptions.py
"""
Custom exceptions for Time Parser Agent
"""


class TimeParserError(Exception):
    """Base exception for time parser errors"""
    pass


class InvalidTimeExpressionError(TimeParserError):
    """Raised when time expression cannot be parsed"""
    pass


class TimeRangeError(TimeParserError):
    """Raised when time range is invalid (e.g., end before start)"""
    pass


class TimezoneMismatchError(TimeParserError):
    """Raised when timezone handling fails"""
    pass
