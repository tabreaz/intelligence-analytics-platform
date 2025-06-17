# src/agents/time_parser/models.py
"""
Data models for Time Parser Agent
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ConstraintType(str, Enum):
    """Type of constraint - inclusion or exclusion"""
    INCLUDE = "include"
    EXCLUDE = "exclude"


class DateGranularity(str, Enum):
    """Granularity for date expansion"""
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"


@dataclass
class DateRange:
    """Represents a parsed date/time range"""
    type: str  # 'absolute' or 'relative'
    start: str  # ISO format datetime
    end: str  # ISO format datetime
    original_text: str
    confidence: float = 1.0
    constraint_type: ConstraintType = ConstraintType.INCLUDE
    expand_to_dates: bool = False  # If true, expand to individual dates


@dataclass
class ExpandedDates:
    """Individual dates expanded from ranges"""
    dates: List[str]  # List of ISO format dates
    source_range: DateRange
    granularity: DateGranularity = DateGranularity.DAY


@dataclass
class HourConstraint:
    """Represents hour-of-day constraints"""
    start_hour: int  # 0-23
    end_hour: int  # 0-23
    original_text: str
    days_applicable: Optional[List[str]] = None  # ['monday', 'tuesday', etc.]
    constraint_type: ConstraintType = ConstraintType.INCLUDE
    excluded_hours: Optional[List[int]] = None  # Specific hours to exclude


@dataclass
class DayConstraint:
    """Enhanced day constraints with exclusion support"""
    days: List[str]  # ['monday', 'tuesday', 'weekend', etc.]
    original_text: str
    constraint_type: ConstraintType = ConstraintType.INCLUDE


@dataclass
class EventMapping:
    """Maps events or special dates to specific date ranges"""
    event_name: str
    dates: List[str]  # Individual dates for the event
    original_text: str


@dataclass
class TimeParsingResult:
    """Complete result from time parsing"""
    has_time_expressions: bool
    date_ranges: List[DateRange]
    excluded_date_ranges: List[DateRange]  # Ranges to exclude
    expanded_dates: Optional[List[ExpandedDates]]  # Individual dates
    hour_constraints: List[HourConstraint]
    day_constraints: List[DayConstraint]  # Enhanced with include/exclude
    event_mappings: Optional[List[EventMapping]]  # Event-based dates
    composite_constraints: Optional[Dict[str, Any]]  # Complex combined constraints
    default_range: Optional[Dict[str, Any]]
    summary: str
    raw_expressions: List[str]  # Original time expressions found
    parsing_confidence: float
    sql_hints: Optional[Dict[str, Any]] = None  # Hints for SQL generation


@dataclass
class TimeContext:
    """Context for time parsing"""
    current_datetime: datetime
    timezone: str = "UTC"
    previous_time_ranges: Optional[List[DateRange]] = None
    query_context: Optional[Dict[str, Any]] = None
    known_events: Optional[Dict[str, List[str]]] = None  # Event name -> dates mapping
