# src/agents/time_parser/date_expander.py
"""
Date expansion utilities for Time Parser Agent

Handles expanding date ranges into individual dates and applying complex constraints
"""
from calendar import monthrange
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.core.logger import get_logger
from .models import DateRange, ExpandedDates, HourConstraint, DayConstraint, DateGranularity, ConstraintType

logger = get_logger(__name__)


class DateExpander:
    """
    Utility class for expanding date ranges and applying constraints
    """

    @classmethod
    def expand_date_range(
            cls,
            date_range: DateRange,
            granularity: DateGranularity = DateGranularity.DAY,
            day_constraints: Optional[List[DayConstraint]] = None,
            hour_constraints: Optional[List[HourConstraint]] = None
    ) -> ExpandedDates:
        """
        Expand a date range into individual dates/times based on granularity
        
        Args:
            date_range: The date range to expand
            granularity: Level of expansion (day, hour, minute)
            day_constraints: Day-level constraints to apply
            hour_constraints: Hour-level constraints to apply
            
        Returns:
            ExpandedDates object with individual dates/times
        """
        try:
            start_dt = datetime.fromisoformat(date_range.start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(date_range.end.replace('Z', '+00:00'))

            if granularity == DateGranularity.DAY:
                dates = cls._expand_by_day(start_dt, end_dt, day_constraints)
            elif granularity == DateGranularity.HOUR:
                dates = cls._expand_by_hour(start_dt, end_dt, day_constraints, hour_constraints)
            else:  # MINUTE
                dates = cls._expand_by_minute(start_dt, end_dt, day_constraints, hour_constraints)

            return ExpandedDates(
                dates=dates,
                source_range=date_range,
                granularity=granularity
            )

        except Exception as e:
            logger.error(f"Failed to expand date range: {e}")
            return ExpandedDates(dates=[], source_range=date_range, granularity=granularity)

    @classmethod
    def _expand_by_day(
            cls,
            start_dt: datetime,
            end_dt: datetime,
            day_constraints: Optional[List[DayConstraint]] = None
    ) -> List[str]:
        """Expand date range by day"""
        dates = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_dt:
            if cls._matches_day_constraints(current, day_constraints):
                dates.append(current.date().isoformat())
            current += timedelta(days=1)

        return dates

    @classmethod
    def _expand_by_hour(
            cls,
            start_dt: datetime,
            end_dt: datetime,
            day_constraints: Optional[List[DayConstraint]] = None,
            hour_constraints: Optional[List[HourConstraint]] = None
    ) -> List[str]:
        """Expand date range by hour"""
        dates = []
        current = start_dt.replace(minute=0, second=0, microsecond=0)

        while current <= end_dt:
            if (cls._matches_day_constraints(current, day_constraints) and
                    cls._matches_hour_constraints(current, hour_constraints)):
                dates.append(current.isoformat())
            current += timedelta(hours=1)

        return dates

    @classmethod
    def _expand_by_minute(
            cls,
            start_dt: datetime,
            end_dt: datetime,
            day_constraints: Optional[List[DayConstraint]] = None,
            hour_constraints: Optional[List[HourConstraint]] = None
    ) -> List[str]:
        """Expand date range by minute"""
        dates = []
        current = start_dt.replace(second=0, microsecond=0)

        # Limit to prevent excessive expansion
        max_minutes = 24 * 60 * 7  # One week max
        count = 0

        while current <= end_dt and count < max_minutes:
            if (cls._matches_day_constraints(current, day_constraints) and
                    cls._matches_hour_constraints(current, hour_constraints)):
                dates.append(current.isoformat())
            current += timedelta(minutes=1)
            count += 1

        if count >= max_minutes:
            logger.warning(f"Minute expansion limited to {max_minutes} entries")

        return dates

    @classmethod
    def _matches_day_constraints(
            cls,
            dt: datetime,
            constraints: Optional[List[DayConstraint]] = None
    ) -> bool:
        """Check if datetime matches day constraints"""
        if not constraints:
            return True

        day_name = dt.strftime('%A').lower()
        day_num = dt.weekday()

        # Separate include and exclude constraints
        include_constraints = [c for c in constraints if c.constraint_type == ConstraintType.INCLUDE]
        exclude_constraints = [c for c in constraints if c.constraint_type == ConstraintType.EXCLUDE]

        # Check exclusions first
        for constraint in exclude_constraints:
            if cls._day_matches_constraint(day_name, day_num, constraint.days):
                return False

        # If no include constraints, default is to include
        if not include_constraints:
            return True

        # Check inclusions
        for constraint in include_constraints:
            if cls._day_matches_constraint(day_name, day_num, constraint.days):
                return True

        return False

    @classmethod
    def _day_matches_constraint(cls, day_name: str, day_num: int, constraint_days: List[str]) -> bool:
        """Check if a day matches constraint days"""
        for day in constraint_days:
            if day == day_name:
                return True
            elif day == 'weekday' and day_num < 5:
                return True
            elif day == 'weekend' and day_num >= 5:
                return True
        return False

    @classmethod
    def _matches_hour_constraints(
            cls,
            dt: datetime,
            constraints: Optional[List[HourConstraint]] = None
    ) -> bool:
        """Check if datetime matches hour constraints"""
        if not constraints:
            return True

        hour = dt.hour

        # Separate include and exclude constraints
        include_constraints = [c for c in constraints if c.constraint_type == ConstraintType.INCLUDE]
        exclude_constraints = [c for c in constraints if c.constraint_type == ConstraintType.EXCLUDE]

        # Check exclusions first
        for constraint in exclude_constraints:
            # Check excluded specific hours
            if constraint.excluded_hours and hour in constraint.excluded_hours:
                return False
            # Check excluded hour ranges
            if constraint.start_hour <= hour <= constraint.end_hour:
                # Check if this applies to current day
                if not constraint.days_applicable or cls._day_matches_constraint(
                        dt.strftime('%A').lower(),
                        dt.weekday(),
                        constraint.days_applicable
                ):
                    return False

        # If no include constraints, default is to include
        if not include_constraints:
            return True

        # Check inclusions
        for constraint in include_constraints:
            if constraint.start_hour <= hour <= constraint.end_hour:
                # Check if this applies to current day
                if not constraint.days_applicable or cls._day_matches_constraint(
                        dt.strftime('%A').lower(),
                        dt.weekday(),
                        constraint.days_applicable
                ):
                    return True

        return False

    @classmethod
    def apply_composite_constraints(
            cls,
            dates: List[str],
            composite_constraints: Dict[str, Any]
    ) -> List[str]:
        """
        Apply complex composite constraints
        
        Example: "only Fridays between 12-14 hrs for last 3 months"
        """
        if not composite_constraints:
            return dates

        filtered_dates = []

        for date_str in dates:
            dt = datetime.fromisoformat(date_str)

            # Check all composite constraints
            should_include = True

            # Specific day + hour combinations
            if 'day_hour_combinations' in composite_constraints:
                matches_combo = False
                for combo in composite_constraints['day_hour_combinations']:
                    if (dt.strftime('%A').lower() in combo['days'] and
                            combo['start_hour'] <= dt.hour <= combo['end_hour']):
                        matches_combo = True
                        break
                if not matches_combo:
                    should_include = False

            # Nth occurrence constraints (e.g., "first Monday of each month")
            if should_include and 'nth_occurrence' in composite_constraints:
                nth_config = composite_constraints['nth_occurrence']
                if not cls._matches_nth_occurrence(dt, nth_config):
                    should_include = False

            # Periodic constraints (e.g., "every 3rd day")
            if should_include and 'periodic' in composite_constraints:
                periodic_config = composite_constraints['periodic']
                if not cls._matches_periodic(dt, periodic_config):
                    should_include = False

            if should_include:
                filtered_dates.append(date_str)

        return filtered_dates

    @classmethod
    def _matches_nth_occurrence(cls, dt: datetime, config: Dict[str, Any]) -> bool:
        """Check if date matches Nth occurrence pattern"""
        # Example: "first Monday of month"
        if 'day' not in config or 'nth' not in config:
            return True

        target_day = config['day'].lower()
        nth = config['nth']

        # Get first day of month
        first_day = dt.replace(day=1)

        # Find the Nth occurrence
        occurrence_count = 0
        for day in range(1, monthrange(dt.year, dt.month)[1] + 1):
            check_date = dt.replace(day=day)
            if check_date.strftime('%A').lower() == target_day:
                occurrence_count += 1
                if occurrence_count == nth and check_date.day == dt.day:
                    return True

        return False

    @classmethod
    def _matches_periodic(cls, dt: datetime, config: Dict[str, Any]) -> bool:
        """Check if date matches periodic pattern"""
        # Example: "every 3rd day starting from X"
        if 'interval' not in config or 'start_date' not in config:
            return True

        start_dt = datetime.fromisoformat(config['start_date'])
        interval_days = config['interval']

        days_diff = (dt.date() - start_dt.date()).days
        return days_diff % interval_days == 0

    @classmethod
    def generate_sql_hints(
            cls,
            expanded_dates: Optional[List[ExpandedDates]],
            composite_constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate hints for SQL generation based on time constraints
        """
        hints = {
            'use_date_array': False,
            'use_hour_filter': False,
            'use_day_of_week': False,
            'use_complex_where': False,
            'suggested_approach': 'date_range'
        }

        if not expanded_dates:
            return hints

        # Analyze patterns
        total_dates = sum(len(ed.dates) for ed in expanded_dates)

        if total_dates > 0 and total_dates <= 30:
            hints['use_date_array'] = True
            hints['suggested_approach'] = 'date_in_array'
        elif composite_constraints:
            hints['use_complex_where'] = True
            hints['suggested_approach'] = 'complex_conditions'

            if 'day_hour_combinations' in composite_constraints:
                hints['use_day_of_week'] = True
                hints['use_hour_filter'] = True

        return hints
