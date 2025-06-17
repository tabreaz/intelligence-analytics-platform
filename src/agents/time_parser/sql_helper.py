# src/agents/time_parser/sql_helper.py
"""
SQL generation helpers for Time Parser results

Demonstrates how to convert time parsing results into SQL WHERE clauses
"""
from typing import Dict, Any, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class TimeSQLHelper:
    """
    Helper class to generate SQL conditions from time parsing results
    """

    @classmethod
    def generate_where_clause(
            cls,
            time_result: Dict[str, Any],
            timestamp_column: str = "timestamp",
            use_expanded_dates: bool = False
    ) -> str:
        """
        Generate SQL WHERE clause from time parsing result
        
        Args:
            time_result: Result from TimeParserAgent
            timestamp_column: Name of the timestamp column
            use_expanded_dates: Whether to use expanded dates array
            
        Returns:
            SQL WHERE clause string
        """
        conditions = []

        # Handle date ranges
        date_conditions = cls._generate_date_conditions(
            time_result.get('date_ranges', []),
            time_result.get('excluded_date_ranges', []),
            time_result.get('expanded_dates', []),
            timestamp_column,
            use_expanded_dates
        )
        if date_conditions:
            conditions.append(date_conditions)

        # Handle hour constraints
        hour_conditions = cls._generate_hour_conditions(
            time_result.get('hour_constraints', []),
            timestamp_column
        )
        if hour_conditions:
            conditions.append(hour_conditions)

        # Handle day constraints
        day_conditions = cls._generate_day_conditions(
            time_result.get('day_constraints', []),
            timestamp_column
        )
        if day_conditions:
            conditions.append(day_conditions)

        # Handle composite constraints
        composite_conditions = cls._generate_composite_conditions(
            time_result.get('composite_constraints'),
            timestamp_column
        )
        if composite_conditions:
            conditions.append(composite_conditions)

        # Combine all conditions
        if conditions:
            return " AND ".join(f"({cond})" for cond in conditions)
        else:
            # Default range if no conditions
            if time_result.get('default_range'):
                days_back = time_result['default_range'].get('days_back', 2)
                return f"{timestamp_column} >= now() - interval '{days_back} days'"
            return "1=1"  # No time constraints

    @classmethod
    def _generate_date_conditions(
            cls,
            date_ranges: List[Dict[str, Any]],
            excluded_ranges: List[Dict[str, Any]],
            expanded_dates: Optional[List[Dict[str, Any]]],
            column: str,
            use_expanded: bool
    ) -> str:
        """Generate date range conditions"""
        conditions = []

        # Use expanded dates if available and requested
        if use_expanded and expanded_dates:
            all_dates = []
            for ed in expanded_dates:
                all_dates.extend(ed.get('dates', []))

            if all_dates:
                # For small sets, use IN clause
                if len(all_dates) <= 30:
                    date_list = ", ".join(f"'{d}'" for d in all_dates)
                    return f"date({column}) IN ({date_list})"
                else:
                    # For larger sets, fall back to range conditions
                    logger.info(f"Too many dates ({len(all_dates)}) for IN clause, using ranges")

        # Include ranges
        for dr in date_ranges:
            if dr.get('constraint_type', 'include') == 'include':
                conditions.append(
                    f"({column} >= '{dr['start']}' AND {column} <= '{dr['end']}')"
                )

        # Exclude ranges
        exclude_conditions = []
        for dr in excluded_ranges:
            exclude_conditions.append(
                f"({column} < '{dr['start']}' OR {column} > '{dr['end']}')"
            )

        # Combine includes
        include_clause = " OR ".join(conditions) if conditions else ""

        # Combine excludes
        exclude_clause = " AND ".join(exclude_conditions) if exclude_conditions else ""

        # Combine both
        if include_clause and exclude_clause:
            return f"({include_clause}) AND ({exclude_clause})"
        elif include_clause:
            return include_clause
        elif exclude_clause:
            return exclude_clause
        else:
            return ""

    @classmethod
    def _generate_hour_conditions(
            cls,
            hour_constraints: List[Dict[str, Any]],
            column: str
    ) -> str:
        """Generate hour constraint conditions"""
        include_conditions = []
        exclude_conditions = []

        for hc in hour_constraints:
            constraint_type = hc.get('constraint_type', 'include')

            # Base hour condition
            hour_cond = f"extract(hour from {column}) >= {hc['start_hour']} AND extract(hour from {column}) <= {hc['end_hour']}"

            # Add day applicability if specified
            if hc.get('days_applicable'):
                day_conds = []
                for day in hc['days_applicable']:
                    if day == 'weekday':
                        day_conds.append(f"extract(dow from {column}) BETWEEN 1 AND 5")
                    elif day == 'weekend':
                        day_conds.append(f"extract(dow from {column}) IN (0, 6)")
                    else:
                        # Map day name to number (Sunday=0)
                        day_num = cls._day_to_number(day)
                        if day_num is not None:
                            day_conds.append(f"extract(dow from {column}) = {day_num}")

                if day_conds:
                    hour_cond = f"({hour_cond} AND ({' OR '.join(day_conds)}))"

            # Handle excluded hours
            if hc.get('excluded_hours'):
                exclude_hours_cond = f"extract(hour from {column}) NOT IN ({', '.join(map(str, hc['excluded_hours']))})"
                hour_cond = f"({hour_cond} AND {exclude_hours_cond})"

            if constraint_type == 'include':
                include_conditions.append(hour_cond)
            else:
                exclude_conditions.append(f"NOT ({hour_cond})")

        # Combine conditions
        conditions = []
        if include_conditions:
            conditions.append(" OR ".join(f"({c})" for c in include_conditions))
        if exclude_conditions:
            conditions.append(" AND ".join(exclude_conditions))

        return " AND ".join(f"({c})" for c in conditions) if conditions else ""

    @classmethod
    def _generate_day_conditions(
            cls,
            day_constraints: List[Dict[str, Any]],
            column: str
    ) -> str:
        """Generate day constraint conditions"""
        include_conditions = []
        exclude_conditions = []

        for dc in day_constraints:
            constraint_type = dc.get('constraint_type', 'include')
            days = dc.get('days', [])

            day_conds = []
            for day in days:
                if day == 'weekday':
                    day_conds.append(f"extract(dow from {column}) BETWEEN 1 AND 5")
                elif day == 'weekend':
                    day_conds.append(f"extract(dow from {column}) IN (0, 6)")
                else:
                    day_num = cls._day_to_number(day)
                    if day_num is not None:
                        day_conds.append(f"extract(dow from {column}) = {day_num}")

            if day_conds:
                combined = " OR ".join(f"({c})" for c in day_conds)
                if constraint_type == 'include':
                    include_conditions.append(combined)
                else:
                    exclude_conditions.append(f"NOT ({combined})")

        # Combine conditions
        conditions = []
        if include_conditions:
            conditions.append(" OR ".join(f"({c})" for c in include_conditions))
        if exclude_conditions:
            conditions.append(" AND ".join(exclude_conditions))

        return " AND ".join(f"({c})" for c in conditions) if conditions else ""

    @classmethod
    def _generate_composite_conditions(
            cls,
            composite_constraints: Optional[Dict[str, Any]],
            column: str
    ) -> str:
        """Generate composite constraint conditions"""
        if not composite_constraints:
            return ""

        conditions = []

        # Day-hour combinations
        if 'day_hour_combinations' in composite_constraints:
            combo_conditions = []
            for combo in composite_constraints['day_hour_combinations']:
                # Day condition
                day_conds = []
                for day in combo.get('days', []):
                    day_num = cls._day_to_number(day)
                    if day_num is not None:
                        day_conds.append(f"extract(dow from {column}) = {day_num}")

                # Hour condition
                hour_cond = f"extract(hour from {column}) >= {combo['start_hour']} AND extract(hour from {column}) <= {combo['end_hour']}"

                # Combine day and hour
                if day_conds:
                    combo_cond = f"(({' OR '.join(day_conds)}) AND {hour_cond})"
                    combo_conditions.append(combo_cond)

            if combo_conditions:
                conditions.append(" OR ".join(f"({c})" for c in combo_conditions))

        # Nth occurrence (e.g., "first Monday of month")
        if 'nth_occurrence' in composite_constraints:
            nth = composite_constraints['nth_occurrence']
            # This would require a more complex SQL expression
            # Example: "date_part('day', timestamp) <= 7 AND extract(dow from timestamp) = 1"
            logger.info("Nth occurrence constraints require custom SQL implementation")

        # Periodic constraints
        if 'periodic' in composite_constraints:
            periodic = composite_constraints['periodic']
            # Example: "mod(date_part('day', timestamp - '2024-01-01'::date), 3) = 0"
            logger.info("Periodic constraints require custom SQL implementation")

        return " AND ".join(f"({c})" for c in conditions) if conditions else ""

    @classmethod
    def _day_to_number(cls, day_name: str) -> Optional[int]:
        """Convert day name to PostgreSQL DOW number (Sunday=0)"""
        day_map = {
            'sunday': 0,
            'monday': 1,
            'tuesday': 2,
            'wednesday': 3,
            'thursday': 4,
            'friday': 5,
            'saturday': 6
        }
        return day_map.get(day_name.lower())

    @classmethod
    def generate_example_query(
            cls,
            time_result: Dict[str, Any],
            table_name: str = "events",
            timestamp_column: str = "timestamp"
    ) -> str:
        """
        Generate a complete example SQL query
        
        Args:
            time_result: Result from TimeParserAgent
            table_name: Name of the table to query
            timestamp_column: Name of the timestamp column
            
        Returns:
            Complete SQL query string
        """
        where_clause = cls.generate_where_clause(time_result, timestamp_column)

        # Determine if we should use expanded dates
        sql_hints = time_result.get('sql_hints', {})
        use_expanded = sql_hints.get('use_date_array', False)

        # Build query based on hints
        if sql_hints.get('suggested_approach') == 'date_in_array' and use_expanded:
            # Use date array approach
            query = f"""
SELECT 
    date({timestamp_column}) as date,
    count(*) as count
FROM {table_name}
WHERE {where_clause}
GROUP BY 1
ORDER BY 1;
"""
        elif sql_hints.get('use_complex_where'):
            # Complex conditions with hour/day filters
            query = f"""
SELECT 
    date({timestamp_column}) as date,
    extract(hour from {timestamp_column}) as hour,
    count(*) as count
FROM {table_name}
WHERE {where_clause}
GROUP BY 1, 2
ORDER BY 1, 2;
"""
        else:
            # Standard date range query
            query = f"""
SELECT 
    date_trunc('day', {timestamp_column}) as day,
    count(*) as count
FROM {table_name}
WHERE {where_clause}
GROUP BY 1
ORDER BY 1;
"""

        return query.strip()
