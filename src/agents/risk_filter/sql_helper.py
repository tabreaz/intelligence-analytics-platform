# src/agents/risk_filter/sql_helper.py
"""
SQL helper for Risk Filter Agent - converts filters to SQL conditions
"""
import re
from typing import List, Optional

from .models import ScoreFilter, RiskFilterResult


class RiskFilterSQLHelper:
    """Converts risk filters to SQL WHERE conditions"""

    @staticmethod
    def build_sql_conditions(result: RiskFilterResult) -> List[str]:
        """Build SQL WHERE conditions from risk filter result
        
        Args:
            result: RiskFilterResult with extracted filters
            
        Returns:
            List of SQL condition strings
        """
        conditions = []

        # Add inclusion conditions
        # Risk scores
        for field, score_filter in result.risk_scores.items():
            condition = RiskFilterSQLHelper._build_score_condition(field, score_filter)
            if condition:
                conditions.append(condition)

        # Flags
        for field, value in result.flags.items():
            conditions.append(f"{field} = {str(value).lower()}")

        # Crime categories
        if result.crime_categories and result.crime_categories.include:
            category_conditions = []
            for category in result.crime_categories.include:
                category_conditions.append(f"has(crime_categories_en, '{category}')")
            if category_conditions:
                conditions.append(f"({' OR '.join(category_conditions)})")

        # Add exclusion conditions
        # Excluded risk scores
        for field, score_filter in result.exclude_scores.items():
            condition = RiskFilterSQLHelper._build_score_condition(field, score_filter, is_exclusion=True)
            if condition:
                conditions.append(condition)

        # Excluded flags
        for field, value in result.exclude_flags.items():
            conditions.append(f"NOT ({field} = {str(value).lower()})")

        # Excluded crime categories
        if result.crime_categories and result.crime_categories.exclude:
            for category in result.crime_categories.exclude:
                conditions.append(f"NOT has(crime_categories_en, '{category}')")

        # Deduplicate conditions before returning
        return RiskFilterSQLHelper._deduplicate_conditions(conditions)

    @staticmethod
    def _build_score_condition(field: str, score_filter: ScoreFilter,
                               is_exclusion: bool = False) -> Optional[str]:
        """Build SQL condition for a score filter
        
        Args:
            field: Field name
            score_filter: Score filter object
            is_exclusion: Whether this is an exclusion condition
            
        Returns:
            SQL condition string or None
        """
        try:
            operator = score_filter.operator.value
            value = score_filter.value

            if operator == "BETWEEN":
                if score_filter.value2 is not None:
                    condition = f"{field} BETWEEN {value} AND {score_filter.value2}"
                else:
                    return None
            else:
                condition = f"{field} {operator} {value}"

            if is_exclusion:
                condition = f"NOT ({condition})"

            return condition

        except Exception as e:
            return None

    @staticmethod
    def _deduplicate_conditions(conditions: List[str]) -> List[str]:
        """Remove duplicate or conflicting conditions
        
        Args:
            conditions: List of SQL conditions
            
        Returns:
            Deduplicated list of conditions
        """
        # Track seen conditions
        seen_conditions = {}
        deduped = []

        for condition in conditions:
            # Extract field name from condition
            field_match = re.match(r'(\w+)\s*[><=!]', condition)
            if field_match:
                field = field_match.group(1)

                # Check for conflicting boolean conditions
                if condition.endswith('= true') or condition.endswith('= false'):
                    # Skip if we already have a condition for this boolean field
                    if field in seen_conditions and seen_conditions[field].endswith(('= true', '= false')):
                        # Keep the non-NOT version
                        if not condition.startswith('NOT'):
                            # Replace the previous condition
                            deduped = [c for c in deduped if not (field in c and c != condition)]
                            deduped.append(condition)
                            seen_conditions[field] = condition
                        continue

                seen_conditions[field] = condition

            if condition not in deduped:
                deduped.append(condition)

        return deduped

    @staticmethod
    def build_complete_where_clause(result: RiskFilterResult) -> str:
        """Build complete WHERE clause from risk filters
        
        Args:
            result: RiskFilterResult with extracted filters
            
        Returns:
            Complete WHERE clause string (without 'WHERE' keyword)
        """
        conditions = RiskFilterSQLHelper.build_sql_conditions(result)

        if not conditions:
            return "1=1"  # Always true condition

        return " AND ".join(conditions)

    @staticmethod
    def get_required_columns(result: RiskFilterResult) -> List[str]:
        """Get list of columns required for the risk filters
        
        Args:
            result: RiskFilterResult with extracted filters
            
        Returns:
            List of column names
        """
        columns = set()

        # Add risk score columns
        columns.update(result.risk_scores.keys())
        columns.update(result.exclude_scores.keys())

        # Add flag columns
        columns.update(result.flags.keys())
        columns.update(result.exclude_flags.keys())

        # Add crime categories column if needed
        if result.crime_categories:
            columns.add('crime_categories_en')

        return list(columns)
